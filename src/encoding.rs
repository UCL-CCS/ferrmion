//! `pyclass` wrapper for the core [`MajoranaEncoding`] type.

use crate::error::CoreError;
use crate::hamiltonians::{PyFermionHamiltonian, PyQubitHamiltonian};
use crate::operators::PyMajoranaSparse;
use ferrmion_core::encode::majorana::{Encode, MajoranaEncoding, TryEncode};
use ferrmion_core::encode::maxnto::maxnto_symplectic_matrix;
use ferrmion_core::encode::ternarytree::{TTFlatpack, TernaryTree};
use ferrmion_core::hamiltonians::QubitHamiltonian;
use ferrmion_core::operators::{FermionProduct, LadderOperator, MajoranaSparse, SymplecticMatrix};
use ferrmion_core::optimise::{anneal_enumerations, AnnealingParameters};
use ferrmion_core::states::{FockState, State, ZBasisEnsemble, ZBasisState};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use numpy::{Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::HashSet;

/// A fermion-to-qubit encoding defined by its Majorana operator representations,
/// backed by the Rust [`MajoranaEncoding`] type.
///
#[pyclass(name = "MajoranaEncoding", module = "ferrmion.core")]
#[derive(Clone, Debug)]
pub struct PyMajoranaEncoding(pub MajoranaEncoding);

/// The operator types accepted by [`PyMajoranaEncoding::encode`].
///
/// Python has no overload resolution and `#[pymethods]` cannot expose a generic
/// method, so the two accepted types are dispatched at runtime by trying each
/// variant's extraction in turn.
#[derive(FromPyObject)]
enum EncodeInput<'py> {
    #[pyo3(annotation = "FermionHamiltonian")]
    Fermion(PyRef<'py, PyFermionHamiltonian>),
    #[pyo3(annotation = "MajoranaSparse")]
    Majorana(PyRef<'py, PyMajoranaSparse>),
}

/// Build a [`MajoranaEncoding`] from the `[x|z]` numpy exchange layout.
///
/// When `vacuum_state` is `None` the vacuum is determined automatically via
/// GF(2) constraint solving.
fn encoding_from_parts(
    ipowers: ArrayView1<u8>,
    symplectics: ArrayView2<bool>,
    vacuum_state: Option<ArrayView1<bool>>,
) -> Result<MajoranaEncoding, CoreError> {
    if ipowers.len() != symplectics.nrows() {
        return Err(CoreError::Value(
            "ipowers and symplectics must be same length.".to_string(),
        ));
    }
    let n_qubits = symplectics.ncols() / 2;
    let x_block = symplectics.slice(s![.., ..n_qubits]).to_owned();
    let z_block = symplectics.slice(s![.., n_qubits..]).to_owned();
    let ipowers = ipowers.mapv(|v| v % 4);
    let matrix = SymplecticMatrix::from_arrays_with_ipowers(x_block, z_block, ipowers);
    let encoding = match vacuum_state {
        Some(vacuum) => MajoranaEncoding::with_vacuum(
            matrix,
            ZBasisState::new(vacuum.to_owned(), Complex64::ONE),
        )?,
        None => MajoranaEncoding::new(matrix)?,
    };
    Ok(encoding)
}

/// Reconstruct a core [`TernaryTree`] from a flatpack, dispatching between the
/// full and naive (leaf-free) flatpack forms.
pub(crate) fn tree_from_flatpack(flatpack: &TTFlatpack) -> Result<(TernaryTree, usize), CoreError> {
    let flatpack_max_qubit_index: &usize = flatpack
        .iter()
        .map(|(v, _)| v)
        .max()
        .ok_or_else(|| CoreError::Value("Flatpack must be non-empty".to_string()))?;

    let mut empty_leaves: usize = 0;
    for (_, children) in flatpack {
        empty_leaves += usize::from(children.0.is_none())
            + usize::from(children.1.is_none())
            + usize::from(children.2.is_none());
    }
    let tree = match empty_leaves {
        1 => TernaryTree::from_flatpack(flatpack)?,
        v if v == 2 * flatpack.len() + 1 => TernaryTree::from_flatpack_naive(flatpack)?,
        _ => {
            return Err(CoreError::Value(
                "TTFlatpack must have no leaves, or 2*n_modes + 1 leaves.".to_string(),
            ));
        }
    };
    Ok((tree, flatpack_max_qubit_index + 1))
}

impl PyMajoranaEncoding {
    /// Encode a product of ladder operators, optionally summed with its
    /// reversed-index hermitian conjugate.
    fn encode_product_impl(
        &self,
        signature: &str,
        mode_indices: Vec<i64>,
        coeff: Complex64,
        with_conjugate: bool,
    ) -> Result<QubitHamiltonian, CoreError> {
        let n_modes = self.0.n_modes as i64;
        if mode_indices.iter().any(|&i| i < 0 || i >= n_modes) {
            return Err(CoreError::Value("Indices invalid.".to_string()));
        }
        if signature.len() != mode_indices.len() {
            return Err(CoreError::Value(
                "Signature and indices must be same length.".to_string(),
            ));
        }
        let action: Vec<LadderOperator> = signature
            .chars()
            .map(|c| {
                LadderOperator::try_from(c)
                    .map_err(|_| CoreError::Value(format!("Invalid signature character: '{c}'")))
            })
            .collect::<Result<_, _>>()?;
        let indices: Vec<usize> = mode_indices.into_iter().map(|i| i as usize).collect();

        let fproduct = FermionProduct::new(action.clone(), indices.clone(), coeff)?;
        let mut qham = self.0.encode(fproduct);
        if with_conjugate {
            let conjugate =
                FermionProduct::new(action, indices.into_iter().rev().collect(), coeff)?;
            let conjugate_qham: QubitHamiltonian = self.0.encode(conjugate);
            for (key, value) in conjugate_qham.0 {
                *qham.0.entry(key).or_insert(Complex64::ZERO) += value;
            }
            qham.0.retain(|_, v| *v != Complex64::ZERO);
        }
        Ok(qham)
    }

    /// Check that every Majorana index in `hamiltonian` addresses a row of this
    /// encoding's symplectic matrix.
    ///
    /// [`MajoranaSparse`] carries no mode count of its own, and the core encode
    /// indexes `operators` unguarded, so an out-of-range index would panic inside
    /// a rayon worker rather than raise. Terms are ordered lexicographically
    /// rather than by magnitude, so the whole index set has to be scanned; the
    /// cost is negligible beside the symplectic multiplies it guards.
    fn check_majorana_indices(&self, hamiltonian: &MajoranaSparse) -> Result<(), CoreError> {
        let n_operators = 2 * self.0.n_modes;
        match hamiltonian
            .indices
            .iter()
            .flat_map(|term| term.iter())
            .copied()
            .max()
        {
            Some(max_index) if max_index as usize >= n_operators => Err(CoreError::Value(format!(
                "MajoranaSparse has Majorana index {max_index} but encoding has {} modes \
                 ({n_operators} Majorana operators).",
                self.0.n_modes
            ))),
            _ => Ok(()),
        }
    }
}

#[pymethods]
impl PyMajoranaEncoding {
    /// Construct an encoding from explicit Majorana string data.
    ///
    /// Args:
    ///     ipowers: 1D uint8 array of phase exponents (taken mod 4).
    ///     symplectics: 2D boolean array of shape ``(2*n_modes, 2*n_qubits)``,
    ///         left half X-block, right half Z-block.
    ///     `vacuum_state`: Optional 1D boolean array of length ``n_qubits``.
    ///         When omitted, the vacuum is determined automatically.
    #[new]
    #[pyo3(signature = (ipowers, symplectics, vacuum_state = None))]
    fn new(
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        vacuum_state: Option<PyReadonlyArray1<bool>>,
    ) -> Result<Self, CoreError> {
        Ok(Self(encoding_from_parts(
            ipowers.as_array(),
            symplectics.as_array(),
            vacuum_state.as_ref().map(numpy::PyReadonlyArray::as_array),
        )?))
    }

    /// The Jordan-Wigner encoding for ``n_modes`` fermionic modes.
    #[staticmethod]
    #[pyo3(signature = (n_modes, n_qubits = None))]
    fn jordan_wigner(n_modes: usize, n_qubits: Option<usize>) -> Result<Self, CoreError> {
        Ok(Self(
            TernaryTree::naive_jordan_wigner(n_modes)
                .build_encoding(n_qubits.unwrap_or(n_modes))?,
        ))
    }

    /// The Bravyi-Kitaev encoding for ``n_modes`` fermionic modes.
    #[staticmethod]
    #[pyo3(signature = (n_modes, n_qubits = None))]
    fn bravyi_kitaev(n_modes: usize, n_qubits: Option<usize>) -> Result<Self, CoreError> {
        Ok(Self(
            TernaryTree::naive_bravyi_kitaev(n_modes)
                .build_encoding(n_qubits.unwrap_or(n_modes))?,
        ))
    }

    /// The parity encoding for ``n_modes`` fermionic modes.
    #[staticmethod]
    #[pyo3(signature = (n_modes, n_qubits = None))]
    fn parity(n_modes: usize, n_qubits: Option<usize>) -> Result<Self, CoreError> {
        Ok(Self(
            TernaryTree::naive_parity(n_modes).build_encoding(n_qubits.unwrap_or(n_modes))?,
        ))
    }

    /// The JKMN (minimum-height ternary tree) encoding for ``n_modes`` modes.
    #[staticmethod]
    #[pyo3(signature = (n_modes, n_qubits = None))]
    fn jkmn(n_modes: usize, n_qubits: Option<usize>) -> Result<Self, CoreError> {
        Ok(Self(
            TernaryTree::naive_jkmn(n_modes).build_encoding(n_qubits.unwrap_or(n_modes))?,
        ))
    }

    /// The `MaxNTO` k-NTO encoding for ``n_modes`` fermionic modes.
    ///
    /// Requires ``n_modes - 1`` to be odd.
    #[staticmethod]
    fn maxnto(n_modes: usize) -> Result<Self, CoreError> {
        let (_y_count, output) = maxnto_symplectic_matrix(n_modes)?;
        let n_qubits = output.ncols() / 2;
        let x_block = output.slice(s![.., ..n_qubits]).to_owned();
        let z_block = output.slice(s![.., n_qubits..]).to_owned();
        Ok(Self(MajoranaEncoding::new(SymplecticMatrix::from_arrays(
            x_block, z_block,
        ))?))
    }

    /// Build an encoding from a ternary-tree flatpack representation.
    ///
    /// Args:
    ///     flatpack: List of ``(qubit_index, (x_child, y_child, z_child))`` tuples.
    ///     `n_qubits`: Optional number of qubits; defaults to the number of tree nodes.
    #[staticmethod]
    #[pyo3(signature = (flatpack, n_qubits = None))]
    fn from_flatpack(flatpack: TTFlatpack, n_qubits: Option<usize>) -> Result<Self, CoreError> {
        let (tree, default_n_qubits) = tree_from_flatpack(&flatpack)?;
        let n_qubits = n_qubits.unwrap_or(default_n_qubits);
        if n_qubits < default_n_qubits - 1 {
            return Err(CoreError::Value(
                "Passed value of n_qubits less than existing flatpack qubit index.".to_string(),
            ));
        }
        Ok(Self(tree.build_encoding(n_qubits)?))
    }

    /// Reconstruct an encoding from the dictionary produced by ``to_json``.
    #[staticmethod]
    fn from_json(data: &Bound<'_, PyDict>) -> Result<Self, CoreError> {
        let ipowers: Vec<u8> = data
            .get_item("ipowers")?
            .ok_or_else(|| CoreError::Value("Missing 'ipowers' key.".to_string()))?
            .extract()?;
        let symplectics: Vec<Vec<bool>> = data
            .get_item("symplectics")?
            .ok_or_else(|| CoreError::Value("Missing 'symplectics' key.".to_string()))?
            .extract()?;
        let vacuum_state: Option<Vec<bool>> = data
            .get_item("vacuum_state")?
            .map(|v| v.extract())
            .transpose()?;

        let nrows = symplectics.len();
        let ncols = symplectics.first().map_or(0, Vec::len);
        if symplectics.iter().any(|row| row.len() != ncols) {
            return Err(CoreError::Value(
                "All symplectic rows must have equal length.".to_string(),
            ));
        }
        let flat: Vec<bool> = symplectics.into_iter().flatten().collect();
        let symplectics = Array2::from_shape_vec((nrows, ncols), flat)
            .map_err(|e| CoreError::Value(e.to_string()))?;
        Ok(Self(encoding_from_parts(
            Array1::from(ipowers).view(),
            symplectics.view(),
            vacuum_state
                .map(Array1::from)
                .as_ref()
                .map(ndarray::ArrayBase::view),
        )?))
    }

    /// Serialise the encoding to a JSON-compatible dictionary with
    /// ``"ipowers"``, ``"symplectics"`` and ``"vacuum_state"`` keys.
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        // Widen u8 -> u32 so PyO3 produces a list of ints rather than `bytes`
        // (Vec<u8> converts to Python bytes).
        let ipowers: Vec<u32> = self
            .0
            .operators
            .ipowers
            .iter()
            .map(|&v| u32::from(v))
            .collect();
        output.set_item("ipowers", ipowers)?;
        let symplectics: Vec<Vec<bool>> = self
            .0
            .operators
            .to_concatenated()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        output.set_item("symplectics", symplectics)?;
        output.set_item("vacuum_state", self.0.vacuum_state.state_bools().to_vec())?;
        Ok(output)
    }

    /// Number of fermionic modes.
    #[getter]
    fn n_modes(&self) -> usize {
        self.0.n_modes
    }

    /// Number of qubits.
    #[getter]
    fn n_qubits(&self) -> usize {
        self.0.n_qubits
    }

    /// Phase exponents (mod 4) for each Majorana operator row.
    #[getter]
    fn ipowers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.0.operators.ipowers.to_owned().into_pyarray(py)
    }

    /// The symplectic matrix in ``[x_block | z_block]`` layout, of shape
    /// ``(2*n_modes, 2*n_qubits)``.
    #[getter]
    fn symplectic_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        self.0.operators.to_concatenated().into_pyarray(py)
    }

    /// The vacuum state in the Z basis.
    #[getter]
    fn vacuum_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.0.vacuum_state.state_bools().into_pyarray(py)
    }

    /// Encode a fermionic operator into a qubit Hamiltonian.
    ///
    /// Args:
    ///     operator: The operator to encode, either a ``FermionHamiltonian`` or a
    ///         ``MajoranaSparse``. A ``FermionHamiltonian`` is converted to its
    ///         Majorana representation first; passing a ``MajoranaSparse``
    ///         directly skips that conversion.
    ///
    /// Returns:
    ///     The encoded ``QubitHamiltonian``.
    ///
    /// Raises:
    ///     ValueError: If the operator does not match the mode count of this encoding.
    fn encode(
        &self,
        py: Python<'_>,
        operator: EncodeInput<'_>,
    ) -> Result<PyQubitHamiltonian, CoreError> {
        let qham = match operator {
            EncodeInput::Fermion(fham) => {
                let fham_n_modes = fham.inner.n_modes();
                if fham_n_modes != 0 && fham_n_modes != self.0.n_modes {
                    return Err(CoreError::Value(format!(
                        "FermionHamiltonian has {fham_n_modes} modes but encoding has {} modes.",
                        self.0.n_modes
                    )));
                }
                let hamiltonian = fham.inner.to_majorana_sparse();
                py.allow_threads(|| self.0.encode(&hamiltonian))
            }
            EncodeInput::Majorana(msparse) => {
                let hamiltonian: &MajoranaSparse = &msparse.0;
                self.check_majorana_indices(hamiltonian)?;
                py.allow_threads(|| self.0.encode(hamiltonian))
            }
        };
        Ok(PyQubitHamiltonian(qham))
    }

    /// Encode a Hamiltonian, optimising mode enumeration via simulated annealing.
    ///
    /// Args:
    ///     fham: The fermionic Hamiltonian to encode.
    ///     temperature: Initial annealing temperature. Defaults to ``n_modes // 2``.
    ///     `initial_guess`: Starting permutation. Defaults to identity.
    ///     `coefficient_weighted`: If ``True``, minimise coefficient-weighted Pauli weight.
    ///     seed: Seed for the RNG driving permutation moves. Defaults to ``1017``.
    ///
    /// Returns:
    ///     Tuple of ``(QubitHamiltonian, MajoranaEncoding)`` — the encoded
    ///     Hamiltonian and the encoding with the optimised mode enumeration.
    #[pyo3(signature = (fham, temperature = None, initial_guess = None, coefficient_weighted = true, seed = None))]
    fn encode_annealed(
        &mut self,
        py: Python<'_>,
        fham: PyRef<'_, PyFermionHamiltonian>,
        temperature: Option<f64>,
        initial_guess: Option<Vec<usize>>,
        coefficient_weighted: bool,
        seed: Option<usize>,
    ) -> Result<PyQubitHamiltonian, CoreError> {
        let hamiltonian = fham.inner.to_majorana_sparse();
        let temperature = temperature.unwrap_or((fham.inner.n_modes() / 2) as f64);
        let initial_guess =
            Array1::from(initial_guess.unwrap_or_else(|| (0..self.0.n_modes).collect()));
        let params = AnnealingParameters::new(temperature, 1000, seed.unwrap_or(1017));

        let qham = py.allow_threads(|| -> Result<_, CoreError> {
            let mut anneal_hamiltonian = hamiltonian.clone();
            anneal_hamiltonian.constant = 0.0;
            let (_, best_mode_enumeration) = anneal_enumerations(
                anneal_hamiltonian,
                self.0.clone(),
                params,
                initial_guess.view(),
                coefficient_weighted,
            )
            .map_err(|e| CoreError::Runtime(e.to_string()))?;
            self.0 = self
                .0
                .apply_mode_enumeration(best_mode_enumeration.to_vec());
            let qham = self.0.encode(&hamiltonian);
            Ok(qham)
        })?;
        Ok(PyQubitHamiltonian(qham))
    }

    /// Optimise the mode enumeration via simulated annealing without encoding.
    ///
    /// Args:
    ///     fham: The fermionic Hamiltonian whose Pauli weight drives the search.
    ///     temperature: Initial annealing temperature. Defaults to ``n_modes``.
    ///     `initial_guess`: Starting permutation. Defaults to identity.
    ///     `coefficient_weighted`: If ``True``, minimise coefficient-weighted Pauli weight.
    ///     seed: Seed for the RNG driving permutation moves. Defaults to ``1017``.
    ///
    /// Returns:
    ///     Tuple of ``(best_cost, MajoranaEncoding)``.
    #[pyo3(signature = (fham, temperature = None, initial_guess = None, coefficient_weighted = false, seed = None))]
    fn anneal_enumeration(
        &mut self,
        py: Python<'_>,
        fham: PyRef<'_, PyFermionHamiltonian>,
        temperature: Option<f64>,
        initial_guess: Option<Vec<usize>>,
        coefficient_weighted: bool,
        seed: Option<usize>,
    ) -> Result<f64, CoreError> {
        let mut hamiltonian = fham.inner.to_majorana_sparse();
        hamiltonian.constant = 0.0;
        let temperature = temperature.unwrap_or(fham.inner.n_modes() as f64);
        let initial_guess =
            Array1::from(initial_guess.unwrap_or_else(|| (0..self.0.n_modes).collect()));
        let params = AnnealingParameters::new(temperature, 1000, seed.unwrap_or(1017));

        let cost = py.allow_threads(|| -> Result<_, CoreError> {
            let (cost, best_mode_enumeration) = anneal_enumerations(
                hamiltonian,
                self.0.clone(),
                params,
                initial_guess.view(),
                coefficient_weighted,
            )
            .map_err(|e| CoreError::Runtime(e.to_string()))?;
            self.0 = self
                .0
                .apply_mode_enumeration(best_mode_enumeration.to_vec());
            Ok(cost)
        })?;
        Ok(cost)
    }

    /// Decode an ensemble of Z-basis states into fermionic occupation vectors.
    ///
    /// Args:
    ///     states: 2D boolean array of shape ``(n_states, n_qubits)``.
    ///
    /// Returns:
    ///     2D boolean array of shape ``(n_states, n_modes)``.
    ///
    /// Raises:
    ///     `ValueError`: if any state cannot be decoded for this encoding.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        states: PyReadonlyArray2<'py, bool>,
    ) -> Result<Bound<'py, PyArray2<bool>>, CoreError> {
        let states = states.as_array().to_owned();
        let n_states = states.nrows();
        let ensemble = ZBasisEnsemble::new(states, Array1::from_elem(n_states, Complex64::ONE));
        let results = py.allow_threads(|| self.0.decode_zbasis_ensemble(&ensemble));
        let mut occupations = Array2::<bool>::default((n_states, self.0.n_modes));
        for (j, result) in results.into_iter().enumerate() {
            match result {
                Some(fock) => occupations.row_mut(j).assign(&fock.state),
                None => {
                    return Err(CoreError::Value(format!(
                        "state at index {j} could not be decoded"
                    )))
                }
            }
        }
        Ok(occupations.into_pyarray(py))
    }

    /// Compute the Hartree-Fock state in the encoded basis.
    ///
    /// Args:
    ///     `fermionic_hf_state`: 1D boolean array of mode occupations.
    ///     `mode_op_map`: Optional permutation mapping modes to operator pairs.
    ///
    /// Returns:
    ///     1D boolean array — the qubit Hartree-Fock state in the Z basis.
    #[pyo3(signature = (fermionic_hf_state, mode_op_map = None))]
    fn hartree_fock_state<'py>(
        &self,
        py: Python<'py>,
        fermionic_hf_state: PyReadonlyArray1<'py, bool>,
        mode_op_map: Option<PyReadonlyArray1<'py, usize>>,
    ) -> Result<Bound<'py, PyArray1<bool>>, CoreError> {
        let mut fockstate =
            FockState::new(fermionic_hf_state.as_array().to_owned(), Complex64::ONE);
        if let Some(mode_op_map) = mode_op_map {
            let mode_op_map = mode_op_map.as_array().to_vec();
            fockstate.reindex(&mode_op_map);
        }
        let state = self.0.try_encode(fockstate)?;
        Ok(state.state_bools().into_pyarray(py))
    }

    /// The encoded number operator of a mode.
    #[pyo3(signature = (mode, coeff = Complex64::new(1.0, 0.0)))]
    fn number_operator(
        &self,
        mode: i64,
        coeff: Complex64,
    ) -> Result<PyQubitHamiltonian, CoreError> {
        Ok(PyQubitHamiltonian(self.encode_product_impl(
            "+-",
            vec![mode, mode],
            coeff,
            false,
        )?))
    }

    /// The encoded edge operator of a pair of modes.
    #[pyo3(signature = (edge_indices, coeff = Complex64::new(1.0, 0.0), with_conjugate = false))]
    fn edge_operator(
        &self,
        edge_indices: (i64, i64),
        coeff: Complex64,
        with_conjugate: bool,
    ) -> Result<PyQubitHamiltonian, CoreError> {
        Ok(PyQubitHamiltonian(self.encode_product_impl(
            "+-",
            vec![edge_indices.0, edge_indices.1],
            coeff,
            with_conjugate,
        )?))
    }

    /// The encoded interaction operator of four modes.
    #[pyo3(signature = (mode_indices, coeff = Complex64::new(1.0, 0.0), physicist_notation = true, with_conjugate = false))]
    fn interaction_operator(
        &self,
        mode_indices: (i64, i64, i64, i64),
        coeff: Complex64,
        physicist_notation: bool,
        with_conjugate: bool,
    ) -> Result<PyQubitHamiltonian, CoreError> {
        let signature = if physicist_notation { "++--" } else { "+-+-" };
        Ok(PyQubitHamiltonian(self.encode_product_impl(
            signature,
            vec![
                mode_indices.0,
                mode_indices.1,
                mode_indices.2,
                mode_indices.3,
            ],
            coeff,
            with_conjugate,
        )?))
    }

    /// Encode a single product of ladder operators.
    ///
    /// Args:
    ///     signature: The fermionic operator signature, composed of "+" and "-".
    ///     `mode_indices`: The mode index for each ladder operator.
    ///     coeff: The operator coefficient.
    ///     `with_conjugate`: Also add the reversed-index hermitian conjugate.
    #[pyo3(signature = (signature, mode_indices, coeff = Complex64::new(1.0, 0.0), with_conjugate = false))]
    fn encode_fermion_product(
        &self,
        signature: &str,
        mode_indices: Vec<i64>,
        coeff: Complex64,
        with_conjugate: bool,
    ) -> Result<PyQubitHamiltonian, CoreError> {
        Ok(PyQubitHamiltonian(self.encode_product_impl(
            signature,
            mode_indices,
            coeff,
            with_conjugate,
        )?))
    }

    /// Compute plain and coefficient-weighted Pauli weights for a batch of
    /// mode permutations in a single parallelised call.
    ///
    /// Args:
    ///     fham: The fermionic Hamiltonian to weigh.
    ///     permutations: 2D uint array of shape ``(n_perms, n_modes)``.
    ///
    /// Returns:
    ///     Tuple ``(plain, weighted)`` of two 1D float64 arrays.
    #[allow(clippy::type_complexity)]
    fn batch_pauli_weights<'py>(
        &self,
        py: Python<'py>,
        fham: PyRef<'_, PyFermionHamiltonian>,
        permutations: PyReadonlyArray2<'py, usize>,
    ) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>), CoreError> {
        let mut hamiltonian = fham.inner.to_majorana_sparse();
        hamiltonian.constant = 0.0;
        let perms: Vec<Vec<usize>> = permutations
            .as_array()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        let (plain, weighted) =
            py.allow_threads(|| self.0.batch_pauli_weights(&hamiltonian, &perms));
        Ok((
            Array1::from(plain).into_pyarray(py),
            Array1::from(weighted).into_pyarray(py),
        ))
    }

    /// Return a new encoding with the fermionic modes reordered.
    fn apply_mode_enumeration(&self, mode_op_map: Vec<usize>) -> Result<Self, CoreError> {
        let expected: HashSet<usize> = (0..self.0.n_modes).collect();
        let got: HashSet<usize> = mode_op_map.iter().copied().collect();
        if mode_op_map.len() != self.0.n_modes || got != expected {
            return Err(CoreError::Value(
                "mode_op_map must be a permutation of range(n_modes).".to_string(),
            ));
        }
        Ok(Self(self.0.apply_mode_enumeration(mode_op_map)))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|o| self.0 == o.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "MajoranaEncoding(n_modes={}, n_qubits={})",
            self.0.n_modes, self.0.n_qubits
        )
    }

    #[allow(clippy::type_complexity)]
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(
        Bound<'py, PyType>,
        (
            Bound<'py, PyArray1<u8>>,
            Bound<'py, PyArray2<bool>>,
            Bound<'py, PyArray1<bool>>,
        ),
    )> {
        let py = slf.py();
        let this = slf.borrow_mut();
        Ok((
            slf.get_type(),
            (
                this.ipowers(py),
                this.symplectic_matrix(py),
                this.vacuum_state(py),
            ),
        ))
    }
}
