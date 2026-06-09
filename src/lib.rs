//! Fast, reliable and easy optimisation of fermion-qubit encodings.
//!
//! To simulate fermionic Hamiltonians with gate-based quantum computers,
//! it is necessary to encode the fermionic operators to qubit operators
//! which obey commutation fermionic relations.
//!
//! This file contains the PyO3 interop layer which wraps rust functions and exposes
//! these to a python API

use ferrmion_core::encode::majorana::{Encode, MajoranaEncoding, MajoranaEncodingError, TryEncode};
use ferrmion_core::encode::maxnto::{maxnto_symplectic_matrix, MaxNTOError};
use ferrmion_core::encode::ternarytree::{TTFlatpack, TernaryTree, TernaryTreeError};
use ferrmion_core::hamiltonians::{QubitHamiltonian, SymplecticHamiltonian};
use ferrmion_core::operators::{
    FermionProduct, FermionProductError, LadderOperator, MajoranaSparse, SymplecticMatrix,
    SymplecticOperator,
};
use ferrmion_core::optimise::*;
use ferrmion_core::states::{FockState, State, ZBasisEnsemble, ZBasisState};
use ferrmion_core::utils::*;
use log::debug;
use ndarray::s;
use numpy::ndarray::{Array1, Array2};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::{IntoPyDict, PyComplex, PyDict, PyInt, PyString, PyTuple};
use pyo3::{prelude::*, pymodule, Bound};
use std::collections::HashMap;
use std::str::FromStr;

/// Local error type bridging `ferrmion_core` errors to `PyErr`.
///
/// The orphan rule prevents `impl From<ForeignError> for PyErr` when both
/// types come from external crates. This local type acts as a bridge:
/// `impl From<CoreError> for PyErr` is allowed (local → foreign), and
/// `impl From<ForeignError> for CoreError` is allowed (foreign → local).
#[derive(Debug)]
enum CoreError {
    Value(String),
    Runtime(String),
    Py(PyErr),
}

impl From<CoreError> for PyErr {
    fn from(e: CoreError) -> PyErr {
        match e {
            CoreError::Value(s) => PyValueError::new_err(s),
            CoreError::Runtime(s) => PyRuntimeError::new_err(s),
            CoreError::Py(e) => e,
        }
    }
}

impl From<PyErr> for CoreError {
    fn from(e: PyErr) -> Self {
        CoreError::Py(e)
    }
}

impl From<MajoranaEncodingError> for CoreError {
    fn from(e: MajoranaEncodingError) -> Self {
        CoreError::Value(e.to_string())
    }
}

impl From<TernaryTreeError> for CoreError {
    fn from(e: TernaryTreeError) -> Self {
        CoreError::Value(e.to_string())
    }
}

impl From<ToppHattError> for CoreError {
    fn from(e: ToppHattError) -> Self {
        CoreError::Runtime(e.to_string())
    }
}

impl From<HattError> for CoreError {
    fn from(e: HattError) -> Self {
        CoreError::Runtime(e.to_string())
    }
}

impl From<FermionProductError> for CoreError {
    fn from(_: FermionProductError) -> Self {
        CoreError::Value(
            "Invalid FermionProduct: operators and indices must have equal length".to_string(),
        )
    }
}

impl From<MaxNTOError> for CoreError {
    fn from(e: MaxNTOError) -> Self {
        CoreError::Value(e.to_string())
    }
}

impl From<CliffordHeuristicError> for CoreError {
    fn from(e: CliffordHeuristicError) -> Self {
        CoreError::Value(e.to_string())
    }
}

/// Apply γ²=1 simplification, merge duplicate Majorana keys, and drop
/// terms whose summed coefficient falls below the near-zero threshold.
///
/// Both `fermionic_to_sparse_majorana` and `wrap_hatt` consume the result;
/// going through a single helper guarantees they see identical term sets
/// (a per-entry accumulate-and-filter loop can leave stale entries when a
/// running coefficient cancels below the threshold).
fn simplified_majorana_terms(
    hamiltonian: MajoranaSparse,
) -> std::collections::BTreeMap<Vec<u16>, Complex64> {
    let mut merged: std::collections::BTreeMap<Vec<u16>, Complex64> =
        std::collections::BTreeMap::new();
    for (key, val) in std::iter::zip(hamiltonian.indices, hamiltonian.coefficients) {
        let mut simplified: Vec<u16> = Vec::with_capacity(key.len());
        for &idx in key.as_slice() {
            if simplified.last() == Some(&idx) {
                simplified.pop();
            } else {
                simplified.push(idx);
            }
        }
        if simplified.is_empty() {
            continue;
        }
        *merged.entry(simplified).or_insert(Complex64::new(0., 0.)) += val;
    }
    merged.retain(|_, v| v.norm() > 1e-16);
    merged
}

/// A Python module implemented in Rust.
#[allow(clippy::type_complexity)]
#[pymodule]
#[pyo3(name = "core")]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    debug!("Initializing Python module 'core'");

    /// Compute the symplectic product of two Pauli operators in symplectic representation.
    ///
    /// Each operator is encoded as a 1D boolean array of length ``2n`` where the first
    /// ``n`` entries are the X-block and the last ``n`` entries are the Z-block.
    ///
    /// Args:
    ///     left: 1D boolean numpy array — symplectic representation of the left operator.
    ///     right: 1D boolean numpy array — symplectic representation of the right operator.
    ///
    /// Returns:
    ///     Tuple of ``(ipower, product)`` where ``ipower`` is the accumulated phase exponent
    ///     (i.e. the overall phase is ``i**ipower``) and ``product`` is the symplectic
    ///     representation of the resulting operator.
    ///
    /// Example:
    ///     ```python
    ///     import ferrmion
    ///     import numpy as np
    ///     a = np.array([True, False, True, False])
    ///     b = np.array([False, True, False, True])
    ///     ipower, product = ferrmion.symplectic_product(a, b)
    ///     ```
    #[pyfn(m)]
    #[pyo3(name = "symplectic_product")]
    fn wrap_symplectic_product_py<'py>(
        py: Python<'py>,
        left: PyReadonlyArray1<bool>,
        right: PyReadonlyArray1<bool>,
    ) -> Result<(usize, Bound<'py, PyArray1<bool>>), CoreError> {
        let left = left.as_array();
        let right = right.as_array();
        let n = left.len() / 2;
        let left_op = SymplecticOperator::new(
            0,
            left.slice(ndarray::s![..n]).to_owned(),
            left.slice(ndarray::s![n..]).to_owned(),
        );
        let right_op = SymplecticOperator::new(
            0,
            right.slice(ndarray::s![..n]).to_owned(),
            right.slice(ndarray::s![n..]).to_owned(),
        );
        let result = left_op * right_op.view();
        let combined =
            ndarray::concatenate(ndarray::Axis(0), &[result.x_block(), result.z_block()])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            result.ipower() as usize,
            PyArray1::from_owned_array(py, combined),
        ))
    }

    /// Compute the Hartree-Fock state in the ternary-tree encoding basis.
    ///
    /// Applies the ternary-tree encoding operators to the fermionic Hartree-Fock
    /// occupation vector, returning the corresponding qubit state expressed as
    /// a superposition over Z-basis states.
    ///
    /// Args:
    ///     fermionic_hf_state: 1D boolean array — fermionic occupation vector
    ///         (``True`` = occupied, ``False`` = unoccupied).
    ///     mode_op_map: 1D uint array mapping fermionic modes to encoding operators.
    ///     ipowers: 1D uint8 array of phase exponents for each encoding operator.
    ///     symplectic_matrix: 2D boolean array of shape ``(2*n_modes, 2*n_qubits)``
    ///         representing the full symplectic encoding matrix.
    ///
    /// Returns:
    ///     1D boolean array — the qubit Hartree-Fock state in the Z basis.
    ///
    /// Example:
    ///     ```python
    ///     import ferrmion
    ///     import numpy as np
    ///     hf = np.array([True, True, False, False, False, False])
    ///     mode_op_map = np.array([0, 1, 2, 3, 4, 5])
    ///     ipowers = np.zeros(6, dtype=np.uint8)
    ///     symplectic = np.eye(6, 12, dtype=bool)
    ///     state = ferrmion.hartree_fock_state(hf, mode_op_map, ipowers, symplectic)
    ///     ```
    #[pyfn(m)]
    #[pyo3(name = "hartree_fock_state")]
    fn wrap_hartree_fock_state<'py>(
        py: Python<'py>,
        fermionic_hf_state: PyReadonlyArray1<bool>,
        mode_op_map: PyReadonlyArray1<usize>,
        ipowers: PyReadonlyArray1<u8>,
        symplectic_matrix: PyReadonlyArray2<bool>,
        vacuum_state: PyReadonlyArray1<bool>,
    ) -> Result<Bound<'py, PyArray1<bool>>, CoreError> {
        let fermionic_hf_state = fermionic_hf_state.as_array();
        let mode_op_map = mode_op_map.as_array();
        let symplectic_matrix = symplectic_matrix.as_array();
        let n_qubits = symplectic_matrix.ncols() / 2;
        let x_block = symplectic_matrix
            .slice(ndarray::s![.., ..n_qubits])
            .to_owned();
        let z_block = symplectic_matrix
            .slice(ndarray::s![.., n_qubits..])
            .to_owned();
        let ipowers = ipowers.as_array().to_owned();
        let vacuum = ZBasisState::new(
            Array1::from(vacuum_state.as_array().to_vec()),
            num_complex::Complex::ONE,
        );
        let encoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::with_ipowers(x_block, z_block, ipowers),
            vacuum,
        )?;
        let mut fockstate = FockState::new(
            Array1::from(fermionic_hf_state.to_vec()),
            num_complex::Complex::ONE,
        );
        fockstate.reindex(
            mode_op_map.as_slice().ok_or_else(|| {
                PyValueError::new_err("mode_op_map must be a contiguous 1-D array")
            })?,
        );
        let zstate = encoding.try_encode(fockstate);
        match zstate {
            Ok(state) => Ok(PyArray1::from_owned_array(py, state.state)),
            Err(e) => Err(e.into()),
        }
    }

    /// Decode an ensemble of Z-basis states into fermionic occupation vectors.
    ///
    /// Args:
    ///     states: 2D boolean array of shape ``(n_states, n_qubits)``.  Each row
    ///         is a Z-basis computational-basis state.
    ///     ipowers: 1D uint8 array of length ``2 * n_modes`` — phase exponents for
    ///         each Majorana operator row.
    ///     symplectic_matrix: 2D boolean array of shape
    ///         ``(2 * n_modes, 2 * n_qubits)``.  Left half is the X-block, right
    ///         half is the Z-block (same layout as all other functions here).
    ///     vacuum_state: 1D boolean array of length ``n_qubits``.
    ///
    /// Returns:
    ///     2D boolean array of shape ``(n_states, n_modes)``.  Row ``j`` is the
    ///     fermionic occupation vector decoded from ``states[j]``.
    ///
    /// Raises:
    ///     ValueError: if any state cannot be decoded (i.e., does not correspond
    ///         to a valid encoded Fock state for this encoding).
    #[pyfn(m)]
    #[pyo3(name = "decode")]
    fn wrap_decode<'py>(
        py: Python<'py>,
        states: PyReadonlyArray2<bool>,
        ipowers: PyReadonlyArray1<u8>,
        symplectic_matrix: PyReadonlyArray2<bool>,
        vacuum_state: PyReadonlyArray1<bool>,
    ) -> Result<Bound<'py, PyArray2<bool>>, CoreError> {
        let states = states.as_array().to_owned();
        let n_states = states.nrows();
        let symplectic_matrix = symplectic_matrix.as_array();
        let n_qubits = symplectic_matrix.ncols() / 2;
        let x_block = symplectic_matrix
            .slice(ndarray::s![.., ..n_qubits])
            .to_owned();
        let z_block = symplectic_matrix
            .slice(ndarray::s![.., n_qubits..])
            .to_owned();
        let ipowers = ipowers.as_array().to_owned();
        let vacuum = ZBasisState::new(
            Array1::from(vacuum_state.as_array().to_vec()),
            num_complex::Complex::ONE,
        );
        let encoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::with_ipowers(x_block, z_block, ipowers),
            vacuum,
        )?;
        let ensemble = ZBasisEnsemble::new(
            states,
            Array1::from_elem(n_states, num_complex::Complex::ONE),
        );
        let results = encoding.decode_zbasis_ensemble(&ensemble);
        let n_modes = encoding.n_modes;
        let mut occupations = numpy::ndarray::Array2::<bool>::default((n_states, n_modes));
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
        Ok(PyArray2::from_owned_array(py, occupations))
    }

    /// Convert a symplectic operator representation to a Pauli string.
    ///
    /// Args:
    ///     symplectic: 1D boolean array of length ``2n`` (X-block then Z-block).
    ///     ipower: Phase exponent — overall phase is ``i**ipower``.
    ///
    /// Returns:
    ///     Tuple of ``(pauli_string, ipower)`` where ``pauli_string`` is a string
    ///     over ``{I, X, Y, Z}`` of length ``n``.
    #[pyfn(m)]
    #[pyo3(name = "symplectic_to_pauli")]
    fn wrap_symplectic_to_pauli<'py>(
        py: Python<'py>,
        symplectic: PyReadonlyArray1<bool>,
        ipower: u8,
    ) -> Result<(Bound<'py, PyString>, Bound<'py, PyInt>), CoreError> {
        let symplectic = symplectic.as_array();
        let n = symplectic.len() / 2;
        let op = SymplecticOperator::new(
            ipower,
            symplectic.slice(ndarray::s![..n]).to_owned(),
            symplectic.slice(ndarray::s![n..]).to_owned(),
        );
        let (pauli, ipower) = op.to_pauli_string();
        Ok((PyString::new(py, &pauli), PyInt::new(py, ipower)))
    }

    /// Convert a Pauli string to symplectic representation.
    ///
    /// Args:
    ///     pauli: Pauli string over ``{I, X, Y, Z}``.
    ///     ipower: Phase exponent — overall phase is ``i**ipower``.
    ///
    /// Returns:
    ///     Tuple of ``(symplectic, ipower)`` where ``symplectic`` is a 1D boolean
    ///     array of length ``2n``.
    #[pyfn(m)]
    #[pyo3(name = "pauli_to_symplectic")]
    fn wrap_pauli_to_symplectic(
        py: Python<'_>,
        pauli: String,
        ipower: usize,
    ) -> Result<(Bound<'_, PyArray1<bool>>, Bound<'_, PyInt>), CoreError> {
        let (symplectic, ipower) = pauli_to_symplectic(pauli, ipower);
        Ok((
            PyArray1::from_owned_array(py, symplectic),
            PyInt::new(py, ipower),
        ))
    }

    /// Convert a symplectic operator to a sparse matrix representation.
    ///
    /// Args:
    ///     symplectic: 1D boolean array of length ``2n``.
    ///     ipower: Phase exponent — overall phase is ``i**ipower``.
    ///
    /// Returns:
    ///     Tuple of ``(pauli_string, positions, coefficient)`` where ``positions``
    ///     is a 1D uint array of non-zero column indices and ``coefficient`` is the
    ///     complex scalar weight.
    #[pyfn(m)]
    #[pyo3(name = "symplectic_to_sparse")]
    fn wrap_symplectic_to_sparse<'py>(
        py: Python<'py>,
        symplectic: PyReadonlyArray1<bool>,
        ipower: usize,
    ) -> Result<
        (
            Bound<'py, PyString>,
            Bound<'py, PyArray1<usize>>,
            Bound<'py, PyComplex>,
        ),
        CoreError,
    > {
        let symplectic = symplectic.as_array();
        let (pauli_string, position_vec, coeff) = symplectic_to_sparse(symplectic, ipower);
        Ok((
            PyString::new(py, &pauli_string),
            PyArray1::from_owned_array(py, position_vec),
            PyComplex::from_complex_bound(py, coeff),
        ))
    }

    /// Optimise the mode enumeration of a Majorana encoding by simulated annealing.
    ///
    /// Searches for the mode permutation that minimises the Pauli weight of the
    /// encoded Hamiltonian using a simulated annealing approach.
    ///
    /// Args:
    ///     ipowers: 1D uint8 array — phase exponents for each encoding operator.
    ///     symplectics: 2D boolean array — symplectic encoding matrix.
    ///     signatures: List of fermionic operator signature strings.
    ///     coeffs: List of coefficient arrays, one per signature.
    ///     temperature: Annealing temperature (higher = more random exploration).
    ///     initial_guess: 1D uint array — initial mode-to-qubit permutation.
    ///     coefficient_weighted: If ``True``, weight moves by Hamiltonian coefficients.
    ///     seed: Seed for the RNG driving permutation moves. Defaults to ``1017``.
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectics)`` for the best encoding found.
    #[allow(clippy::too_many_arguments)]
    #[pyfn(m)]
    #[pyo3(
        name = "anneal_enumerations",
        signature = (
            ipowers,
            symplectics,
            signatures,
            coeffs,
            temperature,
            initial_guess,
            coefficient_weighted,
            seed = None,
        ),
    )]
    fn wrap_anneal_enumerations<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        temperature: f64,
        initial_guess: PyReadonlyArray1<usize>,
        coefficient_weighted: bool,
        seed: Option<usize>,
    ) -> Result<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray2<bool>>), CoreError> {
        let initial_guess = initial_guess.as_array();

        let msparse = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            0.,
        );
        let symplectics = symplectics.as_array();
        let n_qubits = symplectics.ncols() / 2;
        let x_block = symplectics.slice(ndarray::s![.., ..n_qubits]).to_owned();
        let z_block = symplectics.slice(ndarray::s![.., n_qubits..]).to_owned();
        let ipowers = ipowers.as_array().to_owned();
        let encoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::with_ipowers(x_block.clone(), z_block.clone(), ipowers),
            ZBasisState::zeros(n_qubits),
        )?;
        let best_mode_enumeration: Array1<usize>;
        (_, best_mode_enumeration) = anneal_enumerations(
            msparse,
            encoding,
            AnnealingParameters::new(temperature, 1000, seed.unwrap_or(1017)),
            initial_guess,
            coefficient_weighted,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let encoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::new(x_block, z_block),
            ZBasisState::zeros(n_qubits),
        )?
        .apply_mode_enumeration(best_mode_enumeration.to_vec());

        let combined = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                encoding.operators.x_block.view(),
                encoding.operators.z_block.view(),
            ],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            encoding.operators.ipowers.into_pyarray(py),
            combined.into_pyarray(py),
        ))
    }

    /// Optimise a qubit Hamiltonian by searching over Clifford circuits.
    ///
    /// Runs a simulated annealing search over sequences of Clifford gates (H, S, CNOT)
    /// and returns the qubit Hamiltonian for the best circuit found.
    ///
    /// Args:
    ///     qham: Mapping from Pauli strings to complex coefficients.
    ///     n_qubits: Number of qubits the Hamiltonian acts on.
    ///     temperature: Annealing temperature (higher = more random exploration).
    ///     coefficient_weighted: If ``True``, minimise coefficient-weighted Pauli weight.
    ///     seed: Seed for the RNG driving gate choices. Defaults to ``1017``.
    ///     clifford_subset: Gate families to sample from. One of ``"all"``, ``"ch"``,
    ///         ``"cs"``, ``"chs"`` (default).
    ///
    /// Returns:
    ///     Dictionary mapping Pauli strings to complex coefficients.
    #[pyfn(m)]
    #[pyo3(
        name = "clifford_heuristic",
        signature = (qham, n_qubits, temperature, coefficient_weighted, seed = None, clifford_subset = "chs".to_string()),
    )]
    fn wrap_clifford_heuristic<'py>(
        py: Python<'py>,
        qham: HashMap<String, Complex64>,
        n_qubits: usize,
        temperature: f64,
        coefficient_weighted: bool,
        seed: Option<u64>,
        clifford_subset: String,
    ) -> Result<Bound<'py, PyDict>, CoreError> {
        let clifford_subset = CliffordSubset::from_str(&clifford_subset)?;
        let qham_rust = QubitHamiltonian(qham.into_iter().collect());
        let mut sym_ham = SymplecticHamiltonian::from_qubit_hamiltonian(&qham_rust, n_qubits);

        let result = clifford_heuristic_optimisation(
            &mut sym_ham,
            temperature,
            coefficient_weighted,
            seed.unwrap_or(1017),
            None,
            Some(clifford_subset),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        sym_ham.operators.apply_clifford_chain(&result.chain);
        Ok(sym_ham.to_qubit_hamiltonian().0.into_py_dict(py)?)
    }

    /// Iteratively optimise a qubit Hamiltonian by Clifford descent on randomly sampled subsystems.
    ///
    /// Args:
    ///     qham: Mapping from Pauli strings to complex coefficients.
    ///     n_qubits: Number of qubits the Hamiltonian acts on.
    ///     iterations: Number of subsystem-local Clifford descents to perform.
    ///     temperature: Annealing temperature used for each subsystem descent.
    ///     subsystem_dimension: Number of qubits in each sampled subsystem.
    ///     coefficient_weighted: If ``True``, minimise coefficient-weighted Pauli weight.
    ///     sampler: Subsystem sampling strategy. One of ``"full_system"``, ``"uniform"``, ``"hamming"``.
    ///     seed: Seed for the RNG. Defaults to ``1017``.
    ///     clifford_subset: Gate families to sample from. One of ``"all"``, ``"ch"``,
    ///         ``"cs"``, ``"chs"`` (default).
    ///
    /// Returns:
    ///     Dictionary mapping Pauli strings to complex coefficients.
    #[allow(clippy::too_many_arguments)]
    #[pyfn(m)]
    #[pyo3(
        name = "randomised_subsystem_descent",
        signature = (
            qham,
            n_qubits,
            iterations,
            temperature,
            subsystem_dimension,
            coefficient_weighted = false,
            sampler = "uniform".to_string(),
            seed = None,
            clifford_subset = "chs".to_string(),
        ),
    )]
    fn wrap_randomised_subsystem_descent<'py>(
        py: Python<'py>,
        qham: HashMap<String, Complex64>,
        n_qubits: usize,
        iterations: usize,
        temperature: f64,
        subsystem_dimension: usize,
        coefficient_weighted: bool,
        sampler: String,
        seed: Option<usize>,
        clifford_subset: String,
    ) -> Result<Bound<'py, PyDict>, CoreError> {
        let sampler = match sampler.as_str() {
            "full_system" => SubsystemSampler::FullSystem,
            "uniform" => SubsystemSampler::Uniform,
            "hamming" => SubsystemSampler::Hamming,
            other => {
                return Err(CoreError::Value(format!(
                    "unknown sampler '{other}'; expected one of full_system, uniform, hamming"
                )))
            }
        };
        let clifford_subset = CliffordSubset::from_str(&clifford_subset.to_lowercase())?;
        let qham_rust = QubitHamiltonian(qham.into_iter().collect());
        let sym_ham = SymplecticHamiltonian::from_qubit_hamiltonian(&qham_rust, n_qubits);
        let opt = randomised_subsystem_descent(
            sym_ham,
            AnnealingParameters::new(temperature, iterations, seed.unwrap_or(1017)),
            coefficient_weighted,
            sampler,
            subsystem_dimension,
            Some(clifford_subset),
        );
        Ok(opt.to_qubit_hamiltonian().0.into_py_dict(py)?)
    }

    /// Encode a fermionic Hamiltonian under multiple mode permutations and return
    /// both Pauli weight vectors in a single parallelised Rust call.
    ///
    /// This is equivalent to calling ``encode`` followed by computing the Pauli weights
    /// for each permutation individually, but significantly faster because all
    /// encodings run in parallel on the Rust side.
    ///
    /// Args:
    ///     ipowers: 1D uint8 array — phase exponents for each encoding operator.
    ///     symplectics: 2D boolean array — symplectic encoding matrix
    ///         of shape ``(2 * n_modes, 2 * n_qubits)``.
    ///     vacuum_state: 1D boolean array of length ``n_qubits``.
    ///     signatures: List of fermionic operator signature strings.
    ///     coeffs: List of coefficient arrays, one per signature.
    ///     permutations: 2D uint array of shape ``(n_perms, n_modes)``.
    ///         Each row is a permutation of ``range(n_modes)``.
    ///
    /// Returns:
    ///     Tuple ``(plain, weighted)`` of two 1D float64 arrays of length
    ///     ``n_perms``.  ``plain[i]`` is the plain Pauli weight and
    ///     ``weighted[i]`` is the coefficient-weighted Pauli weight for
    ///     permutation ``i``, in the same order as the input rows.
    #[pyfn(m)]
    #[pyo3(name = "batch_pauli_weights")]
    fn wrap_batch_pauli_weights<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        vacuum_state: PyReadonlyArray1<bool>,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        permutations: PyReadonlyArray2<usize>,
    ) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>), CoreError> {
        let symplectics = symplectics.as_array();
        let n_qubits = symplectics.ncols() / 2;
        let x_block = symplectics.slice(ndarray::s![.., ..n_qubits]).to_owned();
        let z_block = symplectics.slice(ndarray::s![.., n_qubits..]).to_owned();
        let ipowers = ipowers.as_array().to_owned();
        let vacuum = ZBasisState::new(
            Array1::from(vacuum_state.as_array().to_vec()),
            Complex64::ONE,
        );
        let encoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::with_ipowers(x_block, z_block, ipowers),
            vacuum,
        )?;
        let msparse = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            0.,
        );
        let permutations = permutations.as_array();
        let perms: Vec<Vec<usize>> = permutations.outer_iter().map(|row| row.to_vec()).collect();
        let (plain, weighted) = encoding.batch_pauli_weights(&msparse, &perms);
        Ok((
            PyArray1::from_owned_array(py, Array1::from(plain)),
            PyArray1::from_owned_array(py, Array1::from(weighted)),
        ))
    }

    /// Build a symplectic encoding matrix from a ternary-tree flatpack representation.
    ///
    /// Args:
    ///     flatpack: List of ``(qubit_index, (left, mid, right))`` tuples describing
    ///         the ternary tree structure.
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectic_matrix)`` for the encoding derived from
    ///     the given tree.
    #[pyfn(m)]
    #[pyo3(name = "flatpack_symplectic_matrix")]
    fn wrap_flatpack_symplectic_matrix(
        py: Python<'_>,
        flatpack: TTFlatpack,
        n_qubits: Option<usize>,
    ) -> Result<
        (
            Bound<'_, PyArray1<u8>>,
            Bound<'_, PyArray2<bool>>,
            Bound<'_, PyArray1<bool>>,
        ),
        CoreError,
    > {
        // ) -> PyResult<()> {
        let flatplack_max_qubit_index: &usize = flatpack
            .iter()
            .map(|(v, _)| v)
            .max()
            .ok_or_else(|| PyValueError::new_err("Flatpack must be non-empty"))?;

        if n_qubits.unwrap_or(*flatplack_max_qubit_index + 1) < *flatplack_max_qubit_index {
            return Err(CoreError::Value(
                "Passed value of n_qubits less than existing flatpack qubit index.".to_string(),
            ));
        }

        let mut empty_leaves: usize = 0;
        for (_, children) in flatpack.iter() {
            empty_leaves += children.0.is_none() as usize
                + children.1.is_none() as usize
                + children.2.is_none() as usize
        }
        let tree = match empty_leaves {
            1 => TernaryTree::from_flatpack(&flatpack)?,
            v if v == 2 * flatpack.len() + 1 => TernaryTree::from_flatpack_naive(&flatpack)?,
            _ => {
                return Err(CoreError::Value(
                    "TTFlatpack must have no leaves, or 2*n_modes + 1 leaves.".to_string(),
                ));
            }
        };

        debug!("Got Tree");
        let encoding = tree.build_encoding(n_qubits.unwrap_or(*flatplack_max_qubit_index + 1))?;
        debug!("Got encoding");

        debug!("Got qham");
        let combined = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                encoding.operators.x_block.view(),
                encoding.operators.z_block.view(),
            ],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            encoding.operators.ipowers.into_pyarray(py),
            combined.into_pyarray(py),
            encoding.vacuum_state.state.into_pyarray(py),
        ))
    }

    /// Convert a fermionic Hamiltonian to a sparse Majorana representation.
    ///
    /// Decomposes the fermionic Hamiltonian expressed via ladder operator signatures
    /// into a dictionary keyed by tuples of Majorana mode indices.
    ///
    /// Args:
    ///     signatures: List of fermionic operator signature strings (e.g. ``"+-+-"``).
    ///     coeffs: List of coefficient arrays, one per signature.
    ///     constant_energy: Constant energy offset to include in the result.
    ///
    /// Returns:
    ///     Dictionary mapping tuples of Majorana indices to complex coefficients.
    #[pyfn(m)]
    #[pyo3(name = "fermionic_to_sparse_majorana")]
    fn fermionic_to_sparse_majorana<'py>(
        py: Python<'py>,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        constant_energy: f64,
    ) -> Result<Bound<'py, PyDict>, CoreError> {
        if signatures.len() != coeffs.len() {
            return Err(CoreError::Value(
                "signatures and coefficients must have equal length".to_string(),
            ));
        }

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            constant_energy,
        );

        let output = PyDict::new(py);
        for (key, val) in simplified_majorana_terms(hamiltonian) {
            output.set_item(PyTuple::new(py, key.as_slice())?, val)?;
        }
        Ok(output)
    }

    /// Encode a single fermionic operator product into a qubit Hamiltonian.
    ///
    /// Args:
    ///     ipowers: 1D uint8 array — phase exponents for each encoding operator.
    ///     symplectics: 2D boolean array — symplectic encoding matrix.
    ///     signatures: Signature string for the fermionic product (e.g. ``"+-"``).
    ///     indices: Mode indices for each ladder operator in the product.
    ///     coeff: Complex coefficient for this operator product.
    ///
    /// Returns:
    ///     Dictionary mapping symplectic Pauli keys to complex coefficients.
    #[pyfn(m)]
    #[pyo3(name = "encode_fermion_product")]
    fn wrap_encode_product<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        signatures: String,
        indices: Vec<usize>,
        coefficient: Complex64,
    ) -> Result<Bound<'py, PyDict>, CoreError> {
        // ) -> PyResult<()> {
        if signatures.len() != indices.len() {
            return Err(CoreError::Value(
                "signatures and indices must have equal length".to_string(),
            ));
        }
        let symplectics = symplectics.as_array();
        let n_qubits = symplectics.ncols() / 2;
        let n_modes = symplectics.nrows() / 2;
        let vec_sig: Vec<LadderOperator> = signatures
            .chars()
            .map(|v| {
                LadderOperator::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("Invalid signature character: '{v}'"))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        if n_qubits < n_modes {
            return Err(CoreError::Value("n_qubits must be >= n_modes".to_string()));
        }

        let fproduct = FermionProduct::new(vec_sig, indices, coefficient)?;
        let x_block = symplectics.slice(ndarray::s![.., ..n_qubits]).to_owned();
        let z_block = symplectics.slice(ndarray::s![.., n_qubits..]).to_owned();
        let ipowers = ipowers.as_array().to_owned();
        let encoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::with_ipowers(x_block, z_block, ipowers),
            ZBasisState::zeros(n_qubits),
        )?;
        debug!("Got encoding");
        let qham: QubitHamiltonian = encoding.encode(fproduct);
        debug!("Got Hamiltonian");

        debug!("Got qham");
        Ok(qham.0.into_py_dict(py)?)
    }

    /// Encode a full fermionic Hamiltonian into a qubit Hamiltonian.
    ///
    /// Uses the symplectic Majorana encoding provided to map all fermionic
    /// operator products to Pauli operators.
    ///
    /// Args:
    ///     ipowers: 1D uint8 array — phase exponents for each encoding operator.
    ///     symplectics: 2D boolean array — symplectic encoding matrix.
    ///     signatures: List of fermionic operator signature strings.
    ///     coeffs: List of coefficient arrays, one per signature.
    ///     constant_energy: Constant energy offset to include in the result.
    ///
    /// Returns:
    ///     Dictionary mapping symplectic Pauli keys to complex coefficients.
    #[pyfn(m)]
    #[pyo3(name = "encode")]
    fn wrap_encode<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        vacuum_state: PyReadonlyArray1<bool>,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        constant_energy: f64,
    ) -> Result<Bound<'py, PyDict>, CoreError> {
        // ) -> PyResult<()> {
        if signatures.len() != coeffs.len() {
            return Err(CoreError::Value(
                "signatures and coefficients must have equal length".to_string(),
            ));
        }
        let symplectics = symplectics.as_array();
        let n_qubits = symplectics.ncols() / 2;
        let n_modes = symplectics.nrows() / 2;
        let vacuum_state = vacuum_state.as_array();

        if n_qubits < n_modes {
            return Err(CoreError::Value("n_qubits must be >= n_modes".to_string()));
        }

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            constant_energy,
        );
        let x_block = symplectics.slice(ndarray::s![.., ..n_qubits]).to_owned();
        let z_block = symplectics.slice(ndarray::s![.., n_qubits..]).to_owned();
        let ipowers = ipowers.as_array().to_owned();
        let encoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::with_ipowers(x_block, z_block, ipowers),
            ZBasisState::new(vacuum_state.to_owned(), Complex64::ONE),
        )?;
        debug!("Got encoding");
        let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        debug!("Got Hamiltonian");

        debug!("Got qham");
        Ok(qham.0.into_py_dict(py)?)
        // Ok(())
    }

    /// Encode a fermionic Hamiltonian using a standard named encoding.
    ///
    /// Convenience wrapper that builds the ternary tree for the named encoding
    /// and then calls the full encoding pipeline.
    ///
    /// Args:
    ///     encoding: One of ``"Jordan-Wigner"`` / ``"JW"``, ``"Bravyi-Kitaev"`` / ``"BK"``,
    ///         ``"Parity"`` / ``"PE"``, or ``"JKMN"``.
    ///     n_modes: Number of fermionic modes.
    ///     n_qubits: Number of qubits (must be >= ``n_modes``).
    ///     signatures: List of fermionic operator signature strings.
    ///     coeffs: List of coefficient arrays, one per signature.
    ///     constant_energy: Constant energy offset to include in the result.
    ///
    /// Returns:
    ///     Dictionary mapping symplectic Pauli keys to complex coefficients.
    #[pyfn(m)]
    #[pyo3(name = "encode_standard")]
    fn wrap_encode_standard<'py>(
        py: Python<'py>,
        encoding: String,
        n_modes: usize,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        constant_energy: f64,
    ) -> Result<Bound<'py, PyDict>, CoreError> {
        // ) -> PyResult<()> {
        if signatures.len() != coeffs.len() {
            return Err(CoreError::Value(
                "signatures and coefficients must have equal length".to_string(),
            ));
        }
        if n_qubits < n_modes {
            return Err(CoreError::Value("n_qubits must be >= n_modes".to_string()));
        }

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            constant_energy,
        );
        debug!("Got MSparse");
        debug!("Got Hamiltonian");
        let tree: TernaryTree = match encoding.as_str() {
            "Jordan-Wigner" | "JW" => TernaryTree::naive_jordan_wigner(n_modes),
            "Bravyi-Kitaev" | "BK" => TernaryTree::naive_bravyi_kitaev(n_modes),
            "Parity" | "PE" => TernaryTree::naive_parity(n_modes),
            "JKMN" => TernaryTree::naive_jkmn(n_modes),
            _ => return Err(CoreError::Value(
                "Encoding must be one of 'Jordan-Wigner'/'JW', 'Bravyi-Kitaev'/'BK', 'Parity'/'PE', or 'JKMN'.".to_string(),
            )),
        };
        debug!("Got Tree");
        debug!("Hamiltonian {:?}", hamiltonian);
        debug!("Hamiltonian {:?}", hamiltonian);
        let encoding = tree.build_encoding(n_qubits)?;
        debug!("Got encoding {:?}", encoding);
        debug!("Got encoding {:?}", encoding);

        let qham: QubitHamiltonian = encoding.encode(&hamiltonian);

        debug!("Got qham");
        debug!("Got qham {:?}", qham);
        Ok(qham.0.into_py_dict(py)?)
        // Ok(())
    }

    /// Run the TOPPHATT algorithm to optimise a ternary-tree encoding structure.
    ///
    /// TOPPHATT (Tree-Optimised Pauli-weight for Hamiltonian-Adapted Ternary Trees)
    /// modifies the ternary tree topology to minimise the Pauli weight of the
    /// encoded Hamiltonian.
    ///
    /// Args:
    ///     flatpack: List of ``(qubit_index, (left, mid, right))`` tuples — initial tree.
    ///     n_qubits: Total number of qubits in the system.
    ///     signatures: List of fermionic operator signature strings.
    ///     coeffs: List of coefficient arrays, one per signature.
    ///     parallelize: If ``True``, use multi-threaded evaluation via Rayon.
    ///     heuristic: Node-selection strategy. One of ``"min_weight"``
    ///         (default — try every active node and keep the lowest Pauli
    ///         weight), ``"x_first"`` (lowest-indexed active node),
    ///         ``"z_first"`` (highest-indexed active node), or ``"random"``
    ///         (uniformly random active node using ``seed``).
    ///     seed: RNG seed for ``heuristic="random"``. Ignored otherwise.
    ///         Defaults to ``0`` when not provided.
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectic_matrix)`` for the optimised encoding.
    #[pyfn(m)]
    #[pyo3(
        name = "topphatt",
        signature = (
            flatpack,
            n_qubits,
            signatures,
            coeffs,
            parallelize = true,
            heuristic = "min_weight",
            seed = None,
        ),
    )]
    #[allow(clippy::too_many_arguments)]
    fn wrap_topphatt<'py>(
        py: Python<'py>,
        flatpack: TTFlatpack,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        parallelize: bool,
        heuristic: &str,
        seed: Option<u64>,
    ) -> Result<
        (
            Bound<'py, PyArray1<u8>>,
            Bound<'py, PyArray2<bool>>,
            Bound<'py, PyArray1<bool>>,
        ),
        CoreError,
    > {
        // ) -> PyResult<()> {
        debug!("Starting TOPPHATT");
        let flatpack: TTFlatpack = flatpack;
        debug!("Got flatpack");

        let heuristic = NodeOrderHeuristic::parse(heuristic, seed).map_err(CoreError::Value)?;

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            0.,
        );

        debug!("Got MSparse");
        debug!("Got Hamiltonian");
        let mut tree: TernaryTree = TernaryTree::from_flatpack_naive(&flatpack)?;
        debug!("Got Tree");
        debug!("Hamiltonian {:?}", hamiltonian);
        tree = topphatt(hamiltonian, tree, parallelize, heuristic)?;

        let encoding = tree.build_encoding(n_qubits)?;
        debug!("Got encoding");
        let combined = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                encoding.operators.x_block.view(),
                encoding.operators.z_block.view(),
            ],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            encoding.operators.ipowers.into_pyarray(py),
            combined.into_pyarray(py),
            encoding.vacuum_state.state.into_pyarray(py),
        ))
    }

    /// Run the HATT algorithm to construct a ternary-tree encoding adapted to a Hamiltonian.
    ///
    /// HATT (Hamiltonian-Adaptive Ternary Tree) greedily constructs a ternary
    /// tree that minimises the Pauli weight of the encoded Hamiltonian. Unlike
    /// ``topphatt``, which optimises an existing tree, HATT builds the tree
    /// from scratch.
    ///
    /// Args:
    ///     n_modes: Number of fermionic modes in the system.
    ///     signatures: List of fermionic operator signature strings.
    ///     coeffs: List of coefficient arrays, one per signature.
    ///
    /// Returns:
    ///     Tuple of ``(flatpack, total_pauli_weight)`` where ``flatpack`` is
    ///     the ternary-tree flatpack representation and the weight is the
    ///     total Pauli weight of the greedy selections.
    #[pyfn(m)]
    #[pyo3(name = "hatt")]
    fn wrap_hatt(
        n_modes: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
    ) -> Result<(TTFlatpack, usize), CoreError> {
        debug!("Starting HATT");
        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            0.,
        );
        let simplified_terms: Vec<tinyvec::ArrayVec<[u16; 7]>> =
            simplified_majorana_terms(hamiltonian)
                .into_keys()
                .map(|k| {
                    let mut av = tinyvec::ArrayVec::<[u16; 7]>::new();
                    for idx in k {
                        av.push(idx);
                    }
                    av
                })
                .collect();
        let (tree, weight) = hatt(simplified_terms, n_modes)?;
        debug!("HATT finished with weight {weight}");
        Ok((tree.to_flatpack(), weight))
    }

    /// Run TOPPHATT optimisation and return both the encoding and the encoded Hamiltonian.
    ///
    /// Combines ``topphatt`` and ``encode`` in a single call: optimises the ternary
    /// tree and then encodes the full Hamiltonian using the resulting tree.
    ///
    /// Args:
    ///     flatpack: List of ``(qubit_index, (left, mid, right))`` tuples — initial tree.
    ///     n_qubits: Total number of qubits in the system.
    ///     signatures: List of fermionic operator signature strings.
    ///     coeffs: List of coefficient arrays, one per signature.
    ///     constant_energy: Constant energy offset to include in the result.
    ///     parallelize: If ``True``, use multi-threaded evaluation via Rayon.
    ///     heuristic: Node-selection strategy. One of ``"min_weight"``
    ///         (default — try every active node and keep the lowest Pauli
    ///         weight), ``"x_first"`` (lowest-indexed active node),
    ///         ``"z_first"`` (highest-indexed active node), or ``"random"``
    ///         (uniformly random active node using ``seed``).
    ///     seed: RNG seed for ``heuristic="random"``. Ignored otherwise.
    ///         Defaults to ``0`` when not provided.
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectic_matrix, hamiltonian_dict)``.
    #[pyfn(m)]
    #[pyo3(
        name = "encode_topphatt",
        signature = (
            flatpack,
            n_qubits,
            signatures,
            coeffs,
            constant_energy,
            parallelize = true,
            heuristic = "min_weight",
            seed = None,
        ),
    )]
    #[allow(clippy::too_many_arguments)]
    fn wrap_encode_topphatt<'py>(
        py: Python<'py>,
        flatpack: TTFlatpack,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        constant_energy: f64,
        parallelize: bool,
        heuristic: &str,
        seed: Option<u64>,
    ) -> Result<
        (
            Bound<'py, PyArray1<u8>>,
            Bound<'py, PyArray2<bool>>,
            Bound<'py, PyDict>,
            Bound<'py, PyArray1<bool>>,
        ),
        CoreError,
    > {
        debug!("Starting TOPPHATT");
        let flatpack: TTFlatpack = flatpack;
        debug!("Got flatpack");

        let heuristic = NodeOrderHeuristic::parse(heuristic, seed).map_err(CoreError::Value)?;

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            constant_energy,
        );

        debug!("Got MSparse");
        debug!("Got Hamiltonian");
        let mut tree: TernaryTree = TernaryTree::from_flatpack_naive(&flatpack)?;
        debug!("Got Tree");
        debug!("Hamiltonian {:?}", hamiltonian);
        tree = topphatt(hamiltonian.clone(), tree, parallelize, heuristic)?;

        let encoding = tree.build_encoding(n_qubits)?;
        debug!("Got encoding");
        let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        debug!("Got qham");
        let combined = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                encoding.operators.x_block.view(),
                encoding.operators.z_block.view(),
            ],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            encoding.operators.ipowers.into_pyarray(py),
            combined.into_pyarray(py),
            qham.0.into_py_dict(py)?,
            encoding.vacuum_state.state.into_pyarray(py),
        ))
    }

    /// Build the symplectic matrix of Majorana operators for the MaxNTO k-NTO encoding.
    ///
    /// Requires ``n_modes - 1`` to be odd.
    ///
    /// Args:
    ///     n_modes: Number of fermionic modes.
    ///
    /// Returns:
    ///     Tuple of ``(y_count, symplectic_matrix)`` where ``y_count`` is a 1D uint8 array
    ///     of phase exponents (mod 4) of length ``2 * n_modes``, and ``symplectic_matrix``
    ///     is a 2D boolean array of shape ``(2 * n_modes, 2 * n_modes)``.
    ///
    /// Example:
    ///     ```python
    ///     import ferrmion
    ///     y_count, sympl = ferrmion.maxnto_symplectic_matrix(14)
    ///     ```
    #[pyfn(m)]
    #[pyo3(name = "maxnto_symplectic_matrix")]
    fn wrap_maxnto_symplectic_matrix(
        py: Python<'_>,
        n_modes: usize,
    ) -> Result<
        (
            Bound<'_, PyArray1<u8>>,
            Bound<'_, PyArray2<bool>>,
            Bound<'_, PyArray1<bool>>,
        ),
        CoreError,
    > {
        let (y_count, output) = maxnto_symplectic_matrix(n_modes)?;
        let x_block: Array2<bool> = output.slice(s![.., ..output.ncols() / 2]).to_owned();
        let z_block: Array2<bool> = output.slice(s![.., output.ncols() / 2..]).to_owned();
        let encoding = MajoranaEncoding::new(SymplecticMatrix::new(x_block, z_block))
            .expect("Should be able to construct maxnto encoding.");
        Ok((
            y_count.into_pyarray(py),
            output.into_pyarray(py),
            encoding.vacuum_state.state.into_pyarray(py),
        ))
    }
    Ok(())
}
