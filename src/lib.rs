//! Fast, reliable and easy optimisation of fermion-qubit encodings.
//!
//! To simulate fermionic Hamiltonians with gate-based quantum computers,
//! it is necessary to encode the fermionic operators to qubit operators
//! which obey commutation fermionic relations.
//!
//! This file contains the PyO3 interop layer which wraps rust functions and exposes
//! these to a python API

use log::debug;
use numpy::ndarray::Array1;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::{IntoPyDict, PyComplex, PyDict, PyInt, PyString, PyTuple};
use pyo3::{prelude::*, pymodule, Bound};
pub mod operators;
pub mod states;
pub mod utils;
use crate::operators::{
    FermionProduct, FermionProductError, LadderOperator, MajoranaSparse, SymplecticMatrix,
    SymplecticOperator,
};
use crate::optimise::topphatt;
use crate::utils::*;
pub mod hamiltonians;
use crate::hamiltonians::QubitHamiltonian;
pub mod encoding;
use crate::encoding::{Encode, MajoranaEncoding, MajoranaEncodingError, TryEncode};
use crate::states::{FockState, State, ZBasisState};
pub mod optimise;
use crate::optimise::anneal_enumerations;
pub mod ternarytree;
use crate::optimise::ToppHattError;
use crate::ternarytree::{TTFlatPack, TernaryTree, TernaryTreeError};

impl From<MajoranaEncodingError> for PyErr {
    fn from(e: MajoranaEncodingError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}

impl From<TernaryTreeError> for PyErr {
    fn from(e: TernaryTreeError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}

impl From<ToppHattError> for PyErr {
    fn from(e: ToppHattError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

impl From<FermionProductError> for PyErr {
    fn from(_: FermionProductError) -> PyErr {
        PyValueError::new_err(
            "Invalid FermionProduct: operators and indices must have equal length",
        )
    }
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
    ) -> PyResult<(usize, Bound<'py, PyArray1<bool>>)> {
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
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
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
        let encoding = MajoranaEncoding::new(
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
            Ok(None) => Err(PyValueError::new_err("HF state encoded to null.")),
            Ok(Some(state)) => Ok(PyArray1::from_owned_array(py, state.state)),
            Err(e) => Err(PyErr::from(e)),
        }
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
    ) -> PyResult<(Bound<'py, PyString>, Bound<'py, PyInt>)> {
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
    ) -> PyResult<(Bound<'_, PyArray1<bool>>, Bound<'_, PyInt>)> {
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
    ) -> PyResult<(
        Bound<'py, PyString>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyComplex>,
    )> {
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
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectics)`` for the best encoding found.
    #[allow(clippy::too_many_arguments)]
    #[pyfn(m)]
    #[pyo3(name = "anneal_enumerations")]
    fn wrap_anneal_enumerations<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        temperature: f64,
        initial_guess: PyReadonlyArray1<usize>,
        coefficient_weighted: bool,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray2<bool>>)> {
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
        let encoding = MajoranaEncoding::new(
            SymplecticMatrix::with_ipowers(x_block.clone(), z_block.clone(), ipowers),
            ZBasisState::zeros(n_qubits),
        )?;
        let best_mode_enumeration: Array1<usize>;
        (_, best_mode_enumeration) = anneal_enumerations(
            msparse,
            encoding,
            temperature,
            initial_guess,
            coefficient_weighted,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let encoding = MajoranaEncoding::new(
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
        flatpack: TTFlatPack,
    ) -> PyResult<(
        Bound<'_, PyArray1<u8>>,
        Bound<'_, PyArray2<bool>>,
        Bound<'_, PyArray1<bool>>,
    )> {
        // ) -> PyResult<()> {
        let n_qubits: &usize = flatpack
            .iter()
            .map(|(v, _)| v)
            .max()
            .ok_or_else(|| PyValueError::new_err("Flatpack must be non-empty"))?;

        let tree: TernaryTree = TernaryTree::from_flatpack_naive(&flatpack)?;

        debug!("Got Tree");
        let encoding = tree.build_encoding(*n_qubits + 1)?;
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

    /// Build the symplectic encoding matrix for a standard named encoding.
    ///
    /// Args:
    ///     encoding: One of ``"Jordan-Wigner"`` / ``"JW"``, ``"Bravyi-Kitaev"`` / ``"BK"``,
    ///         ``"Parity"`` / ``"PE"``, or ``"JKMN"``.
    ///     n_modes: Number of fermionic modes.
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectic_matrix)``.
    #[pyfn(m)]
    #[pyo3(name = "standard_symplectic_matrix")]
    fn wrap_standard_symplectic_matrix(
        py: Python<'_>,
        encoding: String,
        n_modes: usize,
    ) -> PyResult<(
        Bound<'_, PyArray1<u8>>,
        Bound<'_, PyArray2<bool>>,
        Bound<'_, PyArray1<bool>>,
    )> {
        // ) -> PyResult<()> {

        let tree: TernaryTree = match encoding.as_str() {
            "Jordan-Wigner" | "JW" => TernaryTree::naive_jordan_wigner(n_modes),
            "Bravyi-Kitaev" | "BK" => TernaryTree::naive_bravyi_kitaev(n_modes),
            "Parity" | "PE" => TernaryTree::naive_parity(n_modes),
            "JKMN" => TernaryTree::naive_jkmn(n_modes),
            _ => return Err(PyValueError::new_err(
                "Encoding must be one of 'Jordan-Wigner'/'JW', 'Bravyi-Kitaev'/'BK', 'Parity'/'PE', or 'JKMN'.",
            )),
        };
        debug!("Got Tree");
        let encoding = tree.build_encoding(n_modes)?;
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
    ) -> PyResult<Bound<'py, PyDict>> {
        if signatures.len() != coeffs.len() {
            return Err(PyValueError::new_err(
                "signatures and coefficients must have equal length",
            ));
        }

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            constant_energy,
        );

        let output = PyDict::new(py);
        for (key, val) in std::iter::zip(hamiltonian.indices, hamiltonian.coefficients) {
            let py_key = PyTuple::new(py, key.as_slice())?;
            let existing: Option<numpy::Complex64> = output.get_item(&py_key)?.map(|v| v.extract()).transpose()?;
            match existing {
                Some(prev) => output.set_item(&py_key, prev + val)?,
                None => output.set_item(&py_key, val)?,
            }
        }
        Ok(output.into())
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
    ) -> PyResult<Bound<'py, PyDict>> {
        // ) -> PyResult<()> {
        if signatures.len() != indices.len() {
            return Err(PyValueError::new_err(
                "signatures and indices must have equal length",
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
            return Err(PyValueError::new_err("n_qubits must be >= n_modes"));
        }

        let fproduct = FermionProduct::new(vec_sig, indices, coefficient)?;
        let x_block = symplectics.slice(ndarray::s![.., ..n_qubits]).to_owned();
        let z_block = symplectics.slice(ndarray::s![.., n_qubits..]).to_owned();
        let ipowers = ipowers.as_array().to_owned();
        let encoding = MajoranaEncoding::new(
            SymplecticMatrix::with_ipowers(x_block, z_block, ipowers),
            ZBasisState::zeros(n_qubits),
        )?;
        debug!("Got encoding");
        let qham: QubitHamiltonian = encoding.encode(fproduct);
        debug!("Got Hamiltonian");

        debug!("Got qham");
        qham.into_py_dict(py)
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
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        constant_energy: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        // ) -> PyResult<()> {
        if signatures.len() != coeffs.len() {
            return Err(PyValueError::new_err(
                "signatures and coefficients must have equal length",
            ));
        }
        let symplectics = symplectics.as_array();
        let n_qubits = symplectics.ncols() / 2;
        let n_modes = symplectics.nrows() / 2;

        if n_qubits < n_modes {
            return Err(PyValueError::new_err("n_qubits must be >= n_modes"));
        }

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            constant_energy,
        );
        let x_block = symplectics.slice(ndarray::s![.., ..n_qubits]).to_owned();
        let z_block = symplectics.slice(ndarray::s![.., n_qubits..]).to_owned();
        let ipowers = ipowers.as_array().to_owned();
        let encoding = MajoranaEncoding::new(
            SymplecticMatrix::with_ipowers(x_block, z_block, ipowers),
            ZBasisState::zeros(n_qubits),
        )?;
        debug!("Got encoding");
        let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        debug!("Got Hamiltonian");

        debug!("Got qham");
        qham.into_py_dict(py)
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
    ) -> PyResult<Bound<'py, PyDict>> {
        // ) -> PyResult<()> {
        if signatures.len() != coeffs.len() {
            return Err(PyValueError::new_err(
                "signatures and coefficients must have equal length",
            ));
        }
        if n_qubits < n_modes {
            return Err(PyValueError::new_err("n_qubits must be >= n_modes"));
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
            _ => return Err(PyValueError::new_err(
                "Encoding must be one of 'Jordan-Wigner'/'JW', 'Bravyi-Kitaev'/'BK', 'Parity'/'PE', or 'JKMN'.",
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
        qham.into_py_dict(py)
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
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectic_matrix)`` for the optimised encoding.
    #[pyfn(m)]
    #[pyo3(name = "topphatt")]
    fn wrap_topphatt<'py>(
        py: Python<'py>,
        flatpack: TTFlatPack,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        parallelize: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        // ) -> PyResult<()> {
        debug!("Starting TOPPHATT");
        let flatpack: TTFlatPack = flatpack;
        debug!("Got flatpack");

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
        tree = topphatt(hamiltonian, tree, parallelize)?;

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
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectic_matrix, hamiltonian_dict)``.
    #[pyfn(m)]
    #[pyo3(name = "encode_topphatt")]
    fn wrap_encode_topphatt<'py>(
        py: Python<'py>,
        flatpack: TTFlatPack,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        constant_energy: f64,
        parallelize: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyDict>,
        Bound<'py, PyArray1<bool>>,
    )> {
        debug!("Starting TOPPHATT");
        let flatpack: TTFlatPack = flatpack;
        debug!("Got flatpack");

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
        tree = topphatt(hamiltonian.clone(), tree, parallelize)?;

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
            qham.into_py_dict(py)?,
            encoding.vacuum_state.state.into_pyarray(py),
        ))
    }

    /// Run TOPPHATT using a standard named encoding as the initial tree.
    ///
    /// Constructs the ternary tree for the named encoding and then runs the
    /// TOPPHATT optimisation algorithm to minimise Pauli weight.
    ///
    /// Args:
    ///     encoding: One of ``"Jordan-Wigner"`` / ``"JW"``, ``"Bravyi-Kitaev"`` / ``"BK"``,
    ///         ``"Parity"`` / ``"PE"``, or ``"JKMN"``.
    ///     n_modes: Number of fermionic modes.
    ///     n_qubits: Total number of qubits (must be >= ``n_modes``).
    ///     signatures: List of fermionic operator signature strings.
    ///     coeffs: List of coefficient arrays, one per signature.
    ///     parallelize: If ``True``, use multi-threaded evaluation via Rayon.
    ///
    /// Returns:
    ///     Tuple of ``(ipowers, symplectic_matrix)`` for the optimised encoding.
    #[pyfn(m)]
    #[pyo3(name = "topphatt_standard")]
    fn wrap_topphatt_standard<'py>(
        py: Python<'py>,
        encoding: String,
        n_modes: usize,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        parallelize: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        // ) -> PyResult<Bound<'py, PyDict>> {
        if signatures.len() != coeffs.len() {
            return Err(PyValueError::new_err(
                "signatures and coefficients must have equal length",
            ));
        }
        if n_qubits < n_modes {
            return Err(PyValueError::new_err("n_qubits must be >= n_modes"));
        }

        debug!("Starting TOPPHATT");
        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            0.,
        );
        debug!("Got MSparse");
        debug!("Got Hamiltonian");
        let mut tree: TernaryTree = match encoding.as_str() {
            "Jordan-Wigner" | "JW" => TernaryTree::naive_jordan_wigner(n_modes),
            "Bravyi-Kitaev" | "BK" => TernaryTree::naive_bravyi_kitaev(n_modes),
            "Parity" | "PE" => TernaryTree::naive_parity(n_modes),
            "JKMN" => TernaryTree::naive_jkmn(n_modes),
            _ => return Err(PyValueError::new_err(
                "Encoding must be one of 'Jordan-Wigner'/'JW', 'Bravyi-Kitaev'/'BK', 'Parity'/'PE', or 'JKMN'.",
            )),
        };
        debug!("Got Tree");
        debug!("Hamiltonian {:?}", hamiltonian);
        tree = topphatt(hamiltonian.clone(), tree, parallelize)?;
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
        // let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        // debug!("Got qham");
        // Ok(qham
        //     .into_py_dict(py)
        //     .expect("Should be able to convert QubitHamiltonian to PyDict."))
        // Ok(())
    }
    Ok(())
}
