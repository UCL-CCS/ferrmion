//! Free functions exposed in the `ferrmion.core` module.

use crate::encoding::PyMajoranaEncoding;
use crate::error::CoreError;
use crate::hamiltonians::{PyFermionHamiltonian, PyQubitHamiltonian};
use crate::operators::PyMajoranaSparse;
use ferrmion_core::encode::majorana::Encode;
use ferrmion_core::encode::ternarytree::{TTFlatpack, TernaryTree};
use ferrmion_core::hamiltonians::QubitHamiltonian;
use ferrmion_core::operators::{MajoranaSparse, SymplecticOperator};
use ferrmion_core::optimise::{
    hatt, topphatt, DenseTransposeBackend, NodeOrderHeuristic, SparseTransposeBackend,
};
use ferrmion_core::utils;
use log::debug;
use numpy::{Complex64, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyInt, PyString};

/// Apply γ²=1 simplification, merge duplicate Majorana keys, and drop
/// terms whose summed coefficient falls below the near-zero threshold.
///
/// Both `FermionHamiltonian.to_sparse_majorana` and `hatt` consume the result;
/// going through a single helper guarantees they see identical term sets
/// (a per-entry accumulate-and-filter loop can leave stale entries when a
/// running coefficient cancels below the threshold).
pub(crate) fn simplified_majorana_terms(
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
    merged.retain(|_, v| v.norm() > utils::COEFFICIENT_TOLERANCE);
    merged
}

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
#[pyfunction]
pub(crate) fn symplectic_product<'py>(
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
    let combined = ndarray::concatenate(ndarray::Axis(0), &[result.x_block(), result.z_block()])
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        result.ipower() as usize,
        PyArray1::from_owned_array(py, combined),
    ))
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
#[pyfunction]
pub(crate) fn symplectic_to_pauli<'py>(
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
#[pyfunction]
pub(crate) fn pauli_to_symplectic(
    py: Python<'_>,
    pauli: String,
    ipower: usize,
) -> Result<(Bound<'_, PyArray1<bool>>, Bound<'_, PyInt>), CoreError> {
    let (symplectic, ipower) = utils::pauli_to_symplectic(pauli, ipower);
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
#[pyfunction]
#[allow(clippy::type_complexity)]
pub(crate) fn symplectic_to_sparse<'py>(
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
    let (pauli_string, position_vec, coeff) = utils::symplectic_to_sparse(symplectic, ipower);
    Ok((
        PyString::new(py, &pauli_string),
        PyArray1::from_owned_array(py, position_vec),
        PyComplex::from_complex_bound(py, coeff),
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
///     fham: The fermionic Hamiltonian whose terms drive the greedy search.
///     n_modes: Number of fermionic modes. Defaults to ``fham.n_modes``.
///
/// Returns:
///     Tuple of ``(flatpack, total_pauli_weight)`` where ``flatpack`` is
///     the ternary-tree flatpack representation and the weight is the
///     total Pauli weight of the greedy selections.
#[pyfunction(name = "hatt")]
#[pyo3(signature = (fham, n_modes = None))]
pub(crate) fn hatt_py(
    py: Python<'_>,
    fham: PyRef<'_, PyFermionHamiltonian>,
    n_modes: Option<usize>,
) -> Result<(TTFlatpack, usize), CoreError> {
    debug!("Starting HATT");
    let n_modes = n_modes.unwrap_or(fham.inner.n_modes());
    let mut hamiltonian = fham.inner.to_majorana_sparse();
    hamiltonian.constant = 0.0;
    let (tree, weight) = py.allow_threads(|| -> Result<_, CoreError> {
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
        Ok(hatt(simplified_terms, n_modes)?)
    })?;
    debug!("HATT finished with weight {weight}");
    Ok((tree.to_flatpack(), weight))
}

/// Run TOPP-HATT over the requested Majorana term-store backend.
///
/// `"sparse"` (default) uses the production `Vec<ArrayVec<..>>` store;
/// `"dense_transpose"` uses the transposed `MajoranaDenseTranspose` (one `u64` bit-vector
/// per Majorana index); `"sparse_transpose"` uses the sparse `SparseListTermStore`
/// (one sorted list of term indices per Majorana index). The transposed backends
/// do no term deduplication, so they can produce a different (but valid) encoding
/// — they are provided for performance comparison.
fn run_topphatt(
    hamiltonian: MajoranaSparse,
    flatpack: TTFlatpack,
    parallelize: bool,
    heuristic: &str,
    backend: &str,
    seed: Option<u64>,
) -> Result<TernaryTree, CoreError> {
    let heuristic = NodeOrderHeuristic::parse(heuristic, seed).map_err(CoreError::Value)?;
    let tree = TernaryTree::from_flatpack_naive(&flatpack)?;

    match backend {
        "sparse" => Ok(topphatt(hamiltonian, tree, parallelize, heuristic)?),
        "dense_transpose" => {
            let store = DenseTransposeBackend::from_arrayvecs(&hamiltonian.indices, tree.n_nodes);
            Ok(topphatt(store, tree, parallelize, heuristic)?)
        }
        "sparse_transpose" => {
            let store = SparseTransposeBackend::from_arrayvecs(&hamiltonian.indices, tree.n_nodes);
            Ok(topphatt(store, tree, parallelize, heuristic)?)
        }
        other => Err(CoreError::Value(format!(
            "unknown topphatt backend: {other:?} \
             (expected \"sparse\", \"dense_transpose\" or \"sparse_transpose\")"
        ))),
    }
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
///     hamiltonian: The Majorana sparse Hamiltonian driving the optimisation.
///     parallelize: If ``True``, use multi-threaded evaluation via Rayon.
///     heuristic: Node-selection strategy. One of ``"min_weight"``
///         (default), ``"x_first"``, ``"z_first"``, or ``"random"``.
///     seed: RNG seed for ``heuristic="random"``. Ignored otherwise.
///     backend: Term-store backend, ``"sparse"`` (default) or
///         ``"dense_transpose"`` / ``"sparse_transpose"`` (transposed layouts, for benchmarking).
///
/// Returns:
///     MajoranaEncoding: The optimised encoding.
#[pyfunction(name = "topphatt")]
#[pyo3(signature = (flatpack, n_qubits, hamiltonian, parallelize = true, heuristic = "min_weight", seed = None, backend = "dense_transpose"))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python API
pub(crate) fn topphatt_py(
    py: Python<'_>,
    flatpack: TTFlatpack,
    n_qubits: usize,
    hamiltonian: PyMajoranaSparse,
    parallelize: bool,
    heuristic: &str,
    seed: Option<u64>,
    backend: &str,
) -> Result<PyMajoranaEncoding, CoreError> {
    debug!("Starting TOPPHATT");
    // let mut hamiltonian = hamiltonian.0;
    // hamiltonian.constant = 0.0;

    let encoding = py.allow_threads(|| -> Result<_, CoreError> {
        let tree = run_topphatt(
            hamiltonian.0,
            flatpack,
            parallelize,
            heuristic,
            backend,
            seed,
        )?;
        Ok(tree.build_encoding(n_qubits)?)
    })?;
    Ok(PyMajoranaEncoding(encoding))
}

/// Run TOPPHATT optimisation and return both the encoded Hamiltonian and the encoding.
///
/// Combines ``topphatt`` and ``MajoranaEncoding.encode`` in a single call.
///
/// Args:
///     flatpack: List of ``(qubit_index, (left, mid, right))`` tuples — initial tree.
///     n_qubits: Total number of qubits in the system.
///     fham: The fermionic Hamiltonian to optimise for and encode.
///     parallelize: If ``True``, use multi-threaded evaluation via Rayon.
///     heuristic: Node-selection strategy. One of ``"min_weight"``
///         (default), ``"x_first"``, ``"z_first"``, or ``"random"``.
///     seed: RNG seed for ``heuristic="random"``. Ignored otherwise.
///     backend: Term-store backend, ``"sparse"`` (default) or
///         ``"dense_transpose"`` / ``"sparse_transpose"`` (transposed layouts, for benchmarking).
///
/// Returns:
///     Tuple of ``(QubitHamiltonian, MajoranaEncoding)``.
#[pyfunction(name = "encode_topphatt")]
#[pyo3(signature = (flatpack, n_qubits, fham, parallelize = true, heuristic = "min_weight", seed = None, backend = "dense_transpose"))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python API
pub(crate) fn encode_topphatt_py(
    py: Python<'_>,
    flatpack: TTFlatpack,
    n_qubits: usize,
    fham: PyRef<'_, PyFermionHamiltonian>,
    parallelize: bool,
    heuristic: &str,
    seed: Option<u64>,
    backend: &str,
) -> Result<(PyQubitHamiltonian, PyMajoranaEncoding), CoreError> {
    debug!("Starting TOPPHATT");
    // let heuristic = NodeOrderHeuristic::parse(heuristic, seed).map_err(CoreError::Value)?;
    let hamiltonian = fham.inner.to_majorana_sparse();

    let (qham, encoding) = py.allow_threads(|| -> Result<_, CoreError> {
        let tree = run_topphatt(
            hamiltonian.clone(),
            flatpack,
            parallelize,
            heuristic,
            backend,
            seed,
        )?;
        let encoding = tree.build_encoding(n_qubits)?;
        let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        Ok((qham, encoding))
    })?;
    Ok((PyQubitHamiltonian(qham), PyMajoranaEncoding(encoding)))
}
