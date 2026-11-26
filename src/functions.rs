//! Free functions exposed in the `ferrmion.core` module.

use crate::encoding::{EncodeInput, PyMajoranaEncoding};
use crate::error::CoreError;
use crate::hamiltonians::PyQubitHamiltonian;
use crate::operators::PyMajoranaSparse;
use ferrmion_core::encode::majorana::Encode;
use ferrmion_core::encode::ternarytree::{TTFlatpack, TernaryTree};
use ferrmion_core::hamiltonians::QubitHamiltonian;
use ferrmion_core::operators::{MajoranaSparse, SymplecticOperator};
use ferrmion_core::optimise::geometry::{
    exhaustive, parity_canonical_terms, topdown, topdown_expected, tree_pauli_weight,
    HammingColumns, SlotModel,
};
use ferrmion_core::optimise::{
    hatt, topphatt_with_options, DenseTransposeBackend, NodeOrderHeuristic, SparseTransposeBackend,
    ToppHattOptions,
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
/// Both `MajoranaSparse.to_dict()` and `hatt` consume the result;
/// going through a single helper guarantees they see identical term sets
/// (a per-entry accumulate-and-filter loop can leave stale entries when a
/// running coefficient cancels below the threshold).
///
/// A term whose indices fully cancel (γ²=I) contributes to the identity —
/// folded, together with `hamiltonian.constant`, into the returned map's
/// empty-tuple `()` key when the combined result is non-negligible. A
/// self-cancelling term with a non-negligible imaginary part is instead
/// kept as an ordinary empty-key entry, so complex precision from e.g. an
/// arbitrary (non-Hermitian) coefficient isn't silently discarded either.
pub(crate) fn simplified_majorana_terms(
    hamiltonian: MajoranaSparse,
) -> std::collections::BTreeMap<Vec<u16>, Complex64> {
    let mut merged: std::collections::BTreeMap<Vec<u16>, Complex64> =
        std::collections::BTreeMap::new();
    let mut constant = hamiltonian.constant;
    for (key, val) in std::iter::zip(hamiltonian.indices, hamiltonian.coefficients) {
        let mut simplified: Vec<u16> = Vec::with_capacity(key.len());
        for &idx in key.as_slice() {
            if simplified.last() == Some(&idx) {
                simplified.pop();
            } else {
                simplified.push(idx);
            }
        }
        if simplified.is_empty() && val.im.abs() < utils::COEFFICIENT_TOLERANCE {
            constant += val.re;
        } else {
            *merged.entry(simplified).or_insert(Complex64::new(0., 0.)) += val;
        }
    }
    if constant.abs() > utils::COEFFICIENT_TOLERANCE {
        *merged.entry(Vec::new()).or_insert(Complex64::new(0., 0.)) +=
            Complex64::new(constant, 0.0);
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
#[pyo3(signature = (left, right))]
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
    let x = result.x.to_bool_array();
    let z = result.z.to_bool_array();
    let combined = ndarray::concatenate(ndarray::Axis(0), &[x.view(), z.view()])
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        result.ipower as usize,
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
#[pyo3(signature = (symplectic, ipower=0))]
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
#[pyo3(signature = (pauli, ipower=0))]
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
///     operator: The operator whose terms drive the greedy search, either a
///         ``FermionHamiltonian`` or a ``MajoranaSparse``. A
///         ``FermionHamiltonian`` is converted to its Majorana representation
///         first; passing a ``MajoranaSparse`` directly skips that conversion.
///     `n_modes`: Number of fermionic modes. Defaults to the operator's own
///         mode count.
///
/// Returns:
///     Tuple of ``(flatpack, total_pauli_weight)`` where ``flatpack`` is
///     the ternary-tree flatpack representation and the weight is the
///     total Pauli weight of the greedy selections.
#[pyfunction(name = "hatt")]
#[pyo3(signature = (operator, n_modes = None))]
pub(crate) fn hatt_py(
    py: Python<'_>,
    operator: EncodeInput<'_>,
    n_modes: Option<usize>,
) -> Result<(TTFlatpack, usize), CoreError> {
    debug!("Starting HATT");
    let (mut hamiltonian, operator_n_modes) = match operator {
        EncodeInput::Fermion(fham) => (fham.inner.to_majorana_sparse(), fham.inner.n_modes()),
        EncodeInput::Majorana(msparse) => (msparse.inner.clone(), msparse.n_modes),
    };
    let n_modes = n_modes.unwrap_or(operator_n_modes);
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
/// Options shared by the TOPP-HATT entry points.
pub(crate) struct ToppHattArgs<'a> {
    pub parallelize: bool,
    pub heuristic: &'a str,
    pub backend: &'a str,
    pub seed: Option<u64>,
    pub strategy: &'a str,
    pub options: ToppHattOptions,
}

fn run_bottom_up(
    hamiltonian: &MajoranaSparse,
    tree: TernaryTree,
    args: &ToppHattArgs<'_>,
) -> Result<TernaryTree, CoreError> {
    let heuristic =
        NodeOrderHeuristic::parse(args.heuristic, args.seed).map_err(CoreError::Value)?;
    let options = args.options;
    // With deduplication off, canonicalise to parity sets once: the exact term set.
    let terms = if options.dedup {
        hamiltonian.indices.clone()
    } else {
        parity_canonical_terms(hamiltonian)
    };
    let n = tree.n_nodes;
    let p = args.parallelize;
    let result = match args.backend {
        "sparse" => {
            if !options.dedup {
                return Err(CoreError::Value(
                    "dedup=False requires a transposed backend (\"dense_transpose\" or \"sparse_transpose\")".to_string(),
                ));
            }
            topphatt_with_options(hamiltonian.clone(), tree, p, heuristic, options)?
        }
        "dense_transpose" => {
            let store = DenseTransposeBackend::with_dedup(&terms, n, options.dedup);
            topphatt_with_options(store, tree, p, heuristic, options)?
        }
        "sparse_transpose" => {
            let store = SparseTransposeBackend::with_dedup(&terms, n, options.dedup);
            topphatt_with_options(store, tree, p, heuristic, options)?
        }
        other => {
            return Err(CoreError::Value(format!(
                "unknown topphatt backend: {other:?} \
                 (expected \"sparse\", \"dense_transpose\" or \"sparse_transpose\")"
            )))
        }
    };
    Ok(result.0)
}

fn run_topphatt(
    hamiltonian: MajoranaSparse,
    flatpack: TTFlatpack,
    args: &ToppHattArgs<'_>,
) -> Result<TernaryTree, CoreError> {
    let tree = TernaryTree::from_flatpack_naive(&flatpack)?;
    let geometry_err =
        |e: ferrmion_core::optimise::geometry::GeometryError| CoreError::Value(e.to_string());
    match args.strategy {
        "bottom_up" => run_bottom_up(&hamiltonian, tree, args),
        "top_down" | "top_down_seeded" | "top_down_expected" | "exact" => {
            let model = SlotModel::new(&tree).map_err(geometry_err)?;
            let cols = HammingColumns::from_hamiltonian(&hamiltonian, tree.n_nodes);
            let seed: Vec<usize> = if args.strategy == "top_down_seeded" {
                let greedy = run_bottom_up(&hamiltonian, tree.clone(), args)?;
                match model.mode_of_slot_from(&greedy) {
                    Some(perm) => perm,
                    // The greedy may flip a pair's orientation, which is outside
                    // the slot model; fall back to its result unchanged.
                    None => return Ok(greedy),
                }
            } else {
                (0..tree.n_nodes).collect()
            };
            let perm = if args.strategy == "exact" {
                exhaustive(&model, &cols).map_err(geometry_err)?.0
            } else if args.strategy == "top_down_expected" {
                topdown_expected(
                    &model,
                    &parity_canonical_terms(&hamiltonian),
                    &seed,
                    20,
                    512,
                )
            } else {
                topdown(&model, &cols, &seed, 20, args.options.lookahead)
            };
            Ok(model.apply(&tree, &perm))
        }
        "best" => {
            // Every candidate is scored exactly on the Hamming cube. The input
            // assignment (the flatpack's own leaves, when it has a complete set)
            // is both a candidate and a top-down seed, so the result is never
            // worse than the tree passed in.
            let model = SlotModel::new(&tree).map_err(geometry_err)?;
            let terms = parity_canonical_terms(&hamiltonian);
            let cols = HammingColumns::new(&terms, tree.n_nodes);
            let mu = if args.options.lookahead > 0.0 {
                args.options.lookahead
            } else {
                1.0
            };
            let mut candidates = vec![tree.clone()];
            let mut seeds: Vec<Vec<usize>> = vec![(0..tree.n_nodes).collect()];
            if let Ok(input) = TernaryTree::from_flatpack(&flatpack) {
                if tree_pauli_weight(&cols, &input).is_ok() {
                    if let Some(perm) = model.mode_of_slot_from(&input) {
                        seeds.push(perm);
                    }
                    candidates.push(input);
                }
            }
            for seed in &seeds {
                candidates.push(model.apply(&tree, &topdown(&model, &cols, seed, 20, mu)));
                candidates
                    .push(model.apply(&tree, &topdown_expected(&model, &terms, seed, 20, 512)));
            }
            candidates.push(run_bottom_up(&hamiltonian, tree, args)?);
            let mut best: Option<(usize, TernaryTree)> = None;
            for candidate in candidates {
                let w = tree_pauli_weight(&cols, &candidate).map_err(geometry_err)?;
                if best.as_ref().is_none_or(|(bw, _)| w < *bw) {
                    best = Some((w, candidate));
                }
            }
            Ok(best.expect("at least one candidate").1)
        }
        other => Err(CoreError::Value(format!(
            "unknown topphatt strategy: {other:?} (expected \"bottom_up\", \"top_down\", \
             \"top_down_seeded\", \"top_down_expected\", \"best\" or \"exact\")"
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
///     `n_qubits`: Total number of qubits in the system.
///     hamiltonian: The Majorana sparse Hamiltonian driving the optimisation.
///     parallelize: If ``True``, use multi-threaded evaluation via Rayon.
///     heuristic: Node-selection strategy. One of ``"min_weight"``
///         (default), ``"x_first"``, ``"z_first"``, or ``"random"``.
///     seed: RNG seed for ``heuristic="random"``. Ignored otherwise.
///     backend: Term-store backend, ``"sparse"`` (default) or
///         ``"dense_transpose"`` / ``"sparse_transpose"`` (transposed layouts, for benchmarking).
///     dedup: Merge terms that coincide after each contraction (original
///         behaviour). ``False`` counts every distinct Pauli string exactly.
///     lookahead: Weight of the parent-edge penalty (terms left "open" by a
///         node's children). ``0.0`` is the original greedy.
///     `depth_scaled`: Scale ``lookahead`` by node depth.
///     `contract_forced`: Contract nodes whose leaves were fixed by pair
///         propagation, so ancestors see their subtree (exact objective).
///     strategy: ``"bottom_up"`` (TOPP-HATT), ``"top_down"`` (hierarchical
///         partition from the root), ``"top_down_seeded"`` (top-down refinement
///         seeded with the bottom-up result), ``"top_down_expected"`` (top-down
///         scored by exact node weight plus expected random-completion weight of
///         each child subtree), ``"best"`` (run naive, bottom-up and both
///         top-down variants and keep the lowest exact weight) or ``"exact"``
///         (exhaustive, ≤ 10 modes).
///
/// Returns:
///     `MajoranaEncoding`: The optimised encoding.
#[pyfunction(name = "topphatt")]
#[pyo3(signature = (flatpack, n_qubits, hamiltonian, parallelize = true, heuristic = "min_weight", seed = None, backend = "dense_transpose", dedup = true, lookahead = 0.0, depth_scaled = false, contract_forced = false, strategy = "bottom_up"))]
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
    dedup: bool,
    lookahead: f64,
    depth_scaled: bool,
    contract_forced: bool,
    strategy: &str,
) -> Result<PyMajoranaEncoding, CoreError> {
    debug!("Starting TOPPHATT");
    // let mut hamiltonian = hamiltonian.inner;
    // hamiltonian.constant = 0.0;

    let encoding = py.allow_threads(|| -> Result<_, CoreError> {
        let tree = run_topphatt(
            hamiltonian.inner,
            flatpack,
            &ToppHattArgs {
                parallelize,
                heuristic,
                backend,
                seed,
                strategy,
                options: ToppHattOptions {
                    dedup,
                    lookahead,
                    depth_scaled,
                    contract_forced,
                },
            },
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
///     `n_qubits`: Total number of qubits in the system.
///     operator: The operator to optimise for and encode, either a
///         ``FermionHamiltonian`` or a ``MajoranaSparse``. A
///         ``FermionHamiltonian`` is converted to its Majorana representation
///         first; passing a ``MajoranaSparse`` directly skips that conversion.
///     parallelize: If ``True``, use multi-threaded evaluation via Rayon.
///     heuristic: Node-selection strategy. One of ``"min_weight"``
///         (default), ``"x_first"``, ``"z_first"``, or ``"random"``.
///     seed: RNG seed for ``heuristic="random"``. Ignored otherwise.
///     backend: Term-store backend, ``"sparse"`` (default) or
///         ``"dense_transpose"`` / ``"sparse_transpose"`` (transposed layouts, for benchmarking).
///     dedup: Merge terms that coincide after each contraction (original
///         behaviour). ``False`` counts every distinct Pauli string exactly.
///     lookahead: Weight of the parent-edge penalty (terms left "open" by a
///         node's children). ``0.0`` is the original greedy.
///     `depth_scaled`: Scale ``lookahead`` by node depth.
///     `contract_forced`: Contract nodes whose leaves were fixed by pair
///         propagation, so ancestors see their subtree (exact objective).
///     strategy: ``"bottom_up"`` (TOPP-HATT), ``"top_down"`` (hierarchical
///         partition from the root), ``"top_down_seeded"`` (top-down refinement
///         seeded with the bottom-up result), ``"top_down_expected"``,
///         ``"best"`` or ``"exact"``; see ``topphatt``.
///
/// Returns:
///     Tuple of ``(QubitHamiltonian, MajoranaEncoding)``.
#[pyfunction(name = "encode_topphatt")]
#[pyo3(signature = (flatpack, n_qubits, operator, parallelize = true, heuristic = "min_weight", seed = None, backend = "dense_transpose", dedup = true, lookahead = 0.0, depth_scaled = false, contract_forced = false, strategy = "bottom_up"))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python API
pub(crate) fn encode_topphatt_py(
    py: Python<'_>,
    flatpack: TTFlatpack,
    n_qubits: usize,
    operator: EncodeInput<'_>,
    parallelize: bool,
    heuristic: &str,
    seed: Option<u64>,
    backend: &str,
    dedup: bool,
    lookahead: f64,
    depth_scaled: bool,
    contract_forced: bool,
    strategy: &str,
) -> Result<(PyQubitHamiltonian, PyMajoranaEncoding), CoreError> {
    debug!("Starting TOPPHATT");
    // let heuristic = NodeOrderHeuristic::parse(heuristic, seed).map_err(CoreError::Value)?;
    let hamiltonian = match operator {
        EncodeInput::Fermion(fham) => fham.inner.to_majorana_sparse(),
        EncodeInput::Majorana(msparse) => msparse.inner.clone(),
    };

    let (qham, encoding) = py.allow_threads(|| -> Result<_, CoreError> {
        let tree = run_topphatt(
            hamiltonian.clone(),
            flatpack,
            &ToppHattArgs {
                parallelize,
                heuristic,
                backend,
                seed,
                strategy,
                options: ToppHattOptions {
                    dedup,
                    lookahead,
                    depth_scaled,
                    contract_forced,
                },
            },
        )?;
        let encoding = tree.build_encoding(n_qubits)?;
        let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        Ok((qham, encoding))
    })?;
    Ok((PyQubitHamiltonian(qham), PyMajoranaEncoding(encoding)))
}
