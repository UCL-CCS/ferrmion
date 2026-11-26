//! `pyclass` wrappers for ordered circuits and the TOPP-HATT circuit compiler.

use crate::encoding::{EncodeInput, PyMajoranaEncoding};
use crate::error::CoreError;
use crate::operators::PyMajoranaSparse;
use ferrmion_core::encode::ternarytree::{TTFlatpack, TernaryTree};
use ferrmion_core::operators::{MajoranaCircuit, PauliCircuit};
use ferrmion_core::optimise::{compile_circuit, NodeOrderHeuristic};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// An ordered circuit of Majorana rotation gates ``exp(-iθ Γ_S)``.
///
/// ``Γ_S = i^m γ_{s1}⋯γ_{sk}`` with ``m = k(k-1)/2 mod 2``, which makes
/// ``Γ_S`` Hermitian. Unlike a Hamiltonian, gate order and multiplicity are
/// preserved: each entry is one gate and repeats count once per occurrence.
///
/// Construction canonicalises each term: indices are sorted, negating the
/// angle once per swap of unequal indices (``γ1γ0 = -γ0γ1``), and repeated
/// indices cancel pairwise (``γ² = I``). Angles are wrapped into ``(-π, π]``.
/// Gates that reduce to the identity or have a zero angle are dropped, so
/// ``len(circuit)`` may be smaller than the input.
///
/// Args:
///     terms: Ordered Majorana index lists, one per gate (at most 7 each).
///     angles: Rotation angle of each gate, parallel to ``terms``.
///     `n_modes`: Number of fermionic modes; indices must be ``< 2 * n_modes``.
#[pyclass(name = "MajoranaCircuit", module = "ferrmion.core")]
#[derive(Clone, Debug)]
pub struct PyMajoranaCircuit(pub MajoranaCircuit);

#[pymethods]
impl PyMajoranaCircuit {
    #[new]
    fn new(terms: Vec<Vec<u16>>, angles: Vec<f64>, n_modes: usize) -> Result<Self, CoreError> {
        Ok(Self(MajoranaCircuit::from_index_lists(
            terms, angles, n_modes,
        )?))
    }

    /// Build ``∏_j exp(-i c_j γ_{S_j})`` from a Hamiltonian's terms, in order.
    ///
    /// This is one first-order Trotter step of unit time. Each term must be
    /// Hermitian; the constant is a global phase and is dropped.
    ///
    /// Args:
    ///     operator: A ``FermionHamiltonian`` or ``MajoranaSparse``.
    ///     `n_modes`: Number of modes. Defaults to the operator's own.
    #[staticmethod]
    #[pyo3(signature = (operator, n_modes = None))]
    fn from_hamiltonian(
        operator: EncodeInput<'_>,
        n_modes: Option<usize>,
    ) -> Result<Self, CoreError> {
        let (sparse, own_n_modes) = match operator {
            EncodeInput::Fermion(fham) => (fham.inner.to_majorana_sparse(), fham.inner.n_modes()),
            EncodeInput::Majorana(msparse) => (msparse.inner.clone(), msparse.n_modes),
        };
        Ok(Self(MajoranaCircuit::from_majorana_sparse(
            &sparse,
            n_modes.unwrap_or(own_n_modes),
        )?))
    }

    /// Canonical Majorana indices of each gate, in order.
    #[getter]
    fn terms(&self) -> Vec<Vec<u16>> {
        self.0
            .terms
            .iter()
            .map(|term| term.as_slice().to_vec())
            .collect()
    }

    /// Rotation angle of each gate, wrapped into ``(-π, π]``.
    #[getter]
    fn angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.0.angles.clone().into_pyarray(py)
    }

    /// Number of fermionic modes.
    #[getter]
    fn n_modes(&self) -> usize {
        self.0.n_modes
    }

    fn __len__(&self) -> usize {
        self.0.n_terms()
    }

    /// Merge consecutive gates on the same term by summing their angles.
    ///
    /// Non-adjacent repeats are not merged. Gates whose merged angle is ~0
    /// are dropped, and the gates either side of them then become adjacent:
    /// ``A, B(θ), B(-θ), A`` merges to a single ``A``.
    fn merge_adjacent(&self) -> Self {
        Self(self.0.merge_adjacent())
    }

    /// Total absolute angle applied with each distinct operator.
    ///
    /// Args:
    ///     normalise: If ``True``, divide by the total so the weights sum to 1.
    ///
    /// Returns:
    ///     ``(terms, weights)``: distinct terms in order of first occurrence
    ///     and their summed ``|θ|``.
    #[pyo3(signature = (normalise = false))]
    fn operator_weights<'py>(
        &self,
        py: Python<'py>,
        normalise: bool,
    ) -> (Vec<Vec<u16>>, Bound<'py, PyArray1<f64>>) {
        let (terms, weights) = if normalise {
            self.0.relative_operator_weights()
        } else {
            self.0.operator_weights()
        };
        (
            terms.iter().map(|t| t.as_slice().to_vec()).collect(),
            weights.into_pyarray(py),
        )
    }

    /// Relabel the Majorana operators: every ``γ_a`` becomes ``γ_perm[a]``.
    ///
    /// Gate order and repeats are kept; a gate's angle is negated when its
    /// relabelled indices sort with an odd permutation. With ``perm`` from
    /// ``MajoranaEncoding.majorana_permutation_to``, this moves a circuit
    /// between two encodings of the same tree.
    ///
    /// Args:
    ///     perm: A permutation of ``range(2 * n_modes)``.
    ///
    /// Returns:
    ///     MajoranaCircuit: The relabelled circuit.
    ///
    /// Raises:
    ///     ValueError: If ``perm`` is not a permutation of the right size.
    fn permuted(&self, perm: Vec<usize>) -> Result<Self, CoreError> {
        Ok(Self(self.0.permuted(&perm)?))
    }

    /// Reorder the gates by where their Majoranas sit in an encoding's tree.
    ///
    /// Gates are stably sorted by their Majoranas' depth-first ranks
    /// (``MajoranaEncoding.majorana_tree_order``), so rotations acting near
    /// each other in the tree become neighbours and a transpiler can cancel
    /// their shared CNOT ladders.
    ///
    /// This changes the circuit's unitary unless the gates commute. Use it
    /// only where gate order is free, such as within one Trotter step.
    ///
    /// Args:
    ///     encoding: The encoding whose tree defines the order.
    ///
    /// Returns:
    ///     MajoranaCircuit: A new circuit with the same gates, reordered.
    fn sorted_by_tree_order(
        &self,
        encoding: PyRef<'_, PyMajoranaEncoding>,
    ) -> Result<Self, CoreError> {
        if encoding.0.n_modes < self.0.n_modes {
            return Err(CoreError::Value(format!(
                "circuit acts on {} modes but the encoding has {}",
                self.0.n_modes, encoding.0.n_modes
            )));
        }
        Ok(Self(
            self.0.sorted_by_rank(&encoding.0.majorana_tree_order()),
        ))
    }

    /// The gate generators ``θ · i^m · γ_S`` as a ``MajoranaSparse``, one
    /// term per gate, unmerged.
    fn to_majorana_sparse(&self) -> PyMajoranaSparse {
        PyMajoranaSparse {
            inner: self.0.to_majorana_sparse(),
            n_modes: self.0.n_modes,
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|o| self.0 == o.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "MajoranaCircuit(n_gates={}, n_modes={})",
            self.0.n_terms(),
            self.0.n_modes
        )
    }
}

/// An ordered circuit of Pauli rotation gates ``exp(-iθ P)``.
///
/// Pauli strings are qubit-0-first: character ``q`` acts on qubit ``q``.
/// Produced by ``MajoranaEncoding.encode_circuit`` and ``compile_circuit``.
#[pyclass(name = "PauliCircuit", module = "ferrmion.core")]
#[derive(Clone, Debug)]
pub struct PyPauliCircuit(pub PauliCircuit);

#[pymethods]
impl PyPauliCircuit {
    /// Pauli string of each gate, in order.
    #[getter]
    fn paulis(&self) -> Vec<String> {
        self.0.paulis.clone()
    }

    /// Rotation angle of each gate, wrapped into ``(-π, π]``.
    #[getter]
    fn angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.0.angles.clone().into_pyarray(py)
    }

    /// Number of qubits.
    #[getter]
    fn n_qubits(&self) -> usize {
        self.0.n_qubits
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Summed Pauli weight over all gates.
    fn pauli_weight(&self) -> usize {
        self.0.pauli_weight()
    }

    /// Summed ``|θ| · weight(P)`` over all gates.
    fn angle_weighted_pauli_weight(&self) -> f64 {
        self.0.angle_weighted_pauli_weight()
    }

    /// The gates as a list of ``(pauli, angle)`` pairs.
    fn to_list(&self) -> Vec<(String, f64)> {
        self.0
            .paulis
            .iter()
            .cloned()
            .zip(self.0.angles.iter().copied())
            .collect()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|o| self.0 == o.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliCircuit(n_gates={}, n_qubits={})",
            self.0.len(),
            self.0.n_qubits
        )
    }
}

/// Compile a Majorana circuit with TOPP-HATT.
///
/// Runs ``topphatt`` on the circuit, counting each distinct operator once per
/// gate that applies it after merging adjacent gates on the same operator,
/// then encodes every input gate in order.
///
/// Args:
///     flatpack: Initial ternary tree; its topology and qubit labels are kept.
///     `n_qubits`: Total number of qubits.
///     circuit: The ``MajoranaCircuit`` to compile.
///     parallelize: Use multi-threaded evaluation via Rayon.
///     heuristic: Node-selection strategy. One of ``"min_weight"``
///         (default), ``"x_first"``, ``"z_first"``, or ``"random"``.
///     seed: RNG seed for ``heuristic="random"``. Ignored otherwise.
///
/// Returns:
///     Tuple of ``(PauliCircuit, MajoranaEncoding)``.
#[pyfunction(name = "compile_circuit")]
#[pyo3(signature = (flatpack, n_qubits, circuit, parallelize = true, heuristic = "min_weight", seed = None))]
pub(crate) fn compile_circuit_py(
    py: Python<'_>,
    flatpack: TTFlatpack,
    n_qubits: usize,
    circuit: PyRef<'_, PyMajoranaCircuit>,
    parallelize: bool,
    heuristic: &str,
    seed: Option<u64>,
) -> Result<(PyPauliCircuit, PyMajoranaEncoding), CoreError> {
    let heuristic = NodeOrderHeuristic::parse(heuristic, seed).map_err(CoreError::Value)?;
    let circuit = circuit.0.clone();
    let (encoding, compiled) = py.allow_threads(|| -> Result<_, CoreError> {
        let tree = TernaryTree::from_flatpack_naive(&flatpack)?;
        Ok(compile_circuit(
            &circuit,
            tree,
            n_qubits,
            parallelize,
            heuristic,
        )?)
    })?;
    Ok((PyPauliCircuit(compiled), PyMajoranaEncoding(encoding)))
}
