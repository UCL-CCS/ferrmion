"""Tests for building Qiskit circuits from ferrmion Pauli rotation circuits.

``exp(-iθP)`` is one unitary, but a CNOT ladder is only one decomposition of
it. These tests check the ``chain`` and ``fountain`` ladders against Qiskit's
own Pauli-evolution synthesis and against the matrix exponential, then check a
whole compiled circuit end to end.
"""

import numpy as np
import pytest

pytest.importorskip("qiskit")

import ferrmion as fr
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import LieTrotter
from scipy.linalg import expm

from ferrmion import MajoranaCircuit
from ferrmion.encode.ternary_tree import TernaryTree
from ferrmion.interop.circuits import (
    CX_STRUCTURES,
    pauli_circuit_to_qiskit,
    pauli_rotation,
)

from .test_optimize.test_circuit_compile import (
    TREES,
    majorana_unitary,
    pauli_unitary,
    random_circuit,
)

# Operator() is exponential in qubit count; a rotation's decomposition does not
# depend on the qubits around it, so 5 qubits covers every ladder pattern.
MAX_QUBITS = 5


@st.composite
def pauli_strings(draw, min_qubits: int = 2, max_qubits: int = MAX_QUBITS):
    """A non-identity qubit-0-first Pauli string."""
    n_qubits = draw(st.integers(min_value=min_qubits, max_value=max_qubits))
    string = draw(
        st.lists(st.sampled_from("IXYZ"), min_size=n_qubits, max_size=n_qubits).map(
            "".join
        )
    )
    assume(set(string) != {"I"})
    return string


angles = st.floats(
    min_value=-2.0 * np.pi, max_value=2.0 * np.pi, allow_nan=False, allow_infinity=False
)


def qiskit_rotation(string: str, angle: float, cx_structure: str) -> QuantumCircuit:
    """``exp(-i angle P)`` via Qiskit; its labels are qubit-0-last, hence the reversal."""
    operator = SparsePauliOp([string[::-1]], np.array([angle]))
    circuit = QuantumCircuit(len(string))
    circuit.append(
        PauliEvolutionGate(
            operator, time=1.0, synthesis=LieTrotter(reps=1, cx_structure=cx_structure)
        ),
        range(len(string)),
    )
    return circuit.decompose(reps=2)


def phase_free_deviation(a: np.ndarray, b: np.ndarray) -> float:
    """Largest entrywise gap between two unitaries, ignoring a global phase."""
    phase = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
    return float(np.abs(b - phase * a).max())


@pytest.mark.parametrize("cx_structure", CX_STRUCTURES)
@given(string=pauli_strings(), angle=angles)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ladders_match_qiskit_synthesis(string, angle, cx_structure):
    ours = Operator(pauli_rotation(string, angle, cx_structure)).data
    theirs = Operator(qiskit_rotation(string, angle, cx_structure)).data
    assert phase_free_deviation(ours, theirs) < 1e-9


@pytest.mark.parametrize("cx_structure", CX_STRUCTURES)
@given(string=pauli_strings(), angle=angles)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_rotation_is_exactly_exp_minus_i_theta_pauli(string, angle, cx_structure):
    """Pin the convention exactly (including global phase), not just agreement."""
    pauli = Operator(SparsePauliOp([string[::-1]], np.array([1.0]))).data
    expected = expm(-1j * angle * pauli)
    actual = Operator(pauli_rotation(string, angle, cx_structure)).data
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@given(string=pauli_strings(), angle=angles.filter(lambda a: abs(a) > 0.05))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_a_perturbed_angle_is_detected(string, angle):
    """Guard against the equivalence checks passing vacuously."""
    ours = Operator(pauli_rotation(string, angle)).data
    perturbed = Operator(qiskit_rotation(string, angle * 1.05, "fountain")).data
    assert phase_free_deviation(ours, perturbed) > 1e-6


def test_unknown_cx_structure_raises():
    with pytest.raises(ValueError, match="cx_structure"):
        pauli_rotation("XZ", 0.1, cx_structure="staircase")


def test_identity_rotation_is_empty():
    assert pauli_rotation("III", 0.3).size() == 0


@pytest.mark.parametrize("cx_structure", CX_STRUCTURES)
@pytest.mark.parametrize("encoding", TREES)
def test_compiled_circuit_matches_majorana_unitary(encoding, cx_structure):
    n_modes = 3
    circuit = random_circuit(n_modes, 12, seed=5)
    tree = encoding(n_modes)
    compiled = tree.compile_circuit(circuit, parallelize=False)
    qc = pauli_circuit_to_qiskit(compiled, cx_structure)
    # Qiskit orders qubit 0 least significant; the reference puts it first.
    actual = Operator(qc).reverse_qargs().data
    np.testing.assert_allclose(actual, pauli_unitary(compiled), atol=1e-9)
    np.testing.assert_allclose(actual, majorana_unitary(tree._encoding, circuit), atol=1e-9)


def test_h2_trotter_step_round_trip(h2_mol_data_sets):
    fham = fr.molecular_hamiltonian(
        h2_mol_data_sets["ones"],
        h2_mol_data_sets["twos"],
        h2_mol_data_sets["constant_energy"],
    )
    circuit = MajoranaCircuit.from_hamiltonian(fham)
    tree = TernaryTree.JKMN(fham.n_modes)
    compiled = tree.compile_circuit(circuit)
    qc = pauli_circuit_to_qiskit(compiled, "fountain")
    n_cx = qc.count_ops().get("cx", 0)
    expected_cx = sum(
        2 * max(sum(c != "I" for c in p) - 1, 0) for p in compiled.paulis
    )
    assert n_cx == expected_cx
    if fham.n_modes <= 4:
        actual = Operator(qc).reverse_qargs().data
        np.testing.assert_allclose(actual, pauli_unitary(compiled), atol=1e-9)
