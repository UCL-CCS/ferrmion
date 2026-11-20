"""Tests for MajoranaCircuit, PauliCircuit and the TOPP-HATT circuit compiler."""

from functools import reduce

import ferrmion as fr
import numpy as np
import pytest
from ferrmion import MajoranaCircuit, PauliCircuit
from ferrmion.core import compile_circuit, topphatt
from ferrmion.encode.ternary_tree import TernaryTree

PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

TREES = [
    TernaryTree.JordanWigner,
    TernaryTree.BravyiKitaev,
    TernaryTree.Parity,
    TernaryTree.JKMN,
]


def pauli_matrix(pauli: str) -> np.ndarray:
    return reduce(np.kron, (PAULIS[c] for c in pauli))


def rotation(generator: np.ndarray, theta: float) -> np.ndarray:
    """exp(-iθA) for an involution A."""
    return np.cos(theta) * np.eye(len(generator)) - 1j * np.sin(theta) * generator


def majorana_unitary(encoding, circuit: MajoranaCircuit) -> np.ndarray:
    dim = 2**encoding.n_qubits
    u = np.eye(dim, dtype=complex)
    for term, theta in zip(circuit.terms, circuit.angles):
        gamma = np.eye(dim, dtype=complex)
        for index in term:
            pauli, coeff = encoding.encode_majorana_product([index])
            gamma = gamma @ (coeff * pauli_matrix(pauli))
        k = len(term)
        gamma = gamma * 1j ** ((k * (k - 1) // 2) % 2)
        u = rotation(gamma, theta) @ u
    return u


def pauli_unitary(circuit: PauliCircuit) -> np.ndarray:
    u = np.eye(2**circuit.n_qubits, dtype=complex)
    for pauli, theta in circuit.to_list():
        u = rotation(pauli_matrix(pauli), theta) @ u
    return u


def random_circuit(n_modes: int, n_gates: int, seed: int) -> MajoranaCircuit:
    rng = np.random.default_rng(seed)
    terms = [
        [int(i) for i in rng.choice(2 * n_modes, size=rng.integers(1, 5), replace=False)]
        for _ in range(n_gates)
    ]
    angles = rng.uniform(-7, 7, size=n_gates).tolist()
    return MajoranaCircuit(terms, angles, n_modes)


def test_circuit_construction_keeps_order_and_repeats():
    circuit = MajoranaCircuit([[1, 0], [2, 3], [0, 1], [0, 0]], [0.5, 0.25, 0.5, 1.0], 2)
    # [0, 0] is the identity and is dropped; [1, 0] = -[0, 1].
    assert len(circuit) == 3
    assert circuit.terms == [[0, 1], [2, 3], [0, 1]]
    np.testing.assert_allclose(circuit.angles, [-0.5, 0.25, 0.5])
    np.testing.assert_allclose(
        MajoranaCircuit([[0, 1]], [2 * np.pi - 0.1], 1).angles, [-0.1]
    )


def test_circuit_operator_weights():
    circuit = MajoranaCircuit([[2, 3], [0, 1], [2, 3]], [0.5, 0.25, -1.0], 2)
    terms, weights = circuit.operator_weights()
    assert terms == [[2, 3], [0, 1]]
    np.testing.assert_allclose(weights, [1.5, 0.25])
    _, relative = circuit.operator_weights(normalise=True)
    np.testing.assert_allclose(relative.sum(), 1.0)


def test_circuit_rejects_bad_input():
    with pytest.raises(ValueError):
        MajoranaCircuit([[0, 4]], [0.1], 2)
    with pytest.raises(ValueError):
        MajoranaCircuit([[0, 1]], [0.1, 0.2], 2)


@pytest.mark.parametrize("encoding", TREES)
@pytest.mark.parametrize("seed", [0, 1])
def test_encode_circuit_matches_majorana_unitary(encoding, seed):
    n_modes = 3
    circuit = random_circuit(n_modes, 12, seed)
    enc = encoding(n_modes).build_encoding()
    compiled = enc.encode_circuit(circuit)
    assert len(compiled) == len(circuit)
    np.testing.assert_allclose(
        pauli_unitary(compiled), majorana_unitary(enc, circuit), atol=1e-10
    )


@pytest.mark.parametrize("encoding", TREES)
@pytest.mark.parametrize("split_factor", [1.0, 2.0, 4.0])
def test_compile_circuit_from_h2(encoding, split_factor, h2_mol_data_sets):
    fham = fr.molecular_hamiltonian(
        h2_mol_data_sets["ones"],
        h2_mol_data_sets["twos"],
        h2_mol_data_sets["constant_energy"],
    )
    circuit = MajoranaCircuit.from_hamiltonian(fham)
    tree = encoding(fham.n_modes)
    initial_children = tree.root_node.child_strings
    compiled = tree.compile_circuit(circuit, split_factor=split_factor)
    # Topology is preserved, and the circuit keeps its order and length.
    assert tree.root_node.child_strings == initial_children
    assert len(compiled) == len(circuit)
    assert compiled == tree._encoding.encode_circuit(circuit)
    if fham.n_modes <= 4:
        np.testing.assert_allclose(
            pauli_unitary(compiled),
            majorana_unitary(tree._encoding, circuit),
            atol=1e-10,
        )


def test_compile_circuit_weights_heavy_operators():
    """Repeated, large rotations should not cost more than ignoring angles."""
    n_modes = 4
    circuit = MajoranaCircuit(
        [[0, 7]] * 3 + [[1, 2], [3, 4], [5, 6]],
        [1.5] * 3 + [0.01] * 3,
        n_modes,
    )
    tree = TernaryTree.JordanWigner(n_modes)
    plain_encoding = topphatt(
        tree.flatpack(), n_modes, circuit.to_majorana_sparse(), parallelize=False
    )
    plain = plain_encoding.encode_circuit(circuit)
    weighted, _ = compile_circuit(tree.flatpack(), n_modes, circuit, parallelize=False)
    assert weighted.angle_weighted_pauli_weight() <= plain.angle_weighted_pauli_weight()


@pytest.mark.parametrize("heuristic", ["min_weight", "x_first", "z_first", "random"])
def test_compile_circuit_heuristics_and_seed(heuristic):
    circuit = random_circuit(5, 30, 7)
    tree = TernaryTree.JKMN(5)
    first, _ = compile_circuit(tree.flatpack(), 5, circuit, heuristic=heuristic, seed=3)
    second, _ = compile_circuit(tree.flatpack(), 5, circuit, heuristic=heuristic, seed=3)
    assert first == second


def test_compile_circuit_invalid_split_factor():
    circuit = random_circuit(3, 5, 0)
    with pytest.raises(ValueError, match="split_factor"):
        compile_circuit(TernaryTree.JordanWigner(3).flatpack(), 3, circuit, split_factor=0.5)
