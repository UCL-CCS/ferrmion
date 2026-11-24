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
def test_compile_circuit_from_h2(encoding, h2_mol_data_sets):
    fham = fr.molecular_hamiltonian(
        h2_mol_data_sets["ones"],
        h2_mol_data_sets["twos"],
        h2_mol_data_sets["constant_energy"],
    )
    circuit = MajoranaCircuit.from_hamiltonian(fham)
    tree = encoding(fham.n_modes)
    initial_children = tree.root_node.child_strings
    compiled = tree.compile_circuit(circuit)
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


def test_compile_circuit_weights_repeated_operators():
    """Counting repeats should not cost more than counting each operator once."""
    n_modes = 4
    # Repeats are spread out: adjacent ones would merge into one gate.
    circuit = MajoranaCircuit(
        [[0, 7], [1, 2], [0, 7], [3, 4], [0, 7], [5, 6]],
        [0.01, 1.5] * 3,
        n_modes,
    )
    tree = TernaryTree.JordanWigner(n_modes)
    plain_encoding = topphatt(
        tree.flatpack(), n_modes, circuit.to_majorana_sparse(), parallelize=False
    )
    plain = plain_encoding.encode_circuit(circuit)
    counted, _ = compile_circuit(tree.flatpack(), n_modes, circuit, parallelize=False)
    assert counted.pauli_weight() <= plain.pauli_weight()


@pytest.mark.parametrize("encoding", TREES)
def test_topphatt_takes_a_circuit_directly(encoding):
    circuit = random_circuit(5, 40, 3)
    flatpack = encoding(5).flatpack()
    direct = topphatt(flatpack, 5, circuit, parallelize=False)
    compiled, via_compile = compile_circuit(flatpack, 5, circuit, parallelize=False)
    assert direct.encode_circuit(circuit) == compiled
    assert direct.majorana_permutation_to(via_compile) == list(range(10))
    tree = encoding(5)
    assert tree.topphatt(circuit, parallelize=False).majorana_permutation_to(
        direct
    ) == list(range(10))
    # Only occurrences matter: new angles on the same gates change nothing.
    reangled = MajoranaCircuit(
        circuit.terms, np.linspace(0.1, 3.0, len(circuit)).tolist(), 5
    )
    again = topphatt(flatpack, 5, reangled, parallelize=False)
    assert again.majorana_permutation_to(direct) == list(range(10))


def test_adjacent_repeats_merge_before_counting():
    """Adjacent gates on one operator are one gate; others stay separate."""
    circuit = random_circuit(5, 40, 8)
    terms, angles = circuit.terms, circuit.angles.tolist()
    doubled = MajoranaCircuit(
        [t for t in terms for _ in range(2)],
        [a / 2 for a in angles for _ in range(2)],
        5,
    )
    flatpack = TernaryTree.JKMN(5).flatpack()
    single = topphatt(flatpack, 5, circuit, parallelize=False)
    assert topphatt(flatpack, 5, doubled, parallelize=False).majorana_permutation_to(
        single
    ) == list(range(10))
    # A gate followed by its inverse cancels and costs nothing.
    inverse_pair = MajoranaCircuit(
        terms[:1] + [[0, 1], [0, 1]] + terms[1:], angles[:1] + [0.4, -0.4] + angles[1:], 5
    )
    assert topphatt(
        flatpack, 5, inverse_pair, parallelize=False
    ).majorana_permutation_to(single) == list(range(10))


def test_topphatt_rejects_backend_for_a_circuit():
    circuit = random_circuit(3, 5, 0)
    flatpack = TernaryTree.JordanWigner(3).flatpack()
    with pytest.raises(ValueError, match="backend"):
        topphatt(flatpack, 3, circuit, backend="sparse")
    with pytest.raises(ValueError, match="split_budget"):
        topphatt(flatpack, 3, circuit, split_budget=10)


@pytest.mark.parametrize("heuristic", ["min_weight", "x_first", "z_first", "random"])
def test_compile_circuit_heuristics_and_seed(heuristic):
    circuit = random_circuit(5, 30, 7)
    tree = TernaryTree.JKMN(5)
    first, _ = compile_circuit(tree.flatpack(), 5, circuit, heuristic=heuristic, seed=3)
    second, _ = compile_circuit(tree.flatpack(), 5, circuit, heuristic=heuristic, seed=3)
    assert first == second


def branch_qubit_order_key(encoding):
    """The circuit branch's JW-spine ordering: (first X/Y qubit, index parity)."""
    qubits = []
    for index in range(2 * encoding.n_modes):
        pauli, _ = encoding.encode_majorana_product([index])
        qubits.append(next(q for q, c in enumerate(pauli) if c in "XY"))
    return lambda term: tuple(sorted((qubits[i], i & 1) for i in term))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_order_matches_jw_spine_ordering(seed):
    """On a JW spine, tree order reproduces the branch's (qubit, parity) sort,
    including after TOPP-HATT relabels the leaves."""
    n_modes = 5
    circuit = random_circuit(n_modes, 40, seed)
    tree = TernaryTree.JordanWigner(n_modes)
    assert tree.build_encoding().majorana_tree_order() == list(range(2 * n_modes))
    tree.compile_circuit(circuit, parallelize=False)
    encoding = tree._encoding
    sorted_circuit = circuit.sorted_by_tree_order(encoding)
    key = branch_qubit_order_key(encoding)
    assert sorted_circuit.terms == sorted(circuit.terms, key=key)


@pytest.mark.parametrize("encoding", TREES)
def test_sorted_by_tree_order_keeps_every_gate(encoding):
    circuit = random_circuit(4, 30, 9)
    enc = encoding(4).build_encoding()
    rank = enc.majorana_tree_order()
    assert sorted(rank) == list(range(8))
    sorted_circuit = circuit.sorted_by_tree_order(enc)
    assert sorted(zip(map(tuple, sorted_circuit.terms), sorted_circuit.angles)) == sorted(
        zip(map(tuple, circuit.terms), circuit.angles)
    )
    keys = [sorted(rank[i] for i in term) for term in sorted_circuit.terms]
    assert keys == sorted(keys)


def test_sorted_by_tree_order_rejects_small_encoding():
    circuit = random_circuit(4, 5, 0)
    with pytest.raises(ValueError, match="modes"):
        circuit.sorted_by_tree_order(TernaryTree.JordanWigner(3).build_encoding())


@pytest.mark.parametrize("encoding", TREES)
def test_permuting_to_naive_reproduces_compiled_circuit(encoding):
    """Prepare a circuit, compile it, then recover the compiled circuit from the
    naive encoding by relabelling Majoranas leaf for leaf."""
    n_modes = 5
    circuit = random_circuit(n_modes, 40, 13)
    tree = encoding(n_modes)
    naive = tree.build_encoding()
    compiled = tree.compile_circuit(circuit, parallelize=False)
    perm = tree._encoding.majorana_permutation_to(naive)
    assert sorted(perm) == list(range(2 * n_modes))
    assert naive.encode_circuit(circuit.permuted(perm)) == compiled


def test_mode_enumeration_permutation():
    naive = TernaryTree.JKMN(4).build_encoding()
    enumerated = naive.apply_mode_enumeration([3, 0, 2, 1])
    assert enumerated.majorana_permutation_to(naive) == [6, 7, 0, 1, 4, 5, 2, 3]


def test_permutation_errors():
    circuit = random_circuit(3, 5, 0)
    with pytest.raises(ValueError, match="permutation"):
        circuit.permuted([0, 1, 2, 3, 4, 4])
    jw = TernaryTree.JordanWigner(3).build_encoding()
    bk = TernaryTree.BravyiKitaev(3).build_encoding()
    with pytest.raises(ValueError, match="leaves"):
        bk.majorana_permutation_to(jw)
