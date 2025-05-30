"""Tests for base fermion to qubit encoding class"""

import numpy as np
import pytest
from ferrmion.slow import slow_hartree_fock_state
from ferrmion.encode.ternary_tree import TernaryTree

np.random.seed(1710)


@pytest.fixture
def four_mode_tt():
    return TernaryTree.from_hamiltonian_coefficients((np.random.random((4, 4)), np.random.random((4, 4, 4, 4))))


@pytest.fixture
def sixteen_mode_tt():
    return TernaryTree.from_hamiltonian_coefficients((np.random.random((16, 16)), np.random.random((16, 16, 16, 16))))


def test_edge_operator_map(four_mode_tt):
    edge_map, weights = (
        four_mode_tt.JW()._edge_operator_map()
    )
    assert edge_map == {
        (0, 0): {b"\x00": 0.25, b"\x08": -0.25},
        (0, 1): {b"\xc0": 0.5, b"\xcc": -0.5},
        (0, 2): {b"\xa4": 0.5, b"\xae": -0.5},
        (0, 3): {b"\x96": 0.5, b"\x9f": -0.5},
        (1, 1): {b"\x00": 0.25, b"\x04": -0.25},
        (1, 2): {b"`": 0.5, b"f": -0.5},
        (1, 3): {b"R": 0.5, b"W": -0.5},
        (2, 2): {b"\x00": 0.25, b"\x02": -0.25},
        (2, 3): {b"0": 0.5, b"3": -0.5},
        (3, 3): {b"\x00": 0.25, b"\x01": -0.25},
    }
    assert np.all(
        weights
        == [
            [-0.125, 0.0, 0.0, 0.0],
            [0.0, -0.125, 0.0, 0.0],
            [0.0, 0.0, -0.125, 0.0],
            [0.0, 0.0, 0.0, -0.125],
        ]
    )

def test_default_vacuum_state(four_mode_tt):
    assert np.all(four_mode_tt.vacuum_state == np.array([0] * 4))

def test_valid_vacuum_state(four_mode_tt):
    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = [0] * 3
    assert "4" in str(excinfo.value)
    assert "length" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = [0] * 5
    assert "4" in str(excinfo.value)
    assert "length" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = np.array([[0], [0]])
    assert "dimension" in str(excinfo.value)


def test_hartree_fock_state(sixteen_mode_tt):
    jw = sixteen_mode_tt.JW()
    hartree_fock_state = jw.hartree_fock_state
    nq = jw.n_qubits // 2
    print(hartree_fock_state(np.array([True] * nq + [False] * nq, dtype=bool)))
    assert (
        hartree_fock_state(np.array([True] * nq + [False] * nq, dtype=bool))[0]
    ) == [1.0]
    assert np.all(
        hartree_fock_state(np.array([True] * nq + [False] * nq, dtype=bool))[1]
        == np.array([[True] * nq + [False] * nq], dtype=bool)
    )
    assert np.all(
        hartree_fock_state(
            np.array([True] * (nq + 1) + [False] * (nq - 1), dtype=bool)
        )[1]
        == np.array([[True] * (nq + 1) + [False] * (nq - 1)], dtype=bool)
    )


def test_slow_hartree_fock_state(sixteen_mode_tt):
    jw = sixteen_mode_tt.JW()
    mode_op_map = jw.default_mode_op_map
    nq = jw.n_qubits // 2

    assert np.all(
        slow_hartree_fock_state(jw, [1] * nq + [0] * nq, mode_op_map)[0] == [1]
    )
    assert np.all(
        slow_hartree_fock_state(jw, [1] * nq + [0] * nq, mode_op_map)[1]
        == np.array([1] * nq + [0] * nq)
    )
    assert np.all(
        slow_hartree_fock_state(jw, [1] * (nq + 1) + [0] * (nq - 1), mode_op_map)[1]
        == np.array([1] * (nq + 1) + [0] * (nq - 1))
    )


def test_slow_hartree_fock_state_errors(four_mode_tt):
    with pytest.raises(ValueError) as excinfo:
        slow_hartree_fock_state(
            four_mode_tt.JW(), [1] * 3 + [0] * 2, four_mode_tt.JW().default_mode_op_map
        )[1] == np.array([1, 1, 0, 0])
    with pytest.raises(ValueError) as excinfo:
        slow_hartree_fock_state(
            four_mode_tt.JW(), [1] * 4 + [0] * 2, four_mode_tt.JW().default_mode_op_map
        )[1] == np.array([1, 1, 0, 0, 0])

    # add some tests here for other encodings, do them by hand to be confident if you like


def test_benchmark_hf_state(benchmark, sixteen_mode_tt):
    result = benchmark(test_hartree_fock_state, sixteen_mode_tt)


def test_benchmark_slow_hf_state(benchmark, sixteen_mode_tt):
    result = benchmark(test_slow_hartree_fock_state, sixteen_mode_tt)


def test_four_benchmark_hf_state(benchmark, four_mode_tt):
    result = benchmark(test_hartree_fock_state, four_mode_tt)


def test_four_benchmark_slow_hf_state(benchmark, four_mode_tt):
    result = benchmark(test_slow_hartree_fock_state, four_mode_tt)

def test_number_operator(four_mode_tt):
    tree = four_mode_tt.JW()
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    assert tree.edge_operator((0,0)) == tree.number_operator(0)
    assert tree.edge_operator((1,1)) == tree.number_operator(1)
    assert tree.edge_operator((2,2)) == tree.number_operator(2)
    assert tree.edge_operator((3,3)) == tree.number_operator(3)

    with pytest.raises(ValueError) as excinfo:
        tree.number_operator(tree.n_modes+1)
    assert "indices invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tree.number_operator(-1)
    assert "indices invalid" in str(excinfo.value)

def test_edge_operator(four_mode_tt):
    tree = four_mode_tt.JW()
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    assert tree.edge_operator((0,0)) == tree.number_operator(0)
    tree.edge_operator((0,1)) == [('YXII', -0.25j), ('YYII', 0.25), ('XXII', 0.25), ('XYII', 0.25j)]

    left = np.array([t[1] for t in tree.edge_operator((1,0))], dtype=np.complexfloating)
    right = np.array([np.conjugate(t[1]) for t in tree.edge_operator((0,1))], dtype=np.complexfloating)
    assert np.all(left == right)

    tree = four_mode_tt.JKMN()
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    assert tree.edge_operator((0,3)) == [('YZIX', -0.25j), ('YZIY', 0.25), ('XIZX', 0.25), ('XIZY', 0.25j)]

    with pytest.raises(ValueError) as excinfo:
        tree.edge_operator((0, tree.n_modes+1))
    assert "indices invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tree.edge_operator((tree.n_modes+1, 0))
    assert "indices invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tree.number_operator((0, -1))
    assert "indices invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tree.number_operator((-1, 0))
    assert "indices invalid" in str(excinfo.value)
