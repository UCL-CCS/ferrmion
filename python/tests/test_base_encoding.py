"""Tests for base fermion to qubit encoding class"""

import numpy as np
import pytest
from ferrmion.slow import slow_hartree_fock_state
from ferrmion.encode.ternary_tree import TernaryTree

np.random.seed(1710)


@pytest.fixture
def four_mode_tt():
    return TernaryTree(n_modes=4)


@pytest.fixture
def sixteen_mode_tt():
    return TernaryTree(n_modes=16)

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
    # numpy doesn't like comparing empty arrays
    assert str(TernaryTree(n_modes=4).JW().edge_operator((0,0))) == str(TernaryTree(n_modes=4).JW().number_operator(0))
    assert str(TernaryTree(n_modes=4).JW().edge_operator((1,1))) == str(TernaryTree(n_modes=4).JW().number_operator(1))
    assert str(TernaryTree(n_modes=4).JW().edge_operator((2,2))) == str(TernaryTree(n_modes=4).JW().number_operator(2))
    assert str(TernaryTree(n_modes=4).JW().edge_operator((3,3))) == str(TernaryTree(n_modes=4).JW().number_operator(3))

    with pytest.raises(ValueError) as excinfo:
        tree.number_operator(tree.n_modes+1)
    assert "indices invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tree.number_operator(-1)
    assert "indices invalid" in str(excinfo.value)

def test_edge_operator(four_mode_tt):
    tree = four_mode_tt.JKMN()
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    left = np.array([t[2] for t in tree.edge_operator((1,0))], dtype=complex)
    right = np.array([np.conjugate(t[2]) for t in tree.edge_operator((0,1))], dtype=complex)
    assert np.all(right == left[[0,2,1,3]])
    assert np.all(left == np.array([ 0.  -0.25j,  -0.25+0.j  , 0.25+0.j  ,  0.  +0.25j]))
    assert np.all(right == np.array([ 0.  -0.25j,  0.25+0.j  , -0.25+0.j  ,  0.  +0.25j]))

    assert str(tree.edge_operator((0,3))[0]) == str(('YZX', np.array([0,1,3]), -0-0.25j))
    assert str(tree.edge_operator((0,3))[1]) == str(('YZY', np.array([0,1,3]), 0.25))
    assert str(tree.edge_operator((0,3))[2]) == str(('XZX', np.array([0,2,3]), 0.25))
    assert str(tree.edge_operator((0,3))[3]) == str(('XZY', np.array([0,2,3]), 0+0.25j))

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
