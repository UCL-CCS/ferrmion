"""Tests for base fermion to qubit encoding class"""
import pytest
import numpy as np
from ferrmion.ternary_tree import TernaryTree
from ferrmion import hartree_fock_state
from ferrmion.slow import slow_hartree_fock_state

np.random.seed(1710)

@pytest.fixture
def four_mode_tt():
    return TernaryTree(np.random.random((4,4)), np.random.random((4,4,4,4)))

def test_edge_operator_map():
    edge_map, weights = TernaryTree(np.ones((4,4)), np.zeros((4,4,4,4))).JW()._edge_operator_map()
    assert edge_map == {
        (0, 0): {b'\x00': 1, b'\x08': -1},
        (0, 1): {b'\xc0': 2, b'\xcc': -2},
        (0, 2): {b'\xa4': 2, b'\xae': -2},
        (0, 3): {b'\x96': 2, b'\x9f': -2},
        (1, 1): {b'\x00': 1, b'\x04': -1},
        (1, 2): {b'`': 2, b'f': -2},
        (1, 3): {b'R': 2, b'W': -2},
        (2, 2): {b'\x00': 1, b'\x02': -1},
        (2, 3): {b'0': 2, b'3': -2},
        (3, 3): {b'\x00': 1, b'\x01': -1},
    }
    assert np.all(weights == [[-0.5,  0. ,  0. ,  0. ],
        [ 0. , -0.5,  0. ,  0. ],
        [ 0. ,  0. , -0.5,  0. ],
        [ 0. ,  0. ,  0. , -0.5]])

def test_hamiltonian_coefficients_agree(four_mode_tt):
    coefficents, _ = four_mode_tt.BK().to_symplectic_hamiltonian()
    pauli_ham = four_mode_tt.BK().to_qubit_hamiltonian()

    assert coefficents == [*pauli_ham.values()]

def test_default_vaccum_state(four_mode_tt):
    assert np.all(four_mode_tt.vaccum_state == np.array([0]*4))

def test_valid_vaccum_state(four_mode_tt):

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vaccum_state = [0]*3
    assert "4" in str(excinfo.value)
    assert "length" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vaccum_state = [0]*5
    assert "4" in str(excinfo.value)
    assert "length" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vaccum_state = np.array([[0],[0]])
    assert "dimension" in str(excinfo.value)

def test_hartree_fock_state(four_mode_tt):
    jw = four_mode_tt.JW()
    print(jw.hartree_fock_state(np.array([1]*2 + [0]*2, dtype=bool)))
    assert(jw.hartree_fock_state(np.array([1]*2 + [0]*2, dtype=bool))[0]) == [1.]
    assert(np.all(jw.hartree_fock_state(np.array([1]*2 + [0]*2, dtype=bool))
[1] == np.array([[1,1,0,0]], dtype=np.bool)))
    assert(np.all(jw.hartree_fock_state(np.array([1]*3 + [0]*1, dtype=bool))[1] == np.array([[1,1,1,0]], dtype=np.bool)))

def test_slow_hartree_fock_state(four_mode_tt):
    jw = four_mode_tt.JW()
    mode_op_map = jw.default_mode_op_map

    assert np.all(slow_hartree_fock_state(jw, [1]*2 + [0]*2, mode_op_map)[0] == [1])
    assert np.all(slow_hartree_fock_state(jw, [1]*2 + [0]*2, mode_op_map)[1] == np.array([1,1,0,0]))
    assert np.all(slow_hartree_fock_state(jw, [1]*3 + [0]*1, mode_op_map)[1] == np.array([1,1,1,0]))

def test_slow_hartree_fock_state_errors(four_mode_tt):
    with pytest.raises(ValueError) as excinfo:
        slow_hartree_fock_state(four_mode_tt.JW(), [1]*3 + [0]*2, four_mode_tt.JW().default_mode_op_map)[1] == np.array([1,1,0,0])
    with pytest.raises(ValueError) as excinfo:
        slow_hartree_fock_state(four_mode_tt.JW(), [1]*4 + [0]*2, four_mode_tt.JW().default_mode_op_map)[1] == np.array([1,1,0,0,0])

    # add some tests here for other encodings, do them by hand to be confident if you like

def test_benchmark_hf_state(benchmark, four_mode_tt):
    result = benchmark(test_hartree_fock_state, four_mode_tt)

def test_benchmark_slow_hf_state(benchmark, four_mode_tt):
    result = benchmark(test_slow_hartree_fock_state, four_mode_tt)
