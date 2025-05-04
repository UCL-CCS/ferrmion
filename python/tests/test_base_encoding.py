"""Tests for base fermion to qubit encoding class"""
import pytest
import numpy as np
from ferrmion.ternary_tree import TernaryTree

np.random.seed(1710)

@pytest.fixture
def four_mode_tt():
    return TernaryTree(np.random.random((4,4)), np.random.random((4,4,4,4)))

def test_edge_operator_map():
    edge_map, weights = TernaryTree(np.ones((4,4)), np.zeros((4,4,4,4))).JW()._edge_operator_map()
    assert edge_map == {
        (0, 0): {'[0 0 0 0 0 0 0 0]': 1, '[0 0 0 0 1 0 0 0]': -1},
        (0, 1): {'[1 1 0 0 1 1 0 0]': -2, '[1 1 0 0 0 0 0 0]': 2},
        (0, 2): {'[1 0 1 0 1 1 1 0]': -2, '[1 0 1 0 0 1 0 0]': 2},
        (0, 3): {'[1 0 0 1 1 1 1 1]': -2, '[1 0 0 1 0 1 1 0]': 2},
        (1, 1): {'[0 0 0 0 0 0 0 0]': 1, '[0 0 0 0 0 1 0 0]': -1},
        (1, 2): {'[0 1 1 0 0 1 1 0]': -2, '[0 1 1 0 0 0 0 0]': 2},
        (1, 3): {'[0 1 0 1 0 1 1 1]': -2, '[0 1 0 1 0 0 1 0]': 2},
        (2, 2): {'[0 0 0 0 0 0 0 0]': 1, '[0 0 0 0 0 0 1 0]': -1},
        (2, 3): {'[0 0 1 1 0 0 1 1]': -2, '[0 0 1 1 0 0 0 0]': 2},
        (3, 3): {'[0 0 0 0 0 0 0 0]': 1, '[0 0 0 0 0 0 0 1]': -1}}
    assert np.all(weights == [[-0.5,  0. ,  0. ,  0. ],
        [ 0. , -0.5,  0. ,  0. ],
        [ 0. ,  0. , -0.5,  0. ],
        [ 0. ,  0. ,  0. , -0.5]])

def test_hamiltonian_coefficients_agree(four_mode_tt):
    coefficents, _ = four_mode_tt.BK().to_symplectic_hamiltonian()
    pauli_ham = four_mode_tt.BK().to_qubit_hamiltonian()

    assert coefficents == [*pauli_ham.values()]

def test_default_vacuum_state(four_mode_tt):
    assert np.all(four_mode_tt.vacuum_state == np.array([0]*4))

def test_valid_vacuum_state(four_mode_tt):

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = [0]*3
    assert "4" in str(excinfo.value)
    assert "length" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = [0]*5
    assert "4" in str(excinfo.value)
    assert "length" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = np.array([[0],[0]])
    assert "dimension" in str(excinfo.value)

def test_hartree_fock_state(four_mode_tt):
    assert np.all(four_mode_tt.JW().hartree_fock_state([1]*2 + [0]*2)[0] == [1])

    assert np.all(four_mode_tt.JW().hartree_fock_state([1]*2 + [0]*2)[1] == np.array([1,1,0,0]))
    assert np.all(four_mode_tt.JW().hartree_fock_state([1]*3 + [0]*1)[1] == np.array([1,1,1,0]))
    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.JW().hartree_fock_state([1]*3 + [0]*2)[1] == np.array([1,1,0,0])
    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.JW().hartree_fock_state([1]*4 + [0]*2)[1] == np.array([1,1,0,0,0])

    # add some tests here for other encodings, do them by hand to be confident if you like

def test_benchmark_hf_state(benchmark, four_mode_tt):
    result = benchmark(test_hartree_fock_state, four_mode_tt)
