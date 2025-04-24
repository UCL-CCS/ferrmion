"""Tests for base fermion to qubit encoding class"""
import pytest
import numpy as np
from ferrmion import TernaryTree

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