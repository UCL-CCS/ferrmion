"""Tests for Hamiltonan Functions."""

from ferrmion.hamiltonians import molecular_hamiltonian_template, fill_template, molecular_hamiltonian
import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh
from pytest import fixture

@fixture(scope="module")
def filled_template(water_integrals, water_tt):
    symplectic_operators = water_tt.JW()._build_symplectic_matrix()
    # func_ham = molecular_hamiltonian_template(symplectic_operators[0], symplectic_operators[1])
    func_ham = molecular_hamiltonian_template(symplectic_operators[0], symplectic_operators[1], True)
    filled_template = fill_template(func_ham, 0, water_integrals[0], 0.5*water_integrals[1], water_tt.default_mode_op_map)
    return filled_template

def test_basic_molecular_hamiltonian(filled_template, water_tt, water_integrals):
    mh = molecular_hamiltonian(water_tt.JW(), water_integrals[0], water_integrals[1])
    assert filled_template.keys() == mh.keys()

def test_template(filled_template, water_eigenvalues):
    ofop3 = QubitOperator()
    for k, v in filled_template.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop3 += QubitOperator(term=string, coefficient=v)
    diag3, _ = eigsh(get_sparse_operator(ofop3), k=6, which="SA")

    assert np.allclose(sorted(diag3), sorted(water_eigenvalues))
