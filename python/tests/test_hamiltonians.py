"""Tests for Hamiltonan Functions."""

from ferrmion.hamiltonians import molecular_hamiltonian_template
from ferrmion import core
import numpy as np
from ferrmion.hamiltonians.utils import fill_template
from ferrmion.hamiltonians.utils import to_qubit_hamiltonian, to_symplectic_hamiltonian
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh
from pytest import fixture

@fixture(scope="module")
def filled_template(water_integrals, water_tt):
    symplectic_operators = water_tt.JW()._build_symplectic_matrix()
    # func_ham = molecular_hamiltonian_template(symplectic_operators[0], symplectic_operators[1])
    func_ham = core.molecular_hamiltonian_template(symplectic_operators[0], symplectic_operators[1])
    filled_template = fill_template(water_integrals[0], 0.5*water_integrals[1], func_ham, water_tt.default_mode_op_map)
    return filled_template

def test_water_template(filled_template, water_eigenvalues, water_tt):
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

def test_hamiltonian_coefficients_agree(water_tt, filled_template):
    symplectic_ham = to_symplectic_hamiltonian(water_tt.n_qubits, filled_template)
    pauli_ham = to_qubit_hamiltonian(water_tt.n_qubits, filled_template)

    assert symplectic_ham[0] == [*pauli_ham.values()]
