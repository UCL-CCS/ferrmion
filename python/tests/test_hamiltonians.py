"""Tests for Hamiltonan Functions."""
from ferrmion import TernaryTree
from ferrmion.hamiltonians import (
    molecular_hamiltonian_template,
    fill_template,
    molecular_hamiltonian,
)
import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh
from pytest import fixture
import logging
logger = logging.getLogger(__name__)


@fixture(scope="module")
def filled_template(water_integrals, water_tt):
    symplectic_operators = water_tt.JW()._build_symplectic_matrix()
    # func_ham = molecular_hamiltonian_template(symplectic_operators[0], symplectic_operators[1])
    func_ham = molecular_hamiltonian_template(
        symplectic_operators[0], symplectic_operators[1], True
    )
    filled_template = fill_template(
        func_ham,
        0,
        water_integrals[0],
        0.5 * water_integrals[1],
        water_tt.default_mode_op_map,
    )
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

from ferrmion.core import encode_standard

def test_core_standard(water_eigenvalues, water_integrals):
    ones = water_integrals[0]
    twos = 0.5*water_integrals[1]
    mh = molecular_hamiltonian(TernaryTree(14).JW(), ones, twos)
    qham = encode_standard("JW", 14,14, ["+-","++--"], [ones, twos])

    logger.debug([((i[0], mh[i[0]]), (i[0],qham[i[1]])) for i in zip(sorted(mh)[:50], sorted(qham)[:50])])

    ofop4 = QubitOperator()
    for k, v in qham.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop4+= QubitOperator(term=string, coefficient=v)
    diag4, _ = eigsh(get_sparse_operator(ofop4), k=6, which="SA")
    print(diag4)
    print(water_eigenvalues)
    assert np.allclose(sorted(diag4), sorted(water_eigenvalues))
