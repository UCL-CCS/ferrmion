"""Tests for Hamiltonan Functions."""
from ferrmion import TernaryTree
from ferrmion.encode import JKMN, BravyiKitaev, JordanWigner, ParityEncoding
from ferrmion.hamiltonians import (
    molecular_hamiltonian_template,
    fill_template,
    molecular_hamiltonian,
)
import pytest
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

@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN"])
def test_core_standard(encoding, water_eigenvalues, water_integrals):
    ones = water_integrals[0]
    twos = 0.5*water_integrals[1]

    qham = encode_standard(encoding, 14,14, ["+-","++--"], [ones, twos])

    ofop = QubitOperator()
    for k, v in qham.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop+= QubitOperator(term=string, coefficient=v)
    diag, _ = eigsh(get_sparse_operator(ofop), k=6, which="SA")
    print(diag)
    print(water_eigenvalues)
    assert np.allclose(sorted(diag), sorted(water_eigenvalues))
