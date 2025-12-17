"""Tests for Hamiltonan Functions."""
from ferrmion import TernaryTree
from ferrmion.hamiltonians import (
    molecular_hamiltonian,
    FermionHamiltonian,
    hubbard_hamiltonian
)
from ferrmion.core import encode_standard
import pytest
import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh
from pytest import fixture
import logging
logger = logging.getLogger(__name__)


def test_molecular_hamiltonian():
    ones = np.eye(4)
    twos = np.ones((4,4,4,4))
    constant_energy = 10.
    molh = molecular_hamiltonian(one_e_coeffs=ones, two_e_coeffs=twos, constant_energy=constant_energy)
    explicit_molh = FermionHamiltonian()
    explicit_molh.creation().annihilation().with_coefficients(ones)
    explicit_molh.creation().creation().annihilation().annihilation().with_coefficients(twos)
    assert molh._terms.keys() == explicit_molh._terms.keys()
    assert np.all(molh._terms["+-"] == explicit_molh._terms["+-"]) 
    assert np.all(molh._terms["++--"] == explicit_molh._terms["++--"])

@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN"])
def test_core_standard(encoding, water_eigenvalues, water_integrals):
    ones = water_integrals[0]
    twos = 0.5*water_integrals[1]

    qham = encode_standard(encoding, 14,14, ["+-","++--"], [ones, twos], 0.)

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
