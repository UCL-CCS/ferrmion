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


def test_molecular_hamiltonian_equivalent_explicit_fermion_hamiltonian():
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
def test_encode_standard_water_eigvals_equal_expected(encoding, water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]

    qham = encode_standard(encoding, 14,14, ["+-","++--"], [ones, twos], e_nuc)

    ofop = QubitOperator()
    for k, v in qham.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop+= QubitOperator(term=string, coefficient=v)
    print(expected:=water_data["eigvals"])
    diag, _ = eigsh(get_sparse_operator(ofop), k=2, which="SA")
    print(diag)
    assert np.allclose(np.sort(diag), np.sort(expected)[:2])

@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN"])
def test_encode_standard_h2_eigvals_equal_expected(encoding, h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]
    qham = encode_standard(encoding, n_modes, n_modes, ["+-","++--"], [ones, twos], e_nuc)

    ofop = QubitOperator()
    for k, v in qham.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop+= QubitOperator(term=string, coefficient=v)
    print(expected:=h2_mol_data_sets["eigvals"])
    diag, _ = eigsh(get_sparse_operator(ofop), k=2*n_modes, which="SA")
    print(diag)
    assert np.allclose(np.sort(diag), np.sort(h2_mol_data_sets["eigvals"]))
