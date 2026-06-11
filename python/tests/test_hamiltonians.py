"""Tests for Hamiltonan Functions."""
from ferrmion import TernaryTree
from ferrmion.hamiltonians import (
    molecular_hamiltonian,
    FermionHamiltonian,
    hubbard_hamiltonian
)
from ferrmion.core import MajoranaEncoding
import pytest
import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh
from pytest import fixture
import logging
from hypothesis import given, settings, strategies as st
from .conftest import diagonalise_pauli_hamiltonian

rng = np.random.default_rng(46392034)
logger = logging.getLogger(__name__)

def test_molecular_hamiltonian_equivalent_explicit_fermion_hamiltonian(n_modes: int = 5):
    print(n_modes)
    ones = rng.random((n_modes,n_modes))
    twos = rng.random((n_modes,n_modes,n_modes,n_modes))
    constant_energy = rng.random()

    molh = molecular_hamiltonian(one_e_coeffs=ones, two_e_coeffs=twos, constant_energy=constant_energy)

    assert ones.shape == (n_modes,n_modes)
    assert twos.shape == (n_modes,n_modes, n_modes,n_modes)
    explicit_molh = FermionHamiltonian()
    explicit_molh.creation().annihilation().with_coefficients(ones)
    explicit_molh.creation().creation().annihilation().annihilation().with_coefficients(twos)
    # explicit_molh.add_constant(constant_energy)

    assert molh.terms.keys() == explicit_molh.terms.keys()
    assert np.all(molh.terms["+-"] == explicit_molh.terms["+-"])
    assert np.all(molh.terms["++--"] == explicit_molh.terms["++--"])

@pytest.mark.parametrize(
    "encoding",
    [
        MajoranaEncoding.jordan_wigner,
        MajoranaEncoding.bravyi_kitaev,
        MajoranaEncoding.parity,
        MajoranaEncoding.jkmn,
    ],
)
def test_encode_water_eigvals_equal_expected(encoding, water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    qham = encoding(14).encode(fham)
    assert np.isclose(qham["I"*14], -46.465600781952176)

    diag = diagonalise_pauli_hamiltonian(qham, 2)

    assert np.allclose(np.sort(diag), np.sort(water_data["eigvals"])[:2])

@pytest.mark.parametrize(
    "encoding",
    [
        MajoranaEncoding.jordan_wigner,
        MajoranaEncoding.bravyi_kitaev,
        MajoranaEncoding.parity,
        MajoranaEncoding.jkmn,
    ],
)
def test_encode_h2_eigvals_equal_expected(encoding, h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]
    fham = molecular_hamiltonian(ones, twos, e_nuc)
    qham = encoding(n_modes).encode(fham)

    diag = diagonalise_pauli_hamiltonian(qham, 2*n_modes)
    assert np.allclose(np.sort(diag), np.sort(h2_mol_data_sets["eigvals"]))


@given(n_modes=st.integers(2,5))
@settings(max_examples=10, deadline=None)
def test_encode_eigenvalues_constant(n_modes):
    ones = rng.random((n_modes,n_modes))
    twos = rng.random((n_modes,n_modes,n_modes,n_modes))
    e_nuc = rng.random()

    diags = []
    for encoding in [
        MajoranaEncoding.jordan_wigner,
        MajoranaEncoding.bravyi_kitaev,
        MajoranaEncoding.parity,
        MajoranaEncoding.jkmn,
    ]:
        fham = molecular_hamiltonian(ones, twos, e_nuc)
        qham = encoding(n_modes).encode(fham)
        diags.append(diagonalise_pauli_hamiltonian(qham, n_modes))

    for d in diags[1:]:
        assert np.allclose(np.sort(diags[0]), np.sort(d))
