"""Parity tests between the Python and Rust HATT implementations."""

import numpy as np

from ferrmion.core import fermionic_to_sparse_majorana
from ferrmion.hamiltonians import FermionHamiltonian
from ferrmion.optimize.hatt import (
    fast_hatt_rust,
    hamiltonian_adaptive_ternary_tree,
)


def test_hatt_rust_matches_python_small():
    """Rust HATT reproduces the Python tree on a small, hand-built fixture.

    The Python ``hamiltonian_adaptive_ternary_tree`` is treated as the
    reference and given the Majorana dict produced by
    ``fermionic_to_sparse_majorana`` from the same ``FermionHamiltonian`` the
    Rust path receives. Both paths therefore consume identical Majorana
    terms, so exact structural parity is expected.
    """
    n_modes = 3
    ones = np.zeros((n_modes, n_modes))
    ones[0, 0] = 1.0
    ones[1, 1] = -1.0
    ones[2, 2] = 0.5
    twos = np.zeros((n_modes, n_modes, n_modes, n_modes))
    twos[0, 1, 1, 0] = 0.3
    twos[1, 2, 2, 1] = -0.2

    fham = FermionHamiltonian(terms={"+-": ones, "++--": twos})

    sigs, coeffs = fham.signatures_and_coefficients
    majorana_ham = fermionic_to_sparse_majorana(sigs, coeffs, 0)

    py = hamiltonian_adaptive_ternary_tree(majorana_ham, n_modes)
    rs = fast_hatt_rust(fham, n_modes)

    assert rs.pauli_weight == py.pauli_weight
    assert rs.as_dict() == py.as_dict()
    assert rs.enumeration_scheme == py.enumeration_scheme
    assert rs.root_node.branch_majorana_map == py.root_node.branch_majorana_map


def test_hatt_rust_matches_python_water(water_fham, water_sparse_majorana):
    """Rust HATT reproduces the Python tree on the H2O/STO-3G Hamiltonian."""
    n_modes = 14
    py = hamiltonian_adaptive_ternary_tree(water_sparse_majorana, n_modes)
    rs = fast_hatt_rust(water_fham, n_modes)

    assert rs.pauli_weight == py.pauli_weight
    assert rs.as_dict() == py.as_dict()
    assert rs.enumeration_scheme == py.enumeration_scheme
