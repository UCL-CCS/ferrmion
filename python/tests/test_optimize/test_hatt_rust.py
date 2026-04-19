"""Property-based parity tests between the Python and Rust HATT implementations."""

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from ferrmion.core import fermionic_to_sparse_majorana
from ferrmion.hamiltonians import FermionHamiltonian
from ferrmion.optimize.hatt import (
    fast_hatt_rust,
    hamiltonian_adaptive_ternary_tree,
)


def _coeff_array(shape: tuple[int, ...]) -> st.SearchStrategy[np.ndarray]:
    """Draw a float64 coefficient tensor with entries in [-1, 1]."""
    return arrays(
        dtype=np.float64,
        shape=shape,
        elements=st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    )


@st.composite
def fermion_hamiltonians(draw, min_modes: int = 2, max_modes: int = 5):
    """Draw a random FermionHamiltonian with "+-" and "++--" terms.

    The mode count is drawn jointly with the coefficient tensors so their
    shapes are consistent. Coefficients are bounded in [-1, 1] to keep HATT's
    near-zero filter from kicking in stochastically on every draw.
    """
    n_modes = draw(st.integers(min_value=min_modes, max_value=max_modes))
    ones = draw(_coeff_array((n_modes, n_modes)))
    twos = draw(_coeff_array((n_modes, n_modes, n_modes, n_modes)))
    return n_modes, FermionHamiltonian(terms={"+-": ones, "++--": twos})


@given(fham_data=fermion_hamiltonians())
@settings(max_examples=25, deadline=None)
def test_hatt_rust_matches_python_property(fham_data):
    """On random FermionHamiltonians, Rust and Python HATT produce the same tree.

    Both paths consume the same Majorana terms: the Python reference is fed
    the dict returned by ``fermionic_to_sparse_majorana`` on the drawn
    ``FermionHamiltonian``, which is the same transformation the Rust wrapper
    applies internally. Exact structural parity is therefore expected.
    """
    n_modes, fham = fham_data
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
    assert rs.root_node.branch_majorana_map == py.root_node.branch_majorana_map
