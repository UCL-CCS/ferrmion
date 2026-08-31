"""Tests for Utils functions"""

import numpy as np
from ferrmion.core import FermionHamiltonian
from ferrmion.core import (
    pauli_to_symplectic,
    symplectic_to_pauli,
    symplectic_to_sparse,
)

def test_symplectic_pauli_conversion() -> None:
    symplectic = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=bool)

    assert symplectic_to_pauli(symplectic, 0) == ("IZXY", 3)
    assert symplectic_to_pauli(symplectic) == ("IZXY", 3)
    inverse_symplectic, inverse_ipower = pauli_to_symplectic(
        *symplectic_to_pauli(symplectic, 0)
    )
    assert inverse_ipower == 0
    assert np.all(inverse_symplectic == symplectic)


def test_symplectic_sparse_conversion() -> None:
    symplectic = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=bool)

    assert symplectic_to_sparse(symplectic, 1)[0] == "ZXY"
    assert symplectic_to_sparse(symplectic, 1)[2] == 1.0
    assert np.array_equal(symplectic_to_sparse(symplectic, 1)[1], [1, 2, 3])


def test_fermionic_to_sparse_majorana() -> None:
    n_modes = 3
    ones = np.zeros((n_modes, n_modes))
    twos = np.zeros((n_modes, n_modes, n_modes, n_modes))

    ones[0, 0] = 1
    twos[1, 2, 1, 2] = 2

    majorana_ham = FermionHamiltonian(terms={"+-": ones, "++--": twos}).to_majorana_sparse().to_dict()
    assert majorana_ham == {
        (0, 1): np.complex128(0.5j),
        (4, 5): np.complex128(-0.5j),
        (2, 3): np.complex128(-0.5j),
        (2, 3, 4, 5): np.complex128(0.5 + 0j),
    }


def test_fermionic_to_sparse_majorana_includes_constant() -> None:
    """A self-cancelling diagonal term and constant_energy must both
    reach to_dict() as the identity/empty-tuple key, not be dropped."""
    n_modes = 2
    ones = np.zeros((n_modes, n_modes))
    ones[0, 0] = 1  # a†_0 a_0 = 0.5*I + 0.5i*gamma_0*gamma_1

    majorana_ham = (
        FermionHamiltonian(terms={"+-": ones}, constant_energy=0.25)
        .to_majorana_sparse()
        .to_dict()
    )
    assert majorana_ham == {
        (): np.complex128(0.75 + 0j),  # 0.5 (from a†_0 a_0) + 0.25 (constant_energy)
        (0, 1): np.complex128(0.5j),
    }
