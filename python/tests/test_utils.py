"""Tests for Utils functions"""

import numpy as np
from ferrmion.utils import (
    icount_to_sign,
    pauli_to_symplectic,
    symplectic_hash,
    symplectic_to_pauli,
    symplectic_to_sparse,
    symplectic_unhash,
    fermionic_to_sparse_majorana,
)


def test_icount_to_sign() -> None:
    assert icount_to_sign(0) == 1
    assert icount_to_sign(1) == 1j
    assert icount_to_sign(2) == -1
    assert icount_to_sign(3) == -1j


def test_symplectic_hashing() -> None:
    symplectic = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
    print(symplectic_hash(symplectic))
    print(symplectic_unhash(symplectic_hash(symplectic), len(symplectic)))


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

    majorana_ham = fermionic_to_sparse_majorana([(ones, "+-"), (twos, "++--")])
    assert majorana_ham == {
        (0, 1): np.complex128(0.5j),
        (4, 5): np.complex128(-0.5j),
        (2, 3): np.complex128(-0.5j),
        (2, 3, 4, 5): np.complex128(0.5 + 0j),
    }
