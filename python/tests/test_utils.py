"""Tests for Utils functions"""

import numpy as np
from ferrmion.utils import (
    icount_to_sign,
    pauli_to_symplectic,
    symplectic_hash,
    symplectic_to_pauli,
    symplectic_to_sparse,
    symplectic_unhash,
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

    assert symplectic_to_pauli(0, symplectic) == (3, "IZXY")
    inverse_ipower, inverse_symplectic = pauli_to_symplectic(*symplectic_to_pauli(0, symplectic))
    assert inverse_ipower == 0
    assert np.all(inverse_symplectic == symplectic)


def test_symplectic_sparse_conversion() -> None:
    ipower = 1
    symplectic = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=bool)

    assert symplectic_to_sparse(1, symplectic)[0] == 0
    assert symplectic_to_sparse(1, symplectic)[1] == "ZXY"
    assert np.array_equal(symplectic_to_sparse(1, symplectic)[2], [1,2,3])
