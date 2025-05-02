"""Tests for Utils functions"""

import numpy as np
from ferrmion.utils import (
    icount_to_sign,
    symplectic_product,
    symplectic_to_pauli,
    pauli_to_symplectic,
    symplectic_hash,
    symplectic_unhash,
)


def test_symplectic_product() -> None:
    xyz = np.array([1, 1, 0, 0, 1, 1])
    xxx = np.array([1, 1, 1, 0, 0, 0])
    zzz = np.array([0, 0, 0, 1, 1, 1])
    yyy = np.array([1, 1, 1, 1, 1, 1])
    yzx = np.array([1, 0, 1, 1, 1, 0])
    assert symplectic_product(xxx, zzz)[0] == 0
    assert np.all(symplectic_product(xxx, zzz)[1] == np.array([1, 1, 1, 1, 1, 1]))
    assert symplectic_product(zzz, xxx)[0] == 2
    assert np.all(symplectic_product(zzz, xxx)[1] == np.array([1, 1, 1, 1, 1, 1]))

    assert symplectic_product(xxx, yyy)[0] == 0
    assert np.all(symplectic_product(xxx, yyy)[1] == np.array([0, 0, 0, 1, 1, 1]))
    assert symplectic_product(yyy, xxx)[0] == 2
    assert np.all(symplectic_product(yyy, xxx)[1] == np.array([0, 0, 0, 1, 1, 1]))

    assert symplectic_product(zzz, yyy)[0] == 2
    assert np.all(symplectic_product(zzz, yyy)[1] == np.array([1, 1, 1, 0, 0, 0]))
    assert symplectic_product(yyy, zzz)[0] == 0
    assert np.all(symplectic_product(yyy, zzz)[1] == np.array([1, 1, 1, 0, 0, 0]))

    assert symplectic_product(xxx, xyz)[0] == 0
    assert np.all(symplectic_product(xxx, xyz)[1] == np.array([0, 0, 1, 0, 1, 1]))
    assert symplectic_product(xyz, xxx)[0] == 0
    assert np.all(symplectic_product(xyz, xxx)[1] == np.array([0, 0, 1, 0, 1, 1]))

    assert symplectic_product(yzx, xyz)[0] == 0
    assert np.all(symplectic_product(yzx, xyz)[1] == np.array([0, 1, 1, 1, 0, 1]))
    assert symplectic_product(xyz, yzx)[0] == 2
    assert np.all(symplectic_product(xyz, yzx)[1] == np.array([0, 1, 1, 1, 0, 1]))


def test_icount_to_sign() -> None:
    assert icount_to_sign(0) == 1
    assert icount_to_sign(1) == 1j
    assert icount_to_sign(2) == -1
    assert icount_to_sign(3) == -1j


def test_symplectic_hashing() -> None:
    symplectic = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8)
    print(symplectic_hash(symplectic))
    print(symplectic_unhash(symplectic_hash(symplectic), len(symplectic)))


def test_symplectic_pauli_conversion() -> None:
    symplectic = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8)

    print(pauli_to_symplectic(symplectic_to_pauli(symplectic)[1])[1])
    assert symplectic_to_pauli(symplectic) == (3, "IZXY")
    assert np.all(
        symplectic == pauli_to_symplectic(symplectic_to_pauli(symplectic)[1])[1]
    )
    assert (
        pauli_to_symplectic(symplectic_to_pauli(symplectic)[1])[0]
        + symplectic_to_pauli(symplectic)[0]
        == 4
    )