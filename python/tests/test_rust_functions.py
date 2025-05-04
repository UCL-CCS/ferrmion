import pytest
import ferrmion
import numpy as np

def test_sum_as_string():
    assert ferrmion.sum_as_string(1, 1) == "2"

def test_rust_symplectic_product():
    symplectic_product = ferrmion.rust_symplectic_product
    xyz = np.array([1, 1, 0, 0, 1, 1], dtype=np.bool)
    xxx = np.array([1, 1, 1, 0, 0, 0], dtype=np.bool)
    zzz = np.array([0, 0, 0, 1, 1, 1], dtype=np.bool)
    yyy = np.array([1, 1, 1, 1, 1, 1], dtype=np.bool)
    yzx = np.array([1, 0, 1, 1, 1, 0], dtype=np.bool)
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