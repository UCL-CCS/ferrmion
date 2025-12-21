import numpy as np
from ferrmion.core import symplectic_product, topphatt, topphatt_standard, encode, encode_standard, standard_symplectic_matrix
import pytest

def test_symplectic_product():
    xyz = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
    xxx = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    zzz = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    yyy = np.array([1, 1, 1, 1, 1, 1], dtype=bool)
    yzx = np.array([1, 0, 1, 1, 1, 0], dtype=bool)
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

def test_core_topphatt():
    ones = np.random.random((4,4))
    twos = np.random.random((4,4,4,4))
    flatpack = [(0,(None, None, 1)), (1, (None,None,2)), (2,(None, None,3)), (3, (None, None, None))]
    topphatt(flatpack, 4, signatures=["+-", "++--"], coeffs=[ones, twos])

def test_core_topphatt_flatpack_runs(water_data):
    flatpack = [(i, (None, None, i+1)) for i in range(13)] + [(13, (None, None, None))]
    topphatt(flatpack,14, signatures=["+-", "++--"], coeffs=[water_data["ones"], water_data["twos"]])

@pytest.mark.parametrize("encoding", ["JW", "PE", "BK", "JKMN"])
def test_core_topphatt_standard_runs(encoding, water_data):
    topphatt_standard(encoding, 14, 14,signatures=["+-", "++--"], coeffs=[water_data["ones"], water_data["twos"]])


@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN"])
def test_core_standard(encoding, water_eigenvalues, water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    one_step = encode_standard(encoding, 14,14, ["+-","++--"], [ones, twos], 0.)

    ipow, sym = standard_symplectic_matrix(encoding, ones.shape[0])
    two_step = encode(ipow, sym,["+-","++--"], [ones, twos], 0.)

    assert len(one_step) == len(two_step)
    for k,v in one_step.items():
        assert two_step[k] == v
