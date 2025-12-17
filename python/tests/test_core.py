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

def test_core_topphatt_water(water_integrals):
    ones, twos = water_integrals

    flatpack = [(i, (None, None, i+1)) for i in range(13)] + [(13, (None, None, None))]
    topphatt(flatpack,14, signatures=["+-", "++--"], coeffs=[ones, twos])

def test_core_topphatt_jw(water_integrals):
    ones, twos = water_integrals
    topphatt_standard("JW", 14, 14,signatures=["+-", "++--"], coeffs=[ones, twos])

def test_core_topphatt_pe(water_integrals):
    ones, twos = water_integrals
    topphatt_standard("PE",14, 14,signatures=["+-", "++--"], coeffs=[ones, twos])

def test_core_topphatt_bk(water_integrals):
    ones, twos = water_integrals
    ones =np.random.random((6,6))
    twos =np.random.random((6,6, 6,6))
    topphatt_standard("BK",6, 6, signatures=["+-", "++--"], coeffs=[ones, twos])

def test_core_topphatt_jkmn(water_integrals):
    ones, twos = water_integrals
    topphatt_standard("JKMN",14,14, signatures=["+-", "++--"], coeffs=[ones, twos])


@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN"])
def test_core_standard(encoding, water_eigenvalues, water_integrals):
    ones = water_integrals[0]
    twos = 0.5*water_integrals[1]
    one_step = encode_standard(encoding, 14,14, ["+-","++--"], [ones, twos], 0.)

    ipow, sym = standard_symplectic_matrix(encoding, ones.shape[0])
    two_step = encode(ipow, sym,["+-","++--"], [ones, twos], 0.) 

    assert len(one_step) == len(two_step)
    for k,v in one_step.items():
        assert two_step[k] == v
