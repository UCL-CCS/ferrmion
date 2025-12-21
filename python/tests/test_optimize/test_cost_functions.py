"""Tests for functions in the optimize submodule."""

from ferrmion.optimize.cost_functions import (
    minimise_mi_distance,
    distance_squared,
    coefficient_pauli_weight,
)
from ferrmion.hamiltonians import molecular_hamiltonian
from ferrmion.encode import TernaryTree
import numpy as np
import numpy as np
from pytest import fixture


@fixture
def n2mi():
    return np.array(
        [
            [
                8.95411720e-05,
                8.00313442e-07,
                3.57202788e-05,
                5.64570141e-06,
                3.39388829e-06,
                3.67820340e-06,
                3.71248133e-06,
                8.39012020e-06,
                7.60808045e-06,
                4.00424690e-07,
            ],
            [
                8.00313442e-07,
                1.88782428e-04,
                4.61021178e-06,
                1.43974455e-04,
                3.97230035e-06,
                4.34316525e-06,
                4.25262029e-06,
                1.40315431e-05,
                1.31042878e-05,
                4.73322942e-07,
            ],
            [
                3.57202788e-05,
                4.61021178e-06,
                5.97013599e-02,
                1.53473952e-03,
                1.14739409e-03,
                1.67868542e-02,
                2.03055787e-03,
                1.57647091e-02,
                2.20614397e-03,
                2.26381776e-02,
            ],
            [
                5.64570141e-06,
                1.43974455e-04,
                1.53473952e-03,
                9.22768981e-02,
                8.06189477e-03,
                8.02634734e-03,
                3.59937207e-03,
                3.44315829e-02,
                3.39878461e-02,
                1.58240762e-03,
            ],
            [
                3.39388829e-06,
                3.97230035e-06,
                1.14739409e-03,
                8.06189477e-03,
                4.94601967e-01,
                1.27072663e-01,
                2.87313949e-03,
                9.00062433e-02,
                2.49380494e-01,
                2.23093410e-03,
            ],
            [
                3.67820340e-06,
                4.34316525e-06,
                1.67868542e-02,
                8.02634734e-03,
                1.27072663e-01,
                5.27514044e-01,
                8.66763070e-03,
                2.58110275e-01,
                8.95621909e-02,
                2.18442824e-02,
            ],
            [
                3.71248133e-06,
                4.25262029e-06,
                2.03055787e-03,
                3.59937207e-03,
                2.87313949e-03,
                8.66763070e-03,
                5.41448119e-02,
                1.25017005e-02,
                8.08334819e-03,
                1.85391136e-02,
            ],
            [
                8.39012020e-06,
                1.40315431e-05,
                1.57647091e-02,
                3.44315829e-02,
                9.00062433e-02,
                2.58110275e-01,
                1.25017005e-02,
                5.62235070e-01,
                1.28998497e-01,
                2.20469210e-02,
            ],
            [
                7.60808045e-06,
                1.31042878e-05,
                2.20614397e-03,
                3.39878461e-02,
                2.49380494e-01,
                8.95621909e-02,
                8.08334819e-03,
                1.28998497e-01,
                5.30526412e-01,
                2.39364166e-03,
            ],
            [
                4.00424690e-07,
                4.73322942e-07,
                2.26381776e-02,
                1.58240762e-03,
                2.23093410e-03,
                2.18442824e-02,
                1.85391136e-02,
                2.20469210e-02,
                2.39364166e-03,
                9.04878107e-02,
            ],
        ]
    )


def test_minimise_mi_distance(n2mi):
    unpaired = minimise_mi_distance(n2mi, pair_spins=False, spinless_mi=False)
    assert set(unpaired).symmetric_difference({*range(n2mi.shape[0])}) == set()

    paired = minimise_mi_distance(n2mi, pair_spins=True, spinless_mi=False)
    assert np.all((paired[1::2] - paired[0::2]) == 1)
    assert set(paired).symmetric_difference({*range(n2mi.shape[0])}) == set()


def test_distance_squared(n2mi):
    permutation = [*range(n2mi.shape[0])]
    forwards = distance_squared(n2mi, permutation)
    permutation.reverse()
    backwards = distance_squared(n2mi, permutation)

    assert len(forwards) == len(backwards) == 1
    assert forwards[0] == backwards[0] == np.float64(21.85687074218722)
    assert distance_squared(n2mi, [0, 9, 1, 8, 2, 7, 3, 6, 4, 5]) == np.float64(
        37.944815941146125
    )
    assert distance_squared(n2mi, [*range(n2mi.shape[0] - 1)]) == [np.inf]
    assert distance_squared(n2mi, [*range(n2mi.shape[0] + 1)]) == [np.inf]
    assert distance_squared(n2mi, [*range(1, n2mi.shape[0] + 1)]) == [np.inf]


def test_coefficient_pauli_weight(water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]

    jw = TernaryTree(14).JW()
    jw_qham = jw.encode(molecular_hamiltonian(ones, twos))
    jw_norm = coefficient_pauli_weight(jw_qham)[0]

    assert isinstance(jw_norm, np.float64)
    assert np.allclose(int(jw_norm), 191)

    pe = TernaryTree(14).ParityEncoding()
    pe_mol_ham = molecular_hamiltonian(ones, twos)
    pe_qham = pe.encode(pe_mol_ham)
    pe_norm = coefficient_pauli_weight(pe_qham)[0]

    assert isinstance(pe_norm, np.float64)
    assert pe_norm > jw_norm
    assert np.allclose(int(pe_norm), 256)
