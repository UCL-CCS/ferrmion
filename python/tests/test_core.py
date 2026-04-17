import numpy as np
import pytest
from ferrmion.core import symplectic_product, batch_pauli_weights, encode
from ferrmion.encode.ternary_tree import JordanWigner, BravyiKitaev

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


def _pauli_weight(qham: dict) -> int:
    return sum(len(k) - k.count("I") for k in qham)


@pytest.mark.parametrize("encoding_cls", [JordanWigner, BravyiKitaev])
def test_batch_pauli_weights_matches_individual_encode(encoding_cls, h2_631g_data):
    """Each batch weight must equal encoding individually then computing Pauli weight."""
    ones = h2_631g_data["ones"]
    twos = h2_631g_data["twos"]
    n_modes = ones.shape[0]

    enc = encoding_cls(n_modes)
    ipow, sym = enc._build_symplectic_matrix()
    vacuum = enc.vacuum_state.astype(bool)
    sigs = ["+-", "++--"]
    coeffs = [ones, twos]

    rng = np.random.default_rng(seed=42)
    perms = np.array(
        [np.arange(n_modes)] + [rng.permutation(n_modes) for _ in range(99)],
        dtype=np.uintp,
    )  # 100 permutations total

    plain, weighted = batch_pauli_weights(ipow, sym, vacuum, sigs, coeffs, perms)

    assert len(plain) == 100
    assert len(weighted) == 100
    # Cross-check a sample of permutations against individual encode calls
    for i in [0, 1, 25, 50, 99]:
        mperm = np.array([2 * perms[i], 2 * perms[i] + 1]).T.flatten()
        qham = encode(
            ipowers=ipow[mperm],
            symplectics=sym[mperm],
            vacuum_state=vacuum,
            signatures=sigs,
            coeffs=coeffs,
            constant_energy=0.0,
        )
        assert plain[i] == pytest.approx(_pauli_weight(qham))


def test_batch_pauli_weights_returns_correct_length(h2_631g_data):
    ones = h2_631g_data["ones"]
    twos = h2_631g_data["twos"]
    n_modes = ones.shape[0]
    enc = JordanWigner(n_modes)
    ipow, sym = enc._build_symplectic_matrix()
    vacuum = enc.vacuum_state.astype(bool)

    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(n_modes) for _ in range(100)], dtype=np.uintp)

    plain, weighted = batch_pauli_weights(ipow, sym, vacuum, ["+-", "++--"], [ones, twos], perms)
    assert plain.shape == (100,)
    assert weighted.shape == (100,)


def test_batch_pauli_weights_coefficient_weighted_differs(h2_631g_data):
    """Coefficient-weighted and plain Pauli weights should generally differ."""
    ones = h2_631g_data["ones"]
    twos = h2_631g_data["twos"]
    n_modes = ones.shape[0]
    enc = JordanWigner(n_modes)
    ipow, sym = enc._build_symplectic_matrix()
    vacuum = enc.vacuum_state.astype(bool)
    perms = np.array([np.arange(n_modes)], dtype=np.uintp)

    plain, weighted = batch_pauli_weights(ipow, sym, vacuum, ["+-", "++--"], [ones, twos], perms)
    assert not np.allclose(plain, weighted)


def test_batch_pauli_weights_empty_permutations(h2_631g_data):
    ones = h2_631g_data["ones"]
    twos = h2_631g_data["twos"]
    n_modes = ones.shape[0]
    enc = JordanWigner(n_modes)
    ipow, sym = enc._build_symplectic_matrix()
    vacuum = enc.vacuum_state.astype(bool)
    perms = np.empty((0, n_modes), dtype=np.uintp)

    plain, weighted = batch_pauli_weights(ipow, sym, vacuum, ["+-", "++--"], [ones, twos], perms)
    assert plain.shape == (0,)
    assert weighted.shape == (0,)


@pytest.mark.parametrize("encoding_cls", [JordanWigner, BravyiKitaev])
@pytest.mark.parametrize("n_perms", [10, 100, 500, 1000])
def test_benchmark_batch_pauli_weights(benchmark, encoding_cls, n_perms, water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    n_modes = ones.shape[0]
    enc = encoding_cls(n_modes)
    ipow, sym = enc._build_symplectic_matrix()
    vacuum = enc.vacuum_state.astype(bool)
    rng = np.random.default_rng(seed=0)
    perms = np.array([rng.permutation(n_modes) for _ in range(n_perms)], dtype=np.uintp)
    benchmark(lambda: batch_pauli_weights(ipow, sym, vacuum, ["+-", "++--"], [ones, twos], perms))
