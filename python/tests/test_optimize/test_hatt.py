"""Snapshot regression tests for the Rust-backed HATT entry point.

Both flatpacks captured from the prior Python reference implementation
at the time the Python path was removed; they pin the Rust output so
future Rust-side changes can't silently drift.
"""

import json
from pathlib import Path

import numpy as np

from ferrmion.hamiltonians import FermionHamiltonian
from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree


def test_hatt_small_flatpack_snapshot():
    n_modes = 3
    ones = np.zeros((n_modes, n_modes))
    ones[0, 0] = 1.0
    ones[1, 1] = -1.0
    ones[2, 2] = 0.5
    twos = np.zeros((n_modes, n_modes, n_modes, n_modes))
    twos[0, 1, 1, 0] = 0.3
    twos[1, 2, 2, 1] = -0.2
    fham = FermionHamiltonian(terms={"+-": ones, "++--": twos})

    tree = hamiltonian_adaptive_ternary_tree(fham, n_modes)

    assert tree.flatpack() == [
        (2, (5, 6, 1)),
        (1, (7, 8, 0)),
        (0, (3, 4, None)),
    ]
    assert tree.pauli_weight == 6


def test_hatt_water_flatpack_snapshot():
    folder = Path(__file__).parent.parent
    with open(folder / "data/h2o_sto-3g.json") as f:
        data = json.load(f)
    ones = np.array(data["ones"])
    twos = np.array(data["twos"])
    fham = FermionHamiltonian(terms={"+-": ones, "++--": twos})

    tree = hamiltonian_adaptive_ternary_tree(fham, 14)

    assert tree.flatpack() == [
        (13, (5, 6, 12)),
        (5, (34, 35, 18)),
        (6, (26, 27, 19)),
        (12, (11, 39, 2)),
        (11, (10, 8, 38)),
        (2, (14, 15, 1)),
        (10, (3, 9, 7)),
        (8, (22, 23, 37)),
        (1, (32, 33, 0)),
        (3, (16, 17, 24)),
        (9, (40, 41, 4)),
        (7, (20, 21, 36)),
        (0, (30, 31, None)),
        (4, (28, 29, 25)),
    ]
    assert tree.pauli_weight == 4668
