"""Performance benchmarks for ferrmion encoding operations.

Run with: uv run pytest python/benchmarks/ --benchmark-only
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ferrmion.encode import TernaryTree
from ferrmion.hamiltonians import FermionHamiltonian


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2_data():
    folder = Path(__file__).parent.parent / "tests" / "data"
    with open(folder / "h2_sto-3g.json", "rb") as f:
        data = json.load(f)
    data["ones"] = np.array(data["ones"])
    data["twos"] = np.array(data["twos"])
    return data


@pytest.fixture(scope="module")
def water_data():
    folder = Path(__file__).parent.parent / "tests" / "data"
    with open(folder / "h2o_sto-3g.json", "rb") as f:
        data = json.load(f)
    data["ones"] = np.array(data["ones"])
    data["twos"] = np.array(data["twos"])
    return data


# ---------------------------------------------------------------------------
# TernaryTree construction
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="ternary-tree-construction")
@pytest.mark.parametrize("n_modes", [4, 8, 14, 20])
def test_bench_ternary_tree_init(benchmark, n_modes):
    """Benchmark TernaryTree construction for varying mode counts."""
    benchmark(TernaryTree, n_modes)


# ---------------------------------------------------------------------------
# Standard encodings (Jordan-Wigner, Bravyi-Kitaev, Parity)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="standard-encodings")
@pytest.mark.parametrize("n_modes", [4, 8, 14])
def test_bench_jordan_wigner(benchmark, n_modes):
    """Benchmark Jordan-Wigner encoding."""
    tree = TernaryTree(n_modes)
    benchmark(tree.JW)


@pytest.mark.benchmark(group="standard-encodings")
@pytest.mark.parametrize("n_modes", [4, 8, 14])
def test_bench_bravyi_kitaev(benchmark, n_modes):
    """Benchmark Bravyi-Kitaev encoding."""
    tree = TernaryTree(n_modes)
    benchmark(tree.BK)


@pytest.mark.benchmark(group="standard-encodings")
@pytest.mark.parametrize("n_modes", [4, 8, 14])
def test_bench_parity(benchmark, n_modes):
    """Benchmark Parity encoding."""
    tree = TernaryTree(n_modes)
    benchmark(tree.Parity)


# ---------------------------------------------------------------------------
# Full Hamiltonian encoding pipeline
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="hamiltonian-encoding")
def test_bench_h2_jw_encoding(benchmark, h2_data):
    """Benchmark full Jordan-Wigner encoding of H2 Hamiltonian."""
    fham = FermionHamiltonian(
        terms={"+-": h2_data["ones"], "++--": h2_data["twos"]}
    )
    tree = TernaryTree(fham.n_modes)

    def encode():
        enc = tree.JW()
        return enc.encode(fham)

    benchmark(encode)


@pytest.mark.benchmark(group="hamiltonian-encoding")
def test_bench_water_jw_encoding(benchmark, water_data):
    """Benchmark full Jordan-Wigner encoding of H2O Hamiltonian."""
    fham = FermionHamiltonian(
        terms={"+-": water_data["ones"], "++--": water_data["twos"]}
    )
    tree = TernaryTree(fham.n_modes)

    def encode():
        enc = tree.JW()
        return enc.encode(fham)

    benchmark(encode)


@pytest.mark.benchmark(group="hamiltonian-encoding")
def test_bench_water_bk_encoding(benchmark, water_data):
    """Benchmark full Bravyi-Kitaev encoding of H2O Hamiltonian."""
    fham = FermionHamiltonian(
        terms={"+-": water_data["ones"], "++--": water_data["twos"]}
    )
    tree = TernaryTree(fham.n_modes)

    def encode():
        enc = tree.BK()
        return enc.encode(fham)

    benchmark(encode)
