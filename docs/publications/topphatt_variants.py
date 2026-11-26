"""Benchmark TOPP-HATT variants on fixed ternary-tree topologies.

Compares, for each molecule and tree topology, the Pauli weight (PW),
coefficient Pauli weight (CPW) and wall time of:

- ``naive``: the tree's own leaf assignment;
- ``topphatt``: the original bottom-up TOPP-HATT (``strategy="bottom_up"``);
- ``exact_obj``: bottom-up with an exact objective (``dedup=False``,
  ``contract_forced=True``);
- ``lookahead``: bottom-up with the parent-edge penalty (``lookahead=0.5``);
- ``top_down``: top-down partitioning with the ``|π_c|`` bound (``lookahead=1``);
- ``top_down_exp``: top-down with the expected-completion objective;
- ``best``: best of naive, bottom-up and both top-down variants;
- ``exact``: exhaustive optimum (at most 10 modes).

Run from the repository root::

    uv run python docs/publications/topphatt_variants.py

Results are written to ``docs/publications/data/topphatt_variants.json``.
"""

import json
import time
from pathlib import Path

import numpy as np

import ferrmion as fr
from ferrmion import core
from ferrmion.encode.ternary_tree import TernaryTree
from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree
from ferrmion.optimize.huffman import huffman_ternary_tree

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "python" / "tests" / "data"
OUT = ROOT / "docs" / "publications" / "data" / "topphatt_variants.json"

MOLECULES = ["h2_sto-3g", "h2_6-31g", "h2o_sto-3g", "h2o_6-31g"]

VARIANTS = {
    "topphatt": dict(),
    "exact_obj": dict(dedup=False, contract_forced=True),
    "lookahead": dict(lookahead=0.5),
    "top_down": dict(strategy="top_down", lookahead=1.0),
    "top_down_exp": dict(strategy="top_down_expected"),
    "best": dict(strategy="best"),
    "exact": dict(strategy="exact"),
}


def trees(n_modes, ones, twos, fham):
    """Tree constructors for each topology."""
    out = {
        "Jordan-Wigner": lambda: TernaryTree.JordanWigner(n_modes),
        "Parity": lambda: TernaryTree.Parity(n_modes),
        "Bravyi-Kitaev": lambda: TernaryTree.BravyiKitaev(n_modes),
        "JKMN": lambda: TernaryTree.JKMN(n_modes),
        "Huffman": lambda: huffman_ternary_tree(ones, twos),
    }
    if n_modes <= 14:
        hatt_tree = hamiltonian_adaptive_ternary_tree(fham, n_modes)
        flatpack = hatt_tree.flatpack()
        out["HATT"] = lambda: TernaryTree.from_flatpack(flatpack)
    return out


def main():
    """Run the benchmark and write the results."""
    results = []
    for name in MOLECULES:
        data = json.loads((DATA / f"{name}.json").read_text())
        ones, twos = np.array(data["ones"]), np.array(data["twos"])
        fham = fr.molecular_hamiltonian(ones, twos, data["constant_energy"])
        msparse = fham.to_majorana_sparse()
        n_modes = ones.shape[0]
        for tree_name, make in trees(n_modes, ones, twos, fham).items():
            naive = make().encode_naive(fham)
            row = {
                "molecule": name,
                "n_modes": n_modes,
                "tree": tree_name,
                "naive": [naive.pauli_weight(), naive.coeff_pauli_weight(), 0.0],
            }
            for variant, options in VARIANTS.items():
                if variant == "exact" and n_modes > 10:
                    continue
                tree = make()
                start = time.perf_counter()
                qham, _ = core.encode_topphatt(
                    tree.flatpack(), tree.n_qubits, msparse, **options
                )
                elapsed = time.perf_counter() - start
                row[variant] = [qham.pauli_weight(), qham.coeff_pauli_weight(), elapsed]
            results.append(row)
            print(
                f"{name:11} {tree_name:14} "
                + " ".join(
                    f"{k}={v[0]}" for k, v in row.items() if isinstance(v, list)
                ),
                flush=True,
            )
    OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
