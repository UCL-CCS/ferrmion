"""Compare coefficient-aware TOPP-HATT backends on a qDRIFT cost metric.

qDRIFT samples term ``j`` with probability ``|c_j| / lambda``, so the expected
cost of one step is ``sum_j |c_j| * cost(P_j) / lambda``. This script compares,
for several starting trees:

- ``naive``: the tree's default encoding (no optimisation);
- ``count``: default TOPP-HATT (minimises the number of non-identity terms per qubit);
- ``weighted``: TOPP-HATT minimising ``sum |c| * weight`` (quantised coefficients);
- ``split xB``: TOPP-HATT on terms split into ``B * n_terms`` equal-coefficient pieces.

Reported metrics, each relative to ``naive`` (lower is better):

- ``PW``: total Pauli weight ``sum_j weight(P_j)``;
- ``cPW``: qDRIFT expected Pauli weight ``sum_j |c_j| weight(P_j) / lambda``;
- ``cCX``: qDRIFT expected CNOT count of the ladder circuit,
  ``sum_j |c_j| 2 (weight(P_j) - 1) / lambda``.

Run with ``uv run python docs/publications/coefficient_topphatt.py``.
"""

import json
import time
from pathlib import Path

import numpy as np

import ferrmion as fr
from ferrmion.encode.ternary_tree import TernaryTree
from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree

DATA = Path(__file__).resolve().parents[2] / "python" / "tests" / "data"
SPLIT_FACTORS = [1, 2, 4, 16, 64]


def metrics(qham) -> tuple[float, float, float]:
    pw = cpw = ccx = lam = 0.0
    for string, coeff in qham.items():
        w = sum(ch != "I" for ch in string)
        if w == 0:
            continue
        a = abs(coeff)
        lam += a
        pw += w
        cpw += a * w
        ccx += a * 2 * (w - 1)
    return pw, cpw / lam, ccx / lam


def run(molecule: str) -> dict:
    data = json.loads((DATA / f"{molecule}.json").read_text())
    fham = fr.molecular_hamiltonian(
        np.array(data["ones"]), np.array(data["twos"]), data["constant_energy"]
    )
    n = fham.n_modes
    n_terms = len(fham.to_majorana_sparse().to_dict())
    trees = {
        "JW": lambda: TernaryTree.JordanWigner(n),
        "BK": lambda: TernaryTree.BravyiKitaev(n),
        "JKMN": lambda: TernaryTree.JKMN(n),
        "HATT": lambda: hamiltonian_adaptive_ternary_tree(fham, n),
    }
    variants = [("count", {"backend": "sparse"}), ("weighted", {"backend": "weighted"})]
    variants += [
        (f"split x{b}", {"backend": "split", "split_budget": b * n_terms})
        for b in SPLIT_FACTORS
    ]

    results = {}
    print(f"\n## {molecule}: {n} modes, {n_terms} Majorana terms")
    print(f"{'tree':6} {'variant':10} {'PW':>7} {'cPW':>7} {'cCX':>7} {'time/s':>8}")
    for tree_name, make in trees.items():
        base = metrics(make().encode_naive(fham))
        results[tree_name] = {"naive": base}
        print(f"{tree_name:6} {'naive':10} {base[0]:7.0f} {base[1]:7.3f} {base[2]:7.3f}")
        for name, kwargs in variants:
            tree = make()
            t0 = time.perf_counter()
            qham = tree.encode(fham, parallelize=True, **kwargs)
            dt = time.perf_counter() - t0
            m = metrics(qham)
            results[tree_name][name] = m
            rel = [m[i] / base[i] for i in range(3)]
            print(
                f"{'':6} {name:10} {rel[0]:7.3f} {rel[1]:7.3f} {rel[2]:7.3f} {dt:8.2f}"
            )
    return results


if __name__ == "__main__":
    for mol in ["h2o_sto-3g", "h2o_6-31g"]:
        run(mol)
