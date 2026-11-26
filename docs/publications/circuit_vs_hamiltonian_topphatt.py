"""Compare circuit-level and Hamiltonian-level TOPP-HATT on Trotter and qDRIFT circuits.

For each molecule and starting tree, two circuits are built from the
Hamiltonian ``H = sum_j c_j Γ_j``, where ``Γ_j`` is a Hermitian Majorana product:

- ``trotter``: one first-order Trotter step, the gates ``exp(-i c_j dt Γ_j)`` in
  order, once each. Every gate costs the same whatever its angle, and ``r``
  steps scale every cost by ``r``, so one step is enough to compare.
- ``qdrift``: ``N = n_terms`` gates sampled with probability ``|c_j| / λ``,
  each with angle ``sign(c_j) λ dt / N``. Repeated operators are kept as
  separate gates.

Each circuit is encoded with:

- ``naive``: the starting tree's encoding;
- ``H count``: default TOPP-HATT on ``H`` (each distinct term counts once);
- ``H weighted``: TOPP-HATT on ``H`` weighting terms by ``|c_j|``;
- ``H split x4``: TOPP-HATT on ``H`` split into ``4 n_terms`` equal pieces;
- ``circuit``: ``topphatt`` on the circuit itself, counting each operator once
  per gate that applies it.

The Trotter step is costed in two gate orders, since any order is an equally
valid first-order step:

- ``key order``: terms sorted by Majorana index tuple. This is the naive
  Jordan-Wigner qubit order, so it flatters that encoding.
- ``tree order``: each method's own encoding reorders the step with
  ``MajoranaCircuit.sorted_by_tree_order``, so rotations acting near each
  other in its tree are neighbours and share CNOT ladders.

The qDRIFT sample keeps its sampled order.

Each encoded circuit is converted to Qiskit with
:func:`ferrmion.interop.circuits.pauli_circuit_to_qiskit`, using both the
``chain`` and ``fountain`` CNOT ladders, and transpiled at optimisation level
3 to ``{cx, rz, sx, x}`` with all-to-all connectivity. Reported (lower is
better): ``raw CX``, the CNOTs before transpilation (``sum 2 (w - 1)``, the same
in any order), and the transpiled CNOT count for each ladder layout. Transpiled
depths are saved with ``--json``.

The circuit compiler only sees the sampled qDRIFT circuit; results are for a
single sample (``SEED``).

Run with ``uv run python docs/publications/circuit_vs_hamiltonian_topphatt.py``.
Pass molecule names (``h2_sto-3g h2o_6-31g`` ...) to restrict the run, and
``--json PATH`` to save the results.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import ferrmion as fr
from ferrmion import MajoranaCircuit
from ferrmion.core import compile_circuit, topphatt
from ferrmion.encode.ternary_tree import TernaryTree
from ferrmion.interop.circuits import CX_STRUCTURES, pauli_circuit_to_qiskit
from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree
from qiskit import transpile

DATA = Path(__file__).resolve().parents[2] / "python" / "tests" / "data"
MOLECULES = ["h2_sto-3g", "h2_6-31g", "h2o_sto-3g", "h2o_6-31g"]
DT = 1e-3
SEED = 1234
BASIS_GATES = ["cx", "rz", "sx", "x"]
OPTIMIZATION_LEVEL = 3


def hermitian_terms(msparse) -> tuple[list[list[int]], np.ndarray]:
    """Distinct Majorana terms of ``H`` and real coefficients of ``Γ_S``.

    ``c γ_S = c i^{-m} Γ_S`` with ``m = k(k-1)/2 mod 2``.
    """
    terms, coeffs = [], []
    for key, c in msparse.to_dict().items():
        if len(key) == 0:
            continue
        k = len(key)
        theta = c * (-1j) ** ((k * (k - 1) // 2) % 2)
        assert abs(theta.imag) <= 1e-9 * max(1.0, abs(theta)), (key, c)
        terms.append(list(key))
        coeffs.append(theta.real)
    return terms, np.array(coeffs)


def trotter_circuit(terms, coeffs, n_modes) -> MajoranaCircuit:
    return MajoranaCircuit(terms, (coeffs * DT).tolist(), n_modes)


def qdrift_circuit(terms, coeffs, n_modes, rng) -> MajoranaCircuit:
    lam = np.abs(coeffs).sum()
    n_samples = len(terms)
    picks = rng.choice(len(terms), size=n_samples, p=np.abs(coeffs) / lam)
    angle = lam * DT / n_samples
    return MajoranaCircuit(
        [terms[j] for j in picks],
        [float(np.sign(coeffs[j]) * angle) for j in picks],
        n_modes,
    )


def gate_costs(pauli_circuit) -> tuple[int, int]:
    pw = cx = 0
    for pauli in pauli_circuit.paulis:
        w = sum(ch != "I" for ch in pauli)
        pw += w
        cx += 2 * max(w - 1, 0)
    return pw, cx


def transpiled_costs(pauli_circuit) -> dict:
    """CNOT count and depth after Qiskit transpilation, for each ladder layout."""
    out = {}
    for cx_structure in CX_STRUCTURES:
        qc = pauli_circuit_to_qiskit(pauli_circuit, cx_structure)
        tqc = transpile(
            qc,
            basis_gates=BASIS_GATES,
            optimization_level=OPTIMIZATION_LEVEL,
            seed_transpiler=0,
        )
        out[cx_structure] = {"cx": tqc.count_ops().get("cx", 0), "depth": tqc.depth()}
    return out


def costs(pauli_circuit) -> dict:
    pw, cx = gate_costs(pauli_circuit)
    return {"pw": pw, "raw_cx": cx, "transpiled": transpiled_costs(pauli_circuit)}


def run(molecule: str) -> dict:
    data = json.loads((DATA / f"{molecule}.json").read_text())
    fham = fr.molecular_hamiltonian(
        np.array(data["ones"]), np.array(data["twos"]), data["constant_energy"]
    )
    n = fham.n_modes
    msparse = fham.to_majorana_sparse()
    terms, coeffs = hermitian_terms(msparse)
    trotter = trotter_circuit(terms, coeffs, n)
    # Construction drops negligible gates; keep terms and coefficients aligned
    # with the gates that survive.
    terms, coeffs = trotter.terms, np.asarray(trotter.angles) / DT
    qdrift = qdrift_circuit(terms, coeffs, n, np.random.default_rng(SEED))
    n_distinct_qdrift = len(qdrift.operator_weights()[0])
    print(
        f"\n## {molecule}: {n} modes, {len(trotter)} Trotter gates; qDRIFT sample has "
        f"{len(qdrift)} gates on {n_distinct_qdrift} distinct operators"
    )

    trees = {
        "JW": lambda: TernaryTree.JordanWigner(n),
        "BK": lambda: TernaryTree.BravyiKitaev(n),
        "JKMN": lambda: TernaryTree.JKMN(n),
        "HATT": lambda: hamiltonian_adaptive_ternary_tree(fham, n),
    }
    hamiltonian_variants = [
        ("H count", {"backend": "sparse"}),
        ("H weighted", {"backend": "weighted"}),
        ("H split x4", {"backend": "split", "split_budget": 4 * len(terms)}),
    ]

    level = f"L{OPTIMIZATION_LEVEL}"
    print(
        f"{'':18}| {'Trotter':>7} | {'Trotter key order':>17} | "
        f"{'Trotter tree order':>18} | {'qDRIFT sample':>17}"
    )
    header = (
        f"{'tree':5} {'method':11} | {'raw CX':>7} | {'chain':>8} {'fount':>8} | "
        f"{'chain':>8} {'fount':>9} | {'chain':>8} {'fount':>8}"
    )
    print(header + f"   (transpiled {level} CNOTs)")
    print("-" * len(header))
    results = {}
    for tree_name, make in trees.items():
        flatpack = make().flatpack()
        # (encoding used for Trotter, PauliCircuit for the qDRIFT sample)
        methods = {}

        naive = make().build_encoding()
        methods["naive"] = (naive, naive.encode_circuit(qdrift))

        for name, kwargs in hamiltonian_variants:
            enc = topphatt(flatpack, n, msparse, parallelize=True, **kwargs)
            methods[name] = (enc, enc.encode_circuit(qdrift))

        # Each circuit is optimised for itself: the qDRIFT encoding only sees
        # the sampled gates. Reordering a Trotter step does not change its
        # operator counts, so it optimises to the same encoding.
        trotter_enc = topphatt(flatpack, n, trotter)
        qdrift_pc, _ = compile_circuit(flatpack, n, qdrift)
        methods["circuit"] = (trotter_enc, qdrift_pc)

        rows = {}
        for name, (enc, q_pc) in methods.items():
            key_order = costs(enc.encode_circuit(trotter))
            tree_order = costs(enc.encode_circuit(trotter.sorted_by_tree_order(enc)))
            q = costs(q_pc)
            rows[name] = {
                "trotter_key_order": key_order,
                "trotter_tree_order": tree_order,
                "qdrift": q,
            }
            k, t, qt = key_order["transpiled"], tree_order["transpiled"], q["transpiled"]
            print(
                f"{tree_name if name == 'naive' else '':5} {name:11} | "
                f"{key_order['raw_cx']:7d} | "
                f"{k['chain']['cx']:8d} {k['fountain']['cx']:8d} | "
                f"{t['chain']['cx']:8d} {t['fountain']['cx']:9d} | "
                f"{qt['chain']['cx']:8d} {qt['fountain']['cx']:8d}",
                flush=True,
            )
        results[tree_name] = rows
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("molecules", nargs="*", default=MOLECULES)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    all_results = {mol: run(mol) for mol in args.molecules}
    if args.json is not None:
        args.json.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved {args.json}")
