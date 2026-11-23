"""How TOPP-HATT reduces the depth of a Trotter step.

TOPP-HATT keeps a tree's shape and qubit labels and only changes which Majorana
operator sits on each leaf. So instead of encoding a circuit with the optimised
encoding, the same circuit can be produced by *relabelling the circuit* and
encoding it with the naive one:

1. Build one first-order Trotter step ``prod_j exp(-i c_j dt Γ_j)`` as a
   ``MajoranaCircuit``.
2. Optimise the tree's leaf assignment with TOPP-HATT, giving an encoding
   ``E_opt`` of the same tree as the naive encoding ``E_naive``.
3. ``perm = E_opt.majorana_permutation_to(E_naive)`` matches leaves: Majorana
   ``a`` sits where ``E_naive`` puts ``perm[a]``.
4. Relabel the circuit with ``perm``, order the step by position in the
   (shared) tree, and encode with ``E_naive``. This equals encoding the
   tree-ordered step with ``E_opt`` (checked below), so the gain is entirely
   from the relabelling.

Both circuits are converted to Qiskit and transpiled at optimisation level 3 to
``{cx, rz, sx, x}`` with all-to-all connectivity. Each method gets whichever
CNOT ladder (``chain`` or ``fountain``) gives the lower depth.

Two TOPP-HATT variants are shown: ``TOPP-HATT`` (default, optimised for the
Hamiltonian) and ``circuit`` (``topphatt`` on the step itself, counting each
operator once per gate; terms that merge during optimisation keep both counts).

Run with ``uv run python docs/publications/trotter_depth_topphatt.py``.
Pass molecule names to restrict the run and ``--json PATH`` to save results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import transpile

import ferrmion as fr
from ferrmion import MajoranaCircuit
from ferrmion.core import topphatt
from ferrmion.encode.ternary_tree import TernaryTree
from ferrmion.interop.circuits import CX_STRUCTURES, pauli_circuit_to_qiskit

DATA = Path(__file__).resolve().parents[2] / "python" / "tests" / "data"
MOLECULES = ["h2_sto-3g", "h2_6-31g", "h2o_sto-3g", "h2o_6-31g"]
TREES = {
    "JW": TernaryTree.JordanWigner,
    "Parity": TernaryTree.Parity,
    "BK": TernaryTree.BravyiKitaev,
    "JKMN": TernaryTree.JKMN,
}
DT = 1e-3
BASIS_GATES = ["cx", "rz", "sx", "x"]


def trotter_step(fham) -> MajoranaCircuit:
    """One first-order Trotter step: each Hermitian Majorana term once."""
    terms, angles = [], []
    for key, c in fham.to_majorana_sparse().to_dict().items():
        if not key:
            continue
        k = len(key)
        theta = c * (-1j) ** ((k * (k - 1) // 2) % 2)  # c γ_S = θ Γ_S
        assert abs(theta.imag) <= 1e-9 * max(1.0, abs(theta))
        terms.append(list(key))
        angles.append(theta.real * DT)
    return MajoranaCircuit(terms, angles, fham.n_modes)


def transpiled(pauli_circuit) -> dict:
    """Best-depth transpilation over the two CNOT ladder layouts."""
    best = None
    for cx_structure in CX_STRUCTURES:
        qc = transpile(
            pauli_circuit_to_qiskit(pauli_circuit, cx_structure),
            basis_gates=BASIS_GATES,
            optimization_level=3,
            seed_transpiler=0,
        )
        result = {
            "ladder": cx_structure,
            "depth": qc.depth(),
            "cx_depth": qc.depth(lambda inst: inst.operation.num_qubits == 2),
            "cx": qc.count_ops().get("cx", 0),
        }
        if best is None or (result["depth"], result["cx"]) < (best["depth"], best["cx"]):
            best = result
    return best


def run(molecule: str) -> dict:
    data = json.loads((DATA / f"{molecule}.json").read_text())
    fham = fr.molecular_hamiltonian(
        np.array(data["ones"]), np.array(data["twos"]), data["constant_energy"]
    )
    n = fham.n_modes
    step = trotter_step(fham)
    print(f"\n## {molecule}: {n} modes, {len(step)} rotations per Trotter step")
    print(
        f"{'tree':6} {'method':12} | {'depth':>7} {'CX depth':>8} {'CX':>7} "
        f"{'ladder':>8} | {'depth vs naive':>14}"
    )
    print("-" * 74)

    results = {}
    for tree_name, make in TREES.items():
        naive = make(n).build_encoding()
        flatpack = make(n).flatpack()
        optimised = {
            "TOPP-HATT": topphatt(flatpack, n, fham.to_majorana_sparse()),
            "circuit": topphatt(flatpack, n, step),
        }

        rows = {"naive": transpiled(naive.encode_circuit(step.sorted_by_tree_order(naive)))}
        for name, encoding in optimised.items():
            perm = encoding.majorana_permutation_to(naive)
            relabelled = step.permuted(perm).sorted_by_tree_order(naive)
            pauli_circuit = naive.encode_circuit(relabelled)
            # Same circuit as the optimised encoding on its own tree order.
            assert pauli_circuit == encoding.encode_circuit(
                step.sorted_by_tree_order(encoding)
            )
            rows[name] = transpiled(pauli_circuit) | {"permutation": perm}

        base = rows["naive"]["depth"]
        for name, r in rows.items():
            change = "" if name == "naive" else f"{100 * (r['depth'] / base - 1):+.1f}%"
            print(
                f"{tree_name if name == 'naive' else '':6} {name:12} | "
                f"{r['depth']:7d} {r['cx_depth']:8d} {r['cx']:7d} {r['ladder']:>8} | "
                f"{change:>14}",
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
