"""Build Qiskit circuits from ferrmion Pauli rotation circuits.

Requires the optional Qiskit dependencies, e.g. ``pip install ferrmion[qiskit]``.

ferrmion Pauli strings are qubit-0-first: character ``q`` acts on qubit ``q``.
Qiskit's ``Pauli`` labels are qubit-0-last, but here every gate is placed on
explicit qubit indices, so qubit ``q`` of a ferrmion string is Qiskit qubit ``q``.
"""

from collections.abc import Iterable

from qiskit import QuantumCircuit

from ferrmion.core import PauliCircuit

CX_STRUCTURES = ("chain", "fountain")


def _ladder(qubits: list[int], cx_structure: str) -> list[tuple[int, int]]:
    if cx_structure == "chain":
        return list(zip(qubits, qubits[1:]))
    if cx_structure == "fountain":
        return [(qubit, qubits[-1]) for qubit in qubits[:-1]]
    raise ValueError(
        f"unknown cx_structure {cx_structure!r}; expected one of {CX_STRUCTURES}"
    )


def append_pauli_rotation(
    circuit: QuantumCircuit,
    pauli: str,
    angle: float,
    cx_structure: str = "chain",
) -> None:
    """Append ``exp(-i * angle * P)`` to ``circuit`` for a Pauli string ``P``.

    Each non-identity qubit is rotated into the Z basis, a CNOT ladder
    computes the parity onto a pivot, ``RZ(2 * angle)`` acts on it, and the
    ladder and basis changes are undone in reverse.

    Args:
        circuit: The circuit to append to; it needs at least ``len(pauli)``
            qubits.
        pauli: Qubit-0-first Pauli string.
        angle: Rotation angle.
        cx_structure: ``"chain"`` (each CNOT feeds the next) or
            ``"fountain"`` (every CNOT targets the pivot). Both realise the
            same unitary, but leave a transpiler different amounts to cancel
            between neighbouring rotations.
    """
    active = [q for q, p in enumerate(pauli) if p != "I"]
    if not active:
        return
    ladder = _ladder(active, cx_structure)
    for q in active:
        if pauli[q] == "X":
            circuit.h(q)
        elif pauli[q] == "Y":
            circuit.sdg(q)
            circuit.h(q)
    for control, target in ladder:
        circuit.cx(control, target)
    circuit.rz(2 * angle, active[-1])
    # Undo the ladder in reverse: chain CNOTs share qubits and do not commute.
    for control, target in reversed(ladder):
        circuit.cx(control, target)
    for q in active:
        if pauli[q] == "X":
            circuit.h(q)
        elif pauli[q] == "Y":
            circuit.h(q)
            circuit.s(q)


def pauli_rotation(
    pauli: str, angle: float, cx_structure: str = "chain"
) -> QuantumCircuit:
    """Circuit implementing ``exp(-i * angle * P)`` for a qubit-0-first ``P``."""
    circuit = QuantumCircuit(len(pauli))
    append_pauli_rotation(circuit, pauli, angle, cx_structure)
    return circuit


def rotations_to_qiskit(
    rotations: Iterable[tuple[str, float]],
    n_qubits: int,
    cx_structure: str = "chain",
) -> QuantumCircuit:
    """Circuit applying ``exp(-i θ_j P_j)`` for each ``(P_j, θ_j)`` in order."""
    circuit = QuantumCircuit(n_qubits)
    for pauli, angle in rotations:
        append_pauli_rotation(circuit, pauli, angle, cx_structure)
    return circuit


def pauli_circuit_to_qiskit(
    pauli_circuit: PauliCircuit, cx_structure: str = "chain"
) -> QuantumCircuit:
    """Convert a ferrmion ``PauliCircuit`` to a Qiskit ``QuantumCircuit``.

    Gate order and repeats are kept: gate ``j`` of the result is
    ``exp(-i θ_j P_j)``.

    Args:
        pauli_circuit: The circuit, e.g. from ``compile_circuit`` or
            ``MajoranaEncoding.encode_circuit``.
        cx_structure: CNOT ladder layout, ``"chain"`` or ``"fountain"``.

    Returns:
        QuantumCircuit: An un-transpiled circuit of H, S, S†, CX and RZ gates.
    """
    return rotations_to_qiskit(
        pauli_circuit.to_list(), pauli_circuit.n_qubits, cx_structure
    )
