"""Interop for ffsim."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from ferrmion import FermionQubitEncoding
from ffsim import variational
from ffsim.qiskit.gates.orbital_rotation import (
    OrbitalRotationJW,
)
from numpy.typing import NDArray
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


class UCJOpSpinBalancedGeneric(Gate):
    """Qiskit Gate for the UCJ Operator with arbitrary encoding."""

    def __init__(
        self,
        ucj_op: variational.UCJOpSpinBalanced,
        encoding: FermionQubitEncoding,
        *,
        label: str | None = None,
    ):
        """Create a new spin-balanced unitary cluster Jastrow (UCJ) gate.

        Args:
            ucj_op: The UCJ operator.
            encoding: A fermion qubit encoding.
            label: The label of the gate.
        """
        self.ucj_op = ucj_op
        self.encoding = encoding

        super().__init__("ucj_balanced_generic", 2 * ucj_op.norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _ucj_op_spin_balanced_generic(qubits, self.ucj_op, encoding=self.encoding),
            qubits=qubits,
            name=self.name,
        )


def _ucj_op_spin_balanced_generic(
    qubits: Sequence[Qubit],
    ucj_op: variational.UCJOpSpinBalanced,
    encoding: FermionQubitEncoding,
) -> Iterator[CircuitInstruction]:
    """Create circuit instructions for the UCJ oprator in any encoding."""
    for (diag_coulomb_mat_aa, diag_coulomb_mat_ab), orbital_rotation in zip(
        ucj_op.diag_coulomb_mats, ucj_op.orbital_rotations
    ):
        # I think the rotations can stay as they are.
        # These come directly from the t2 amplitudes
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, orbital_rotation.T.conj()),
            qubits,
        )
        # This part will need to change
        yield CircuitInstruction(
            DiagCoulombEvolutionGeneric(
                ucj_op.norb,
                (diag_coulomb_mat_aa, diag_coulomb_mat_ab, diag_coulomb_mat_aa),
                -1.0,
                encoding=encoding,
            ),
            qubits,
        )
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, orbital_rotation), qubits
        )
    if ucj_op.final_orbital_rotation is not None:
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, ucj_op.final_orbital_rotation), qubits
        )


class DiagCoulombEvolutionGeneric(Gate):
    """Qiskit Gate for the diagonal coulomb term in arbitrary encoding."""

    def __init__(
        self,
        norb: int,
        mat: NDArray | tuple[NDArray | None, NDArray | None, NDArray | None],
        time: float,
        *,
        z_representation: bool = False,
        label: str | None = None,
        encoding: FermionQubitEncoding,
    ):
        r"""Create new diagonal Coulomb evolution gate.

        Args:
            norb: The number of spatial orbitals.
            mat: The diagonal Coulomb matrix :math:`Z`.
                You can pass either a single Numpy array specifying the coefficients
                to use for all spin interactions, or you can pass a tuple of three Numpy
                arrays specifying independent coefficients for alpha-alpha, alpha-beta,
                and beta-beta interactions (in that order). If passing a tuple, you can
                set a tuple element to ``None`` to indicate the absence of interactions
                of that type. The alpha-alpha and beta-beta matrices are assumed to be
                symmetric, and only their upper triangular entries are used.
            time: The evolution time.
            z_representation: Whether the input matrices are in the "Z" representation.
            label: The label of the gate.
            encoding (FermionQubitEncoding): A fermion-qubit encoding method.
        """
        self.norb = norb
        self.mat = mat
        self.time = time
        self.z_representation = z_representation
        self.encoding = encoding
        super().__init__("diag_coulomb_jw", 2 * norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        generate_instructions = (
            # _diag_coulomb_evo_z_rep_jw
            # if self.z_representation
            # else _diag_coulomb_evo_num_rep_jw
            _diag_coulomb_evo_num_rep_generic
        )
        self.definition = QuantumCircuit.from_instructions(
            generate_instructions(
                qubits,
                mat=self.mat,
                time=self.time,
                norb=self.norb,
                encoding=self.encoding,
            ),
            qubits=qubits,
        )

    def inverse(self):
        """Inverse gate."""
        return DiagCoulombEvolutionGeneric(
            self.norb, self.mat, -self.time, z_representation=self.z_representation
        )


def _diag_coulomb_evo_num_rep_generic(
    qubits: Sequence[Qubit],
    mat: NDArray | tuple[NDArray | None, NDArray | None, NDArray | None],
    time: float,
    norb: int,
    encoding: FermionQubitEncoding,
) -> Iterator[CircuitInstruction]:
    """Craeate circuit instructions for the Diagonal Coulomb Term in a generic encoding."""
    assert len(qubits) == 2 * norb
    mat_aa: NDArray | None
    mat_ab: NDArray | None
    mat_bb: NDArray | None
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        mat_aa, mat_ab, mat_bb = mat, mat, mat
    else:
        mat_aa, mat_ab, mat_bb = mat

    # gates that involve a single spin sector
    # NOTE Need to change the way this is ordered so that
    # there are no complex coefficients
    for sigma, this_mat in enumerate([mat_aa, mat_bb]):
        if this_mat is not None:
            for i in range(norb):
                for j in range(i, norb):
                    if i == j and this_mat[i, i]:
                        sparse_op = SparsePauliOp.from_list(encoding.number_operator(i))
                        sparse_op = sparse_op.compose(sparse_op).simplify()
                        yield CircuitInstruction(
                            PauliEvolutionGate(sparse_op, -0.5 * this_mat[i, i] * time),
                        )
                    if this_mat[i, j]:
                        # NOTE assuming the number operators commute
                        left = SparsePauliOp.from_list(encoding.number_operator(i))
                        right = SparsePauliOp.from_list(encoding.number_operator(j))
                        yield CircuitInstruction(
                            PauliEvolutionGate(
                                left.compose(right).simplify(),
                                -0.5 * this_mat[i, j] * time,
                            ),
                        )
                    if this_mat[j, i]:
                        # NOTE assuming the number operators commute
                        left = SparsePauliOp.from_list(encoding.number_operator(j))
                        right = SparsePauliOp.from_list(encoding.number_operator(i))
                        yield CircuitInstruction(
                            PauliEvolutionGate(
                                left.compose(right).simplify(),
                                -0.5 * this_mat[i, j] * time,
                            ),
                        )

    # gates that involve both spin sectors
    if mat_ab is not None:
        for i in range(norb):
            for j in range(i, norb):
                if mat_ab[i, i]:
                    sparse_op = SparsePauliOp.from_list(encoding.number_operator(i))
                    sparse_op = sparse_op.compose(sparse_op).simplify()
                    yield CircuitInstruction(
                        PauliEvolutionGate(
                            left.compose(right).simplify(), -mat_ab[i, i] * time
                        ),
                    )
                if mat_ab[i, j]:
                    left = SparsePauliOp.from_list(encoding.number_operator(i))
                    right = SparsePauliOp.from_list(encoding.number_operator(j))
                    yield CircuitInstruction(
                        PauliEvolutionGate(
                            left.compose(right).simplify(), -mat_ab[i, j] * time
                        ),
                    )
                if mat_ab[j, i]:
                    left = SparsePauliOp.from_list(encoding.number_operator(j))
                    right = SparsePauliOp.from_list(encoding.number_operator(i))
                    yield CircuitInstruction(
                        PauliEvolutionGate(
                            left.compose(right).simplify(), -mat_ab[j, i] * time
                        ),
                    )
