"""Defines the interface to Qiskit-Nature."""

import logging
from itertools import product

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from ferrmion import FermionQubitEncoding
from ferrmion.utils import symplectic_product, symplectic_to_sparse

logger = logging.getLogger(__name__)


class QiskitAdapter(FermionicMapper):
    """Wrapper class enabling the use of ferrmion in qiskit.

    In qiskit_nature, encodings are handled by a general `QubitMapper` class,
    and a `FermionMapper` for Fermion to Qubit encodings.

    These classes have an abstract method `_map_single` which transforms a
    single qiskit operator into a `qiskit.SparsePauliOp`.

    NOTE: You must have installed the optional dependencies with
     `pip install ferrmion[qiskit]` for this functionality to be available.

    Example:
    >>> from ferrmion.encode import JordanWigner
    >>> from ferrmion.interop.qiskit import QiskitAdapter
    >>> from qiskit_nature.second_q.operators import FermionOp
    >>> fop = FermionicOp(
    >>>     {
    >>>         "+_0 -_0": 1.0,
    >>>         "+_1 -_1": -1.0,
    >>>     },
    >>>     num_spin_orbitals=2,
    >>> )
    >>> mapper = QiskitAdapter(JordanWigner(2))
    >>> mapper.map(fop)

    )

    """

    def __init__(self, encoding: FermionQubitEncoding) -> None:
        """Initialise QiskitAdapter.

        Args:
            encoding (FermionQubitEncoding): A valid ferrmion encoding.
        """
        self.encoding = encoding
        super().__init__()

    def _map_single(
        self, second_q_op: FermionicOp, *, register_length: int | None = None
    ) -> SparsePauliOp:
        """Function required to adapt ferrmion encodings to qiskit_nature.

        Allows the use of a ferrmion.FermionQubitEncoding to encode
        qiskit_nature.

        Args:
            second_q_op (qiskit.FermionicOp): A fermionic Operator.
            register_length (int): Number of qubits to use for operator (typically equals the number of modes of encoding).
        """
        if register_length is None:
            register_length = second_q_op.register_length
        ipowers, symplectics = self.encoding._build_symplectic_matrix()

        term_ops = []
        for term, term_coeff in second_q_op.terms():
            majoranas = []
            term_ipowers = []
            logger.debug(f"{term=}, {term_coeff=}")
            for signature, index in term:
                logger.debug(f"{signature=}, {index=}")
                match signature:
                    case "+":
                        term_ipowers.append([0, 3])
                    case "-":
                        term_ipowers.append([0, 1])
                    case _:
                        raise ValueError("Term signature should be + or -")

                majoranas.append([2 * int(index), 2 * int(index) + 1])

            sparse_list = []
            for iterm, comb in zip(product(*term_ipowers), product(*majoranas)):
                logger.debug(f"{comb=}")
                ipower = sum(ipowers[m] for m in comb) + sum(it for it in iterm)
                left = symplectics[comb[0]]
                logger.debug(f"{comb[0]=},{ipower=}, {left=}")
                for m_index in comb[1:]:
                    right = symplectics[m_index]
                    iprod, left = symplectic_product(left, right)
                    ipower += iprod
                    logger.debug(f"{m_index=}, {ipower=}, {left=}")
                sparse = symplectic_to_sparse(left, ipower)
                logger.debug(f"{sparse=}")

                coeff = term_coeff
                coeff *= 0.5 ** len(comb)
                coeff *= sparse[2]

                sparse_list.append((sparse[0], list(sparse[1]), coeff))
                logger.debug(f"{sparse_list=}")
            term_ops.append(
                SparsePauliOp.from_sparse_list(
                    sparse_list, num_qubits=register_length
                ).simplify()
            )
        logger.debug(f"{term_ops=}")

        sparse_op: SparsePauliOp = term_ops[0]
        for term in term_ops[1:]:
            sparse_op += term
        return sparse_op.simplify()
