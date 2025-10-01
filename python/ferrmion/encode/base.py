"""Base FermionQubitEncoding class."""

import logging
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
from numpy.typing import NDArray

from ferrmion.core import hartree_fock_state, symplectic_product_map
from ferrmion.utils import (
    icount_to_sign,
    pauli_to_symplectic,
    symplectic_to_pauli,
    symplectic_to_sparse,
)

logger = logging.getLogger(__name__)


class FermionQubitEncoding(ABC):
    """Fermion Encodings for the Electronic Structure Hamiltonian in symplectic form.

    Attributes:
        one_e_coeffs (NDArray): One electron coefficients.
        two_e_coeffs (NDArray): Two electron coefficients.
        modes (set[int]): A set of modes.
        n_qubits (int): The number of qubits.

    Methods:
        default_mode_op_map: Get the default mode operator map.
        _build_symplectic_matrix: Build a symplectic matrix representing terms for each operator in the Hamiltonian.
        hartree_fock_state: Find the Hartree-Fock state of a majorana string encoding.
        _symplectic_to_pauli: Convert a symplectic matrix to a Pauli string.
        _pauli_to_symplectic: Convert a Pauli string to a symplectic matrix.
        fill_template: Fill a template with Hamiltonian coefficients.
        to_symplectic_hamiltonian: Output the hamiltonian in symplectic form.
        to_qubit_hamiltonian: Create qubit representation Hamiltonian.

    NOTE: A 'Y' pauli operator is mapped to -iXY so a (0+n)**3 term is needed.

    Example:
        >>> from ferrmion.encode.base import FermionQubitEncoding
        >>> class DummyEncoding(FermionQubitEncoding):
        ...     def _build_symplectic_matrix(self):
        ...         import numpy as np
        ...         return np.zeros(1), np.zeros((1, 2))
        >>> enc = DummyEncoding(2, 2)
        >>> enc.n_modes
        2
    """

    def __init__(
        self,
        n_modes: int,
        n_qubits: int,
    ):
        """Initialise encoding.

        Args:
            n_modes (int): Number of Fermion modes to encode.
            n_qubits (int): Number of Qubits used to encode.
            vacuum_state (NDArray | None): The vacuum state of the encoding.
        """
        self.n_modes = n_modes
        self.n_qubits = n_qubits
        self.default_mode_op_map = np.array([*range(self.n_modes)], dtype=np.uint)

    def __eq__(self, other: object) -> bool:
        """Checks if two encodings are exactly equivalent."""
        if isinstance(other, FermionQubitEncoding):
            if self.n_modes != other.n_modes:
                return False

            if self.n_qubits != other.n_qubits:
                return False

            left = self._build_symplectic_matrix()
            right = other._build_symplectic_matrix()

            if not np.all(left[0] == right[0]) or not np.all(left[1] == right[1]):
                return False

            return True
        else:
            return False

    @property
    def default_mode_op_map(self):
        """Create a default mode operator map for the tree."""
        return self._default_mode_op_map

    @default_mode_op_map.setter
    def default_mode_op_map(self, permutation: list[int]):
        """Set the default mode operator map.

        Args:
            permutation (list[int]): A list containing a permutation of mode indices.
        """
        logger.debug("Setting default mode operator map.")
        error_string = ""
        if set(permutation) != {*range(self.n_modes)}:
            error_string += "Default Mode op map does not cover all modes.\n"

        if error_string != "":
            logger.error(error_string)
            logger.error(permutation)
            raise ValueError(error_string)

        self._default_mode_op_map = np.array(permutation, dtype=np.uint)

    @property
    def vacuum_state(self):
        """Return the vacuum state."""
        return self._vacuum_state

    @vacuum_state.setter
    def vacuum_state(self, state: NDArray):
        """Validate and set the vacuum state.

        Args:
            state (NDArray): The vacuum state.
        """
        logger.debug("Setting vacuum state as %s", state)
        error_string = []
        state = np.array(state, dtype=np.float64)

        if len(state) != self.n_qubits:
            error_string.append("vacuum state must be length " + str(self.n_qubits))
        if state.ndim != 1:
            error_string.append("vacuum state must be vector (dimension==1)")

        if error_string != []:
            logger.error("\n".join(error_string))
            raise ValueError("\n".join(error_string))
        else:
            self._vacuum_state = state

    @abstractmethod
    def _build_symplectic_matrix(
        self,
    ) -> tuple[NDArray[np.uint8], NDArray[bool]]:
        """Build a symplectic matrix representing terms for each operator in the Hamitonian."""
        pass

    def hartree_fock_state(
        self,
        fermionic_hf_state: NDArray[bool],
        mode_op_map: list[int] | None = None,
    ):
        """Find the Hartree-Fock state of a majorana string encoding.

        This function calls to the rust implementatin in `src/lib.rs`.
        It assumes that the vacuum state is a single state vector, though the HF state may not be
        The global phase so that the first component state has 0 phase.

        Args:
            fermionic_hf_state (NDArray[int]): An array of mode occupations.
            mode_op_map (dict[int, int]): A dictionary mapping modes to sets of majorana strings i->(j,j+1).

        Returns:
            NDArray: The Hartree-Fock ground state in computational basis.
        """
        if mode_op_map is None:
            mode_op_map = self.default_mode_op_map

        return hartree_fock_state(
            self.vacuum_state,
            fermionic_hf_state,
            mode_op_map,
            self._build_symplectic_matrix()[1],
        )

    @staticmethod
    def _symplectic_to_pauli(ipower: int, symplectic: NDArray) -> tuple[int, str]:
        """Convert a symplectic matrix to a Pauli string.

        Args:
            ipower (NDArray[np.uint]): power of i coefficient
            symplectic (NDArray): A symplectic vector.
        """
        return symplectic_to_pauli(ipower, symplectic)

    @staticmethod
    def _pauli_to_symplectic(ipower: int, pauli: str) -> tuple[int, NDArray[bool]]:
        """Convert a Pauli string to a symplectic matrix.

        Args:
            ipower (NDArray[np.uint]): power of i coefficient
            pauli (str): A Pauli-string.
        """
        return pauli_to_symplectic(ipower, pauli)

    @property
    def symplectic_product_map(self):
        """Calculate the product of symplectic terms and cache them."""
        logger.debug("Building symplectic product map")
        ipowers, symplectics = self._build_symplectic_matrix()
        return symplectic_product_map(ipowers, symplectics)

    def number_operator(
        self, mode: int
    ) -> list[tuple[str, NDArray, np.complexfloating]]:
        """Return the number operator of a mode for this encoding.

        Args:
            mode (int): The mode index to obtain a number operator for.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTee(4)
            >>> tree.number_operator(0)
        """
        return number_operator(self, mode)

    def edge_operator(
        self, edge_indices: tuple[int, int]
    ) -> list[tuple[str, NDArray, np.complexfloating]]:
        """Return the edge operator of a pair of modes for this encoding.

        Args:
            edge_indices (tuple[int, int]): The mode index to obtain a number operator for.

            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTee(4)
            >>> tree.edge_operator(0, 1)
        """
        return edge_operator(self, edge_indices)


def number_operator(
    encoding: FermionQubitEncoding, mode: int
) -> list[tuple[str, NDArray, np.complexfloating]]:
    """Return the number operator for a given encoding and mode.

    Args:
        encoding (FermionQubitEncoding): A Fermion to qubit encoding object.
        mode (int): The mode index to obtain a number operator for.

    Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTee(4)
            >>> tree.number_operator(0)
    """
    return edge_operator(encoding, (mode, mode))


def edge_operator(
    encoding: FermionQubitEncoding, edge_indices: tuple[int, int]
) -> list[tuple[str, NDArray, np.complexfloating]]:
    """Return the edge operator for a given encoding and pair of modes.

    Args:
        encoding (FermionQubitEncoding): A Fermion to qubit encoding object.
        edge_indices (tuple[int, int]): The mode index to obtain a number operator for.

    Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTee(4)
            >>> tree.edge_operator(0,1)
    """
    return double_fermionic_operator(
        encoding=encoding, mode_indices=edge_indices, signature="+-"
    )


def double_fermionic_operator(
    encoding: FermionQubitEncoding, mode_indices: tuple[int, int], signature: str
) -> list[tuple[str, NDArray, np.complexfloating]]:
    """Returns the sparse pauli form of a double fermionic operator.

    Args:
        encoding (FermionQubitEncoding): A Fermion to qubit encoding object.
        mode_indices (tuple[int, int]): The mode indices to obtain a number operator for.
        signature (str): The fermionic operator signature, one of "++", "+-", "-+", "--".

    Returns:
        list[tuple[str, NDArray, np.complexfloating]]: A list of tuples each containing a Pauli string, its qubit indices and a complex coefficient.

    Example:
        >>> from ferrmion import TernaryTree
        >>> tree = TernaryTree(4)
        >>> tree.double_fermionic_operator((0,1), "+-")
        [('ZZ', array([0, 1]), 0.25+0j), ('YX', array([0, 1]), 0.25j), ('XY', array([0, 1]), -0.25j), ('II', array([0, 1]), 0.25+0j)]
    """
    match signature:
        case "++":
            signature_iterm = [0, 3, 3, 2]
        case "+-":
            signature_iterm = [0, 1, 3, 0]
        case "-+":
            signature_iterm = [0, 3, 1, 0]
        case "--":
            signature_iterm = [0, 1, 1, 2]
        case _:
            logger.error(
                "Operator signature can only contain + or -, %s not valid", signature
            )
            raise ValueError(
                "Operator signature can only contain + or -, %s not valid", signature
            )

    logger.debug("Finding edge operator %s", mode_indices)
    if not set(mode_indices).issubset(set(range(encoding.n_modes))):
        logger.error("Edge operator indices invalid %s", mode_indices)
        raise ValueError("Edge operator indices invalid %s", mode_indices)

    icount, sym_products = encoding.symplectic_product_map
    m, n = mode_indices
    m = int(encoding.default_mode_op_map[m])
    n = int(encoding.default_mode_op_map[n])
    terms: list[tuple[int, str, NDArray]] = [
        symplectic_to_sparse(
            icount[2 * m + l, 2 * n + r], sym_products[2 * m + l, 2 * n + r]
        )
        for l, r in product([0, 1], [0, 1])
    ]
    factors = (
        0.25 * icount_to_sign(terms[0][0] + signature_iterm[0]),
        0.25 * icount_to_sign(terms[1][0] + signature_iterm[1]),
        0.25 * icount_to_sign(terms[2][0] + signature_iterm[2]),
        0.25 * icount_to_sign(terms[3][0] + signature_iterm[3]),
    )

    return [(t[1], t[2], f) for t, f in zip(terms, factors)]
