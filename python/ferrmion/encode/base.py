"""Base FermionQubitEncoding class."""

import logging
from abc import ABC, abstractmethod

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
        self.default_mode_op_map = {i: i for i in range(self.n_modes)}

    @property
    def default_mode_op_map(self):
        """Create a default mode operator map for the tree."""
        return self._default_mode_op_map

    @default_mode_op_map.setter
    def default_mode_op_map(self, map_dict: dict[int, int]):
        """Set the default mode operator map.

        Args:
            map_dict (dict[int, int]): A dictionary mapping modes to operators.
        """
        logger.debug("Setting default mode operator map.")
        error_string = ""
        if set(map_dict.keys()) != {*range(self.n_modes)}:
            error_string += "Default Mode op map does not cover all modes.\n"
        if set(map_dict.values()) != {*range(self.n_modes)}:
            error_string += "Default Mode op map does not cover all operators.\n"

        if error_string != "":
            logger.error(error_string)
            logger.error(map_dict)
            raise ValueError(error_string)

        self._default_mode_op_map = map_dict

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
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
        """Build a symplectic matrix representing terms for each operator in the Hamitonian."""
        pass

    def hartree_fock_state(
        self, fermionic_hf_state: NDArray[np.bool_], mode_op_map: dict | None = None
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
    def _symplectic_to_pauli(symplectic: NDArray) -> tuple[int, str]:
        """Convert a symplectic matrix to a Pauli string.

        Args:
            symplectic (NDArray): A symplectic vector.
        """
        return symplectic_to_pauli(symplectic)

    @staticmethod
    def _pauli_to_symplectic(pauli: str) -> tuple[int, NDArray[np.bool_]]:
        """Convert a Pauli string to a symplectic matrix.

        Args:
            pauli (str): A Pauli-string.
        """
        return pauli_to_symplectic(pauli)

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
        """
        return number_operator(self, mode)

    def edge_operator(
        self, edge_indices: tuple[int, int]
    ) -> list[tuple[str, NDArray, np.complexfloating]]:
        """Return the edge operator of a pair of modes for this encoding.

        Args:
            edge_indices (tuple[int, int]): The mode index to obtain a number operator for.
        """
        return edge_operator(self, edge_indices)


def number_operator(
    encoding: FermionQubitEncoding, mode: int
) -> list[tuple[str, NDArray, np.complexfloating]]:
    """Return the number operator for a given encoding and mode.

    Args:
        encoding (FermionQubitEncoding): A Fermion to qubit encoding object.
        mode (int): The mode index to obtain a number operator for.
    """
    return edge_operator(encoding, (mode, mode))


def edge_operator(
    encoding: FermionQubitEncoding, edge_indices: tuple[int, int]
) -> list[tuple[str, NDArray, np.complexfloating]]:
    """Return the number operator for a given encoding and pair of modes.

    Args:
        encoding (FermionQubitEncoding): A Fermion to qubit encoding object.
        edge_indices (tuple[int, int]): The mode index to obtain a number operator for.
    """
    logger.debug("Finding edge operator %s", edge_indices)
    if not set(edge_indices).issubset(set(encoding.default_mode_op_map.keys())):
        logger.error("Edge operator indices invalid %s", edge_indices)
        raise ValueError("Edge operator indices invalid %s", edge_indices)

    icount, sym_products = encoding.symplectic_product_map
    m, n = edge_indices
    m = encoding.default_mode_op_map[m]
    n = encoding.default_mode_op_map[n]

    first_term = sym_products[2 * m, 2 * n]
    second_term = sym_products[2 * m, 2 * n + 1]
    third_term = sym_products[2 * m + 1, 2 * n]
    fourth_term = sym_products[2 * m + 1, 2 * n + 1]

    terms = [first_term, second_term, third_term, fourth_term]
    terms: list[tuple[int, str, NDArray]] = [symplectic_to_sparse(t) for t in terms]
    factors = (
        0.25 * icount_to_sign(icount[2 * m, 2 * n] + terms[0][0]),
        0.25 * icount_to_sign(icount[2 * m, 2 * n + 1] + 1 + terms[1][0]),
        0.25 * icount_to_sign(icount[2 * m + 1, 2 * n] + 3 + terms[2][0]),
        0.25 * icount_to_sign(icount[2 * m + 1, 2 * n + 1] + terms[3][0]),
    )

    return [(t[1], t[2], f) for t, f in zip(terms, factors)]
