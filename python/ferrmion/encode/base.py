"""Base FermionQubitEncoding class."""

import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferrmion.core import (
    anneal_enumerations,
    encode,
    encode_fermion_product,
    symplectic_product_map,
)
from ferrmion.hamiltonians import FermionHamiltonian, QubitHamiltonian
from ferrmion.utils import (
    pauli_to_symplectic,
    symplectic_to_pauli,
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
    def default_mode_op_map(self) -> NDArray[np.uint]:
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

    def encode_annealed(
        self,
        fham: FermionHamiltonian,
        temperature: int | None = None,
        initial_guess: list[int] | None = None,
        coefficient_weighted: bool = True,
    ):
        """Encode a Hamiltonian, using simulated annealing optimisation."""
        sigs, coeffs = fham.signatures_and_coefficients
        ipow, sym = self._build_symplectic_matrix()

        if temperature is None:
            temperature = fham.n_modes

        if isinstance(initial_guess, list):
            initial_guess: NDArray[np.uint] = np.array(initial_guess, dtype=np.uint)
        else:
            initial_guess: NDArray[np.uint] = np.array(
                [*range(self.n_modes)], dtype=np.uint
            )

        ipow, sym = anneal_enumerations(
            ipowers=ipow,
            symplectics=sym,
            signatures=sigs,
            coeffs=coeffs,
            temperature=temperature,
            initial_guess=initial_guess,
            coefficient_weighted=coefficient_weighted,
        )

        self._build_symplectic_matrix: Callable = lambda: (ipow, sym)
        self.default_mode_op_map = [*range(self.n_modes)]

        return encode(
            ipowers=ipow,
            symplectics=sym,
            signatures=sigs,
            coeffs=coeffs,
            constant_energy=fham.constant_energy,
        )

    def to_json(self) -> dict:
        """Store the encoding as a JSON-readable dict."""
        ipow, sym = self._build_symplectic_matrix()
        dict_output = {}
        dict_output["ipowers"] = ipow.tolist()
        dict_output["symplectics"] = sym.tolist()
        return dict_output

    def encode(self, fham: FermionHamiltonian) -> QubitHamiltonian:
        """Encode a Hamiltonian."""
        logger.debug("Encoding fermionic Hamiltonian.")
        ipowers, symplectic = self._build_symplectic_matrix()
        signatures, coeffs = fham.signatures_and_coefficients
        return encode(
            ipowers=ipowers,
            symplectics=symplectic,
            signatures=signatures,
            coeffs=coeffs,
            constant_energy=fham.constant_energy,
        )

    @staticmethod
    def _symplectic_to_pauli(
        symplectic: NDArray,
        ipower: int = 0,
    ) -> tuple[str, int]:
        """Convert a symplectic matrix to a Pauli string.

        Args:
            ipower (NDArray[np.uint]): power of i coefficient
            symplectic (NDArray): A symplectic vector.
        """
        return symplectic_to_pauli(symplectic, ipower)

    @staticmethod
    def _pauli_to_symplectic(
        pauli: str,
        ipower: int = 0,
    ) -> tuple[NDArray[bool], int]:
        """Convert a Pauli string to a symplectic matrix.

        Args:
            ipower (NDArray[np.uint]): power of i coefficient
            pauli (str): A Pauli-string.
        """
        return pauli_to_symplectic(pauli, ipower)

    @property
    def symplectic_product_map(self):
        """Calculate the product of symplectic terms and cache them."""
        logger.debug("Building symplectic product map")
        ipowers, symplectics = self._build_symplectic_matrix()
        return symplectic_product_map(ipowers, symplectics)

    def number_operator(
        self,
        mode: int,
        coeff: float | np.complex64 = np.complex64(1.0, 0.0),
    ) -> QubitHamiltonian:
        """Return the number operator of a mode for this encoding.

        Args:
            mode (int): The mode index to obtain a number operator for.
            coeff (float): Optional complex coeffient

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(4)
            >>> n_zero = tree.number_operator(0)
        """
        return self._encode_fermion_product(
            signature="+-",
            mode_indices=[mode, mode],
            coeff=coeff,
        )

    def edge_operator(
        self,
        edge_indices: tuple[int, int],
        coeff: float | np.complex64 = np.complex64(1.0, 0.0),
    ) -> QubitHamiltonian:
        """Return the edge operator of a pair of modes for this encoding.

        Args:
            edge_indices (tuple[int, int]): The mode index to obtain a number operator for.
            coeff (float): Optional complex coeffient

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(4)
            >>> tree.edge_operator((0, 1))
        """
        return self._encode_fermion_product(
            signature="+-",
            mode_indices=list(edge_indices),
            coeff=coeff,
        )

    def interaction_operator(
        self,
        mode_indices: tuple[int, int, int, int],
        coeff: float | np.complex64 = np.complex64(1.0, 0.0),
        physicist_notation=True,
    ) -> QubitHamiltonian:
        """Return an interaction operator of four modes for this encoding.

        Args:
            mode_indices (tuple[int, int, int, int]): The mode index to obtain an interaction operator for.
            coeff (float): Optional complex coeffient
            physicist_notation (bool): Use Physicist's notation, Chemist's if False.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(4)
            >>> tree.interaction_operator((0, 1, 2, 3))
        """
        if physicist_notation:
            return self._encode_fermion_product("++--", list(mode_indices), coeff=coeff)
        else:
            return self._encode_fermion_product("+-+-", list(mode_indices), coeff=coeff)

    def _encode_fermion_product(
        self,
        signature: str,
        mode_indices: list[int],
        coeff: np.complex64 | float,
    ) -> QubitHamiltonian:
        """Returns the sparse pauli form of a double fermionic operator.

        Args:
            signature (str): The fermionic operator signature, composed of "+" and "-".
            mode_indices (list[int]): The mode indices to obtain a number operator for.
            coeff (float): The operator coefficient.

        Returns:
            list[tuple[str, NDArray, np.complexfloating]]: A list of tuples each containing a Pauli string, its qubit indices and a complex coefficient.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(4)
            >>> _encode_fermion_product(tree, "++--", (0, 1, 2, 3), 1.0)
        """
        if not all([ind in range(self.n_modes) for ind in mode_indices]):
            raise ValueError("Indices invalid.")
        if len(signature) != len(mode_indices):
            raise ValueError("Signature and indices must be same length.")
        if isinstance(coeff, float | int):
            coeff = np.complex64(coeff, 0.0)
        return encode_fermion_product(
            *self._build_symplectic_matrix(), signature, mode_indices, coeff
        )


class MajoranaStringEncoding(FermionQubitEncoding):
    """Majorana string encoding defined by input."""

    def __init__(self, ipowers: ArrayLike, symplectics: ArrayLike):
        """Initialize MajoranaStringEncoding.

        Args:
            ipowers (ArrayLike): Array of integers representing powers of the imaginary unit (i).
            symplectics (ArrayLike): Array of booleans representing Pauli strings in XZ format.
        """
        self._ipowers = np.array(ipowers, dtype=np.uint8)
        self._ipowers %= 4

        self._symplectics = np.array(symplectics, dtype=np.bool)

        if len(self._ipowers) != len(self._symplectics):
            raise ValueError("ipowers and symplectics must be same length.")
        self.n_modes = self._symplectics.shape[0] // 2
        self.n_qubits = self._symplectics.shape[1] // 2
        super().__init__(self.n_modes, self.n_qubits)

    def _build_symplectic_matrix(
        self,
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool]]:
        """Build the symplectic matrix for the Majorana string encoding.

        Returns:
            Tuple[ArrayLike, ArrayLike]: The symplectic matrix in XZ format.
        """
        return self._ipowers, self._symplectics
