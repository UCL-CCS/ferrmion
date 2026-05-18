"""Class and methods to easily build general Fermion Hamiltonians."""

import logging
from typing import Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from ferrmion.core import FermionProduct, MatrixFermion, SparseFermion

logger = logging.getLogger(__name__)

"""
Type alias for qubit hamiltonians.
"""
type QubitHamiltonian = dict[str, float]

FermionTerm = Union[FermionProduct, SparseFermion, MatrixFermion]


class FermionHamiltonian:
    """Class for building Fermionic Hamiltonians.

    A container of one or more FermionProduct, SparseFermion, or MatrixFermion
    terms, with an optional constant energy offset.

    The ``terms`` constructor argument accepts either:
    - a ``list`` of operator objects (FermionProduct, SparseFermion, MatrixFermion), or
    - a ``dict[str, NDArray]`` for backward compatibility with the previous API.
    """

    def __init__(
        self,
        *,
        terms: Union[list[FermionTerm], dict[str, NDArray], None] = None,
        constant_energy: float = 0.0,
    ):
        """Initialiser for FermionHamiltonian."""
        logger.debug("Initialising FermionHamiltonian")
        self.constant_energy = constant_energy
        self._term_list: list[FermionTerm] = []
        self._next_action: list[str] = []
        self.n_modes: int = 0

        if terms is None:
            terms = {}

        if isinstance(terms, dict):
            for sig, coeff in terms.items():
                action = list(sig)
                self._add_term(MatrixFermion(action, np.asarray(coeff, dtype=float)))
        else:
            for term in terms:
                self._add_term(term)

    def _add_term(self, term: FermionTerm) -> None:
        self._term_list.append(term)
        if isinstance(term, MatrixFermion):
            n = term.coefficients.shape[0]
            if self.n_modes == 0:
                self.n_modes = n
            elif n != self.n_modes:
                raise ValueError(
                    f"Hamiltonian coefficient shape {term.coefficients.shape} is "
                    f"inconsistent with n_modes={self.n_modes}."
                )

    @property
    def _terms(self) -> dict[str, NDArray]:
        """Dict view of MatrixFermion terms for backward compatibility."""
        result: dict[str, NDArray] = {}
        for term in self._term_list:
            if isinstance(term, MatrixFermion):
                sig = "".join(term.action)
                result[sig] = term.coefficients
        return result

    def __repr__(self) -> str:
        """String representation of FermionHamiltonian."""
        n_terms = len(self._term_list)
        return f"FermionHamiltonian({n_terms} terms, {self.n_modes} modes, constant {self.constant_energy})"

    @property
    def signatures_and_coefficients(self) -> tuple[list[str], list[NDArray]]:
        """Return signature strings and coefficient arrays for all terms.

        MatrixFermion terms map directly. SparseFermion and FermionProduct terms
        are densified into NDArrays using ``n_modes``.
        """
        sigs: list[str] = []
        coeffs: list[NDArray] = []
        for term in self._term_list:
            if isinstance(term, MatrixFermion):
                sigs.append("".join(term.action))
                coeffs.append(term.coefficients)
            elif isinstance(term, SparseFermion):
                sig = "".join(term.action)
                rank = len(term.action)
                arr = np.zeros((self.n_modes,) * rank, dtype=float)
                for idx_row, coeff in zip(term.indices, term.coefficients):
                    arr[tuple(idx_row)] += coeff.real
                sigs.append(sig)
                coeffs.append(arr)
            elif isinstance(term, FermionProduct):
                sig = "".join(term.action)
                rank = len(term.action)
                arr = np.zeros((self.n_modes,) * rank, dtype=float)
                arr[tuple(term.indices)] += term.coefficient.real
                sigs.append(sig)
                coeffs.append(arr)
        return (sigs, coeffs)

    def add_term(self, term: FermionTerm) -> "FermionHamiltonian":
        """Add a FermionProduct, SparseFermion, or MatrixFermion term."""
        self._add_term(term)
        return self

    def creation(self) -> "FermionHamiltonian":
        """Append a creation operator to the current builder action."""
        self._next_action.append("+")
        return self

    def annihilation(self) -> "FermionHamiltonian":
        """Append an annihilation operator to the current builder action."""
        self._next_action.append("-")
        return self

    def with_coefficients(self, coefficients: NDArray) -> "FermionHamiltonian":
        """Finalise the current builder action with a dense coefficient array."""
        if coefficients.ndim != len(self._next_action):
            logger.error(f"Cannot apply coefficients to action {self._next_action}")
        else:
            self._add_term(
                MatrixFermion(self._next_action, np.asarray(coefficients, dtype=float))
            )
            self._next_action = []
        return self

    def add_constant(self, constant_energy: float) -> "FermionHamiltonian":
        """Add a constant energy offset."""
        self.constant_energy += constant_energy
        return self


def molecular_hamiltonian(
    one_e_coeffs: NDArray,
    two_e_coeffs: NDArray,
    constant_energy: float = 0.0,
    physicist_notation: bool = True,
) -> FermionHamiltonian:
    """Return a molecular electronic structure Hamiltonian.

    Args:
        one_e_coeffs (NDArray): One electron hamiltonian coefficients in spinorb format.
        two_e_coeffs (NDArray): Two electron hamiltonian coefficients in spinorb format.
        constant_energy (float): Constant energy offset.
        physicist_notation (bool): Set to False for Chemist Notation.

    Example:
        >>> import numpy as np
        >>> from ferrmion.hamiltonians import molecular_hamiltonian
        >>> one_e = np.eye(2)
        >>> two_e = np.zeros((2, 2, 2, 2))
        >>> fham = molecular_hamiltonian(one_e, two_e, 0.0)
        >>> fham.n_modes
        2
    """
    if physicist_notation:
        terms = {"+-": one_e_coeffs, "++--": two_e_coeffs}
    else:
        terms = {"+-": one_e_coeffs, "+-+-": two_e_coeffs}

    return FermionHamiltonian(terms=terms, constant_energy=constant_energy)


def linear_adjacency_matrix(length: int, periodic: bool) -> npt.NDArray[bool]:
    """Creates an adjacency matrix for a linear Hubbard Hamiltonian.

    Args:
        length (int): The number of sites.
        periodic (bool): If true, periodic boundary conditions are used.

    Returns:
        np.ndarray[bool]: Adjacency matrix for lattice sites.
    """
    return square_lattice_adjacency_matrix((length, 1), periodic=periodic)


def square_lattice_adjacency_matrix(
    shape: tuple[int, int], periodic: bool
) -> npt.NDArray[bool]:
    """Creates an adjacency matrix for a 2D square lattice Hubbard Hamiltonian.

    Args:
        shape (tuple[int, int]): The number of sites.
        periodic (bool): If true, periodic boundary conditions are used.

    Returns:
        np.ndarray[bool]: Adjacency matrix for lattice sites.
    """
    # find the side length to fit nodes into square
    # we'll build a perfect square first before cutting.
    nx, ny = shape
    n_sites = nx * ny

    # initially make a chain
    adjacency_matrix = np.eye(n_sites, k=1)

    # cut chain into rows by removing connections
    for i in range(nx, n_sites, nx):
        adjacency_matrix[i - 1, i] = 0.0

    # Add connection to number below.
    adjacency_matrix += np.eye(n_sites, k=nx)

    if periodic:
        # Wrap rows
        for i in range(ny):
            adjacency_matrix[i * nx, (i + 1) * nx - 1] = 1

        # Wrap columns
        adjacency_matrix += np.eye(n_sites, k=nx * (ny - 1))

    # Hamitian conjugate
    adjacency_matrix += adjacency_matrix.T
    return np.array(adjacency_matrix, dtype=bool)


def cube_lattice_adjacency_matrix(
    shape: tuple[int, int, int], periodic: bool
) -> npt.NDArray[bool]:
    """Creates an adjacency matrix for a 3D square lattice Hubbard Hamiltonian.

    Args:
        shape (tuple[int, int, int]): The number of sites.
        periodic (bool): If true, periodic boundary conditions are used.

    Returns:
        np.ndarray[bool]: Adjacency matrix for lattice sites.
    """
    nx, ny, nz = shape
    n_sites = nx * ny * nz

    adjacency_matrix = np.zeros((n_sites, n_sites))
    # Add each of the layers of a square matrix
    for i in range(0, n_sites, nx * ny):
        adjacency_matrix[i : i + nx * ny, i : i + nx * ny] = np.triu(
            square_lattice_adjacency_matrix((nx, ny), periodic=periodic)
        )

    # Add connection in D3
    adjacency_matrix += np.eye(n_sites, k=nx * ny)

    # Wrap D3
    if periodic:
        adjacency_matrix += np.eye(n_sites, k=nx * ny * (nz - 1))

    adjacency_matrix += adjacency_matrix.T

    return np.array(adjacency_matrix, dtype=bool)


def hubbard_coefficients(
    n_modes: int,
    adjacency_matrix: npt.NDArray,
    onsite_term: float,
    hopping_term: float = 1.0,
    spinless: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients to fill a Hubbard Hamiltonian Template.

    Args:
        n_modes (int): Number of fermion modes in the system.
        adjacency_matrix (npt.NDArray): Adjacency matrix of lattice sites.
        onsite_term (float): Onsite interaction term.
        hopping_term (float): Kinetic term.
        spinless (bool): Set to True to use single spin Hamiltonian.

    Returns:
        tuple: one and two electron coefficients.
    """
    if not spinless:
        # We know which sites are adjacent, we need to restrict to same spin hopping.
        spin_adjacency_matrix = np.zeros(
            (2 * adjacency_matrix.shape[0], 2 * adjacency_matrix.shape[1])
        )
        spin_adjacency_matrix[::2, ::2] += adjacency_matrix
        spin_adjacency_matrix[1::2, 1::2] += adjacency_matrix
    else:
        spin_adjacency_matrix = adjacency_matrix

    one_e_coeffs = hopping_term * spin_adjacency_matrix
    one_e_coeffs = one_e_coeffs[:n_modes, :n_modes]

    two_e_coeffs = np.zeros((n_modes, n_modes, n_modes, n_modes))
    idx = np.arange(n_modes)
    two_e_coeffs[idx, idx, idx, idx] = onsite_term
    return one_e_coeffs, two_e_coeffs


def hubbard_hamiltonian(
    adjacency_matrix: npt.NDArray,
    onsite_term: float,
    hopping_term: float = 1.0,
    spinless: bool = False,
) -> FermionHamiltonian:
    """Return a Hubbard model Hamiltonian.

    As the Hubbard Hamiltonian has the same signature as the Chemists' Molecular Hamiltonian
    (+-, +-+-), the molecular Hamiltonian functions are reused internally.

    Args:
        adjacency_matrix (npt.NDArray): Adjacency matrix of lattice sites.
        onsite_term (float): Onsite two-electron term.
        hopping_term (float): Kinetic term coefficient.
        physicist_notation (bool): Set to False for Chemist Notation.
        spinless (bool): Set to True to use single spin Hamiltonian.

    Returns:
        dict[str, float]: A qubit Hamiltonian.

    Example:
        >>> import numpy as np
        >>> from ferrmion.hamiltonians import hubbard_hamiltonian, linear_adjacency_matrix
        >>> adjacency = linear_adjacency_matrix(4, periodic=False)
        >>> fham = hubbard_hamiltonian(adjacency, onsite_term=2.0)
        >>> fham.n_modes
        8
    """
    n_sites = adjacency_matrix.shape[0]

    one_e_coeffs, two_e_coeffs = hubbard_coefficients(
        n_sites, adjacency_matrix, onsite_term, hopping_term, spinless=spinless
    )
    return FermionHamiltonian(
        terms={"+-": one_e_coeffs, "+-+-": two_e_coeffs}, constant_energy=0.0
    )
