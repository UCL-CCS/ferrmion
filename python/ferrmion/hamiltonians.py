"""Factory functions to easily build general Fermion and Qubit Hamiltonians.

The :class:`QubitHamiltonian` and :class:`FermionHamiltonian` types are
implemented in Rust and re-exported here from :mod:`ferrmion.core`.
"""

import logging

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from ferrmion.core import FermionHamiltonian, QubitHamiltonian

logger = logging.getLogger(__name__)

__all__ = [
    "FermionHamiltonian",
    "QubitHamiltonian",
    "molecular_hamiltonian",
    "hubbard_hamiltonian",
    "hubbard_coefficients",
    "linear_adjacency_matrix",
    "square_lattice_adjacency_matrix",
    "cube_lattice_adjacency_matrix",
]


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
    one_e_coeffs = np.asarray(one_e_coeffs, dtype=np.float64)
    two_e_coeffs = np.asarray(two_e_coeffs, dtype=np.float64)
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
        spinless (bool): Set to True to use single spin Hamiltonian.

    Returns:
        FermionHamiltonian: The Hubbard model Hamiltonian.

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
        terms={
            "+-": np.asarray(one_e_coeffs, dtype=np.float64),
            "+-+-": np.asarray(two_e_coeffs, dtype=np.float64),
        },
        constant_energy=0.0,
    )
