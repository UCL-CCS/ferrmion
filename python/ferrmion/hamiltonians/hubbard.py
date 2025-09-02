"""Hubbard Hamiltonian."""

import numpy as np
import numpy.typing as npt

from ferrmion.core import fill_template, molecular_hamiltonian_template

from ..encode import FermionQubitEncoding


def hubbard_hamiltonian_template(
    ipowers: npt.NDArray[np.uint8], majorana_symplectic: npt.NDArray[np.bool_]
) -> dict:
    """Return a Hamiltonian Template for the Hubbard Hamiltonian.

    Args:
        ipowers (np.ndarray): Imaginary Coefficients.
        majorana_symplectic (np.ndarray): Symplectic Matrix form of encoding.

    Returns:
        dict: A template hamiltonian.
    """
    return molecular_hamiltonian_template(ipowers, majorana_symplectic, False)


def hubbard_coefficients(
    n_modes: int,
    adjacency_matrix: npt.NDArray,
    onsite_term: float,
    hopping_term: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients to fill a Hubbard Hamiltonian Template.

    Args:
        n_modes (int): Number of fermion modes in the system.
        adjacency_matrix (npt.NDArray): Adjacency matrix of lattice sites.
        onsite_term (float): Onsite interaction term.
        hopping_term (float): Kinetic term.

    Returns:
        tuple: one and two electron coefficients.
    """
    one_e_coeffs = hopping_term * adjacency_matrix

    two_e_coeffs = np.zeros((n_modes, n_modes, n_modes, n_modes))
    idx = np.arange(n_modes)
    two_e_coeffs[idx, idx, idx, idx] = onsite_term
    return one_e_coeffs, two_e_coeffs


def hubbard_hamiltonian(
    encoding: FermionQubitEncoding,
    adjacency_matrix: npt.NDArray,
    onsite_term: float,
    hopping_term: float = 1.0,
) -> dict[str, float]:
    """Return an encoded Hubbard hamiltonain with niave enumeration.

    As the Hubbard Hamiltonian has the same signature as the Chemists' Molecular Hamiltonian:
    (+-, +-+-)
    We can use the existing functions for the molecular Hamiltonian to create a template.

    Args:
        encoding (FermionQubitEncoding): The encoding to use.
        adjacency_matrix (npt.NDArray): Adjacency matrix of lattice sites.
        onsite_term (float): Onsite two-electron term.
        hopping_term (float): Kinetic term coefficient.
        physicist_noation (bool): Set to False for Chemist Notation.

    Returns:
        dict[str, float]: A qubit Hamiltonian.

    Example:
        >>> import numpy as np
        >>> from ferrmion.hamiltonians.molecular import molecular_hamiltonian
        >>> from ferrmion.encode import TernaryTree
        >>> tree = TernaryTree(12).JW()
        >>> one_e = np.eye((2,2))
        >>> two_e = np.eye((2,2,2,2))
        >>> molecular_hamiltonian(tree, one_e, two_e, 0.0)
    """
    ipowers, symplectic = encoding._build_symplectic_matrix()
    template = hubbard_hamiltonian_template(ipowers, symplectic)

    n_modes = encoding.n_modes
    one_e_coeffs, two_e_coeffs = hubbard_coefficients(
        n_modes,
        adjacency_matrix,
        onsite_term,
        hopping_term,
    )

    qubit_hamiltonian = fill_template(
        template=template,
        constant_energy=0,
        one_e_terms=one_e_coeffs,
        two_e_terms=two_e_coeffs,
        mode_op_map=encoding.default_mode_op_map,
    )
    return qubit_hamiltonian


def linear_hubbard_hamiltonian(
    encoding: FermionQubitEncoding,
    onsite_term: float,
    hopping_term: float = 1,
    periodic: bool = False,
) -> dict[str, float]:
    """Hubbard Hamiltonian for a chain.

    Args:
        encoding (FermionQubitEncoding): The encoding to use.
        hopping_term (float): Kinetic term coefficient.
        onsite_term (float): Onsite two-electron term.
        periodic (bool): Whether to use a periodic lattice.

    Returns:
        dict[str, float]: A qubit Hamiltonian.
    """
    adjacency_matrix = np.eye(encoding.n_modes, k=1) + np.eye(encoding.n_modes, k=-1)

    if periodic:
        adjacency_matrix[0, encoding.n_modes] = 1.0
        adjacency_matrix[encoding.n_modes, 0] = 1.0

    return hubbard_hamiltonian(encoding, adjacency_matrix, onsite_term, hopping_term)


def square_hubbard_hamiltonian(
    encoding: FermionQubitEncoding,
    onsite_term: float,
    hopping_term: float = 1,
    periodic: bool = False,
) -> dict[str, float]:
    """Hubbard Hamiltonian for a square lattice.

    Note that if he number of modes is not a square number,
    modes wil be missing from the lower right portion of the lattice.

    Args:
        encoding (FermionQubitEncoding): The encoding to use.
        hopping_term (float): Kinetic term coefficient.
        onsite_term (float): Onsite two-electron term.
        periodic (bool): Whether to use a periodic lattice.

    Returns:
        dict[str, float]: A qubit Hamiltonian.
    """
    n_modes = encoding.n_modes
    # find the side length to fit nodes into square
    # we'll build a perfect square first before cutting.
    side_length = int(np.ceil(np.log2(n_modes)))

    # initially make a chain
    adjacency_matrix = np.eye(side_length**2, k=1)
    adjacency_matrix
    # cut chain into rows by removing connections
    for i in range(1, side_length):
        adjacency_matrix[i * side_length - 1, i * side_length] = 0

    # Add connection to number below.
    adjacency_matrix += np.eye(side_length**2, k=side_length)

    if periodic:
        # Wrap rows
        for i in range(side_length):
            adjacency_matrix[i * side_length, i * side_length + side_length - 1] = 1.0

        # Wrap columns
        adjacency_matrix += np.eye(side_length**2, k=side_length * (side_length - 1))

    # Remove excess nodes
    adjacency_matrix = adjacency_matrix[:n_modes, :n_modes]

    # Hermitian conjugate
    adjacency_matrix += adjacency_matrix.T

    return hubbard_hamiltonian(encoding, adjacency_matrix, onsite_term, hopping_term)
