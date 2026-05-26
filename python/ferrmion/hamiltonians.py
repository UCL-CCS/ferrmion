"""Class and methods to easily build general Fermion Hamiltonians."""

import logging

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class QubitHamiltonian(dict[str, complex]):
    """Mapping from Pauli strings to complex coefficients.

    Subclasses ``dict`` so all standard mapping operations (``q[key]``,
    ``q.items()``, ``len(q)``, ``q.get(...)``) work as expected, and adds
    Clifford-based optimisation methods that return a new
    ``QubitHamiltonian``.
    """

    @property
    def n_qubits(self) -> int:
        """Number of qubits, inferred from the length of any Pauli key."""
        if not self:
            raise ValueError("QubitHamiltonian is empty; cannot infer n_qubits")
        return len(next(iter(self)))

    def clifford_heuristic(
        self,
        temperature: float | None = None,
        coefficient_weighted: bool = False,
        seed: int | None = None,
    ) -> "QubitHamiltonian":
        """Optimise this Hamiltonian via Clifford-heuristic simulated annealing.

        Args:
            temperature: Initial annealing temperature. Defaults to ``2 * n_qubits``.
            coefficient_weighted: If ``True``, minimise coefficient-weighted Pauli weight.
            seed: Seed for the RNG. Defaults to ``1017`` when omitted.

        Returns:
            QubitHamiltonian: The optimised Hamiltonian.
        """
        from ferrmion.core import clifford_heuristic

        n = self.n_qubits
        if temperature is None:
            temperature = float(n)
        return QubitHamiltonian(
            clifford_heuristic(
                qham=dict(self),
                n_qubits=n,
                temperature=temperature,
                coefficient_weighted=coefficient_weighted,
                seed=seed,
            )
        )

    def randomised_subsystem_descent(
        self,
        iterations: int,
        subsystem_dimension: int,
        temperature: float | None = None,
        coefficient_weighted: bool = False,
        sampler: str = "hamming",
        seed: int | None = None,
    ) -> "QubitHamiltonian":
        """Iteratively optimise this Hamiltonian via Clifford-heuristic simulated annealing.

        Args:
            iterations: Number of subsystem-local Clifford descents to perform.
            subsystem_dimension: Number of qubits in each sampled subsystem.
            temperature: Annealing temperature for each descent. Defaults to ``2 * n_qubits``.
            coefficient_weighted: If ``True``, minimise coefficient-weighted Pauli weight.
            sampler: Subsystem sampling strategy: ``"full_system"``, ``"uniform"``, or ``"hamming"``.
            seed: Seed for the RNG. Defaults to ``1017`` when omitted.

        Returns:
            QubitHamiltonian: The optimised Hamiltonian.
        """
        from ferrmion.core import randomised_subsystem_descent

        n = self.n_qubits
        if temperature is None:
            temperature = float(n)
        return QubitHamiltonian(
            randomised_subsystem_descent(
                qham=dict(self),
                n_qubits=n,
                iterations=iterations,
                temperature=temperature,
                subsystem_dimension=subsystem_dimension,
                coefficient_weighted=coefficient_weighted,
                sampler=sampler,
                seed=seed,
            )
        )


class FermionHamiltonian:
    """Class for building Fermionic Hamiltonians."""

    def __init__(self, *, terms: dict[str, NDArray] = {}, constant_energy: float = 0.0):
        """Initialiser for FermionHamiltonian."""
        logger.debug("Initialising FermionHamiltonian")
        self._terms: dict[str, NDArray] = terms
        self.constant_energy = constant_energy
        self._next_term = ""
        self.n_modes = 0
        self._check_and_set_n_modes()

    def _check_and_set_n_modes(self):
        if len(self._terms) == 0:
            pass
        elif len(self._terms) > 0 and self.n_modes == 0:
            logger.debug(f"Setting n_modes: {[*self._terms.values()][0]}")
            self.n_modes = [*self._terms.values()][0].shape[0]
        else:
            for term in self._terms.values():
                if np.any([side != self.n_modes for side in term.shape]):
                    logger.error(
                        f"Hamiltonian coefficient {term.shape} must have constant length {self.n_modes} on all dimensions.\n"
                    )
                    raise ValueError(
                        f"Hamiltonian coefficient {term.shape} must have constant length {self.n_modes} on all dimensions."
                    )

    def __repr__(self):
        """String representation of FermionHamiltonian."""
        terms = ", ".join([*self._terms])
        return f"FermionHamiltonian({terms}, {self.n_modes} modes, constant {self.constant_energy})"

    @property
    def signatures_and_coefficients(self) -> tuple[list[str], list[NDArray]]:
        """Return the signature and coefficient of all terms."""
        sigs: list[str] = []
        coeffs: list[NDArray] = []
        for k, v in self._terms.items():
            sigs.append(k)
            coeffs.append(v)

        return (sigs, coeffs)

    def creation(self) -> "FermionHamiltonian":
        """Add a creation term."""
        self._next_term += "+"
        return self

    def annihilation(self) -> "FermionHamiltonian":
        """Add an annihilation term."""
        self._next_term += "-"
        return self

    def with_coefficients(self, coefficients: NDArray) -> "FermionHamiltonian":
        """Add coefficients to the Hamiltonian terms."""
        if coefficients.ndim != len(self._next_term):
            logger.error(f"Cannot apply coefficents to term {self._next_term}")
        else:
            self._terms[self._next_term] = coefficients
            self._next_term = ""
            self._check_and_set_n_modes()
        return self

    def add_constant(self, constant_energy: float) -> "FermionHamiltonian":
        """Add a constant term to the Hamiltonian."""
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
