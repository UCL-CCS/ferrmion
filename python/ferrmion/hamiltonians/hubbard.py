"""Hubbard Hamiltonian."""

import numpy as np

from ferrmion.core import fill_template, molecular_hamiltonian_template

from ..encode import FermionQubitEncoding


def hubbard_hamiltonian_template(encoding: FermionQubitEncoding) -> dict:
    """Return a Hamiltonian Template for the Hubbard Hamiltonian.

    Args:
        encoding (FermionQubitEncoding): A valid encoding.

    Returns:
        dict: A template hamiltonian.
    """
    ipowers, majorana_symplectic = encoding._build_symplectic_matrix()
    return molecular_hamiltonian_template(ipowers, majorana_symplectic, False)


def hubbard_coefficients(
    n_modes: int, onsite_term: float, hopping_term: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients to fill a Hubbard Hamiltonian Template.

    Args:
        n_modes (int): Number of fermion modes in the system.
        onsite_term (float): Onsite interaction term.
        hopping_term (float): Kinetic term.

    Returns:
        tuple: one and two electron coefficients.
    """
    one_e_coeffs = np.eye(n_modes, k=1) + np.eye(n_modes, k=-1)
    one_e_coeffs *= hopping_term

    two_e_coeffs = np.zeros((n_modes, n_modes, n_modes, n_modes))
    idx = np.arange(n_modes)
    two_e_coeffs[idx, idx, idx, idx] = onsite_term
    return one_e_coeffs, two_e_coeffs


def hubbard_hamiltonian(
    encoding: FermionQubitEncoding,
    onsite_term: float,
    hopping_term: float = 1,
    constant_energy: float = 0,
):
    """Return an encoded Hubbard hamiltonain with niave enumeration.

    As the Hubbard Hamiltonian has the same signature as the Chemists' Molecular Hamiltonian:
    (+-, +-+-)
    We can use the existing functions for the molecular Hamiltonian to create a template.

    Args:
        encoding (FermionQubitEncoding): The encoding to use.
        hopping_term (float): Kinetic term coefficient.
        onsite_term (float): Onsite two-electron term.
        constant_energy (float): Constant energy offset.
        physicist_noation (bool): Set to False for Chemist Notation.

    Example:
        >>> import numpy as np
        >>> from ferrmion.hamiltonians.molecular import molecular_hamiltonian
        >>> from ferrmion.encode import TernaryTree
        >>> tree = TernaryTree(12).JW()
        >>> one_e = np.eye((2,2))
        >>> two_e = np.eye((2,2,2,2))
        >>> molecular_hamiltonian(tree, one_e, two_e, 0.0)
    """
    template = hubbard_hamiltonian_template(encoding)

    n_modes = encoding.n_modes
    one_e_coeffs, two_e_coeffs = hubbard_coefficients(
        n_modes, onsite_term, hopping_term
    )

    qubit_hamiltonian = fill_template(
        template=template,
        constant_energy=constant_energy,
        one_e_terms=one_e_coeffs,
        two_e_terms=two_e_coeffs,
        mode_op_map=encoding.default_mode_op_map,
    )
    return qubit_hamiltonian
