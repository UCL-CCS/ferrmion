"""Hubbard Hamiltonian."""

import numpy as np

from ferrmion.core import fill_template, molecular_hamiltonian_template

from ..encode import FermionQubitEncoding


def hubbard_hamiltonian(
    encoding: FermionQubitEncoding,
    hopping_term: float,
    onsite_term: float,
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
    ipowers, majorana_symplectic = encoding._build_symplectic_matrix()
    template = molecular_hamiltonian_template(ipowers, majorana_symplectic, False)

    n_modes = majorana_symplectic.shape[0] // 2
    one_e_coeffs = np.eye(n_modes, k=1) + np.eye(n_modes, k=-1)
    one_e_coeffs *= hopping_term

    two_e_coeffs = np.zeros((n_modes, n_modes, n_modes, n_modes))
    idx = np.arange(n_modes)
    two_e_coeffs[idx, idx, idx, idx] = onsite_term

    qubit_hamiltonian = fill_template(
        template=template,
        constant_energy=constant_energy,
        one_e_terms=one_e_coeffs,
        two_e_terms=two_e_coeffs,
        mode_op_map=encoding.default_mode_op_map,
    )
    return qubit_hamiltonian
