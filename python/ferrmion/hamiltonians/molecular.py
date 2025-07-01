"""Molecular Hamiltonian."""

import logging

from numpy.typing import NDArray

from ferrmion import FermionQubitEncoding
from ferrmion.core import fill_template, molecular_hamiltonian_template

logger = logging.getLogger(__name__)


def molecular_hamiltonian(
    encoding: FermionQubitEncoding,
    one_e_coeffs: NDArray,
    two_e_coeffs: NDArray,
    constant_energy: float = 0,
):
    """Return an encoded electronic stucture hamiltonain with niave enumeration.

    Args:
        encoding (FermionQubitEncoding): The encoding to use.
        one_e_coeffs (NDArray): One electron hamiltonian coefficients in spinorb format.
        two_e_coeffs (NDArray): Two electron hamiltonian coefficients in spinorb format.
        constant_energy (float): Constant energy offset.

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
    template = molecular_hamiltonian_template(ipowers, majorana_symplectic)
    qubit_hamiltonian = fill_template(
        template=template,
        constant_energy=constant_energy,
        one_e_terms=one_e_coeffs,
        two_e_terms=two_e_coeffs,
        mode_op_map=encoding.default_mode_op_map,
    )
    return qubit_hamiltonian
