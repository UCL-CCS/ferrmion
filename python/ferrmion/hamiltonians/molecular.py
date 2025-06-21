"""Molecular Hamiltonian."""

import logging

import numpy as np
from numpy.typing import NDArray

from ferrmion import FermionQubitEncoding
from ferrmion.core import symplectic_product, symplectic_product_map, molecular_hamiltonian_template
from ferrmion.utils import icount_to_sign, symplectic_hash, symplectic_unhash, symplectic_to_pauli

from .utils import fill_template, to_qubit_hamiltonian

logger = logging.getLogger(__name__)


def molecular_hamiltonian(
    encoding: FermionQubitEncoding,
    one_e_coeffs: NDArray,
    two_e_coeffs: NDArray,
    constant_energy: float,
):
    """Return an encoded electronic stucture hamiltonain with niave enumeration.

    Args:
        encoding (FermionQubitEncoding): The encoding to use.
        one_e_coeffs (NDArray): One electron hamiltonian coefficients in spinorb format.
        two_e_coeffs (NDArray): Two electron hamiltonian coefficients in spinorb format.
        constant_energy (float): Constant energy offset.
    """
    ipowers, majorana_symplectic = encoding._build_symplectic_matrix()
    template = molecular_hamiltonian_template(ipowers, majorana_symplectic)
    qubit_hamiltonian = fill_template(
        one_e_coeffs,
        two_e_coeffs,
        template,
        mode_op_map=encoding.default_mode_op_map,
        constant_energy=constant_energy,
    )
    return qubit_hamiltonian
