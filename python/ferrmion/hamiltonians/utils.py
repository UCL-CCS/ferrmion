"""Utilities for Hamiltonian Functions."""

import logging

import numpy as np
from numpy.typing import NDArray

from ferrmion.core import symplectic_product
from ferrmion.utils import (
    icount_to_sign,
    symplectic_hash,
    symplectic_to_pauli,
    symplectic_unhash,
)

logger = logging.getLogger(__name__)


def fill_template(
    one_e_coeffs,
    two_e_coeffs,
    template: dict,
    mode_op_map: dict,
    constant_energy=0,
    precision: float = 1e-12,
) -> dict:
    """Fill a template with Hamiltonian coefficients.

    Args:
        one_e_coeffs (NDArray): One electron hamiltonian coefficients
        two_e_coeffs (NDArray): Two electron hamiltonian coefficients
        template (dict): A template Hamilonian
        mode_op_map (dict): A dictionary mapping the mode indices to their corresponding majorana operator indices.
        constant_energy (float): A constant term to be added to the Hamiltonian.
        precision (float): Cutoff for inclusion of terms, defaults to 1e-12

    Returns:
        dict: A filled template
    """
    logger.debug(f"Filling template with map\n{mode_op_map}")

    total_ham = {t: 0 for t in template}
    total_ham["I" * len(mode_op_map)] += (
        constant_energy
    )

    for term, component in template.items():
        for item, factor in component.items():
            match len(item):
                case 2:
                    total_ham[term] += (
                        factor * one_e_coeffs[*[mode_op_map[i] for i in item]]
                    )
                case 4:
                    total_ham[term] += (
                        factor * two_e_coeffs[*[mode_op_map[i] for i in item]]
                    )

        # print(total_ham[term])
        if np.abs(total_ham[term]) < precision:
            total_ham.pop(term)
    return total_ham


def to_symplectic_hamiltonian(
    n_qubits: int,
    hashed_hamiltonian,
) -> tuple[list[complex], NDArray]:
    """Output the hamiltonian in symplectic form.

    Remember, in symplectic form representation of XZ is literal.
    Convcerting to a Y will require an additional term.

    Args:
        n_qubits (int): The numbe of qubits of operators (needed for unhashing) #TODO this should be handled by code somehow.
        hashed_hamiltonian (dict): A hashed hamiltonian with modes assigned. (i.e. a filled template)

    Returns:
        tuple[list[complex], NDArray]: A tuple of coefficients and symplectic terms.
    """
    logger.debug("Creating symplectic Hamiltonian")

    coeffs: list[complex] = []
    terms = []
    for term, coeff in hashed_hamiltonian.items():
        term = symplectic_unhash(term, 2 * n_qubits)
        half_length = len(term) // 2
        y_count = np.sum(np.bitwise_and(term[half_length:], term[:half_length]))
        coeff = icount_to_sign(y_count * 3) * coeff
        if y_count % 2 == 1:
            coeff = (coeff + np.conj(coeff)) / 2

        if coeff != 0:
            coeffs.append(coeff)
            terms.append(term)

    return coeffs, np.vstack(tuple(terms))


def to_qubit_hamiltonian(n_qubits: int, hashed_hamiltonian) -> dict[str, float]:
    """Create qubit representation Hamiltonian.

    Args:
        n_qubits (int): The numbe of qubits of operators (needed for unhashing) #TODO this should be handled by code somehow.
        hashed_hamiltonian (dict): A hashed hamiltonian with modes assigned. (i.e. a filled template)

    Returns:
        dict[str, float]: A dictionary of Pauli strings and their coefficients.
    """
    logger.debug("Creating qubit Hamiltonian")

    pauli_hamiltonian: dict[str, float] = {}
    for term, coefficient in hashed_hamiltonian.items():
        if np.real(coefficient) == 0:
            continue

        unhashed_symplectic = symplectic_unhash(term, 2 * n_qubits)
        ipower, pauli_term = symplectic_to_pauli(unhashed_symplectic)
        coefficient = icount_to_sign(ipower) * coefficient
        coefficient = (coefficient + np.conj(coefficient)) / 2

        if coefficient == 0:
            continue

        if pauli_hamiltonian.get(pauli_term, None) is not None:
            pauli_hamiltonian[pauli_term] += coefficient
        else:
            pauli_hamiltonian[pauli_term] = coefficient
    return pauli_hamiltonian
