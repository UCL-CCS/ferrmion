"""Functions which have been replaced by rust implementations.

These should be retained for testing backwards compatability.
"""

import logging

import numpy as np

from ferrmion.base import FermionQubitEncoding

logger = logging.getLogger(__name__)


def slow_symplectic_product(left, right) -> tuple[int, np.ndarray[np.uint8]]:
    """Calculate the product of two symplectic vectors.

    Args:
        left (np.ndarray): The first symplectic vector.
        right (np.ndarray): The second symplectic vector.
    """
    term = np.bitwise_xor(left, right)
    # print(symplectics[i], symplectics[j], term)

    n_qubits = len(left) // 2

    # any time S1 @ S2 involves Z on the left of an X we gain a -1
    zx_count = np.sum(np.bitwise_and(left[n_qubits:], right[:n_qubits]))

    # XZ products are stored as XZ and not as Y

    ipower = (2 * zx_count) % 4
    return ipower, term


def slow_hartree_fock_state(
    encoding: FermionQubitEncoding,
    fermionic_hf_state: np.ndarray,
    mode_op_map: dict[int, int],
) -> np.ndarray:
    """Find the Hartree-Fock state of a majorana string encoding.

    NOTE: This is the python-only version, the main codebase uses a rust rewrite.
        This function is retained for testing backwards compatability.

    Args:
        encoding (FermionQubitEncoding): a Majoprana string encoding.
        fermionic_hf_state (np.ndarray[int]): An array of mode occupations.
        mode_op_map (dict[int, int]): A dictionary mapping modes to sets of majorana strings i->(j,j+1).

    Returns:
        np.ndarray: The Hartree-Fock ground state in computational basis.
    """
    mode_op_map = encoding.default_mode_op_map if mode_op_map is None else mode_op_map
    if len(fermionic_hf_state) != len(mode_op_map):
        error_string = "Fermionic HF state must be same length as Mode-Operator Map"
        logger.error(error_string)
        raise ValueError(error_string)

    matrices = {
        "X": np.array([[0, 1], [1, 0]]),
        "XZ": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
        "": np.array([[1, 0], [0, 1]]),
    }

    logger.debug("Creating computational basis HF state")

    modes = [mode_op_map[mode] for mode in np.where(fermionic_hf_state)[0]]
    symplectic_operators = encoding._build_symplectic_matrix()[1]
    symplectic_operators = np.vstack(
        symplectic_operators[[(2 * mode, 2 * mode + 1) for mode in modes]]
    )

    half_length = symplectic_operators.shape[1] // 2
    vaccum_state = [
        np.array([1, 0]) if site == 0 else np.array([0, 1])
        for site in encoding.vaccum_state
    ]
    for index in range(0, symplectic_operators.shape[0], 2):
        xlist = [
            "X" if line == 1 else ""
            for line in symplectic_operators[index, :half_length]
        ]
        zlist = [
            "Z" if line == 1 else ""
            for line in symplectic_operators[index, half_length:]
        ]
        left_operators = [matrices[f"{x}{z}"] for x, z in zip(xlist, zlist)]
        # ipower = (3 * y_count) % 4
        xlist = [
            "X" if line == 1 else ""
            for line in symplectic_operators[index + 1, :half_length]
        ]
        zlist = [
            "Z" if line == 1 else ""
            for line in symplectic_operators[index + 1, half_length:]
        ]
        right_operators = [1j * matrices[f"{x}{z}"] for x, z in zip(xlist, zlist)]

        total_ops = [
            left - right for left, right in zip(left_operators, right_operators)
        ]

        vaccum_state = [op @ state for op, state in zip(total_ops, vaccum_state)]
    # vaccum_state = [(v*np.conj(v))/np.linalg.norm(v*np.conj(v)) for v in vaccum_state]
    total_state = vaccum_state[0]
    for state in vaccum_state[1:]:
        total_state = np.kron(total_state, state)

    coeffs = total_state / np.linalg.norm(total_state * np.conj(total_state))
    hf_components = np.vstack(
        [
            np.array(list(np.binary_repr(val, width=len(vaccum_state))), dtype=np.uint8)
            for val in np.where(coeffs)[0]
        ]
    )
    coeffs = [c for c in coeffs if c != 0]
    coeffs /= coeffs[0]

    return coeffs, hf_components
