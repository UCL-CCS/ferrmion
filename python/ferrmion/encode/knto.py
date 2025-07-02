"""Build the weird k-NTO encodings."""

import logging

import numpy as np
from numpy.typing import NDArray

from .base import FermionQubitEncoding

logger = logging.getLogger(__name__)


class KNTO(FermionQubitEncoding):
    """k-NTO encoding for fermionic operators.

    Attributes:
        n_modes (int): The number of modes.

    Methods:
        _build_symplectic_matrix(): Build the symplectic matrix for the k-NTO encoding.
        _valid_qubit_number(): Check if the number of qubits is valid for the k-NTO encoding.
    """

    def __init__(self, n_modes):
        """Initialise a k-NTO encoding.

        Args:
            n_modes (int): The number of fermionic modes
        """
        self.n_modes = n_modes
        super().__init__(n_modes=n_modes, n_qubits=n_modes)

    def _build_symplectic_matrix(self) -> tuple[NDArray[np.number], NDArray[np.bool_]]:
        """Build the symplectic matrix for the k-NTO encoding.

        Returns:
            NDArray: The symplectic matrix.

        Example:
            >>> from ferrmion.encode.knto import KNTO
            >>> knto = KNTO(5)
            >>> y_count, sympl = knto._build_symplectic_matrix()
        """
        return knto_symplectic_matrix(self.n_modes)

    def _valid_qubit_number(self) -> int:
        """Check if the number of qubits is valid for the k-NTO encoding.

        Returns:
            int: The number of qubits.
        """
        return self.n_modes


def knto_symplectic_matrix(n_modes) -> tuple[NDArray[np.number], NDArray[np.bool_]]:
    """Build a symplectic matrix of majorana operators for the k-NTO encoding.

    Args:
        n_modes (int): The number of modes.

    Returns:
        tuple[NDArray, NDArray]: The y_count of each vector and the symplectic matrix.

    Example:
        >>> from ferrmion.encode.knto import knto_symplectic_matrix
        >>> y_count, sympl = knto_symplectic_matrix(5)
        >>> sympl.shape
    """
    logger.debug(f"Building k-NTO symplectic matrix for {n_modes=}")
    k = n_modes - 1
    if k % 2 != 1:
        raise ValueError("Only works for Odd k")

    # Choice of right and left is arbitary but at least for TNs
    # having the simple block on the left was better.
    right = np.ones(((k + 1) * 2, k + 1), dtype=bool)

    right[::2, :] = right[::2, :] - np.eye(k + 1)
    right[1::2, :] = right[1::2, :] - np.eye(k + 1)

    left = np.zeros(((k + 1) * 2, k + 1), dtype=bool)

    for i in range(k + 1):
        if i % 2 == 1:
            left[2 * i, i] = 1
            left[2 * i + 1, i] = 1

    for i in range(1, (k + 1) * 2):
        if i % 2 == 1:
            left[i, np.ma.where(right[i] == 1)[0]] = np.logical_not(
                left[i - 1, np.ma.where(right[i] == 1)[0]]
            )
        else:
            left[i] = left[i - 1]

    # Y = iXZ
    y_count = np.sum(np.bitwise_and(left, right), axis=1, dtype=np.uint8) % 4
    output = np.hstack((left, right), dtype=bool)

    return y_count, output
