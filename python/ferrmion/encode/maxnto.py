"""Build the weird k-NTO encodings."""

import logging

import numpy as np
from numpy.typing import NDArray

from ferrmion.utils import xy_swap

from .base import FermionQubitEncoding

logger = logging.getLogger(__name__)


class MaxNTO(FermionQubitEncoding):
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

    def _build_symplectic_matrix(self) -> tuple[NDArray[np.number], NDArray[bool]]:
        """Build the symplectic matrix for the k-NTO encoding.

        Returns:
            NDArray: The symplectic matrix.

        Example:
            >>> from ferrmion.encode.maxnto import MaxNTO
            >>> MaxNTO = MaxNTO(5)
            >>> y_count, sympl = MaxNTO._build_symplectic_matrix()
        """
        return maxnto_symplectic_matrix(self.n_modes)

    def _valid_qubit_number(self) -> int:
        """Check if the number of qubits is valid for the k-NTO encoding.

        Returns:
            int: The number of qubits.
        """
        return self.n_modes


def maxnto_symplectic_matrix(n_modes) -> tuple[NDArray[np.number], NDArray[bool]]:
    """Build a symplectic matrix of majorana operators for the k-NTO encoding.

    Args:
        n_modes (int): The number of modes.

    Returns:
        tuple[NDArray, NDArray]: The y_count of each vector and the symplectic matrix.

    Example:
        >>> from ferrmion.encode.maxnto import maxnto_symplectic_matrix
        >>> y_count, sympl = maxnto_symplectic_matrix(5)
        >>> sympl.shape
    """
    logger.debug(f"Building k-NTO symplectic matrix for {n_modes=}")
    k = n_modes - 1
    if k % 2 != 1:
        raise ValueError("Only works for Odd k")

    # Choice of x_block and z_block is arbitary but at least for TNs
    # having the simple block on the z_block was better.
    x_block = np.zeros((n_modes, n_modes))
    x_block += np.triu(np.ones(n_modes), k=1)
    x_block += np.tril(np.ones(n_modes), k=-1)

    z_block = np.tril(np.ones(n_modes), k=-1)
    for i in range(0, z_block.shape[0], 2):
        z_block[i, i] = True

    # Y = iXZ
    x_block = np.array(x_block, dtype=bool)
    z_block = np.array(z_block, dtype=bool)
    z_block[1::2] = z_block.T[1::2, :]
    odd_majoranas = np.empty((n_modes, n_modes), dtype=np.uint8)
    even_majoranas = np.empty(x_block.shape, dtype=np.uint8)
    odd_majoranas = np.hstack((x_block, z_block), dtype=np.uint8)
    even_majoranas = np.hstack((x_block, z_block), dtype=np.uint8)
    even_majoranas = xy_swap(even_majoranas)

    output = np.empty((2 * n_modes, 2 * n_modes), dtype=bool)
    output[::2, :] = odd_majoranas
    output[1::2, :] = even_majoranas
    output = np.array(output, dtype=bool)
    y_count = np.zeros(output.shape[0], dtype=np.uint8)
    y_count = (
        np.sum(
            np.bitwise_and(output[:, :n_modes], output[:, n_modes:]),
            axis=1,
            dtype=np.uint8,
        )
        % 4
    )
    return y_count, output
