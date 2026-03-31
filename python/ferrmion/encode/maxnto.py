"""Build the weird k-NTO encodings."""

import logging

import numpy as np
from numpy.typing import NDArray

from ferrmion.core import maxnto_symplectic_matrix

from .base import FermionQubitEncoding

logger = logging.getLogger(__name__)


class MaxNTO(FermionQubitEncoding):
    """k-NTO encoding for fermionic operators.

    Attributes:
        n_modes (int): The number of modes.

    Methods:
        _build_symplectic_matrix(): Build the symplectic matrix for the k-NTO encoding.
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
