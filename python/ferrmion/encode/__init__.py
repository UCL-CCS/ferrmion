"""Tools for converting fermionic operators and Hamiltonians into Pauli representations.

You can also `decode()` measurements in Z-basis back into Fock states.
"""

from ferrmion.core import MajoranaEncoding

from .ternary_tree import (
    JKMN,
    BravyiKitaev,
    JordanWigner,
    Parity,
    TernaryTree,
)


def MaxNTO(n_modes: int) -> MajoranaEncoding:
    """The MaxNTO k-NTO encoding for ``n_modes`` fermionic modes.

    Alias for :meth:`ferrmion.core.MajoranaEncoding.maxnto`.
    """
    return MajoranaEncoding.maxnto(n_modes)


__all__ = [
    "MajoranaEncoding",
    "TernaryTree",
    "MaxNTO",
    "JordanWigner",
    "BravyiKitaev",
    "Parity",
    "JKMN",
]
