"""Init for encodings."""

from ferrmion.core import MajoranaEncoding

from .ternary_tree import (
    JKMN,
    BravyiKitaev,
    JordanWigner,
    ParityEncoding,
    TernaryTree,
    bravyi_kitaev,
    bravyi_kitaev_annealed,
    bravyi_kitaev_topphatt,
    jkmn,
    jkmn_annealed,
    jkmn_topphatt,
    jordan_wigner,
    jordan_wigner_annealed,
    jordan_wigner_topphatt,
    parity,
    parity_annealed,
    parity_topphatt,
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
    "jordan_wigner",
    "jordan_wigner_annealed",
    "jordan_wigner_topphatt",
    "parity",
    "parity_annealed",
    "parity_topphatt",
    "bravyi_kitaev",
    "bravyi_kitaev_annealed",
    "bravyi_kitaev_topphatt",
    "jkmn",
    "jkmn_annealed",
    "jkmn_topphatt",
    "JordanWigner",
    "BravyiKitaev",
    "ParityEncoding",
    "JKMN",
]
