"""Init for fermion qubit encodings.

This file is ignored by pre-commit as the pyo3 integration requires importing
rust functions before importing functions from the python module.
"""

from .core import (
    FermionHamiltonian,
    MajoranaEncoding,
    QubitHamiltonian,
    symplectic_product,
)
from .encode import MaxNTO
from .encode.ternary_tree_node import TTNode, node_sorter

from .encode.ternary_tree import(
    TernaryTree,
    JordanWigner,
    BravyiKitaev,
    ParityEncoding,
    JKMN,
    jordan_wigner,
    jordan_wigner_annealed,
    jordan_wigner_topphatt,
    parity,
    parity_annealed,
    parity_topphatt,
    bravyi_kitaev,
    bravyi_kitaev_annealed,
    bravyi_kitaev_topphatt,
    jkmn,
    jkmn_annealed,
    jkmn_topphatt,
)
from .hamiltonians import molecular_hamiltonian, hubbard_hamiltonian
from .utils import (
    icount_to_sign,
    pauli_to_symplectic,
    setup_logs,
    symplectic_hash,
    symplectic_to_pauli,
    symplectic_unhash,
    two_operator_product,

)

__all__ = [
    "MajoranaEncoding",
    "TernaryTree",
    "TTNode",
    "node_sorter",
    "pauli_to_symplectic",
    "symplectic_to_pauli",
    "symplectic_hash",
    "symplectic_unhash",
    "symplectic_product",
    "icount_to_sign",
    "two_operator_product",
    "FermionHamiltonian",
    "QubitHamiltonian",
    "molecular_hamiltonian",
    "hubbard_hamiltonian",
    "JordanWigner",
    "BravyiKitaev",
    "ParityEncoding",
    "JKMN",
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
]

setup_logs()
