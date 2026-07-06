"""Init for fermion qubit encodings.

This file is ignored by pre-commit as the pyo3 integration requires importing
rust functions before importing functions from the python module.
"""

from .core import (
    FermionHamiltonian,
    MajoranaEncoding,
    QubitHamiltonian,
    symplectic_product,
    pauli_to_symplectic,
    symplectic_product,
    symplectic_to_pauli,
    symplectic_to_sparse,
)
from .encode import MaxNTO
from .encode.ternary_tree_node import TTNode, node_sorter

from .encode.ternary_tree import(
    TernaryTree,
    JordanWigner,
    BravyiKitaev,
    Parity,
    JKMN,
)
from .hamiltonians import molecular_hamiltonian, hubbard_hamiltonian
from .utils import (
    setup_logs,

)

__all__ = [
    "MajoranaEncoding",
    "TernaryTree",
    "TTNode",
    "node_sorter",
    "pauli_to_symplectic",
    "symplectic_to_pauli",
    "symplectic_product",
    "FermionHamiltonian",
    "QubitHamiltonian",
    "molecular_hamiltonian",
    "hubbard_hamiltonian",
    "JordanWigner",
    "BravyiKitaev",
    "Parity",
    "JKMN",
    "MaxNTO",
]

setup_logs()
