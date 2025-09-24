"""Init for fermion qubit encodings.

This file is ignored by pre-commit as the pyo3 integration requires importing
rust functions before importing functions from the python module.
"""

from .core import hartree_fock_state, symplectic_product
from .encode import FermionQubitEncoding
from .encode.maxnto import MaxNTO, maxnto_symplectic_matrix
from .encode.ternary_tree import TernaryTree
from .encode.ternary_tree_node import TTNode, node_sorter
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
    "FermionQubitEncoding",
    "TernaryTree",
    "TTNode",
    "node_sorter",
    "pauli_to_symplectic",
    "symplectic_to_pauli",
    "symplectic_hash",
    "symplectic_unhash",
    "symplectic_product",
    "icount_to_sign",
    "MaxNTO",
    "maxnto_symplectic_matrix",
    "hartree_fock_state",
    "two_operator_product",
]

setup_logs()
