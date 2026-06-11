"""Hamiltonian-Adaptive Ternary Tree construction.

The greedy tree construction runs in native Rust (`ferrmion.core.hatt`);
this module exposes a thin Python wrapper that takes a `FermionHamiltonian`
and returns a fully-populated `TernaryTree`.
"""

from typing import TYPE_CHECKING

from ferrmion.encode import TernaryTree
from ferrmion.encode.ternary_tree_node import node_sorter

if TYPE_CHECKING:
    from ferrmion.hamiltonians import FermionHamiltonian


def hamiltonian_adaptive_ternary_tree(
    fham: "FermionHamiltonian", n_modes: int
) -> TernaryTree:
    """Construct an adaptive ternary tree from a fermionic Hamiltonian.

    The tree is built greedily so the resulting qubit Hamiltonian has low
    Pauli weight. The construction runs in native Rust
    (:func:`ferrmion.core.hatt`).

    Args:
        fham (FermionHamiltonian): The Hamiltonian whose terms drive the
            greedy Pauli-weight minimisation.
        n_modes (int): Number of fermionic modes in the system.

    Returns:
        TernaryTree: The constructed tree with ``pauli_weight`` populated.
    """
    from ferrmion import core

    flatpack, weight = core.hatt(fham, n_modes)
    tree = TernaryTree.from_flatpack(flatpack)
    # `from_flatpack` installs a DFS-ordered enumeration; switch to the
    # sort-by-`node_sorter` ordering produced by `default_enumeration_scheme`
    # so the returned tree's enumeration matches every other HATT-style
    # constructor in the library.
    sorted_paths = sorted(tree.enumeration_scheme.keys(), key=node_sorter)
    tree.enumeration_scheme = {
        p: (mode, tree.enumeration_scheme[p][1]) for mode, p in enumerate(sorted_paths)
    }
    tree.pauli_weight = weight
    return tree
