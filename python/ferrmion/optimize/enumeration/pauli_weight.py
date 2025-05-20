"""Functions for optimizing the Pauli Weight of an enumeration."""

import numpy as np

from ferrmion.encode.ternary_tree import TernaryTree
from ferrmion.utils import symplectic_unhash


def scaled_pauli_weight(tree: TernaryTree, permutation: list[int]) -> list[float]:
    """The Pauli-weight of a template scaled by the term coefficients.

    Args:
        tree (TernaryTree): A Ternary Tree with template calculated.
        permutation (list[int]): A list of integer mode labels, assigned to operator pairs [0,...,N]


    Return:
        list[float]: A single value in a list (needed for deap) giving the cost.
    """
    ham = tree.fill_template({i: j for i, j in zip(range(tree.n_qubits), permutation)})

    def hashed_pauli_weight(hashed_term):
        return np.sum(
            np.bitwise_or(*np.hsplit(symplectic_unhash(hashed_term, tree.n_qubits), 2))
        )

    return [np.sum([hashed_pauli_weight(k) * np.abs(v) for k, v in ham.items()])]
