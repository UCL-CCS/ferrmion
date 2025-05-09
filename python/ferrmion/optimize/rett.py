"""Reduced entanglement Ternary Tree."""

from ferrmion.ternary_tree import TernaryTree
import numpy as np

def reduced_entanglement_form(tree: TernaryTree, mutual_information: np.ndarray[np.float]) -> TernaryTree:
    """Creates the reduced entanglement TernaryTree.
    
    Args:
        tree (TernaryTree): A ternary tree encoding.
        mutual_informatin (np.ndarray): A 2D array of mode mutual information.
    
    Returns:
        TernaryTree: A new ternary tree.
    """

    return TernaryTree(tree.one_e_coeffs, tree.two_e_coeffs)