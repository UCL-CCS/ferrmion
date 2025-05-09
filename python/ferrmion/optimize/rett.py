"""Reduced entanglement Ternary Tree."""

from ferrmion.ternary_tree import TernaryTree
from ferrmion.ternary_tree_node import TTNode
import numpy as np
import logging
logger = logging.getLogger(__name__)

def reduced_entanglement_tree(tree: TernaryTree, mutual_information: np.ndarray[np.float64], cutoff: float | None = 0.5) -> TernaryTree:
    """Creates the reduced entanglement TernaryTree.
    
    Args:
        tree (TernaryTree): A ternary tree encoding.
        mutual_informatin (np.ndarray): A 2D array of mode mutual information.
        cutoff (float | None): The average MI between spatial orbitals
    
    Returns:
        TernaryTree: A new ternary tree.

    NOTE:
        Assumes that the MI matrix is in spin-orb ordering 
        So that each block of four contains [[aa, ab], [ba,bb]]
    """
    logger.debug("Creating Reduced entanglement TT")
    enumeration_scheme = {}
    new_tree = TernaryTree(tree.one_e_coeffs, tree.two_e_coeffs, root_node=TTNode())

    # First combine the MI information for alpha and beta spins
    squash_rows = mutual_information[::2]+mutual_information[1::2]
    squash_matrix = squash_rows[:,::2] + squash_rows[:,1::2]

    mi_rank = np.triu(squash_matrix).flatten().argsort()[::-1]
    squash_indices = [np.unravel_index(index, squash_matrix.shape) for index in mi_rank]
    squash_indices = [(int(i[0]), int(i[1])) for i  in squash_indices]
    logger.debug(f"Spatial orbital mutual information rank {squash_indices}")

    branches = []
    unused_indices = {i for i in range(squash_matrix.shape[0])}
    for squash_index in squash_indices:
        if not unused_indices.issuperset(squash_index):
            logger.debug("Indices %s previously assigned to branch.", squash_index)
            continue

        if squash_matrix[squash_index] >= 4*cutoff:
            branch = (
                2*squash_index[0],
                2*squash_index[0]+1, 
                2*squash_index[1],
                2*squash_index[1]+1
                )
            logger.debug("Adding branch %s", branch)
            unused_indices.remove(squash_index[0])
            unused_indices.remove(squash_index[1])
            branches.append(branch)
        
        if len(unused_indices) <= 1:
            break

    unused_modes = {i for i in range(new_tree.n_qubits)}
    for i, branch in enumerate(branches):
        for j, mode in enumerate(branch):
            node_path = "z"*i + "x"*j
            new_tree.add_node(node_path)
            enumeration_scheme[node_path] = (mode,mode)
            unused_modes.remove(mode)


    remaining_modes = new_tree.n_qubits - (4*len(branches))
    new_tree.add_node("z"*(remaining_modes + len(branches) -1))

    for node_path in new_tree.root.child_strings:
        if enumeration_scheme.get(node_path, None) is None:
            mode = unused_modes.pop()
            enumeration_scheme[node_path] = (mode, mode)

    logger.debug("Setting enumeration scheme")
    logger.debug(enumeration_scheme)
    new_tree.enumeration_scheme = enumeration_scheme
    return new_tree