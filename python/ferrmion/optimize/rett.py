"""Reduced entanglement Ternary Tree."""

import logging

import numpy as np
from numpy.typing import NDArray

from ..encode.ternary_tree import TernaryTree
from ..encode.ternary_tree_node import TTNode

logger = logging.getLogger(__name__)


def reduced_entanglement_ternary_tree(
    mutual_information: NDArray,
    cutoff: float = 0.5,
    max_branches: int | None = None,
    squash: bool = False,
) -> TernaryTree:
    """Creates the reduced entanglement TernaryTree.

    Args:
        mutual_information (NDArray): A 2D array of mode mutual information.
        cutoff (float | None): The average MI between spatial orbitals.
        max_branches (int): The maximum allowed number of Parity branches.
        squash (bool): Whether to squash the mutual_information from spin-orbit form to spinless.

    Returns:
        TernaryTree: A new ternary tree.

    Note:
        Assumes that the MI matrix gives MI between spinless spatial orbitals
        So that each block of four contains [[aa, ab], [ba,bb]]

    Example:
        >>> import numpy as np
        >>> from ferrmion.optimize.rett import reduced_entanglement_tree
        >>> mi = 0.5 * np.random.random((6,6))
        >>> mi = mi + mi.T
        >>> tree = reduced_entanglement_tree(mi)
        >>> tree.as_dict()

    Advanced example (with options):
        >>> tree = reduced_entanglement_tree(mi, cutoff=0.1, max_branches=2, squash=False)
    """
    logger.debug("Creating Reduced entanglement TT.")
    enumeration_scheme = {}
    n_modes = mutual_information.shape[0]
    n_modes *= 1 if squash else 2

    new_tree = TernaryTree(n_modes, root_node=TTNode())

    if squash:
        # First combine the MI information for alpha and beta spins
        squash_rows = mutual_information[::2] + mutual_information[1::2]
        squash_matrix = squash_rows[:, ::2] + squash_rows[:, 1::2]
        squash_matrix *= 0.25
    else:
        squash_matrix = mutual_information

    mi_rank = np.triu(squash_matrix).flatten().argsort()[::-1]
    # Convert back to square format from flattened
    sorted_indices = [np.unravel_index(index, squash_matrix.shape) for index in mi_rank]
    sorted_indices = [(int(i[0]), int(i[1])) for i in sorted_indices]
    logger.debug(f"Matrix indices sorted by decreasing MI: {sorted_indices}")

    branches: list[tuple[int, int, int, int]] = []
    unused_indices = {i for i in range(squash_matrix.shape[0])}
    for squash_index in sorted_indices:
        if len(set(squash_index)) == 1:
            logger.warning("MI Matrix contains non-zero diagonal elements, skipping.")
            continue

        if max_branches is not None and len(branches) >= max_branches:
            break

        if not unused_indices.issuperset(squash_index):
            logger.debug("Indices %s previously assigned to branch.", squash_index)
            continue

        if squash_matrix[squash_index] >= cutoff:
            branch = (
                2 * squash_index[0],
                2 * squash_index[0] + 1,
                2 * squash_index[1],
                2 * squash_index[1] + 1,
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
            node_path = "z" * i + "x" * j
            new_tree.add_node(node_path)
            enumeration_scheme[node_path] = (mode, mode)
            unused_modes.remove(mode)

    remaining_modes = new_tree.n_qubits - (4 * len(branches))
    new_tree.add_node("z" * (remaining_modes + len(branches) - 1))

    for node_path in new_tree.root_node.child_strings:
        if enumeration_scheme.get(node_path, None) is None:
            mode = unused_modes.pop()
            enumeration_scheme[node_path] = (mode, mode)

    logger.debug("Setting enumeration scheme")
    logger.debug(enumeration_scheme)
    new_tree.enumeration_scheme = enumeration_scheme
    return new_tree
