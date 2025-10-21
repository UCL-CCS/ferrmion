"""Code to Geneate Hamiltonian Adaptive Ternary Tree from Majorana Hamiltonian."""

from itertools import combinations
from typing import Iterable

import numpy as np

from ferrmion.encode import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode


def _qubit_term_weight(term: Iterable, comb: tuple[int, int, int]) -> int:
    """Find the single-qubit Pauli-weight of majorana terms.

    If any pauli term is found an even number f times, we obtain I, weight = 0.
    If we find all three pauli terms, return I (with an imaginary ccoefficient), weight = 0
    If we find either one pauli or two then the weight = 1.

    Args:
        term (Iterable): Indices of term in our majorana-hamiltonian.
        comb (tuple[int, int, int]): Combination of indices to weigh (x,y,z).

    Returns:
        int: Weight of the term.
    """
    term_array = np.array([t for t in term])
    odd_parity_paulis = np.array(
        [np.count_nonzero(np.array(term_array - index)) % 2 for index in comb]
    )
    non_commuting = np.sum(odd_parity_paulis) % 3
    return int(non_commuting != 0)


def _reduce_hamiltonian(
    majorana_ham: dict[Iterable[int], float],
    parent_index: int,
    selection: tuple[int, int, int],
) -> dict[tuple[int, ...], float]:
    """Simplify the Hamiltonian.

    As we increase the qubit number, we iteratively remove majoranas
    which act trivially on the remaining qubits.
    We replace them with the index of their parent string
    as going forward they are identical to the parent string.

    Args:
        majorana_ham (dict[tuple[int,...],float]): Current Hamiltonian.
        parent_index (int): Qubit index of the parent node.
        selection (tuple[int, int, int]): Indices of the majoranas to be replaced.

    Returns:
        dict[tuple[int,...],float]: Reduced Hamiltonian.
    """
    new_ham = {}
    for term, coeff in majorana_ham.items():
        new_term = tuple(i if i not in selection else parent_index for i in term)
        if len(set(new_term)) != 1:
            new_ham[new_term] = coeff
    return new_ham


def hamiltonian_adaptive_ternary_tree(
    majorana_ham: dict[Iterable[int], float], n_modes: int
) -> TernaryTree:
    """Construct an adaptive ternary tree from a majorana Hamiltonian.

    Args:
        majorana_ham (dict[tuple[int,...],float]): Majorana Hamiltonian to encode.
        n_modes (int): Number of fermionic modes in the system.

    Returns:
        TTNode: Root node of the constructed ternary tree.
    """
    # We need 2*M +1 leaves and M nodes.
    nodes: dict[int, TTNode | None] = {i: None for i in range(2 * n_modes + 1)}
    for i in range(n_modes):
        nodes[2 * n_modes + 1 + i] = TTNode(qubit_label=i)

    # Start with all the leaves unassigned
    unassigned = {*range(2 * n_modes + 1)}

    # We create two maps, of z_ancestors and z_descendants
    ancestor_map = {i: i for i in nodes}
    descendant_map = {i: i for i in nodes}

    total_weight = 0
    for i in range(n_modes):
        parent_index = 2 * n_modes + 1 + i
        parent = nodes[parent_index]

        min = np.inf
        for comb in combinations(unassigned, 2):
            small_y = None
            small_x = None
            # This way x index will be higher term - more often node.
            # z_index, x_index= comb
            x_index, z_index = comb
            small_x = descendant_map[x_index]

            # discard this combination
            if small_x == 2 * n_modes:
                continue

            if small_x % 2 == 0:
                small_y = small_x + 1
            else:
                small_y = small_x - 1
            # We can't use this index for y a
            # it has been used in the combination already
            # so we'd be replacing our x or z!
            if small_y in comb:
                continue

            y_index = ancestor_map[small_y]

            if y_index in comb:
                continue

            if small_x % 2 == 0:
                comb = np.array([x_index, y_index, z_index], dtype=np.uint)
            else:
                comb = np.array([y_index, x_index, z_index], dtype=np.uint)
            comb = [int(i) for i in comb]
            weight = np.sum(
                [_qubit_term_weight(term, comb) for term in majorana_ham.keys()]
            )
            if weight < min:
                min = weight
                selection = comb
            # would be better to break on zero
            # if weight == 0:
            #     break

        total_weight += min
        # Now find the Y pair of the x-node
        for i, char in zip(selection, ["x", "y", "z"]):
            if i in unassigned:
                unassigned.remove(i)

            if isinstance(nodes.get(i, None), TTNode):
                parent.add_child(which_child=char, child_node=nodes.get(i))
            else:
                parent.leaf_majorana_indices[char] = i

        z_index = selection[2]
        z_desc = descendant_map[z_index]
        descendant_map[parent_index] = z_desc
        ancestor_map[z_index] = parent_index
        ancestor_map[z_desc] = parent_index

        unassigned.add(parent_index)

        majorana_ham = _reduce_hamiltonian(majorana_ham, parent_index, selection)

    if len(unassigned) != 1:
        raise ValueError("Not all nodes assigned by HATT.")

    last_node = nodes[unassigned.pop()]
    if isinstance(last_node, TTNode):
        root = last_node
    else:
        raise ValueError("Hatt root node is not a TTNode object.")

    tree = TernaryTree(n_modes=n_modes, root_node=root)
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    tree.pauli_weight = total_weight
    return tree
