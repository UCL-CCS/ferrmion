"""Huffman-code Ternary Tree."""

import numpy as np
import numpy.typing as npt

from ferrmion.encode import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode
from ferrmion.utils import find_pauli_weight, pauli_to_symplectic, symplectic_product


def _majarana_op_frequency(
    ones: npt.NDArray[float], twos: npt.NDArray[float]
) -> npt.NDArray[float]:
    majorana_freq = np.zeros(ones.shape[0])
    for i in range(ones.shape[0]):
        for j in range(ones.shape[1]):
            val = np.abs(ones[i, j])
            positions = {i, j}
            for p in positions:
                majorana_freq[p] += val

    for i in range(ones.shape[0]):
        for j in range(ones.shape[1]):
            for k in range(ones.shape[1]):
                for l in range(ones.shape[1]):
                    val = np.abs(twos[i, j, k, l])
                    positions = {i, j, k, l}
                    for p in positions:
                        majorana_freq[p] += val
    return majorana_freq.repeat(2)


def _build_huffman_tree(majorana_frequencies: npt.NDArray[float]) -> TernaryTree:
    nodes = {i: None for i in range(len(majorana_frequencies))}
    weights = {i: j for i, j in enumerate(majorana_frequencies)}
    n_ops = len(majorana_frequencies)
    for i in range(n_ops // 2):
        parent_index = 2 * n_ops - 1 - i
        mins = sorted(weights.items(), key=lambda kv: (kv[1], kv[0]))[:3]
        print(mins)

        parent = nodes.get(parent_index, TTNode(parent=None, qubit_label=i))

        match len(mins):
            case 0:
                break
            case 1:
                parent.x = nodes[mins[0][0]]
            case 2:
                parent.x = nodes[mins[0][0]]
                parent.y = nodes[mins[1][0]]
            case 3:
                parent.x = nodes[mins[0][0]]
                parent.y = nodes[mins[1][0]]
                parent.z = nodes[mins[2][0]]

        new_weight = 0
        for index, weight in mins:
            new_weight += weight
            weights.pop(index)
            nodes.pop(index)
        print(new_weight)

        print(parent_index, parent.child_strings)

        nodes[parent_index] = parent
        weights[parent_index] = new_weight

    root_node = [*nodes.values()][0]
    huffman_tree = TernaryTree(
        n_modes=len(majorana_frequencies) // 2, root_node=root_node
    )

    # Needed because of a bug to do with node labels.
    relabeled_tree = TernaryTree(14)
    for child in huffman_tree.root.child_strings:
        relabeled_tree.add_node(child)

    return relabeled_tree


def _huffman_mode_op_map(huffman_tree):
    weights = {}
    for index, pair in enumerate(huffman_tree.string_pairs.values()):
        left, right = pair
        left = huffman_tree.branch_operator_map[left]
        right = huffman_tree.branch_operator_map[right]

        weights[index] = {}
        _, left = pauli_to_symplectic(left)
        _, right = pauli_to_symplectic(right)
        pair_weight = find_pauli_weight(np.array([left])) + find_pauli_weight(
            np.array([right])
        )
        _, product = symplectic_product(left, right)
        product_weight = find_pauli_weight(np.array([product]))

        weights[index]["pair_weight"] = pair_weight
        weights[index]["prod_weight"] = product_weight

        operator_order = sorted(
            weights.items(), key=lambda kv: (kv[1]["prod_weight"], kv[1]["pair_weight"])
        )

        operator_order = [index for index, _ in operator_order]


def huffman_ternary_tree(ones, twos):
    majorana_frequencies = _majarana_op_frequency(ones, twos)
    huffman_ternary_tree = _build_huffman_tree(majorana_frequencies)
    mode_op_map = _huffman_mode_op_map(huffman_ternary_tree)
    huffman_ternary_tree.default_mode_op_map = mode_op_map
    return huffman_ternary_tree
