"""Huffman-code Ternary Tree."""

import numpy as np
import numpy.typing as npt

from ferrmion.encode import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode
from ferrmion.utils import find_pauli_weight, pauli_to_symplectic, symplectic_product


def _majarana_op_frequency(
    ones: npt.NDArray[float], twos: npt.NDArray[float]
) -> npt.NDArray[float]:
    n_modes = ones.shape[0]
    majorana_freq = np.zeros(n_modes)

    for i in range(n_modes):
        for j in range(n_modes):
            val = np.abs(ones[i, j])
            positions = {i, j}
            for p in positions:
                majorana_freq[p] += val

    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                for l in range(n_modes):
                    val = np.abs(twos[i, j, k, l])
                    positions = {i, j, k, l}
                    for p in positions:
                        majorana_freq[p] += val
    return majorana_freq.repeat(2)


def _build_huffman_tree(
    n_modes: int, majorana_frequencies: npt.NDArray[float]
) -> TernaryTree:
    nodes = {i: None for i in range(len(majorana_frequencies))}
    weights = {i: j for i, j in enumerate(majorana_frequencies)}
    n_ops = len(majorana_frequencies)
    for i in range(n_ops // 2):
        parent_index = 2 * n_ops - 1 - i
        mins = sorted(weights.items(), key=lambda kv: (kv[1], kv[0]))[:3]

        parent = nodes.get(
            parent_index, TTNode(parent=None, root_path="", qubit_label=i)
        )

        for min, child_string in zip(mins, ["x", "y", "z"][: len(mins)]):
            possible_child = nodes[min[0]]
            if isinstance(possible_child, TTNode):
                parent.add_child(which_child=child_string, child_node=possible_child)

        new_weight = 0
        for index, weight in mins:
            new_weight += weight
            weights.pop(index)
            nodes.pop(index)

        nodes[parent_index] = parent
        weights[parent_index] = new_weight

    root_node = [*nodes.values()][0]
    huffman_tree = TernaryTree(n_modes=n_modes, root_node=root_node)
    huffman_tree.string_pairs

    return huffman_tree


def _two_e_frequency(ones, twos) -> npt.NDArray[float]:
    n_modes = ones.shape[0]
    two_e_freq = np.zeros(ones.shape)
    for j in range(n_modes):
        for i in range(n_modes):
            for l in range(n_modes):
                for k in range(n_modes):
                    val = np.abs(twos[i, j, k, l])
                    two_e_freq[i, j] += val
                    two_e_freq[k, l] += val
    two_e_freq = np.kron(two_e_freq, np.array([[1, 1], [1, 1]]))
    two_e_freq = np.triu(two_e_freq, k=1)
    return two_e_freq


def _mode_priority(two_e_freq):
    vaccum_frequencies = np.diag(two_e_freq, k=1)
    sorted_pairs = [
        (i, i + 1) for i in np.argsort(vaccum_frequencies)[::-1] if i % 2 == 0
    ]
    sorted_modes = [i[0] // 2 for i in sorted_pairs]
    return sorted_modes


def _operator_pair_priority(huffman_tree):
    weights = {}
    for index, pair in enumerate(huffman_tree.string_pairs.values()):
        left, right = pair
        left = huffman_tree.branch_operator_map[left]
        right = huffman_tree.branch_operator_map[right]

        weights[index] = {}
        _, left = pauli_to_symplectic(0, left)
        _, right = pauli_to_symplectic(0, right)
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
    return operator_order


def huffman_ternary_tree(
    one_e_coeffs: npt.NDArray[float], two_e_coeffs: npt.NDArray[float]
) -> TernaryTree:
    """Creates a Huffman-code Ternary Tree.

    Li, Q. S., Liu, H. Y., Wang, Q., Wu, Y. C., & Guo, G. P. (2025).
    Huffman-Code-based Ternary Tree Transformation. Chinese Physics Letters.

    http://iopscience.iop.org/article/10.1088/0256-307X/42/10/100001

    Note: Only vaccum-preserving Huffman-trees are currently supported.

    Args:
        one_e_coeffs (np.ndarray): One electron coefficients.
        two_e_coeffs (np.ndarray): Two electron coefficients.

    Return:
        TernaryTree: A Huffman-code ternary tree.
    """
    n_modes = one_e_coeffs.shape[0]

    majorana_frequencies = _majarana_op_frequency(one_e_coeffs, two_e_coeffs)

    huffman_ternary_tree = _build_huffman_tree(n_modes, majorana_frequencies)
    huffman_ternary_tree.enumeration_scheme = (
        huffman_ternary_tree.default_enumeration_scheme()
    )

    two_e_frequencies = _two_e_frequency(one_e_coeffs, two_e_coeffs)
    sorted_modes = _mode_priority(two_e_frequencies)
    sorted_operators = _operator_pair_priority(huffman_ternary_tree)

    mode_op_map = [0] * len(sorted_modes)
    for operator_index, mode_index in enumerate(sorted_modes):
        mode_op_map[mode_index] = sorted_operators[operator_index]

    huffman_ternary_tree.default_mode_op_map = mode_op_map
    huffman_ternary_tree.enumeration_scheme = (
        huffman_ternary_tree.default_enumeration_scheme()
    )
    return huffman_ternary_tree
