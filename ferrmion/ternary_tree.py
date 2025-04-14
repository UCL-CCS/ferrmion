"""Ternary Tree fermion to qubit mappings"""

import numpy as np
from .devices import Qubit
from .base import FermionQubitEncoding
from .ternary_tree_node import TTNode, node_sorter
from .utils import symplectic_product, icount_to_sign
from functools import cached_property
import logging

logger = logging.getLogger(__name__)

class TernaryTree(FermionQubitEncoding):
    def __init__(
        self,
        one_e_coeffs: np.ndarray,
        two_e_coeffs: np.ndarray,
        qubits: set[Qubit],
        root_node: TTNode = TTNode(),
        enumeration_scheme: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__(one_e_coeffs, two_e_coeffs, qubits)
        self.root = root_node
        self.root.label = ""
        self.enumeration_scheme = enumeration_scheme

    def default_enumeration_scheme(self):
        node_strings = self.root.child_strings
        enumeration_scheme = {node: (None, None) for node in node_strings}
        enumeration_scheme = {
            k: {"mode": i, "qubit": i} for i, k in enumerate(enumeration_scheme)
        }
        return enumeration_scheme

    def _valid_qubit_number(self):
        return len(self.fermionic_modes) == len(self.qubits)

    def as_dict(self):
        return self.root.as_dict()

    def add_node(self, node_string: str):
        node_string = node_string.lower()
        valid_string = np.all([char in ["x", "y", "z"] for char in node_string])
        if not valid_string:
            raise ValueError("Branch string can only contain x,y,z")

        node = self.root
        for char in node_string:
            if isinstance(getattr(node, char), TTNode):
                node = getattr(node, char)
            else:
                self._next_qubit()
                node = node.add_child(char, node.label + char)
        return self

    @property
    def branch_operator_map(self):
        if self.enumeration_scheme is None:
            raise ValueError("enumeration scheme not set")

        branches = self.root.branch_strings

        nodes = self.root.child_strings
        node_indices = {node: i for i, node in enumerate(nodes)}

        branch_operator_map = {}
        num_qubits = len(self.qubits)
        for branch in branches:
            branch_operator_map[branch] = ["I"] * num_qubits
            node = self.root
            for char in branch:
                node_index = node_indices[node.label]
                # qubit_index = self.enumeration_scheme[node.label]["qubit"]
                branch_operator_map[branch][node_index] = char.upper()
                node = getattr(node, char, None)
            branch_operator_map[branch] = "".join(branch_operator_map[branch])

        return branch_operator_map

    @property
    def string_pairs(self):
        """Return the pair of branch strings which correspond to each node.

        Returns:
            dict[str, tuple(str,str)]: A dictionary of all node labels, j,  with branch strings (2j, 2j+1).
        """
        node_set = self.root.child_strings

        pairs = {}
        for node_string in node_set:
            node = self.root
            for char in node_string:
                node = getattr(node, char)

            x_string = node_string + "x"
            y_string = node_string + "y"
            if x_string in node_set:
                while True:
                    x_string += "z"
                    if x_string not in node_set:
                        break

            if y_string in node_set:
                while True:
                    y_string += "z"
                    if y_string not in node_set:
                        break

            if x_string.count("y") % 2 == 0:
                pairs[node.label] = x_string, y_string
            elif y_string.count("y") % 2 == 0:
                pairs[node.label] = y_string, x_string

        return pairs

    def _build_symplectic_matrix(self):
        # If there isn't one provided, assume the naive one
        if self.enumeration_scheme is None:
            self.enumeration_scheme = self.default_enumeration_scheme()

        pauli_string_map = self.branch_operator_map
        n_qubits = len(self.qubits)
        symplectic = np.zeros((2 * n_qubits, 2 * n_qubits), dtype=np.byte)
        ipowers = np.zeros((2 * n_qubits), dtype=np.uint8)
        for node, operators in self.string_pairs.items():
            for offset, operator in enumerate(operators):
                operator = pauli_string_map[operator]
                operator = np.array(list(operator))
                # If the string is X or Y then assign 1
                term_ipower, symplectic_term = self._pauli_to_symplectic(operator)
                fermion_mode = self.enumeration_scheme[node]["mode"]
                ipowers[2 * fermion_mode + offset] = term_ipower
                symplectic[2 * fermion_mode + offset] = symplectic_term
        return ipowers, symplectic

    def JordanWigner(self):
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            qubits=self.qubits,
            root_node=TTNode(),
        )
        new_tree.add_node("z" * (len(self.qubits) - 1))
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    def JW(self):
        return self.JordanWigner()

    def ParityEncoding(self):
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            qubits=self.qubits,
            root_node=TTNode(),
        )
        new_tree.add_node("x" * (len(self.qubits) - 1))
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    def BravyiKitaev(self):
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            qubits=self.qubits,
            root_node=TTNode(),
        )
        branches = ["x"]
        # one is used for root, which is defined
        remaining_qubits = len(self.qubits) - 1
        while remaining_qubits > 0:
            new_branches = set()
            for item in branches:
                if remaining_qubits > 0:
                    new_tree.add_node(item)
                    remaining_qubits -= 1
                else:
                    break

                new_branches.add(item + "x")
                new_branches.add(item + "z")
            branches = sorted(list(new_branches), key=node_sorter)
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    def BK(self):
        return self.BravyiKitaev()

    def JKMN(self):
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            qubits=self.qubits,
            root_node=TTNode(),
        )
        branches = ["x", "y", "z"]
        # one is used for root which is defined
        remaining_qubits = len(self.qubits) - 1
        while remaining_qubits > 0:
            new_branches = set()
            for item in branches:
                if remaining_qubits > 0:
                    new_tree.add_node(item)
                    remaining_qubits -= 1
                else:
                    break

                new_branches.add(item + "x")
                new_branches.add(item + "y")
                new_branches.add(item + "z")
            branches = sorted(list(new_branches), key=node_sorter)
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree