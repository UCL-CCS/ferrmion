"""Ternary Tree fermion to qubit mappings"""

import numpy as np
from .devices import Qubit
from .base import FermionQubitEncoding
from .ternary_tree_node import TTNode, node_sorter
from .utils import icount_to_sign
from functools import cached_property
from ferrmion import symplectic_product
import logging

logger = logging.getLogger(__name__)

class TernaryTree(FermionQubitEncoding):
    def __init__(
        self,
        one_e_coeffs: np.ndarray,
        two_e_coeffs: np.ndarray,
        root_node: TTNode = TTNode(),
        enumeration_scheme: dict[str, tuple[int, int]] | None = None,
    ):
        """Initialise a ternary tree.
        
        Args:
            one_e_coeffs (np.ndarray): The one-electron coefficients.
            two_e_coeffs (np.ndarray): The two-electron coefficients.
            qubits (set[Qubit]): The qubits.
            root_node (TTNode): The root node of the tree.
            enumeration_scheme (dict[str, tuple[int, int]]): The enumeration scheme.
        """
        self.n_qubits = one_e_coeffs.shape[1]
        self.root = root_node
        self.root.label = ""
        self.enumeration_scheme = enumeration_scheme
        vaccum_state = np.array([0]*self.n_qubits,dtype=np.uint8)
        super().__init__(one_e_coeffs, two_e_coeffs, vaccum_state)

    @property
    def default_mode_op_map(self):
        return {i:i for i in range(self.n_qubits)}
    
    def default_enumeration_scheme(self) -> dict[str, dict[str, int]]:
        """Create a default enumeration scheme for the tree.
        
        Returns:
            dict[str, dict[str, int]]: A dictionary of all node labels, j, with mode and qubit indices.
        """
        node_strings = self.root.child_strings
        enumeration_scheme = {node: (None, None) for node in node_strings}
        enumeration_scheme = {
            k: {"mode": i, "qubit": i} for i, k in enumerate(enumeration_scheme)
        }
        return enumeration_scheme
    
    def as_dict(self):
        """Return the tree structure as a dictionary."""
        return self.root.as_dict()

    def add_node(self, node_string: str) -> "TernaryTree":
        """Add a node to the tree.
        
        Args:
            node_string (str): The string representation of the node.
            
        Returns:
            TernaryTree: The tree with the node added.
        """
        logger.debug("Adding node %s to TernaryTree", node_string)
        node_string = node_string.lower()
        valid_string = np.all([char in ["x", "y", "z"] for char in node_string])
        if not valid_string:
            raise ValueError("Branch string can only contain x,y,z")

        node = self.root
        for char in node_string:
            if isinstance(getattr(node, char), TTNode):
                node = getattr(node, char)
            else:
                node = node.add_child(char, node.label + char)
        return self

    @property
    def branch_operator_map(self) -> dict[str, str]:
        """Create a map from each branch string to a Pauli string.
        
        Returns:
            dict[str, str]: A dictionary of all branch strings with their corresponding Pauli strings.
        """
        logger.debug("Building branch operator map for TernaryTree.")
        if self.enumeration_scheme is None:
            logger.error("No enumeration scheme provided, using default.")
            raise ValueError("enumeration scheme not set")

        branches = self.root.branch_strings

        nodes = self.root.child_strings
        node_indices = {node: i for i, node in enumerate(nodes)}

        branch_operator_map = {}
        for branch in branches:
            branch_operator_map[branch] = ["I"] * self.n_qubits
            node = self.root
            for char in branch:
                node_index = node_indices[node.label]
                branch_operator_map[branch][node_index] = char.upper()
                node = getattr(node, char, None)
            branch_operator_map[branch] = "".join(branch_operator_map[branch])

        return branch_operator_map

    @property
    def string_pairs(self) -> dict[str, tuple[str, str]]:
        """Return the pair of branch strings which correspond to each node.

        Returns:
            dict[str, tuple(str,str)]: A dictionary of all node labels, j,  with branch strings (2j, 2j+1).
        """
        logger.debug("Building string pairs for TernaryTree.")
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

    def _build_symplectic_matrix(self) -> tuple[np.ndarray[np.bool], np.ndarray[np.bool]]:
        """Build the symplectic matrix for the tree.
        Returns:
            np.ndarray[np.uint8]: Powers of i for each row of the symplectic matrix.
            np.ndarray[np.uint8]: Symplectic matrix.
        """
        logger.debug("Building symplectic matrix for TernaryTree.")
        # If there isn't one provided, assume the naive one
        if self.enumeration_scheme is None:
            logger.debug("No enumeration scheme provided, using default.")
            self.enumeration_scheme = self.default_enumeration_scheme()

        pauli_string_map = self.branch_operator_map
        
        symplectic = np.zeros((2 * self.n_qubits, 2 * self.n_qubits), dtype=np.bool)
        ipowers = np.zeros((2 * self.n_qubits), dtype=np.uint8)
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

    def JordanWigner(self) -> "TernaryTree":
        """Create a new tree with the Jordan-Wigner encoding."""
        logger.debug("Creating Jordan-Wigner encoding tree")
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            root_node=TTNode(),
        )
        new_tree.add_node("z" * (self.n_qubits - 1))
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    def JW(self) -> "TernaryTree":
        """Alias for Jordan-Wigner encoding."""
        return self.JordanWigner()

    def ParityEncoding(self) -> "TernaryTree":
        """Create a new tree with the parity encoding."""
        logger.debug("Creating parity encoding tree")
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            root_node=TTNode(),
        )
        new_tree.add_node("x" * (self.n_qubits - 1))
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    def BravyiKitaev(self) -> "TernaryTree":
        """Create a new tree with the Bravyi-Kitaev encoding."""
        logger.debug("Creating Bravyi-Kitaev encoding tree")
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            root_node=TTNode(),
        )
        branches = ["x"]
        # one is used for root, which is defined
        remaining_qubits = self.n_qubits - 1
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

    def BK(self) -> "TernaryTree":
        """Alias for Bravyi-Kitaev encoding."""
        return self.BravyiKitaev()

    def JKMN(self) -> "TernaryTree":
        """Create a new tree with the JKMN encoding."""
        logger.debug("Creating JKMN encoding tree.")
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            root_node=TTNode(),
        )
        branches = ["x", "y", "z"]
        # one is used for root which is defined
        remaining_qubits = self.n_qubits - 1
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