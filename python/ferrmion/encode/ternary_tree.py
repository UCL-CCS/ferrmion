"""Ternary Tree fermion to qubit mappings."""

import logging

import numpy as np
from numpy.typing import NDArray

from .base import FermionQubitEncoding
from .ternary_tree_node import TTNode, node_sorter

logger = logging.getLogger(__name__)


class TernaryTree(FermionQubitEncoding):
    """Ternary tree encoding for fermionic operators.

    Attributes:
        n_qubits (int): The number of qubits.
        root (TTNode): The root node of the tree.
        enumeration_scheme (dict[str, tuple[int, int]] | None): The enumeration scheme.

    Methods:
        default_mode_op_map(): Create a default mode operator map for the tree.
        default_enumeration_scheme(): Create a default enumeration scheme for the tree.
        as_dict(): Return the tree structure as a dictionary.
        add_node(node_string: str): Add a node to the tree.
        branch_operator_map(): Create a map from each branch string to a Pauli string.
        string_pairs(): Return the pair of branch strings which correspond to each node.
        _build_symplectic_matrix(): Build the symplectic matrix for the tree.

    Simple Example:
        >>> from ferrmion.encode.ternary_tree import TernaryTree
        >>> tree = TernaryTree(4)
        >>> tree.add_node('x')
        >>> tree.enumeration_scheme = tree.default_enumeration_scheme()
        >>> tree.as_dict()

    Advanced Usage:
        >>> from ferrmion.encode.ternary_tree import TernaryTree
        >>> tree = TernaryTree(4)
        >>> jordan_wigner = tree.JW()
        >>> bravyi_kitaev = tree.BK()
        >>> parity = tree.Parity()
        >>> minimum_height_tree = tree.JKMN()
    """

    def __init__(
        self,
        n_modes: int,
        root_node: TTNode = TTNode(),
    ):
        """Initialise a ternary tree.

        Args:
            n_modes (int): How many fermionic modes in the encoding.
            root_node (TTNode): The root node of the tree.
        """
        self._enumeration_scheme = {}
        self.n_modes = n_modes
        self.n_qubits = n_modes
        self.root_node = root_node
        self.root_node.root_path = ""
        self.vacuum_state = np.array([0] * self.n_qubits, dtype=np.uint8)
        self._enumeration_scheme = {}
        super().__init__(self.n_modes, self.n_qubits)

    @classmethod
    def from_hamiltonian_coefficients(cls, coeffs: tuple) -> "TernaryTree":
        """Create an encoding by passing coefficients.

        Args:
            coeffs (tuple): The electron integrals for some hamiltonian.

        Returns:
            FermionQubitEncoding: An initialised encoding.

        Example:
            >>> import numpy as np
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> coeffs = (np.zeros((4, 4)), np.zeros((4, 4, 4, 4)))
            >>> tree = TernaryTree.from_hamiltonian_coefficients(coeffs)
        """
        if not all([set(coeff.shape) == set(coeffs[0].shape) for coeff in coeffs]):
            logger.error("Coeff axes must be of equal size for all terms.")

        return cls(coeffs[0].shape[0])

    @property
    def enumeration_scheme(self) -> dict[str, tuple[int, int]]:
        """Get the enumeration scheme for the tree.

        Note:
            The tuple is organised as (modes, qubits).

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3).JW()
            >>> tree.enumeration_scheme
            {"": (0,0), "z": (1,1), "zz": (2,2)}
        """
        return self._enumeration_scheme

    @enumeration_scheme.setter
    def enumeration_scheme(self, enumeration_dict: dict[str, tuple[int, int]]):
        """Set the enumeration scheme.

        Args:
            enumeration_dict (dict[str, tuple[int, int]]): An dictionary mapping tree nodes to (mode, qubit) indices
        """
        logger.debug("Setting enumeration scheme.")
        error_string = ""
        if set(self.root_node.child_strings) != set(enumeration_dict.keys()):
            error_string += f"Enumeration scheme {enumeration_dict} must contain all nodes {self.root_node.child_strings}.\n"

        modes = set()
        qubits = set()
        for m, q in enumeration_dict.values():
            logger.debug(f"{m=}{q=}")
            modes.add(m)
            qubits.add(q)
        expected_modes = set(range(self.n_modes))
        expected_qubits = set(range(self.n_qubits))
        if set(modes).symmetric_difference(expected_modes):
            error_string += f"Invalid mode labels {set(modes)} in enumeration scheme ({expected_modes=}).\n"
        if set(qubits).symmetric_difference(expected_qubits):
            error_string += f"Invalid qubit labels {set(qubits)} in enumeration scheme ({expected_qubits=}).\n"

        if error_string != "":
            logger.error(error_string)
            raise ValueError(error_string)

        self.default_mode_op_map = [enum[0] for enum in enumeration_dict.values()]
        self._enumeration_scheme = enumeration_dict

    def default_enumeration_scheme(self) -> dict[str, tuple[int, int]]:
        """Create a default enumeration scheme for the tree.

        Note:
            The tuple is organised as (modes, qubits).

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3).JW()
            >>> tree.default_enumeration_scheme()
            {"": (0,0), "z": (1,1), "zz": (2,2)}
        """
        logger.debug("Setting default enumeration scheme")
        logger.debug("Child strings %s", self.root_node.child_strings)
        enumeration_scheme = {}
        child_labels = self.root_node.child_qubit_labels
        spare_labels: set[int] = set(range(len(child_labels))).difference(
            child_labels.values()
        )
        for child, label in child_labels.items():
            if label is None:
                label = spare_labels.pop()
            enumeration_scheme[child] = (int(label), int(label))
        return enumeration_scheme

    def as_dict(self):
        """Return the tree structure as a dictionary."""
        return self.root_node.as_dict()

    def add_node(self, node_string: str) -> "TernaryTree":
        """Add a node to the tree.

        Args:
            node_string (str): The string representation of the node.

        Returns:
            TernaryTree: The tree with the node added.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3)
            >>> tree.add_node('x')
        """
        logger.debug("Adding node %s to TernaryTree", node_string)
        node_string = node_string.lower()
        valid_string = np.all([char in ["x", "y", "z"] for char in node_string])
        if not valid_string:
            raise ValueError("Branch string can only contain x,y,z")

        node = self.root_node
        for char in node_string:
            if isinstance(getattr(node, char), TTNode):
                node = getattr(node, char)
            else:
                node = node.add_child(char, root_path=f"{node.root_path}{char}")
        return self

    @property
    def branch_operator_map(self) -> dict[str, str]:
        """Create a map from each branch string to a Pauli string.

        Returns:
            dict[str, str]: A dictionary of all branch strings with their corresponding Pauli strings.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3)
            >>> tree.add_node('x')
            >>> tree.add_node('xz')
            >>> tree.branch_operator_map
            {'xx': 'XXI',
            'xzx': 'XZX',
            'y': 'YII',
            'xy': 'XYI',
            'xzz': 'XZZ',
            'xzy': 'XZY',
            'z': 'ZII'}
        """
        logger.debug("Building branch operator map for TernaryTree.")

        branches = self.root_node.branch_strings

        node_indices = {
            node: qubit for node, (_, qubit) in self.enumeration_scheme.items()
        }
        branch_operator_map = {}
        for branch in branches:
            branch_operator_map[branch] = ["I"] * self.n_qubits
            node = self.root_node
            for char in branch:
                node_index = node_indices[node.root_path]
                branch_operator_map[branch][node_index] = char.upper()
                node = getattr(node, char, None)
            branch_operator_map[branch] = "".join(branch_operator_map[branch])

        return branch_operator_map

    @property
    def string_pairs(self) -> dict[str | int, tuple[str, str]]:
        """Return the pair of branch strings which correspond to each node.

        Returns:
            dict[str, tuple(str,str)]: A dictionary of all node labels, j,  with branch strings (2j, 2j+1).

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3)
            >>> tree.add_node('x')
            >>> tree.add_node('xz')
            >>> tree.string_pairs
            {'': ('xzz', 'y'), 'x': ('xx', 'xy'), 'xz': ('xzx', 'xzy')}
        """
        logger.debug("Building string pairs for TernaryTree.")
        node_set = self.root_node.child_strings

        pairs = {}
        for node_string in node_set:
            node = self.root_node
            for char in node_string:
                node = getattr(node, char)

            x_string = node_string + "x"
            y_string = node_string + "y"
            while x_string in node_set:
                x_string += "z"

            while y_string in node_set:
                y_string += "z"

            if x_string.count("y") % 2 == 0:
                pairs[node.root_path] = x_string, y_string
            elif y_string.count("y") % 2 == 0:
                pairs[node.root_path] = y_string, x_string

        return pairs

    def _build_symplectic_matrix(
        self,
    ) -> tuple[NDArray[np.uint8], NDArray[bool]]:
        """Build the symplectic matrix for the tree.

        Returns:
            NDArray[np.uint8]: Powers of i for each row of the symplectic matrix.
            NDArray[np.uint8]: Symplectic matrix.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3)
            >>> tree.add_node('x')
            >>> tree.enumeration_scheme = tree.default_enumeration_scheme()
            >>> tree._build_symplectic_matrix()
            (array([0, 1, 0, 1, 0, 1], dtype=uint8),
            array([[ True, False, False, False,  True,  True],
                    [ True, False, False,  True, False, False],
                    [ True,  True, False, False, False, False],
                    [ True,  True, False, False,  True, False],
                    [ True, False,  True, False,  True, False],
                    [ True, False,  True, False,  True,  True]]))
        """
        logger.debug("Building symplectic matrix for TernaryTree.")
        if self.enumeration_scheme is None:
            logger.error("No enumeration scheme provided, using default.")
            self.enumeration_scheme = self.default_enumeration_scheme()

        pauli_string_map = self.branch_operator_map

        symplectic = np.zeros((2 * self.n_qubits, 2 * self.n_qubits), dtype=bool)
        ipowers = np.zeros((2 * self.n_qubits), dtype=np.uint8)
        for node, operators in self.string_pairs.items():
            for offset, operator in enumerate(operators):
                operator = pauli_string_map[operator]
                operator = np.array(list(operator), dtype=str)
                # If the string is X or Y then assign 1
                term_ipower, symplectic_term = self._pauli_to_symplectic(0, operator)
                fermion_mode = self.enumeration_scheme[node][0]
                ipowers[2 * fermion_mode + offset] = term_ipower
                symplectic[2 * fermion_mode + offset] = symplectic_term
        return ipowers, symplectic

    def JordanWigner(self) -> "TernaryTree":
        """Create a new tree with the Jordan-Wigner encoding.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> jw_tree = TernaryTree(3).JordanWigner()
        """
        return JordanWigner(self.n_modes)

    def JW(self) -> "TernaryTree":
        """Alias for Jordan-Wigner encoding.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> jw_tree = TernaryTree(3).JW()
        """
        return JordanWigner(self.n_modes)

    def ParityEncoding(self) -> "TernaryTree":
        """Create a new tree with the parity encoding.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> parity_tree = TernaryTree(3).ParityEncoding()
        """
        return ParityEncoding(self.n_modes)

    def BravyiKitaev(self) -> "TernaryTree":
        """Create a new tree with the Bravyi-Kitaev encoding.

        Args:
            n_modes (int): The number of fermionic modes.

        Returns:
            TernaryTree: A ternary tree encoding.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> bk_tree = TernaryTree(3).BravyiKitaev()
        """
        return BravyiKitaev(self.n_modes)

    def BK(self) -> "TernaryTree":
        """Alias for Bravyi-Kitaev encoding.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> bk_tree = TernaryTree(3).BK()
        """
        return BravyiKitaev(self.n_modes)

    def JKMN(self) -> "TernaryTree":
        """Create a new tree with the JKMN encoding.

        The JKMN encoding gives a ternary tree with the minimum Pauli-weight.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> min_height_tree = TernaryTree(3).JKMN()
        """
        return JKMN(self.n_modes)


def JordanWigner(n_modes: int) -> TernaryTree:
    """Create a new tree with the Jordan-Wigner encoding.

    Args:
        n_modes (int): The number of fermionic modes.

    Returns:
        TernaryTree: A ternary tree encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import JordanWigner
        >>> jw_tree = JordanWigner(3)
    """
    logger.debug("Creating Jordan-Wigner encoding tree")
    new_tree = TernaryTree(
        n_modes=n_modes,
        root_node=TTNode(),
    )
    new_tree.add_node("z" * (n_modes - 1))
    new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
    return new_tree


def JW(n_modes: int) -> TernaryTree:
    """Alias for Jordan-Wigner encoding.

    Args:
        n_modes (int): The number of fermionic modes.

    Returns:
        TernaryTree: A ternary tree encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import JW
        >>> jw_tree = JW(3)
    """
    return JordanWigner(n_modes)


def ParityEncoding(n_modes: int) -> TernaryTree:
    """Create a new tree with the parity encoding.

    Args:
        n_modes (int): The number of fermionic modes.

    Returns:
        TernaryTree: A ternary tree encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import ParityEncoding
        >>> parity_tree = ParityEncoding(3)
    """
    logger.debug("Creating parity encoding tree")
    new_tree = TernaryTree(
        n_modes=n_modes,
        root_node=TTNode(),
    )
    new_tree.add_node("x" * (n_modes - 1))
    new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
    return new_tree


def BravyiKitaev(n_modes: int) -> "TernaryTree":
    """Create a new tree with the Bravyi-Kitaev encoding.

    Args:
        n_modes (int): The number of fermionic modes.

    Returns:
        TernaryTree: A ternary tree encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import BravyiKitaev
        >>> bk_tree = BravyiKitaev(3)
    """
    logger.debug("Creating Bravyi-Kitaev encoding tree")
    new_tree = TernaryTree(
        n_modes=n_modes,
        root_node=TTNode(),
    )
    branches = ["x"]
    # one is used for root, which is defined
    remaining_qubits = n_modes - 1
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


def BK(n_modes: int) -> TernaryTree:
    """Alias for Bravyi-Kitaev encoding.

    Args:
        n_modes (int): The number of fermionic modes.

    Returns:
        TernaryTree: A ternary tree encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import BK
        >>> bk_tree = BK(3)
    """
    return BravyiKitaev(n_modes)


def JKMN(n_modes: int) -> TernaryTree:
    """Create a new tree with the JKMN encoding.

    The JKMN encoding gives a ternary tree with the minimum Pauli-weight.

    Args:
        n_modes (int): The number of fermionic modes.

    Returns:
        TernaryTree: A ternary tree encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import JKMN
        >>> min_height_tree = JKMN(3)
    """
    logger.debug("Creating JKMN encoding tree.")
    new_tree = TernaryTree(
        n_modes=n_modes,
        root_node=TTNode(),
    )
    branches = ["x", "y", "z"]
    # one is used for root which is defined
    remaining_qubits = n_modes - 1
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
