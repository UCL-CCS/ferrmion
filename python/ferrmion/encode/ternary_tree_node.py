"""Contains the class for ternary tree nodes."""

import logging
from typing import Optional

import rustworkx as rx

logger = logging.getLogger(__name__)


class TTNode:
    """A node in a ternary tree.

    Attributes:
        parent (TTNode): The parent node.
        label (int | str): The qubit label.
        x (TTNode): The x child node.
        y (TTNode): The y child node.
        z (TTNode): The z child node.

    Methods:
        as_dict(): Convert the node to a dictionary.
        branch_strings(): Get the branch strings for the node.
        child_strings(): Get the child strings for the node.
        add_child(which_child, root_path): Add a child node to the current node.

    Simple Example:
        >>> from ferrmion.encode.ternary_tree_node import TTNode
        >>> node = TTNode()
        >>> node.add_child('x')
        >>> node.as_dict()
    """

    def __str__(self):
        """String representation of node."""
        return f"TTNode (root path:{self.root_path})"

    def __init__(
        self,
        parent: Optional["TTNode"] = None,
        qubit_label: int | str | None = None,
    ):
        """Initialise a ternary tree node.

        Args:
            parent (TTNode | None): The parent node.
            root_path (str | None): The path from root to this node.
            qubit_label (int | str | None): The qubit label.
        """
        logger.debug(
            f"Creating TTNode with parent {parent} and qubit label {qubit_label}"
        )
        self.root_path = ""
        self.qubit_label = qubit_label
        self.parent: TTNode | None = parent
        self.x: TTNode | None = None
        self.y: TTNode | None = None
        self.z: TTNode | None = None
        self.z_ancestor: TTNode = self
        self.leaf_majorana_indices: dict[str, int | None] = {
            "x": None,
            "y": None,
            "z": None,
        }

    # def __str__(self) -> str:
    # return f"{self.as_dict()}"

    def as_dict(self) -> dict:
        """Return the node structure as a dictionary.

        Example:
            >>> from ferrmion.encode.ternary_tree_node import TTNode
            >>> node = TTNode()
            >>> node.add_child('x')
            >>> node.as_dict()
        """
        return as_dict(self)

    @property
    def branch_strings(self) -> set[str]:
        """Return a list of all branch strings for the node.

        Example:
            >>> from ferrmion.encode.ternary_tree_node import TTNode
            >>> node = TTNode()
            >>> node.add_child('x')
            >>> node.branch_strings
        """
        return set(branch_majorana_map(self).keys())

    @property
    def branch_majorana_map(self) -> dict[str, int]:
        """Create a map from branch strings to majorana indices.

        Returns:
            dict[str,int]: A map from branch strings to majorana operator indices.

        Example:
            >>> from ferrmion.encode.ternary_tree_node import TTNode, branch_strings
            >>> node = TTNode()
            >>> node.add_child('x')
            >>> branch_majorana_map(node)
        """
        return branch_majorana_map(root_node=self)

    @property
    def child_strings(self) -> list[str]:
        """Return a list of all child strings for the node.

        Example:
            >>> from ferrmion.encode.ternary_tree_node import TTNode
            >>> node = TTNode()
            >>> node.add_child('x')
            >>> node.child_strings
        """
        return sorted(child_strings(self, prefix=""), key=node_sorter)

    @property
    def child_qubit_labels(self) -> dict[str, int | None]:
        """Return a dict of sorted child nodes and their qubit label.

        Example:
            >>> from ferrmion.encode.ternary_tree_node import TTNode
            >>> node = TTNode()
            >>> node.add_child('x', qubit_label=5)
            >>> node.child_qubit_labels
        """
        return child_qubit_labels(self)

    def update_root_path(self, prefix: str) -> None:
        """Prefix the root path of a node and all its children.

        Args:
            prefix (str): String to prefix to root paths.
        """
        return update_root_path(root=self, prefix=prefix)

    def add_child(
        self,
        which_child: str,
        child_node: Optional["TTNode"] = None,
        root_path: str | None = None,
        qubit_label: int | str | None = None,
    ) -> "TTNode":
        """Add a child node to the current node.

        Args:
            which_child (str): The child node to add.
            child_node (TTNode|None): A node object to set as the child.
            root_path (str): Path from root node.
            qubit_label (int | str): The qubit label.

        Returns:
            TTNode: The added child node

        Example:
            >>> from ferrmion.encode.ternary_tree_node import TTNode
            >>> node = TTNode()
            >>> node.add_child('x')
        """
        return add_child(
            self,
            which_child=which_child,
            child_node=child_node,
            qubit_label=qubit_label,
        )

    def to_rustworkx(self) -> rx.PyDiGraph:
        """Create a rustworkx graph from this node and its children.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(10).BK()
            >>> rx_graph = tree.root_node.to_rustworkx()
        """
        return to_rustworkx(self)


def update_root_path(root: TTNode, prefix: str) -> None:
    """Prefix the root path of a node and all its children.

    Args:
        root (TTNode): Root ternary tree node.
        prefix (str): String to prefix to root paths.
    """
    logger.debug("Prefixing node root path with %s.", prefix)
    root.root_path = prefix
    for child in ["x", "y", "z"]:
        child_node = getattr(root, child, None)
        if child_node is not None:
            child_node.update_root_path(prefix + child)


def z_descendant(ancestor: TTNode) -> TTNode:
    """Find the furthest z-descendant of a node."""
    node = ancestor
    while isinstance(node.z, TTNode):
        node = node.z
    return node


def z_ancestor(descendant: TTNode) -> TTNode:
    """Find the further z-ancestor of a node."""
    node: TTNode = descendant
    for char in descendant.root_path[::-1]:
        if char != "z":
            break
        node: TTNode = node.parent
    return node


def add_child(
    parent,
    which_child: str,
    child_node: TTNode | None = None,
    qubit_label: int | str | None = None,
) -> TTNode:
    """Add a child node to a parent node.

    Args:
        parent (TTNode): The parent node.
        which_child (str): The child node to add.
        root_path (str): Path from the root node.
        qubit_label (int | str): The qubit label.
        child_node (TTNode | None): A node to assign as child.

    Returns:
        TTNode: The added child node.

    Example:
        >>> from ferrmion.encode.ternary_tree_node import TTNode, add_child
        >>> node = TTNode()
        >>> add_child(node, 'x')
    """
    logger.debug("Adding child %s to parent %s", which_child, parent)

    if (existing_child := getattr(parent, which_child, None)) is not None:
        logger.warning(
            f"Already has child node {existing_child.root_path} at {which_child}"
        )
        return existing_child

    if child_node is None:
        logger.debug("Creating child node.")
        child_node = TTNode()
    elif isinstance(child_node, TTNode) and isinstance(child_node.parent, TTNode):
        logger.warning("Removing child node from current parent.")
        current_position = child_node.root_path[-1]
        setattr(child_node.parent, current_position, None)

    logger.debug("Setting node relationships.")
    try:
        parent.leaf_majorana_indices.pop(which_child)
    except KeyError:
        logger.debug("No leaf index to remove.")

    setattr(parent, which_child, child_node)
    child_node.parent = parent

    child_node.update_root_path(parent.root_path + which_child)

    if qubit_label is not None:
        logger.debug("Replacing child qubit label.")
        child_node.qubit_label = qubit_label

    if which_child == "z":
        child_node.z_ancestor = parent.z_ancestor

    return getattr(parent, which_child)


def as_dict(node: TTNode) -> dict[str, dict]:
    """Create a dictionary of children for a node.

    Args:
        node (TTNode): The node to convert to a dictionary.

    Returns:
        dict[str, dict]: A nested dictionary of children for the node.

    Example:
        >>> from ferrmion.encode.ternary_tree_node import TTNode, as_dict
        >>> node = TTNode()
        >>> node.add_child('x')
        >>> as_dict(node)
    """
    logger.debug("Converting node to dict %s", node)
    children: dict[str, TTNode] = {"x": node.x, "y": node.y, "z": node.z}
    for key, val in children.items():
        if val is not None:
            children[key] = as_dict(children[key])
        else:
            children[key] = node.leaf_majorana_indices[key]
    return children


def child_strings(node: TTNode, prefix: str = "") -> set[str]:
    """Create a list of all child strings for a node.

    Args:
        node (TTNode): The node to convert to a list of strings.
        prefix (str): The prefix for the string.

    Returns:
        list[str]: A list of all child strings for the node.

    Example:
        >>> from ferrmion.encode.ternary_tree_node import TTNode, child_strings
        >>> node = TTNode()
        >>> node.add_child('x')
        >>> child_strings(node)
    """
    logger.debug("Creating child strings for node %s", node)
    strings = {prefix}
    for pauli in ["x", "y", "z"]:
        child = getattr(node, pauli, None)
        if child is not None:
            strings = strings.union(child_strings(node=child, prefix=f"{prefix+pauli}"))
    logger.debug("Sorting nodes.")
    return strings


def child_qubit_labels(node: TTNode) -> dict[str, int | None]:
    """Return a dict of sorted child nodes and their qubit label.

    Example:
        >>> from ferrmion.encode.ternary_tree_node import TTNode
        >>> node = TTNode()
        >>> node.add_child('x', qubit_label=5)
        >>> node.child_qubit_labels
    """
    label_dict: dict[str, int | str | None] = {}
    for child_string in node.child_strings:
        if child_string == "":
            label_dict[""] = node.qubit_label
            continue

        child: TTNode = node
        for char in child_string:
            # we are using pre-checked strings
            # so we don't need a default
            child = getattr(child, char)
        label_dict[child_string] = child.qubit_label

    return label_dict


def branch_majorana_map(root_node: TTNode) -> dict[str, int]:
    """Create a map from branch strings to majorana indices.

    Args:
        root_node (TTNode): The node to convert to a set of strings.

    Returns:
        dict[str,int]: A map from branch strings to majorana operator indices.

    Example:
        >>> from ferrmion.encode.ternary_tree_node import TTNode, branch_strings
        >>> node = TTNode()
        >>> node.add_child('x')
        >>> branch_majorana_map(node)
    """
    logger.debug("Creating branch strings for node %s", root_node)
    branch_majorana_map = {}
    child_strings = root_node.child_strings
    # possible_leaves = set(*[child+char for char in ["x","y","z"] for child in child_strings])
    # leaves = possible_leaves.difference(child_strings)
    for child in child_strings:
        node = root_node
        for char in child:
            node = getattr(node, char)

        for char in ["x", "y", "z"]:
            if getattr(node, char, None) is None:
                branch_majorana_map[child + char] = node.leaf_majorana_indices[char]

    return branch_majorana_map


def node_sorter(label: str) -> int:
    """This is used to keep the ordring of encodings consistent.

    Args:
        label (str): The label to sort.

    Returns:
        int: Integer label to sort by.

    Example:
        >>> from ferrmion.encode.ternary_tree_node import node_sorter
        >>> node_sorter('xyz')
        123
        >>> node_sorter('xx')
        11
        >>> node_sorter('z')
        3
    """
    if label == "":
        return 0
    pauli_dict = {"x": "1", "y": "2", "z": "3"}
    return int("".join([pauli_dict[item] for item in label.lower()]))


def to_rustworkx(root: TTNode) -> rx.PyDiGraph:
    """Convert a TT node and its children to a rustworkx PyDiGraph.

    Args:
        root (TTNode): A node to be the root of the rx graph.

    Example:
        >>> from ferrmion.encode.ternary_tree import TernaryTree
        >>> tree = TernaryTree(10).BK()
        >>> rx_graph = to_rustworkx(tree.root)
    """
    graph = rx.PyDiGraph(check_cycle=True)
    child_dict = {s: i for i, s in enumerate(root.child_strings)}
    graph.add_nodes_from(child_dict)
    for string in root.child_strings:
        if len(string) == 0:
            continue
        graph.add_edge(child_dict[string[:-1]], child_dict[string], string[-1])
    return graph
