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

    def __init__(
        self,
        parent: Optional["TTNode"] = None,
        root_path: str | None = None,
        qubit_label: int | str | None = None,
    ):
        """Initialise a ternary tree node.

        Args:
            parent (TTNode | None): The parent node.
            root_path (str | None): The path from root to this node.
            qubit_label (int | str | None): The qubit label.
        """
        logger.debug(
            f"Creating TTNode with parent {parent} and qubit label {root_path}"
        )
        self.parent = parent
        self.root_path = root_path
        self.qubit_label = qubit_label
        self.x = None
        self.y = None
        self.z = None

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
        return branch_strings(self, prefix="")

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
    def child_qubit_labels(self) -> dict[str, str | int | None]:
        """Return a dict of sorted child nodes and their qubit label.

        Example:
            >>> from ferrmion.encode.ternary_tree_node import TTNode
            >>> node = TTNode()
            >>> node.add_child('x', qubit_label=5)
            >>> node.child_qubit_labels
        """
        return child_qubit_labels(self)

    def prefix_root_path(self, prefix: str) -> None:
        """Prefix the root path of a node and all its children.

        Args:
            prefix (str): String to prefix to root paths.
        """
        self.root_path = f"{prefix}{self.root_path}"
        for child in ["x", "y", "z"]:
            child_node = getattr(self, child, None)
            if child_node is not None:
                child_node.prefix_root_path(prefix)

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
            root_path=root_path,
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


def add_child(
    parent,
    which_child: str,
    child_node: TTNode | None = None,
    root_path: str | None = None,
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
    if root_path is None:
        root_path = which_child

    if (child := getattr(parent, which_child, None)) is not None:
        logger.warning(f"Already has child node {child.root_path} at {which_child}")
        pass
    elif isinstance(child_node, TTNode):
        if root_path is not None:
            child_node.prefix_root_path(parent.root_path + root_path)
        if qubit_label is not None:
            child_node.qubit_label = qubit_label
        setattr(parent, which_child, child_node)
    else:
        setattr(
            parent,
            which_child,
            TTNode(parent=parent, root_path=root_path, qubit_label=qubit_label),
        )
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
    children = {"x": node.x, "y": node.y, "z": node.z}
    for key, val in children.items():
        if val is not None:
            children[key] = as_dict(children[key])
        else:
            children[key] = {}
    return children


def child_strings(node: TTNode, prefix: str = "") -> list[str]:
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


def child_qubit_labels(node: TTNode) -> dict[str, int | str | None]:
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


def branch_strings(node: TTNode, prefix: str = "") -> set[str]:
    """Create a set of all branch strings for a node.

    Args:
        node (TTNode): The node to convert to a set of strings.
        prefix (str): The prefix for the string.

    Returns:
        set[str]: A set of all branch strings for the node.

    Example:
        >>> from ferrmion.encode.ternary_tree_node import TTNode, branch_strings
        >>> node = TTNode()
        >>> node.add_child('x')
        >>> branch_strings(node)
    """
    logger.debug("Creating branch strings for node %s", node)
    strings = set()
    for pauli in ["x", "y", "z"]:
        child = getattr(node, pauli, None)
        if child is None:
            strings.add(f"{prefix+pauli}")
        else:
            strings = strings.union(
                branch_strings(node=child, prefix=f"{prefix+pauli}")
            )
    return strings


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
