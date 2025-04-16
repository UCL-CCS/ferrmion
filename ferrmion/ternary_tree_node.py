"""Contains the class for ternary tree nodes."""

from typing import Optional, Hashable
import logging

logger = logging.getLogger(__name__)


class TTNode:
    def __init__(
        self, parent: Optional["TTNode"] = None, qubit_label: Hashable | None = None
    ):
        """Initialise a ternary tree node.
        Args:
            parent (TTNode): The parent node.
            qubit_label (Hashable): The qubit label.
        """
        logger.debug(f"Creating TTNode with parent {parent} and qubit label {qubit_label}")
        self.parent = parent
        self.label = qubit_label
        self.x = None
        self.y = None
        self.z = None

    def __str__(self) -> str:
        return f"{self.as_dict()}"

    def as_dict(self) -> dict:
        return as_dict(self)

    @property
    def branch_strings(self) -> list[str]:
        return branch_strings(self, prefix="")

    @property
    def child_strings(self) -> list[str]:
        return child_strings(self, prefix="")

    def add_child(
        self, which_child: str, qubit_label: Hashable | None = None
    ) -> "TTNode":
        return add_child(self, which_child, qubit_label)


def add_child(parent, which_child: str, qubit_label: Hashable | None = None) -> TTNode:
    """Add a child node to a parent node.
    Args:
        parent (TTNode): The parent node.
        which_child (str): The child node to add.
        qubit_label (Hashable): The qubit label.
    Returns:
        TTNode: The parent node with the child added.
    """
    logger.debug("Adding child %s to parent %s", which_child, parent)
    if getattr(parent, which_child, None) is not None:
        logger.warning("Already has child node at %s", which_child)
        pass
    else:
        setattr(parent, which_child, TTNode(parent=parent, qubit_label=qubit_label))
    return getattr(parent, which_child)


def as_dict(node: TTNode) -> dict[str, dict]:
    """Create a dictionary of children for a node.
    Args:
        node (TTNode): The node to convert to a dictionary.
    Returns:
        dict[str, dict]: A nested dictionary of children for the node.
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
    """
    logger.debug("Creating child strings for node %s", node)
    strings = {prefix}
    for pauli in ["x", "y", "z"]:
        child = getattr(node, pauli, None)
        if child is not None:
            strings = strings.union(child_strings(node=child, prefix=f"{prefix+pauli}"))
    return sorted(strings, key=node_sorter)


def branch_strings(node: TTNode, prefix: str = "") -> set[str]:
    """Create a set of all branch strings for a node.
    Args:
        node (TTNode): The node to convert to a set of strings.
        prefix (str): The prefix for the string.
    Returns:
        set[str]: A set of all branch strings for the node.
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
    """
    logger.debug("Sorting node %s", label)
    
    if label == "":
        return 0
    pauli_dict = {"x": "1", "y": "2", "z": "3"}
    return int("".join([pauli_dict[item] for item in label]))
