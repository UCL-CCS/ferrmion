"""Graph visualisation tools."""

import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw

from ferrmion.encode import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode, node_sorter


def draw_tt(
    graph: rx.PyDiGraph | TTNode | TernaryTree,
    type: str = "spaced",
    enumeration_scheme=None,
    node_size=600,
    font_size=10,
):
    """Draws a rustworkx graph with nodes positioned as a ternary tree.

    Args:
        graph (rustworkx.PyDiGraph | ferrmion.TTNode | TernaryTree): A ternary tree.
        type (str): Make the graph prettier, one of "standard", "spaced", "linear".
        enumeration_scheme (dict[str, tuple[int, int]]): A mapping from node labels to a tuple of (mode index, qubit index).
        node_size (int): Size of nodes in the image.
        font_size (int): Size of node and edge labels in the image.

    Example:
        >>> from ferrmion.encode.ternary_tree import TernaryTree
        >>> from ferrmion.visualise.graph import draw_tt
        >>> tree = TernaryTree(3).Parity()
        >>> draw_tt(tree)
        >>> draw_tt(tree.root)
        >>> draw_tt(tree.root_node.to_rustworkx())
    """
    if isinstance(graph, TTNode):
        graph = graph.to_rustworkx()
    elif isinstance(graph, TernaryTree):
        graph = graph.root_node.to_rustworkx()

    def y_pos(label) -> float:
        return -3 * len(label)

    def x_pos(label) -> float:
        if type == "standard":
            pos = sum(
                [
                    (float(val) - 2) / (3**i)
                    for i, val in enumerate(list(str(node_sorter(label))))
                ]
            )
            pos = pos * len(label)

        elif type == "spaced":
            same_len = np.array([l for l in graph.nodes() if len(l) == len(label)])
            this_pos = np.where(same_len == label)[0][0]
            pos = (this_pos + 1) / (len(same_len) + 1) - 0.5

        elif type == "linear":
            same_len = np.array([l for l in graph.nodes() if len(l) == len(label)])
            this_pos = np.where(same_len == label)[0][0]
            pos = (this_pos + 1) / (len(same_len) + 1) - 0.5
            if len(same_len) <= 1:
                pos = len(label)
            if "z" not in label:
                pos *= -1.0
        else:
            raise ValueError("Type must be one of standard,spaced or linear.")
        return pos

    def format_label(label):
        return rf"$f_{{{enumeration_scheme[label][0]}}}q_{{{enumeration_scheme[label][1]}}}$"

    labels: callable = str if enumeration_scheme is None else format_label
    posmap = {
        index: [x_pos(label), y_pos(label)] for index, label in enumerate(graph.nodes())
    }
    posmap[0] = [0, 0]

    mpl_draw(
        graph,
        pos=posmap,
        with_labels=True,
        node_size=node_size,
        node_color="orange",
        edge_labels=str,
        labels=labels,
        font_size=font_size,
    )
