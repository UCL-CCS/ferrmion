"""Graph visualisation tools."""

import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw

from ferrmion.encode import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode


def draw_tt(graph: rx.PyDiGraph | TTNode | TernaryTree, enumeration_scheme=None):
    """Draws a rustworkx graph with nodes positioned as a ternary tree.

    Args:
        graph (rustworkx.PyDiGraph | ferrmion.TTNode | TernaryTree): A ternary tree.
        enumeration_scheme (dict[str, tuple[int, int]]): A mapping from node labels to a tuple of (mode index, qubit index).

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
        same_len = np.array([l for l in graph.nodes() if len(l) == len(label)])
        this_pos = np.where(same_len == label)[0][0]
        pos = (this_pos + 1) / (len(same_len) + 1) - 0.5
        if len(same_len) <= 1:
            pos = len(label)

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
        node_size=600,
        node_color="orange",
        edge_labels=str,
        labels=labels,
        font_size=10,
    )
