"""Graph visualisation tools."""

import rustworkx as rx
from rustworkx.visualization import mpl_draw

from ferrmion.encode.ternary_tree_node import node_sorter


def draw_tt(graph: rx.PyDiGraph, enumeration_scheme=None):
    """Draws a rustworkx graph with nodes positioned as a ternary tree.

    Args:
        graph (rustworkx.PyDiGraph): A ternary tree graph.
        enumeration_scheme (dict[str, tuple[int, int]]): A mapping from node labels to a tuple of (mode index, qubit index).

    Example:
        tree = ferrmion.encode.TernaryTree(10).BK()
        draw_tt(tree.root.to_rustworkx())
    """

    def y_pos(label) -> float:
        return -3 * len(label)

    def x_pos(label) -> float:
        return sum(
            [
                (float(val) - 2) / (3**i)
                for i, val in enumerate(list(str(node_sorter(label))))
            ]
        )

    def format_label(label):
        return rf"$f_{{{enumeration_scheme[label][0]}}}q_{{{enumeration_scheme[label][1]}}}$"

    posmap = {
        index: [x_pos(label), y_pos(label)] for index, label in enumerate(graph.nodes())
    }
    posmap[0] = [0, 0]
    labels: callable = str if enumeration_scheme is None else format_label

    mpl_draw(
        graph,
        pos=posmap,
        with_labels=True,
        node_size=600,
        node_color="orange",
        edge_labels=str,
        labels=labels,
    )
