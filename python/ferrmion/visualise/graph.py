"""Graph visualisation tools."""

from typing import Mapping

import rustworkx as rx
from rustworkx.visualization import mpl_draw

from ferrmion.encode.ternary_tree_node import node_sorter


def draw_tt(graph: rx.PyDiGraph) -> None:
    """Draws a rustworkx graph with nodes positioned as a ternary tree.

    Args:
        graph (rustworkx.PyDiGraph): A ternary tree graph.

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

    posmap: Mapping[int, tuple[float, float]] = {
        index: (x_pos(label), y_pos(label)) for index, label in enumerate(graph.nodes())
    }
    # set the root node to the origin
    posmap[0] = (0, 0)

    mpl_draw(
        graph,
        pos=posmap,
        with_labels=True,
        node_size=600,
        node_color="teal",
        edge_labels=str,
    )
