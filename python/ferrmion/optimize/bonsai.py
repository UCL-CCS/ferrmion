"""Bonsai Algorithm."""

import logging

import numpy as np
import rustworkx as rx

from ferrmion import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode

logger = logging.getLogger(__name__)


def bonsai_algorithm(
    graph: rx.PyGraph,
    homogenous: bool = False,
    max_nodes: None | int = None,
) -> TernaryTree:
    """Create a TernaryTree encoding using the Bonsai Algorithm.

    Args:
        graph (rx.PyGraph): A RustworkX graph of device qubit-connectivity.
        homogenous (bool): "homogenous" labelling if true, else "heterogenous"
        max_nodes (int): Maximum number of nodes to include in output tree.

    Returns:
        TernaryTree: A ternary tree encoding.
    """
    logger.debug("Starting Bonsai Algorithm.")
    if homogenous:
        chars = ["x", "y", "z"]
    else:
        chars = ["z", "x", "y"]

    distances = rx.distance_matrix(graph)
    root_index = int(np.argmin(np.max(distances, axis=1)))
    node_queue = [root_index]
    used_indices = {root_index}
    nodes = [TTNode(parent=None) for _ in range(graph.num_nodes())]
    nodes[root_index].root_path = ""
    nodes[root_index].qubit_label = root_index

    def limit_reached() -> bool:
        return max_nodes is not None and len(used_indices) >= max_nodes

    while len(node_queue) > 0 and not limit_reached():
        logger.debug(f"{node_queue=}")
        logger.debug(f"{used_indices=}")
        node: int = node_queue.pop(0)
        parent = nodes[node]
        logger.debug(node)

        neighbors = sorted(list(set(graph.neighbors(node)).difference(used_indices)))
        logger.debug(f"{neighbors=}")

        for neighbor, char in zip(neighbors[:3], chars):
            if limit_reached():
                break
            node_queue.append(neighbor)
            used_indices.add(neighbor)

            parent.add_child(
                char,
                child_node=nodes[neighbor],
                root_path=f"{parent.root_path}{char}",
                qubit_label=neighbor,
            )
            nodes[neighbor].parent = parent

        logger.debug(node_queue)
        logger.debug("")

    if limit_reached():
        logger.debug(f"Found {max_nodes} nodes.")
    elif len(used_indices) == graph.num_nodes():
        logger.debug("Found spanning tree")
    else:
        logger.debug("Tree does not span the graph.")
        for unused in sorted(set(range(graph.num_nodes())).difference(used_indices)):
            if limit_reached():
                break
            # Attach to the closest node already in the tree with a free branch.
            for used in np.argsort(distances[unused], kind="stable"):
                used = int(used)
                if used not in used_indices:
                    continue
                used_node: TTNode = nodes[used]
                free_branch = next(
                    (c for c in chars if getattr(used_node, c) is None), None
                )
                if free_branch is None:
                    continue
                used_node.add_child(
                    free_branch, child_node=nodes[unused], qubit_label=unused
                )
                used_indices.add(unused)
                break

        if len(used_indices) < graph.num_nodes() and not limit_reached():
            logger.debug("Error, not all qubits assigned to nodes.")
        else:
            logger.debug("All graph nodes assigned to tree.")

    logger.debug("Creating encoding.")

    n_nodes = graph.num_nodes() if max_nodes is None else max_nodes
    tree = TernaryTree(
        n_modes=n_nodes, n_qubits=graph.num_nodes(), root_node=nodes[root_index]
    )
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    tree.build_encoding()
    return tree
