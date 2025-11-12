"""Bonsai Algorithm."""

import logging

import numpy as np
import rustworkx as rx

from ferrmion import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode

logger = logging.getLogger(__name__)


def bonsai_algorithm(
    graph: rx.PyGraph, homogenous: bool = True, max_nodes: None | int = None
) -> TernaryTree:
    """Create a TernayTree encoding using the Bonsai Algorithm.

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

    while len(node_queue) > 0:
        logger.debug(f"{node_queue=}")
        logger.debug(f"{used_indices=}")
        node: int = node_queue.pop(0)
        parent = nodes[node]
        logger.debug(node)

        neighbors = sorted(list(set(graph.neighbors(node)).difference(used_indices)))
        logger.debug(f"{neighbors=}")

        n_neighbors = len(neighbors)

        for neighbor, char in zip(neighbors[:3], chars[:n_neighbors]):
            node_queue.append(neighbor)
            used_indices.add(neighbor)

            parent.add_child(
                char,
                child_node=nodes[neighbor],
                root_path=f"{parent.root_path}{char}",
                qubit_label=neighbor,
            )
            nodes[neighbor].parent = parent

            if max_nodes is not None and len(used_indices) == max_nodes:
                break

        if max_nodes is not None and len(used_indices) == max_nodes:
            break
        logger.debug(node_queue)
        logger.debug("")

    if max_nodes is not None and len(used_indices) == max_nodes:
        logger.debug(f"Found {max_nodes} nodes.")
    elif len(used_indices) == graph.num_nodes():
        logger.debug("Found spanning tree")
    else:
        logger.debug("Tree does not span the graph.")
        unused_indices = set(range(graph.num_nodes())).difference(used_indices)
        for unused in unused_indices:
            closest = np.argsort(distances[unused])
            for used in closest:
                used_node: TTNode = nodes[used]
                for child_branch in chars:
                    if getattr(used_node, child_branch) is None:
                        used_node.add_child(
                            child_branch,
                            child_node=nodes[unused],
                            root_path=f"{used_node.root_path}{child_branch}",
                            qubit_label=unused,
                        )
                        unused_indices.remove(used)
                        break

        if len(unused_indices) > 0:
            logger.debug("Error, not all qubits assigned to nodes.")
        else:
            logger.debug("All graph nodes assigned to tree.")

    logger.debug("Creating encoding.")

    n_nodes = graph.num_nodes() if max_nodes is None else max_nodes
    tree = TernaryTree(
        n_modes=n_nodes, n_qubits=graph.num_nodes(), root_node=nodes[root_index]
    )
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    return tree
