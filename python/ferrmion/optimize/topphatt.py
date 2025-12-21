"""Topology-Preserving Hamiltonian Adaptive Ternary Tree."""

import logging
from itertools import product
from typing import Callable, Iterable

import numpy as np

from ferrmion.encode import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode
from ferrmion.optimize.hatt import _reduce_hamiltonian

logger = logging.getLogger(__name__)


def _get_node(root_node: TTNode, child_string: str):
    node = root_node
    for char in child_string:
        node = getattr(node, char)
    return node


def _add_all_z_restriction(tree, string_index_map, restrictions):
    child_strings = tree.root_node.child_strings
    all_z = sorted(
        {child for child in child_strings if "x" not in child and "y" not in child},
        key=len,
        reverse=True,
    )[0]
    all_z_index = string_index_map[all_z]
    restrictions[all_z_index][2] = 2 * tree.n_modes
    return restrictions


def _add_retain_child_restrictions(tree: TernaryTree, string_index_map, restrictions):
    """Nodes that have a child."""
    child_strings = tree.root_node.child_strings
    for node in child_strings:
        if node == "":
            continue
        parent = node[:-1]
        parent_index = string_index_map[parent]
        node_index = string_index_map[node]
        match node[-1]:
            case "x":
                restrictions[parent_index][0] = node_index
            case "y":
                restrictions[parent_index][1] = node_index
            case "z":
                restrictions[parent_index][2] = node_index
    return restrictions


def _add_xy_parent_restrictions(
    tree: TernaryTree,
    string_index_map: dict[str, int],
    node_objects_map: dict[int, TTNode],
    restrictions: dict[int, list[int | None]],
):
    """Nodes that have an X-parent or Y-parent."""
    child_strings = tree.root_node.child_strings
    for child in child_strings:
        child_index = string_index_map[child]
        child_node = node_objects_map[child_index]
        # child can be it's own ancestor
        ancestor = child_node.z_ancestor

        if ancestor.root_path == "":
            continue
        # Z-ancestors of an x-child have even z-leaf
        if ancestor.root_path[-1] == "x":
            restrictions[child_index][2] = "Even"
        # Z-ancestors of an y-child have odd z-leaf
        elif ancestor.root_path[-1] == "y":
            restrictions[child_index][2] = "Odd"

    return restrictions


def _get_string_index_map(tree):
    n_leaves = 2 * tree.n_modes + 1
    enumeration_scheme = tree.default_enumeration_scheme()
    child_strings = tree.root_node.child_strings
    return {child: n_leaves + enumeration_scheme[child][1] for child in child_strings}


def _get_node_objects_map(tree, string_index_map):
    return {
        string_index_map[child]: _get_node(tree.root_node, child)
        for child in tree.root_node.child_strings
    }


def _initialise_restrictions(tree: TernaryTree):
    string_index_map = _get_string_index_map(tree)
    node_objects_map = _get_node_objects_map(tree, string_index_map)
    for node in [
        node_objects_map[string_index_map[child]]
        for child in tree.root_node.child_strings
    ]:
        node.leaf_majorana_indices = {k: None for k in node.leaf_majorana_indices}

    # each node has restrictions in tuple (restrictions on x, restrictions on y, restrictions on z)
    restrictions = {i: ["Any", "Any", "Any"] for i in string_index_map.values()}
    restrictions = _add_xy_parent_restrictions(
        tree, string_index_map, node_objects_map, restrictions
    )
    restrictions = _add_retain_child_restrictions(tree, string_index_map, restrictions)
    restrictions = _add_all_z_restriction(tree, string_index_map, restrictions)
    return restrictions


def _unpack_single_restriction(
    restriction: None | list[int] | Callable, unassigned_leaves: set
):
    match restriction:
        case "Any":
            restriction = unassigned_leaves
        case "Even":
            restriction = [i for i in unassigned_leaves if i % 2 == 0]
        case "Odd":
            restriction = [i for i in unassigned_leaves if i % 2 == 1]
        # a REQUIRED assignment will be removed from unassigned_leaves
        case int():
            restriction = [restriction]
        case _:
            raise ValueError("Restriction should be 'Any', 'Even', 'Odd' or int.")
    return restriction


def _update_restrictions(restrictions, parent_node, string_index_map):
    # The node may be its own z-ancestor
    restrictor = parent_node.z_ancestor
    logger.debug(f"{restrictor=}")

    logger.debug(f"{restrictions=}")
    if restrictor.root_path == "":
        pass
    elif restrictor.root_path[-1] == "x":
        if isinstance(restrictor.parent.y, TTNode):
            restricted = restrictor.parent.y.z_descendant
            logger.debug(f"{restricted=}")
            restrictions[string_index_map[restricted.root_path]][2] = (
                restrictor.z_descendant.leaf_majorana_indices["z"] + 1
            )
        elif restrictor.parent.y is None:
            restricted = restrictor.parent
            logger.debug(f"{restricted=}")
            restrictions[string_index_map[restricted.root_path]][1] = (
                restrictor.z_descendant.leaf_majorana_indices["z"] + 1
            )

    elif restrictor.root_path[-1] == "y":
        if isinstance(restrictor.parent.x, TTNode):
            restricted = restrictor.parent.x.z_descendant
            logger.debug(f"{restricted=}")
            restrictions[string_index_map[restricted.root_path]][2] = (
                restrictor.z_descendant.leaf_majorana_indices["z"] - 1
            )
        elif restrictor.parent.x is None:
            restricted = restrictor.parent
            logger.debug(f"{restricted=}")
            restrictions[string_index_map[restricted.root_path]][0] = (
                restrictor.z_descendant.leaf_majorana_indices["z"] - 1
            )
    logger.debug(f"{restrictions=}")
    return restrictions


def _initialise_node_dependencies(tree: TernaryTree):
    string_index_map = _get_string_index_map(tree)
    child_strings = tree.root_node.child_strings
    dependencies = {string_index_map[child]: [] for child in child_strings}

    for child in child_strings:
        for char in ["x", "y", "z"]:
            if child + char in child_strings:
                dependencies[string_index_map[child]].append(
                    string_index_map[child + char]
                )
    return dependencies


def _initialise_active_nodes(index_string_map, node_dependencies):
    active_nodes: dict[int, set[TTNode]] = {}
    for node_index, deps in node_dependencies.items():
        root_len = len(index_string_map[node_index])
        if deps != []:
            continue
        if active_nodes.get(root_len, None) is None:
            active_nodes[root_len] = set()
        active_nodes[root_len].add(node_index)
    return active_nodes


def _update_active_nodes(
    active_nodes, min_parent, index_string_map, node_dependencies, completed_nodes
):
    active_nodes[root_len := len(index_string_map[min_parent])].remove(min_parent)
    if active_nodes[root_len] == set():
        active_nodes.pop(root_len)

    for node, deps in node_dependencies.items():
        if completed_nodes.issuperset(deps):
            root_len = len(index_string_map[node])
            if active_nodes.get(root_len, None) is None:
                active_nodes[root_len] = set()
            active_nodes[root_len].add(node)
    return active_nodes


def _build_valid_combination(
    comb: tuple[int, int],
    descendant_map: dict,
    ancestor_map: dict,
    n_modes: int,
    selection: list[int, int, int],
) -> None | list[int, int, int]:
    # print(f"{comb=}")
    z_index, x_index = comb
    if z_index == x_index:
        return None

    small_y = None
    small_x = None

    small_x = descendant_map[x_index]

    # discard this combination
    if small_x == 2 * n_modes:
        return None

    if small_x % 2 == 0:
        small_y = small_x + 1
    else:
        small_y = small_x - 1
    # We can't use this index for y a
    # it has been used in the combination already
    # so we'd be replacing our x or z!
    if small_y in comb:
        return None

    y_index = ancestor_map[small_y]

    if y_index in comb:
        return None

    if small_x % 2 == 0:
        comb = [x_index, y_index, z_index]
    else:
        comb = [y_index, x_index, z_index]

    # W3 can end uop with the same combination in two different ways
    if comb == selection:
        return None
    return comb


def topphatt(
    majorana_ham: dict[Iterable[int], float],
    tree: TernaryTree,
) -> TernaryTree:
    """Construct an adaptive ternary tree from a majorana Hamiltonian.

    Args:
        majorana_ham (dict[tuple[int,...],float]): Majorana Hamiltonian to encode.
        tree (TernaryTree): The TernaryTree encoding to be optimised
    Returns:
        TernaryTree: Optimised Ternary Tree encoding.
    """
    logger.debug(f"Number of Hamiltonian terms {len(majorana_ham)}")
    n_modes = tree.n_modes
    n_leaves = 2 * n_modes + 1
    # We need 2*M +1 leaves and M nodes.
    nodes: dict[int, TTNode | None] = {i: None for i in range(n_leaves - 1)}
    node_dependencies = _initialise_node_dependencies(tree)  # O(N)
    logging.debug(f"Initial Dependencies:\n{node_dependencies}")

    string_index_map = _get_string_index_map(tree)  # O(N)
    index_string_map = {v: k for k, v in string_index_map.items()}

    node_objects_map = _get_node_objects_map(tree, string_index_map)  # O(N)

    for node in node_objects_map.values():  # O(N)
        for char in ["x", "y", "z"]:
            node.branch_majorana_map[char] = None

    nodes.update(_get_node_objects_map(tree, string_index_map))

    active_nodes: dict[int, set[TTNode]] = _initialise_active_nodes(  # O(N)
        index_string_map=index_string_map, node_dependencies=node_dependencies
    )

    # active_nodes:set[int] = {node for node, deps in node_dependencies.items() if deps == []}
    completed_nodes = set()
    restrictions = _initialise_restrictions(tree)  # O(N)

    logger.debug(f"Initial Restrictions:\n{restrictions}")
    # Start with all the leaves unassigned
    unassigned_leaves = [*range(n_leaves)]
    unassigned_leaves.reverse()

    # We create two maps, of z_ancestors and z_descendants
    ancestor_map = {i: i for i in range(n_leaves + n_modes)}
    descendant_map = {i: i for i in range(n_leaves + n_modes)}

    total_weight = 0
    for i in range(n_modes + 1):  # O(N)
        logging.debug(f"\nLoop {i}")
        # # Update the restrictions with the new information about the tree.
        # # Any nodes that are required to be in a certain position
        # # have to be removed from unassigned!
        to_remove = []
        for restriction in restrictions.values():  # O(N2)
            for term in restriction:
                if isinstance(term, int):
                    to_remove.append(term)
        unassigned_leaves = [l for l in unassigned_leaves if l not in to_remove]
        logger.debug(f"{unassigned_leaves=}")

        min_weight = np.inf
        min_parent = None
        selection = [None, None, None]

        # NOTE
        # Another opion would be to check all allowed
        # combinations just once and then to look
        # for the parent that is eligable for the minimum combination.
        logging.debug(f"{active_nodes=}")
        max_len_active = active_nodes[max([*active_nodes.keys()])]
        logging.debug(f"{max_len_active=}")
        unique_restrictions = set()
        max_len_active = sorted([index_string_map[i] for i in max_len_active])
        for parent_string in max_len_active:  # O ()
            parent_index = string_index_map[parent_string]

            logging.debug(f"\n{parent_index=}")

            # Z-child of new node will always be the previous node.
            # We only need to use every second entry in unassigned
            # as we already order odd/even terms
            # we also know for jw that every new node will be
            # the z-ancestor all all other nodes.

            parent_restrictions = restrictions[parent_index]

            if ("Any", "Any", "Any") in unique_restrictions:
                continue
            elif tuple(parent_restrictions) in unique_restrictions:
                continue
            else:
                unique_restrictions.add(tuple(parent_restrictions))

            logger.debug(f"{unique_restrictions=}")

            allowed_x = _unpack_single_restriction(
                parent_restrictions[0], unassigned_leaves
            )
            allowed_y = _unpack_single_restriction(
                parent_restrictions[1], unassigned_leaves
            )
            allowed_z = _unpack_single_restriction(
                parent_restrictions[2], unassigned_leaves
            )
            logger.debug(f"{allowed_x=}")
            logger.debug(f"{allowed_y=}")
            logger.debug(f"{allowed_z=}")

            match parent_restrictions[0], parent_restrictions[1]:
                case ("Any", "Any"):
                    allowed_product = product(allowed_z, allowed_x)
                case (x, "Any") if x != "Any":
                    allowed_product = product(allowed_z, allowed_x)
                case ("Any", y) if y != "Any":
                    allowed_product = product(allowed_z, allowed_y)
                case (x, y) if "Any" not in [x, y]:
                    allowed_product = product(allowed_x, allowed_y, allowed_z)

            # If x and y are both none, just do all x and create a pair
            # if x is none and y is set, just use y
            # if x is set and y is none, just use x

            logging.debug(f"{parent_restrictions=}")
            for comb in allowed_product:  # O(n choose 2)
                match len(comb):
                    case 2:
                        comb = _build_valid_combination(
                            comb, descendant_map, ancestor_map, n_modes, selection
                        )
                        if comb is None:
                            continue
                    case 3:
                        # We can use the combination provided
                        comb = list(comb)
                    case _:
                        raise ValueError("Length of combination should be 2 or 3.")

                weight = 0
                for key in majorana_ham.keys():  # O(M terms)
                    if min(comb) > max(key):
                        continue
                    elif max(comb) < min(key):
                        continue
                    else:
                        odd_parity_paulis = [
                            sum([t == c for t in key]) % 2
                            for c in comb  # O(k term length)
                        ]
                        non_commuting = sum(odd_parity_paulis) % 3
                        weight += int(non_commuting != 0)
                        if weight > min_weight:
                            break

                if weight < min_weight:
                    logging.debug(
                        f"NEW Min Node:{i}, Parent Index: {parent_index}, Comb: {comb}, Old Min:{min_weight}, New Min:{weight}"
                    )
                    min_weight = weight
                    selection = comb
                    min_parent = parent_index
                # elif weight == min_weight:
                # #     # logging.debug(f"SAME Min Node:{i}, Parent Index: {parent_index}, Comb: {comb}, Old Min:{min_weight }, New Min:{weight}")
                # min_weight = weight
                # selection = comb
                # min_parent = parent_index
            logging.debug(f"{parent_index=}, {selection=}")
        total_weight += min_weight
        logging.debug(f"{min_parent=}")
        # Now find the Y pair of the x-node
        unassigned_leaves = [u for u in unassigned_leaves if u not in selection]
        for child_index, char in zip(selection, ["x", "y", "z"]):
            maybe_node = node_objects_map.get(child_index, None)
            if isinstance(maybe_node, TTNode):
                continue
            node_objects_map[min_parent].leaf_majorana_indices[char] = child_index

        if i + 1 == n_modes:
            break

        logging.debug(
            f"Loop: {i}, Selection: {selection}, Chosen Parent: {min_parent},{index_string_map[min_parent]}"
        )

        completed_nodes.add(min_parent)
        node_dependencies.pop(min_parent)

        active_nodes = _update_active_nodes(
            active_nodes=active_nodes,
            min_parent=min_parent,
            index_string_map=index_string_map,
            node_dependencies=node_dependencies,
            completed_nodes=completed_nodes,
        )

        z_index = selection[2]

        # Update the restrictions on the parent node to be its selection.
        restrictions[min_parent] = selection
        # Use these to update restrictions on other nodes.

        parent_node = node_objects_map[min_parent]
        restrictions = _update_restrictions(
            restrictions, parent_node=parent_node, string_index_map=string_index_map
        )

        z_desc = descendant_map[z_index]
        descendant_map[parent_index] = z_desc
        ancestor_map[z_index] = parent_index
        ancestor_map[z_desc] = parent_index

        majorana_ham = _reduce_hamiltonian(majorana_ham, parent_index, selection)
        logger.debug(f"{majorana_ham=}")

    if len(active_nodes) != 1:
        raise ValueError(f"Not all nodes assigned by HATT. {active_nodes=}")

    tree.pauli_weight = total_weight
    return tree
