"""Code to Geneate Hamiltonian Adaptive Ternary Tree from Majorana Hamiltonian."""

from itertools import permutations
from typing import Iterable

import numpy as np

from ferrmion.encode import TernaryTree
from ferrmion.encode.ternary_tree_node import TTNode


def _qubit_term_weight(term: Iterable, comb: tuple[int, int, int]) -> int:
    """Find the single-qubit Pauli-weight of majorana terms.

    If any pauli term is found an even number f times, we obtain I, weight = 0.
    If we find all three pauli terms, return I (with an imaginary ccoefficient), weight = 0
    If we find either one pauli or two then the weight = 1.

    Args:
        term (Iterable): Indices of term in our majorana-hamiltonian.
        comb (tuple[int, int, int]): Combination of indices to weigh (x,y,z).

    Returns:
        int: Weight of the term.
    """
    odd_parity_paulis = [sum([t != c for t in term]) % 2 for c in comb]
    non_commuting = sum(odd_parity_paulis) % 3
    return int(non_commuting != 0)


def _reduce_hamiltonian(
    majorana_ham: dict[Iterable[int], float],
    parent_index: int,
    selection: tuple[int, int, int],
) -> dict[tuple[int, ...], float]:
    """Simplify the Hamiltonian.

    As we increase the qubit number, we iteratively remove majoranas
    which act trivially on the remaining qubits.
    We replace them with the index of their parent string
    as going forward they are identical to the parent string.

    Args:
        majorana_ham (dict[tuple[int,...],float]): Current Hamiltonian.
        parent_index (int): Qubit index of the parent node.
        selection (tuple[int, int, int]): Indices of the majoranas to be replaced.

    Returns:
        dict[tuple[int,...],float]: Reduced Hamiltonian.
    """
    new_ham = {}
    for term, coeff in majorana_ham.items():
        # new_term = tuple(i if i not in selection else parent_index for i in term)
        new_term = tuple(i for i in term if i not in selection) + tuple(
            parent_index for i in term if i in selection
        )
        if len(set(new_term)) > 1:
            new_ham[new_term] = new_ham.get(new_term, 0) + coeff
    return new_ham


def hamiltonian_adaptive_ternary_tree(
    majorana_ham: dict[Iterable[int], float], n_modes: int
) -> TernaryTree:
    """Construct an adaptive ternary tree from a majorana Hamiltonian.

    Args:
        majorana_ham (dict[tuple[int,...],float]): Majorana Hamiltonian to encode.
        n_modes (int): Number of fermionic modes in the system.

    Returns:
        TTNode: Root node of the constructed ternary tree.
    """
    # We need 2*M +1 leaves and M nodes.
    nodes: dict[int, TTNode | None] = {i: None for i in range(2 * n_modes + 1)}
    for i in range(n_modes):
        nodes[2 * n_modes + 1 + i] = TTNode(qubit_label=i)

    # Start with all the leaves unassigned
    unassigned = {*range(2 * n_modes + 1)}

    # We create two maps, of z_ancestors and z_descendants
    ancestor_map = {i: i for i in nodes}
    descendant_map = {i: i for i in nodes}

    total_weight = 0
    for i in range(n_modes):
        parent_index = 2 * n_modes + 1 + i
        parent = nodes[parent_index]

        min = np.inf
        for comb in permutations(unassigned, 2):
            small_y = None
            small_x = None
            # This way x index will be higher term - more often node.
            # z_index, x_index= comb
            x_index, z_index = comb
            small_x = descendant_map[x_index]

            # discard this combination
            if small_x == 2 * n_modes:
                continue

            if small_x % 2 == 0:
                small_y = small_x + 1
            else:
                small_y = small_x - 1
            # We can't use this index for y a
            # it has been used in the combination already
            # so we'd be replacing our x or z!
            if small_y in comb:
                continue

            y_index = ancestor_map[small_y]

            if y_index in comb:
                continue

            if small_x % 2 == 0:
                comb = np.array([x_index, y_index, z_index], dtype=np.uint)
            else:
                comb = np.array([y_index, x_index, z_index], dtype=np.uint)
            comb = [int(i) for i in comb]
            weight = np.sum(
                [_qubit_term_weight(term, comb) for term in majorana_ham.keys()]
            )
            if weight < min:
                min = weight
                selection = comb
            # would be better to break on zero
            # if weight == 0:
            #     break

        total_weight += min
        # Now find the Y pair of the x-node
        for i, char in zip(selection, ["x", "y", "z"]):
            if i in unassigned:
                unassigned.remove(i)

            if isinstance(nodes.get(i, None), TTNode):
                parent.add_child(which_child=char, child_node=nodes.get(i))
            else:
                parent.leaf_majorana_indices[char] = i

        z_index = selection[2]
        z_desc = descendant_map[z_index]
        descendant_map[parent_index] = z_desc
        ancestor_map[z_index] = parent_index
        ancestor_map[z_desc] = parent_index

        unassigned.add(parent_index)

        majorana_ham = _reduce_hamiltonian(majorana_ham, parent_index, selection)

    if len(unassigned) != 1:
        raise ValueError("Not all nodes assigned by HATT.")

    last_node = nodes[unassigned.pop()]
    if isinstance(last_node, TTNode):
        root = last_node
    else:
        raise ValueError("Hatt root node is not a TTNode object.")

    tree = TernaryTree(n_modes=n_modes, root_node=root)
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    tree.pauli_weight = total_weight
    return tree


def fast_hatt(
    majorana_ham: dict[Iterable[int], float],
    n_modes: int,
    coeff_weight: bool = False,
) -> TernaryTree:
    """Construct an adaptive ternary tree from a majorana Hamiltonian.

    Args:
        majorana_ham (dict[tuple[int,...],float]): Majorana Hamiltonian to encode.
        n_modes (int): Number of fermionic modes in the system.
        coeff_weight (bool): Multiply Pauli-weight by coefficient norm.

    Returns:
        TTNode: Root node of the constructed ternary tree.
    """
    # if coeff_weight:
    # majorana_ham = {k:v for k,v in sorted(majorana_ham.items(), key=lambda item: sum(item[0])*item[1], reverse=True)}
    # else:
    # majorana_ham = {k:v for k,v in sorted(majorana_ham.items(), key=lambda item: sum(item[0]), reverse=True)}

    n_leaves = 2 * n_modes + 1
    # We need 2*M +1 leaves and M nodes.
    nodes: dict[int, TTNode | None] = {i: None for i in range(n_leaves)}
    for i in range(n_modes):
        nodes[n_leaves + i] = TTNode(qubit_label=i)

    # Start with all the leaves unassigned
    unassigned = [*range(n_leaves)]
    unassigned.reverse()

    # We create two maps, of z_ancestors and z_descendants
    ancestor_map = {i: i for i in range(n_leaves + n_modes)}
    descendant_map = {i: i for i in range(n_leaves + n_modes)}

    total_weight = 0
    for i in range(n_modes + 1):
        parent_index = n_leaves + i
        parent = nodes[parent_index]

        min_weight = np.inf
        selection = [None, None, None]
        previous = [None, None, None]
        # reverse because best is usually at the end
        # reversed_combinations = [c for c in combinations(unassigned, 2)]
        if i == 0:
            comb_iterator = ([n_leaves - 1, i] for i in range(n_leaves - 1)[::-1])
        else:
            comb_iterator = permutations(unassigned, 2)

        for comb in comb_iterator:
            small_y = None
            small_x = None
            # This way x index will be higher term - more often node.
            z_index, x_index = comb
            # x_index, z_index = comb

            small_x = descendant_map[x_index]

            # discard this combination
            if small_x == 2 * n_modes:
                # print("small x is all z")
                continue

            if small_x % 2 == 0:
                small_y = small_x + 1
            else:
                small_y = small_x - 1
            # We can't use this index for y a
            # it has been used in the combination already
            # so we'd be replacing our x or z!
            if small_y in comb:
                # print("small_y in comb")
                continue

            y_index = ancestor_map[small_y]

            if y_index in comb:
                # print("y index in comb")
                continue

            if small_x % 2 == 0:
                comb = [x_index, y_index, z_index]
            else:
                comb = [y_index, x_index, z_index]
            # comb = [int(i) for i in comb]

            # W3 can end uop with the same combination in two different ways
            if comb == selection:
                continue
            if comb == previous:
                continue
            previous = comb
            # tc = tuple(comb)
            # if tc in checked:
            #     continue
            # checked.add(tc)

            weight = 0
            if coeff_weight:
                for key, val in majorana_ham.items():
                    if min(comb) > key[-1]:
                        continue
                    elif max(comb) < key[0]:
                        continue
                    else:
                        odd_parity_paulis = [
                            sum([t != c for t in key]) % 2 for c in comb
                        ]
                        non_commuting = sum(odd_parity_paulis) % 3
                        weight += int(non_commuting != 0)
                        weight *= abs(val)
                    if weight > min_weight:
                        break
            else:
                for key in majorana_ham.keys():
                    if min(comb) > key[-1]:
                        continue
                    elif max(comb) < key[0]:
                        continue
                    else:
                        odd_parity_paulis = [
                            sum([t != c for t in key]) % 2 for c in comb
                        ]
                        non_commuting = sum(odd_parity_paulis) % 3
                        weight += int(non_commuting != 0)
                        if weight > min_weight:
                            break
                # weight = np.sum(
                #     [_qubit_term_weight(term, comb) for term in majorana_ham.keys()]
                # )
            if weight < min_weight:
                # print(f"NEW Min Node:{i}, Parent Index: {parent_index}, Comb: {comb}, Old Min:{min_weight }, New Min:{weight}")
                min_weight = weight
                selection = comb
            elif weight == min_weight:
                #     # print(f"SAME Min Node:{i}, Parent Index: {parent_index}, Comb: {comb}, Old Min:{min_weight }, New Min:{weight}")
                min_weight = weight
                selection = comb
            # would be better to break on zero
            # if weight == 0:
            #     break

        total_weight += min_weight
        # print(f"{selection=}")
        # Now find the Y pair of the x-node
        unassigned = [u for u in unassigned if u not in selection]
        for child_index, char in zip(selection, ["x", "y", "z"]):
            if isinstance(nodes.get(child_index, None), TTNode):
                # print(f"{child_index} {nodes[child_index]}")
                # print(f"Child node {char=} with index {child_index=} is node {nodes.get(child_index).qubit_label=}")
                parent.add_child(which_child=char, child_node=nodes.get(child_index))
            else:
                parent.leaf_majorana_indices[char] = child_index

        # unassigned.append(parent_index)
        unassigned = [parent_index] + unassigned

        if i + 1 == n_modes:
            break

        z_index = selection[2]
        z_desc = descendant_map[z_index]
        descendant_map[parent_index] = z_desc
        ancestor_map[z_index] = parent_index
        ancestor_map[z_desc] = parent_index

        # print("START reducing hamiltonian")
        majorana_ham = _reduce_hamiltonian(majorana_ham, parent_index, selection)
        # print("STOP reducing hamiltonian")

    if len(unassigned) != 1:
        raise ValueError(f"Not all nodes assigned by HATT. {unassigned=}")

    last_node = nodes[unassigned[0]]
    if isinstance(last_node, TTNode):
        root = last_node
    else:
        raise ValueError("Hatt root node is not a TTNode object.")

    tree = TernaryTree(n_modes=n_modes, root_node=root)
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    tree.pauli_weight = total_weight
    print("Total Weight: ", total_weight)
    return tree


from itertools import product
from typing import Callable

from ferrmion.encode import TernaryTree


def build_valid_combination(
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


def get_node(root_node: TTNode, child_string: str):
    node = root_node
    for char in child_string:
        node = getattr(node, char)
    return node


def add_all_z_restriction(tree, string_index_map, restrictions):
    child_strings = tree.root_node.child_strings
    all_z = sorted(
        {child for child in child_strings if "x" not in child and "y" not in child},
        key=len,
        reverse=True,
    )[0]
    all_z_index = string_index_map[all_z]
    restrictions[all_z_index][2] = [2 * tree.n_modes]
    return restrictions


def majorana_indices_of_node(node_index, n_leaves) -> int:
    x_majorana = 2 * (node_index - n_leaves)
    y_majorana = x_majorana + 1
    return x_majorana, y_majorana


def get_twinned_majorana_index(node: TTNode):
    index = node.leaf_majorana_indices["z"]
    if index is None:
        return None
    if index % 2 == 0:
        return index + 1
    else:
        return index - 1


def add_retain_child_restrictions(tree: TernaryTree, string_index_map, restrictions):
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
                restrictions[parent_index][0] = [node_index]
            case "y":
                restrictions[parent_index][1] = [node_index]
            case "z":
                restrictions[parent_index][2] = [node_index]
    return restrictions


def add_xy_parent_restrictions(
    tree: TernaryTree,
    string_index_map: dict[str, int],
    node_objects_map: dict[int, TTNode],
    restrictions: dict[int, list[int | None]],
):
    """Nodes that have an X-parent or Y-parent."""
    child_strings = tree.root_node.child_strings
    parent_strings = {child[:-1] for child in child_strings}
    n_leaves = 2 * tree.n_modes
    for child in child_strings:
        child_index = string_index_map[child]
        child_node = node_objects_map[child_index]
        # child can be it's own ancestor
        ancestor = child_node.z_ancestor

        if ancestor.root_path == "":
            continue
        # Z-ancestors of an x-child have even z-leaf
        if ancestor.root_path[-1] == "x":
            restrictions[child_index][2] = [*range(0, n_leaves, 2)]
        # Z-ancestors of an y-child have odd z-leaf
        elif ancestor.root_path[-1] == "y":
            restrictions[child_index][2] = [*range(1, n_leaves, 2)]

    return restrictions


def get_string_index_map(tree):
    n_leaves = 2 * tree.n_modes + 1
    enumeration_scheme = tree.default_enumeration_scheme()
    child_strings = tree.root_node.child_strings
    return {child: n_leaves + enumeration_scheme[child][1] for child in child_strings}


def get_node_objects_map(tree, string_index_map):
    return {
        string_index_map[child]: get_node(tree.root_node, child)
        for child in tree.root_node.child_strings
    }


def initialise_restrictions(tree: TernaryTree):
    string_index_map = get_string_index_map(tree)
    node_objects_map = get_node_objects_map(tree, string_index_map)
    # each node has restrictions in tuple (restrictions on x, restrictions on y, restrictions on z)
    restrictions = {i: [None, None, None] for i in string_index_map.values()}
    restrictions = add_xy_parent_restrictions(
        tree, string_index_map, node_objects_map, restrictions
    )
    restrictions = add_retain_child_restrictions(tree, string_index_map, restrictions)
    restrictions = add_all_z_restriction(tree, string_index_map, restrictions)
    return restrictions


def unpack_single_restriction(
    restriction: None | list[int] | Callable, unassigned_leaves: set
):
    if restriction is None:
        restriction = unassigned_leaves
    # a REQUIRED assignment will be removed from unassigned_leaves
    elif isinstance(restriction, list) and len(restriction) == 1:
        restriction = restriction
    elif isinstance(restriction, list):
        restriction = [r for r in restriction if r in unassigned_leaves]
    else:
        raise ValueError("Restriction should be None or list.")
    return restriction


from ferrmion.encode import BK


def initialise_node_dependencies(tree: TernaryTree):
    string_index_map = get_string_index_map(tree)
    node_objects_map = get_node_objects_map(tree, string_index_map)
    child_strings = tree.root_node.child_strings
    dependencies = {string_index_map[child]: [] for child in child_strings}

    for child in child_strings:
        for char in ["x", "y", "z"]:
            if child + char in child_strings:
                dependencies[string_index_map[child]].append(
                    string_index_map[child + char]
                )
    return dependencies


initialise_node_dependencies(BK(4))


def topp_hatt(
    majorana_ham: dict[Iterable[int], float],
    tree: TernaryTree,
    coeff_weight: bool = False,
) -> TernaryTree:
    """Construct an adaptive ternary tree from a majorana Hamiltonian.

    Args:
        majorana_ham (dict[tuple[int,...],float]): Majorana Hamiltonian to encode.
        n_modes (int): Number of fermionic modes in the system.

    Returns:
        TTNode: Root node of the constructed ternary tree.
    """
    # if coeff_weight:
    # majorana_ham = {k:v for k,v in sorted(majorana_ham.items(), key=lambda item: sum(item[0])*item[1], reverse=True)}
    # else:
    # majorana_ham = {k:v for k,v in sorted(majorana_ham.items(), key=lambda item: sum(item[0]), reverse=True)}
    n_modes = tree.n_modes
    n_leaves = 2 * n_modes + 1
    # We need 2*M +1 leaves and M nodes.
    nodes: dict[int, TTNode | None] = {i: None for i in range(n_leaves - 1)}
    node_dependencies = initialise_node_dependencies(tree)
    print(node_dependencies)

    string_index_map = get_string_index_map(tree)
    index_string_map = {v: k for k, v in string_index_map.items()}
    node_objects_map = get_node_objects_map(tree, string_index_map)

    nodes.update(get_node_objects_map(tree, string_index_map))

    active_nodes: set[int] = {
        node for node, deps in node_dependencies.items() if deps == []
    }
    completed_nodes = set()
    print(f"{active_nodes=}")
    restrictions = initialise_restrictions(tree)
    print(f"{restrictions=}")
    # Start with all the leaves unassigned
    unassigned_leaves = [*range(n_leaves)]
    unassigned_leaves.reverse()

    # We create two maps, of z_ancestors and z_descendants
    ancestor_map = {i: i for i in range(n_leaves + n_modes)}
    descendant_map = {i: i for i in range(n_leaves + n_modes)}

    total_weight = 0
    for i in range(n_modes + 1):
        print(f"\n Iteration {i}")
        print(f"{restrictions=}")
        print(f"{active_nodes=}")

        # Update the restrictions with the new information about the tree.
        # Any nodes that are required to be in a certain position
        # have to be removed from unassigned!
        to_remove = []
        for restriction in restrictions.values():
            for term in restriction:
                if isinstance(term, list) and len(term) == 1:
                    to_remove.append(term[0])
        unassigned_leaves = [l for l in unassigned_leaves if l not in to_remove]

        print(f"{unassigned_leaves=}")

        print(f"Index {i},{active_nodes=}, {unassigned_leaves=}")
        print(f"{restrictions=}")
        min_weight = np.inf
        min_parent = None
        selection = [None, None, None]

        # NOTE
        # Another opion would be to check all allowed
        # combinations just once and then to look
        # for the parent that is eligable for the minimum combination.
        for parent_index in active_nodes:
            print(f"\nPossible Parent {parent_index}")

            parent: TTNode = nodes[parent_index]

            # Z-child of new node will always be the previous node.
            # We only need to use every second entry in unassigned
            # as we already order odd/even terms
            # we also know for jw that every new node will be
            # the z-ancestor all all other nodes.

            parent_restrictions = restrictions[parent_index]
            print(f"{parent_restrictions=}")

            allowed_x = unpack_single_restriction(
                parent_restrictions[0], unassigned_leaves
            )
            allowed_y = unpack_single_restriction(
                parent_restrictions[1], unassigned_leaves
            )
            allowed_z = unpack_single_restriction(
                parent_restrictions[2], unassigned_leaves
            )

            match parent_restrictions[0], parent_restrictions[1]:
                case None, None:
                    allowed_product = product(allowed_z, allowed_x)
                case list(), None:
                    allowed_product = product(allowed_z, allowed_x)
                case None, list():
                    allowed_product = product(allowed_z, allowed_y)
                case list(), list():
                    allowed_product = product(allowed_x, allowed_y, allowed_z)

            print(f"{allowed_x=}, {allowed_y=}, {allowed_z=}")
            # If x and y are both none, just do all x and create a pair
            # if x is none and y is set, just use y
            # if x is set and y is none, just use x

            for comb in allowed_product:
                match len(comb):
                    case 2:
                        comb = build_valid_combination(
                            comb, descendant_map, ancestor_map, n_modes, selection
                        )
                        if comb is None:
                            continue
                    case 3:
                        # We can use the combination provided
                        pass
                    case _:
                        raise ValueError("Length of combination should be 2 or 3.")

                weight = 0
                for key in majorana_ham.keys():
                    if min(comb) > max(key):
                        continue
                    elif max(comb) < min(key):
                        continue
                    else:
                        odd_parity_paulis = [
                            sum([t != c for t in key]) % 2 for c in comb
                        ]
                        non_commuting = sum(odd_parity_paulis) % 3
                        weight += int(non_commuting != 0)
                        if weight > min_weight:
                            break

                if weight < min_weight:
                    # print(f"NEW Min Node:{i}, Parent Index: {parent_index}, Comb: {comb}, Old Min:{min_weight }, New Min:{weight}")
                    min_weight = weight
                    selection = comb
                    min_parent = parent_index
                # elif weight == min_weight:
                # #     # print(f"SAME Min Node:{i}, Parent Index: {parent_index}, Comb: {comb}, Old Min:{min_weight }, New Min:{weight}")
                # min_weight = weight
                # selection = comb

        total_weight += min_weight

        print(f"{selection=}")
        # Now find the Y pair of the x-node
        unassigned_leaves = [u for u in unassigned_leaves if u not in selection]
        for child_index, char in zip(selection, ["x", "y", "z"]):
            if isinstance(nodes.get(child_index, None), TTNode):
                # the tree structure already exists
                # we just want to add leaf_majorana_indices
                pass
            else:
                parent.leaf_majorana_indices[char] = child_index

        # TODO this is only correct if the parent only has one child.

        if i + 1 == n_modes:
            break

        completed_nodes.add(min_parent)
        node_dependencies.pop(min_parent)
        active_nodes.remove(min_parent)
        for node, deps in node_dependencies.items():
            if completed_nodes.issuperset(deps):
                active_nodes.add(node)

        z_index = selection[2]

        # Update the restrictions on the parent node to be its selection.
        restrictions[min_parent] = selection
        # Use these to update restrictions on other nodes.

        parent_node = node_objects_map[min_parent]
        print(f"{parent.root_path=}")
        print(f"{parent_node=}")
        print(f"{parent_node.parent=}")

        # The node may be its own z-ancestor
        restrictor = parent_node.z_ancestor
        print(restrictor)

        if restrictor.root_path == "":
            pass
        elif restrictor.root_path[-1] == "x":
            if isinstance(restrictor.parent.y, TTNode):
                restricted = restrictor.parent.y.z_descendant
                restrictions[string_index_map[restricted.root_path]][2] = [
                    restrictor.z_descendant.leaf_majorana_indices["z"] + 1
                ]
            elif restrictor.parent.y is None:
                restricted = restrictor.parent
                restrictions[string_index_map[restricted.root_path]][1] = [
                    restrictor.z_descendant.leaf_majorana_indices["z"] + 1
                ]

        elif parent_node.root_path[-1] == "y":
            if isinstance(parent_node.parent.x, TTNode):
                restricted = parent_node.parent.x.z_descendant
                restrictions[string_index_map[restricted.root_path]][2] = [
                    restrictor.z_descendant.leaf_majorana_indices["z"] - 1
                ]
            elif parent_node.parent.x is None:
                restricted = parent_node.parent
                restrictions[string_index_map[restricted.root_path]][1] = [
                    restrictor.z_descendant.leaf_majorana_indices["z"] - 1
                ]

        z_desc = descendant_map[z_index]
        descendant_map[parent_index] = z_desc
        ancestor_map[z_index] = parent_index
        ancestor_map[z_desc] = parent_index

        majorana_ham = _reduce_hamiltonian(majorana_ham, parent_index, selection)

    if len(active_nodes) != 1:
        print(active_nodes)
        raise ValueError(f"Not all nodes assigned by HATT. {unassigned_leaves=}")

    last_node = nodes[active_nodes.pop()]
    if isinstance(last_node, TTNode):
        root = last_node
    else:
        raise ValueError("Hatt root node is not a TTNode object.")

    tree.pauli_weight = total_weight
    print(total_weight)
    return tree
