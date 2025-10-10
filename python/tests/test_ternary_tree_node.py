"""Tests for TTNode class."""

from ferrmion.encode.ternary_tree_node import TTNode, z_ancestor,z_descendant, node_sorter
from ferrmion.encode import JKMN
import numpy as np

def test_ttnode_update_root_path():
    root = TTNode()
    assert root.root_path == ""
    xchild = root.add_child("x")
    assert xchild.root_path == "x"
    ychild = root.add_child("y")
    assert ychild.root_path == "y"
    yxchild = ychild.add_child("x")
    assert yxchild.root_path == "yx"

    root.update_root_path("xyz")
    assert root.root_path == "xyz"
    assert xchild.root_path == "xyzx"
    assert ychild.root_path == "xyzy"
    assert yxchild.root_path == "xyzyx"

def test_ttnode_z_relatives():
    root = TTNode()
    child = root.add_child("z")
    assert child.root_path == "z"
    grandchild = child.add_child("z")
    assert grandchild.root_path == "zz"

    # descendant
    assert z_descendant(root) is grandchild
    assert z_descendant(child) is grandchild

    # ancestor
    assert z_ancestor(child) is root
    assert z_ancestor(grandchild) is root

def test_ttnode_add_child():
    root = TTNode()
    child = root.add_child("x")
    child = child.add_child("y")
    child = child.add_child("z")
    assert root.parent is None
    assert child == root.x.y.z
    assert child.parent == root.x.y
    assert child.parent.parent == root.x
    assert child.parent.parent.parent == root
    assert root.root_path == ""
    assert root.x.root_path == "x"

    new_root = TTNode()
    moved_child=new_root.add_child(which_child="x", child_node=root.x.y)
    assert root.x.y is None
    assert new_root.x is moved_child
    assert moved_child.root_path == "x"
    assert isinstance(moved_child.z, TTNode)
    assert moved_child.z.root_path == "xz"

def test_ttnode_as_dict():
    root = TTNode()
    child = root.add_child("x")
    child = child.add_child("y")
    child = child.add_child("z")
    assert root.as_dict() == {
        "x": {
            "x": None,
            "y": {"x": None, "y": None, "z": {"x": None, "y": None, "z": None}},
            "z": None,
        },
        "y": None,
        "z": None,
    }



def test_ttnode_child_strings():
    root = TTNode()
    child = root.add_child("x")
    child = child.add_child("y")
    child = child.add_child("z")
    assert root.child_strings == [
        "",
        "x",
        "xy",
        "xyz",
    ]

def test_ttnode_child_qubit_labels():
    root = TTNode()
    child = root.add_child("x")
    child = child.add_child("y")
    child = child.add_child("z")
    assert root.child_qubit_labels == {"":None, "x":None, "xy":None, "xyz":None}

    root.qubit_label = 1
    root.x.qubit_label = 0
    root.x.y.qubit_label = 3
    root.x.y.z.qubit_label = 2
    assert root.child_qubit_labels == {"":1, "x":0, "xy":3, "xyz":2}

def test_ttnode_branch_strings():
    root = TTNode()
    child = root.add_child("x")
    child = child.add_child("y")
    child = child.add_child("z")
    assert root.branch_strings == {
        "xx",
        "xyx",
        "xyy",
        "xyzx",
        "xyzy",
        "xyzz",
        "xz",
        "y",
        "z",
    }

def test_ttnode_node_sorter():
    assert node_sorter("z") == 3
    assert node_sorter("xx") == 11
    assert node_sorter("xyz") == 123

def test_ttnode_branch_majorana_map():
    root = TTNode()
    child = TTNode()
    assert root.leaf_majorana_indices == {"x":None, "y":None, "z":None}
    assert child.leaf_majorana_indices == {"x":None, "y":None, "z":None}
    child.leaf_majorana_indices = {"x":0, "y":1, "z":2}

    root.add_child(which_child="x",child_node=child, root_path=None, qubit_label=None)
    assert root.branch_majorana_map == {"y":None, "z":None,"xx":0,"xy":1,"xz":2}

def test_ttnode_to_rustworkx():
    graph = JKMN(6).root_node.to_rustworkx()
    assert graph.nodes() == ['', 'x', 'y', 'z', 'xx', 'xy']
    assert np.all([*graph.edge_list()] == [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])
