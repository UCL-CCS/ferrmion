"""Tests for TOPP-HATT Algorithm."""

from ferrmion.optimize.topphatt import topphatt
from ferrmion.utils import fermionic_to_sparse_majorana
from ferrmion.encode import (
    JordanWigner,
    BravyiKitaev,
    ParityEncoding,
    JKMN,
)
import pytest
from ferrmion.optimize.huffman import huffman_ternary_tree
from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree, fast_hatt


def test_jw_topphatt(water_sparse_majorana):
    tree = JordanWigner(14)
    tree = topphatt(majorana_ham=water_sparse_majorana, tree=tree)
    assert tree.pauli_weight == 2128
    assert tree.root_node.child_strings == JordanWigner(14).root_node.child_strings
    assert tree.root_node.branch_strings == JordanWigner(14).root_node.branch_strings

@pytest.mark.parametrize("encoding", [JordanWigner, BravyiKitaev, ParityEncoding, JKMN])
def test_topphatt_preserves_topology(water_sparse_majorana, encoding):
    tree = encoding(14)
    tree = topphatt(water_sparse_majorana, tree)
    assert tree.root_node.child_strings == encoding(14).root_node.child_strings
    assert tree.root_node.branch_strings == encoding(14).root_node.branch_strings

def test_topphatt_huffman(water_sparse_majorana, water_integrals):
    ones, twos = water_integrals
    test_tree = huffman_ternary_tree(ones, twos)
    initial_children = test_tree.root_node.child_strings
    initial_branches = test_tree.root_node.branch_strings
    topphatt_tree = topphatt(water_sparse_majorana, test_tree)
    assert topphatt_tree.root_node.child_strings == initial_children
    assert topphatt_tree.root_node.branch_strings == initial_branches

def test_topphatt_hatt(water_sparse_majorana, water_integrals):
    ones, twos = water_integrals
    test_tree = hamiltonian_adaptive_ternary_tree(fermionic_to_sparse_majorana(((ones,"+-"), (twos, "++--"))), n_modes=14)
    initial_children = test_tree.root_node.child_strings
    initial_branches = test_tree.root_node.branch_strings
    topphatt_tree = topphatt(water_sparse_majorana, test_tree)
    assert topphatt_tree.root_node.child_strings == initial_children
    assert topphatt_tree.root_node.branch_strings == initial_branches


def test_topphatt_fasthatt(water_sparse_majorana, water_integrals):
    ones, twos = water_integrals
    test_tree = fast_hatt(fermionic_to_sparse_majorana(((ones,"+-"), (twos, "++--"))), n_modes=14)
    initial_children = test_tree.root_node.child_strings
    initial_branches = test_tree.root_node.branch_strings
    topphatt_tree = topphatt(water_sparse_majorana, test_tree)
    assert topphatt_tree.root_node.child_strings == initial_children
    assert topphatt_tree.root_node.branch_strings == initial_branches


def test_topphatt_bonsai(water_sparse_majorana):
    pass
