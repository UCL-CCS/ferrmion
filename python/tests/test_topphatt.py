"""Tests for TOPP-HATT Algorithm."""

from ferrmion.optimize.topphatt import topphatt
from ferrmion.utils import fermionic_to_sparse_majorana
from ferrmion.encode import (
    TernaryTree,
    JordanWigner,
    BravyiKitaev,
    ParityEncoding,
    JKMN,
)


def test_jw_topphatt(water_sparse_majorana):
    tree = JordanWigner(14)
    tree = topphatt(majorana_ham=water_sparse_majorana, tree=tree)
    assert tree.pauli_weight == 2291
    assert tree.root_node.child_strings == JordanWigner(14).root_node.child_strings
    assert tree.root_node.branch_strings == JordanWigner(14).root_node.branch_strings
