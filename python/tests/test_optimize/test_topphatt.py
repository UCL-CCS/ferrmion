"""Tests for TOPP-HATT Algorithm."""
from autoray import e

import json
import ferrmion as fr
from ferrmion.optimize.cost_functions import pauli_weight, coefficient_pauli_weight
from ferrmion.optimize.topphatt import topphatt
from ferrmion.utils import fermionic_to_sparse_majorana
from ferrmion.encode.ternary_tree import (
    JordanWigner,
    BravyiKitaev,
    ParityEncoding,
    JKMN,
)
import numpy as np
import pytest
from ferrmion.core import topphatt_standard, encode
from ferrmion.optimize.huffman import huffman_ternary_tree
from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree, fast_hatt
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh


@pytest.mark.parametrize("encoding", [JordanWigner, BravyiKitaev, ParityEncoding, JKMN])
def test_topphatt_preserves_topology(water_sparse_majorana, encoding):
    tree = encoding(14)
    tree = topphatt(water_sparse_majorana, tree)
    assert tree.root_node.child_strings == encoding(14).root_node.child_strings
    assert tree.root_node.branch_strings == encoding(14).root_node.branch_strings

def test_topphatt_huffman(water_sparse_majorana, water_data):
    test_tree = huffman_ternary_tree(water_data["ones"], water_data["twos"])
    initial_children = test_tree.root_node.child_strings
    initial_branches = test_tree.root_node.branch_strings
    topphatt_tree = topphatt(water_sparse_majorana, test_tree)
    assert topphatt_tree.root_node.child_strings == initial_children
    assert topphatt_tree.root_node.branch_strings == initial_branches

def test_topphatt_hatt(water_sparse_majorana, water_data):
    test_tree = hamiltonian_adaptive_ternary_tree(fermionic_to_sparse_majorana(((water_data["ones"],"+-"), (water_data["twos"], "++--"))), n_modes=14)
    initial_children = test_tree.root_node.child_strings
    initial_branches = test_tree.root_node.branch_strings
    topphatt_tree = topphatt(water_sparse_majorana, test_tree)
    assert topphatt_tree.root_node.child_strings == initial_children
    assert topphatt_tree.root_node.branch_strings == initial_branches


def test_topphatt_fasthatt(water_sparse_majorana, water_data):
    ones, twos = water_data["ones"], water_data["twos"]
    test_tree = fast_hatt(fermionic_to_sparse_majorana(((ones,"+-"), (twos, "++--"))), n_modes=14)
    initial_children = test_tree.root_node.child_strings
    initial_branches = test_tree.root_node.branch_strings
    topphatt_tree = topphatt(water_sparse_majorana, test_tree)
    assert topphatt_tree.root_node.child_strings == initial_children
    assert topphatt_tree.root_node.branch_strings == initial_branches


def test_topphatt_bonsai(water_sparse_majorana):
    pass


@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN"])
def test_topphatt_standard_h2_eigvals_equal_expected(encoding, h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    tree = fr.TernaryTree(fham.n_modes)
    match encoding:
        case "JW":
            tree = tree.JW()
        case "BK":
            tree = tree.BK()
        case "PE":
            tree = tree.ParityEncoding()
        case "JKMN":
            tree = tree.JKMN()
    qham = tree.encode_topphatt(fham)

    ofop = QubitOperator()
    for k, v in qham.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop+= QubitOperator(term=string, coefficient=v)

    print(expected:=h2_mol_data_sets["eigvals"])
    diag, _ = eigsh(get_sparse_operator(ofop), k=2*fham.n_modes, which="SA")
    print(diag)
    assert np.allclose(np.sort(diag), np.sort(h2_mol_data_sets["eigvals"]))

@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN","HATT", "Huffman"])
def test_topphatt_standard_h2o_weights_not_increased(encoding, water_data, topphatt_weight_snapshot):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    tree = fr.TernaryTree(fham.n_modes)
    match encoding:
        case "JW":
            tree = tree.JW()
        case "BK":
            tree = tree.BK()
        case "PE":
            tree = tree.ParityEncoding()
        case "JKMN":
            tree = tree.JKMN()
        case "HATT":
            tree = hamiltonian_adaptive_ternary_tree(
                fr.core.fermionic_to_sparse_majorana(*fham.signatures_and_coefficients, fham.constant_energy),
                fham.n_modes)
        case "Huffman":
            tree = huffman_ternary_tree(ones, twos)
    qham = tree.encode_topphatt(fham)

    assert pauli_weight(qham)[0] <= topphatt_weight_snapshot[encoding]["pauli_weight"]
    assert coefficient_pauli_weight(qham)[0] <= topphatt_weight_snapshot[encoding]["coefficient_pauli_weight"]
