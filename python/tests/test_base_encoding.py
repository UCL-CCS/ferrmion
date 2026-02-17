"""Tests for base fermion to qubit encoding class"""
from jupyter_lsp.specs import r

import numpy as np
import pytest
from ferrmion.encode import TernaryTree, MaxNTO
from ferrmion.encode.ternary_tree import JordanWigner, BravyiKitaev, JKMN, ParityEncoding
from ferrmion.encode.base import FermionQubitEncoding

np.random.seed(1710)


@pytest.fixture
def four_mode_tt():
    return TernaryTree(n_modes=4)


@pytest.fixture
def sixteen_mode_tt():
    return TernaryTree(n_modes=16)


def test_default_vacuum_state(four_mode_tt):
    assert np.all(four_mode_tt.vacuum_state == np.array([0] * 4))


def test_valid_vacuum_state(four_mode_tt):
    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = [0] * 3
    assert "4" in str(excinfo.value)
    assert "length" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = [0] * 5
    assert "4" in str(excinfo.value)
    assert "length" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        four_mode_tt.vacuum_state = np.array([[0], [0]])
    assert "dimension" in str(excinfo.value)


def test_hartree_fock_state(sixteen_mode_tt):
    jw = sixteen_mode_tt.JW()
    hartree_fock_state = jw.hartree_fock_state
    nq = jw.n_qubits // 2
    print(hartree_fock_state(np.array([True] * nq + [False] * nq, dtype=bool)))
    assert np.all(
        hartree_fock_state(np.array([True] * nq + [False] * nq, dtype=bool))
        == np.array([[True] * nq + [False] * nq], dtype=bool)
    )
    assert np.all(
        hartree_fock_state(
            np.array([True] * (nq + 1) + [False] * (nq - 1), dtype=bool)
        )
        == np.array([[True] * (nq + 1) + [False] * (nq - 1)], dtype=bool)
    )


def test_number_operator(four_mode_tt):
    tree = four_mode_tt.JW()
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    # numpy doesn't like comparing empty arrays
    assert TernaryTree(n_modes=4).JW().edge_operator((0, 0)) == TernaryTree(n_modes=4).JW().number_operator(0)

    assert TernaryTree(n_modes=4).JW().edge_operator((1, 1)) == TernaryTree(n_modes=4).JW().number_operator(1)

    assert TernaryTree(n_modes=4).JW().edge_operator((2, 2)) == TernaryTree(n_modes=4).JW().number_operator(2)

    assert TernaryTree(n_modes=4).JW().edge_operator((3, 3)) == TernaryTree(n_modes=4).JW().number_operator(3)


    with pytest.raises(ValueError) as excinfo:
        tree.number_operator(tree.n_modes + 1)
    assert "Indices invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tree.number_operator(-1)
    assert "Indices invalid" in str(excinfo.value)


def test_edge_operator(four_mode_tt):
    tree = four_mode_tt.JKMN()
    tree.enumeration_scheme = tree.default_enumeration_scheme()

    output = tree.edge_operator((0, 3))
    expected = {"XIZY": -0 - 0.25j,"YIZY":0.25 - 0j, "XZIX":0.25 + 0j, "YZIX":0 + 0.25j,
    }

def test_encode_fermion_product(four_mode_tt):
    jw_expected = {"IIII":0.5, "ZIII":-0.5}
    jw_num_zero = four_mode_tt.JW()._encode_fermion_product("+-", [0,0], 1.)
    assert jw_expected==jw_num_zero

    bk_expected = {"IIII": 0.5, "ZZIZ":-0.5}
    bk_num_zero = four_mode_tt.BK()._encode_fermion_product("+-", [0,0],1.)
    assert bk_expected==bk_num_zero

    maxnto_expected = {"IIII":0.5, "IZZZ":0.5}
    maxnto_num_zero = MaxNTO(4)._encode_fermion_product( "+-", [0,0],1.)
    assert maxnto_expected == maxnto_num_zero
