"""Tests for base fermion to qubit encoding class"""
from typing import Callable
from IPython.utils.data import T
from autoray.lazy import exp
from jupyter_lsp.specs import r

import numpy as np
import pytest
from ferrmion.encode import TernaryTree, MaxNTO
from ferrmion.encode.ternary_tree import JordanWigner, BravyiKitaev, JKMN, ParityEncoding
from ferrmion.encode.base import FermionQubitEncoding, MajoranaStringEncoding
from hypothesis import strategies as st, given

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


@pytest.mark.parametrize("encoding", [JordanWigner, ParityEncoding, BravyiKitaev, JKMN])
@given(n_modes=st.integers(min_value=1, max_value=10), operator_mode=st.integers(min_value=1, max_value=10))
def test_number_operator_equals_edge_operator(encoding: Callable[[int], TernaryTree], n_modes:int, operator_mode:int):
    tree = encoding(n_modes)
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    if operator_mode < n_modes:
        # numpy doesn't like comparing empty arrays
        assert tree.edge_operator((operator_mode, operator_mode)) == tree.number_operator(operator_mode)
    else:
        with pytest.raises(ValueError) as excinfo:
            tree.number_operator(tree.n_modes + 1)
        assert "Indices invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        tree.number_operator(-1)
        assert "Indices invalid" in str(excinfo.value)


def test_edge_operator(four_mode_tt: TernaryTree):
    tree: TernaryTree = four_mode_tt.JKMN()
    tree.enumeration_scheme = tree.default_enumeration_scheme()

    output = tree.edge_operator((0, 3))
    expected = {"YZIX": -0 - 0.25j,"YZIY":0.25 - 0j, "XIZX":0.25 + 0j, "XIZY":0 + 0.25j,
    }

    assert output == expected

    scaled_output = tree.edge_operator((0,3), 0.5)
    expected = {k:0.5*v for k,v in output.items()}

    assert scaled_output == expected

@given(scaler=st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e5))
def test_jw_encode_fermion_product_coefficient_scaling_correct(scaler:np.complex64):
    four_mode_tt = TernaryTree(4)

    jw_expected = {"IIII":0.5, "ZIII":-0.5}
    jw_num_zero = four_mode_tt.JW()._encode_fermion_product("+-", [0,0], 1.)
    jw_num_zero_scaled = four_mode_tt.JW()._encode_fermion_product("+-", [0,0], scaler)
    assert jw_expected==jw_num_zero
    scaler_expected = {k:scaler*v for k,v in jw_expected.items()}
    assert set(jw_num_zero_scaled.keys()) == set(scaler_expected.keys())
    assert all([np.isclose(scaler_expected[k], jw_num_zero_scaled[k]) for k in jw_num_zero_scaled.keys()])

@given(scaler=st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e5))
def test_bk_encode_fermion_product_coefficient_scaling_correct(scaler:np.complex64):
    four_mode_tt = TernaryTree(4)
    bk_expected = {"IIII": 0.5, "ZZIZ":-0.5}
    bk_num_zero = four_mode_tt.BK()._encode_fermion_product("+-", [0,0],1.)
    bk_num_zero_scaled = four_mode_tt.BK()._encode_fermion_product("+-", [0,0],scaler)
    assert bk_expected==bk_num_zero
    scaler_expected = {k:scaler*v for k,v in bk_expected.items()}
    assert set(bk_num_zero_scaled.keys()) == set(scaler_expected.keys())
    assert all([np.isclose(scaler_expected[k], bk_num_zero_scaled[k]) for k in bk_num_zero_scaled.keys()])


@given(scaler=st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e5))
def test_maxnto_encode_fermion_product_coefficient_scaling_correct(scaler:np.complex64):
    four_mode_tt = TernaryTree(4)
    maxnto_expected = {"IIII":0.5, "IZZZ":0.5}
    with pytest.raises(ValueError):
        maxnto_num_zero = MaxNTO(4)._encode_fermion_product( "+-", [0,0],1.)
        maxnto_num_zero_scaled = MaxNTO(4)._encode_fermion_product( "+-", [0,0],scaler)
        assert maxnto_expected == maxnto_num_zero
        scaler_expected = {k:scaler*v for k,v in maxnto_expected.items()}
        assert set(maxnto_num_zero_scaled.keys()) == set(scaler_expected.keys())
        assert all([np.isclose(scaler_expected[k], maxnto_num_zero_scaled[k]) for k in maxnto_num_zero_scaled.keys()])
