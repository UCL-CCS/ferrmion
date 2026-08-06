"""Tests for the Rust-backed MajoranaEncoding class."""
import json
import pickle
from typing import Callable

import numpy as np
import pytest
from hypothesis import given, strategies as st

from ferrmion.core import (
    FermionHamiltonian,
    MajoranaEncoding,
    MajoranaSparse,
    QubitHamiltonian,
)
from ferrmion.encode import MaxNTO

np.random.seed(1710)

FACTORIES = [
    MajoranaEncoding.jordan_wigner,
    MajoranaEncoding.parity,
    MajoranaEncoding.bravyi_kitaev,
    MajoranaEncoding.jkmn,
]


@pytest.fixture
def jw_four():
    return MajoranaEncoding.jordan_wigner(4)


def test_default_vacuum_state(jw_four):
    assert np.all(jw_four.vacuum_state == np.array([False] * 4))


def test_invalid_vacuum_state(jw_four):
    ipowers = jw_four.ipowers
    symplectics = jw_four.symplectic_matrix

    with pytest.raises(ValueError):
        MajoranaEncoding(ipowers, symplectics, np.array([True] * 3))

    with pytest.raises(ValueError):
        MajoranaEncoding(ipowers, symplectics, np.array([True] * 5))


def test_mismatched_ipowers_raises(jw_four):
    symplectics = jw_four.symplectic_matrix
    with pytest.raises(ValueError) as excinfo:
        MajoranaEncoding(np.zeros(3, dtype=np.uint8), symplectics)
    assert "same length" in str(excinfo.value)


def test_from_symplectic_roundtrip(jw_four):
    rebuilt = MajoranaEncoding(
        jw_four.ipowers, jw_four.symplectic_matrix, jw_four.vacuum_state
    )
    assert rebuilt == jw_four

    auto_vacuum = MajoranaEncoding(jw_four.ipowers, jw_four.symplectic_matrix)
    assert np.all(auto_vacuum.vacuum_state == jw_four.vacuum_state)


def test_to_json_from_json_roundtrip(jw_four):
    data = jw_four.to_json()
    assert set(data.keys()) == {"ipowers", "symplectics", "vacuum_state"}
    assert isinstance(data["ipowers"], list)
    assert all(isinstance(v, int) for v in data["ipowers"])
    assert isinstance(data["symplectics"], list)
    assert isinstance(data["vacuum_state"], list)
    assert MajoranaEncoding.from_json(data) == jw_four

    # The output must actually be JSON-serialisable and survive a full
    # serialise/deserialise cycle.
    rebuilt = MajoranaEncoding.from_json(json.loads(json.dumps(data)))
    assert rebuilt == jw_four


def test_hartree_fock_state():
    jw = MajoranaEncoding.jordan_wigner(16)
    nq = jw.n_qubits // 2
    assert np.all(
        jw.hartree_fock_state(np.array([True] * nq + [False] * nq, dtype=bool))
        == np.array([[True] * nq + [False] * nq], dtype=bool)
    )
    assert np.all(
        jw.hartree_fock_state(
            np.array([True] * (nq + 1) + [False] * (nq - 1), dtype=bool)
        )
        == np.array([[True] * (nq + 1) + [False] * (nq - 1)], dtype=bool)
    )


@pytest.mark.parametrize("encoding", FACTORIES)
@given(
    n_modes=st.integers(min_value=1, max_value=10),
    operator_mode=st.integers(min_value=1, max_value=10),
)
def test_number_operator_equals_edge_operator(
    encoding: Callable[[int], MajoranaEncoding], n_modes: int, operator_mode: int
):
    enc = encoding(n_modes)
    if operator_mode < n_modes:
        # numpy doesn't like comparing empty arrays
        assert enc.edge_operator(
            (operator_mode, operator_mode)
        ) == enc.number_operator(operator_mode)
    else:
        with pytest.raises(ValueError) as excinfo:
            enc.number_operator(enc.n_modes + 1)
        assert "Indices invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        enc.number_operator(-1)
    assert "Indices invalid" in str(excinfo.value)


def test_edge_operator():
    enc = MajoranaEncoding.jkmn(4)

    output = enc.edge_operator((0, 3))
    expected = {
        "YZIX": -0 - 0.25j,
        "YZIY": 0.25 - 0j,
        "XIZX": 0.25 + 0j,
        "XIZY": 0 + 0.25j,
    }

    assert output == expected

    scaled_output = enc.edge_operator((0, 3), 0.5)
    expected = {k: 0.5 * v for k, v in output.items()}

    assert scaled_output == expected


@given(
    left=st.integers(min_value=0, max_value=9),
    right=st.integers(min_value=0, max_value=9),
)
@pytest.mark.parametrize("encoding", FACTORIES)
def test_conjugate_values_real(
    left: int, right: int, encoding: Callable[[int], MajoranaEncoding]
):
    enc = encoding(10)
    lr = enc.edge_operator((left, right), with_conjugate=True)
    assert np.all(np.isreal([*lr.values()]))


@given(
    left=st.integers(min_value=0, max_value=9),
    right=st.integers(min_value=0, max_value=9),
)
@pytest.mark.parametrize("encoding", FACTORIES)
def test_with_conjugate_ordering_equivalent(
    left: int, right: int, encoding: Callable[[int], MajoranaEncoding]
):
    enc = encoding(10)
    lr = enc.edge_operator((left, right), with_conjugate=True)
    rl = enc.edge_operator((right, left), with_conjugate=True)
    assert lr == rl


@given(scaler=st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e5))
def test_jw_encode_fermion_product_coefficient_scaling_correct(scaler: complex):
    jw = MajoranaEncoding.jordan_wigner(4)

    jw_expected = {"IIII": 0.5, "ZIII": -0.5}
    jw_num_zero = jw.encode_fermion_product("+-", [0, 0], 1.0)
    jw_num_zero_scaled = jw.encode_fermion_product("+-", [0, 0], scaler)
    assert jw_num_zero == jw_expected
    scaler_expected = {k: scaler * v for k, v in jw_expected.items()}
    assert set(jw_num_zero_scaled.keys()) == set(scaler_expected.keys())
    assert all(
        np.isclose(scaler_expected[k], jw_num_zero_scaled[k])
        for k in jw_num_zero_scaled.keys()
    )


@given(scaler=st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e5))
def test_bk_encode_fermion_product_coefficient_scaling_correct(scaler: complex):
    bk = MajoranaEncoding.bravyi_kitaev(4)
    bk_expected = {"IIII": 0.5, "ZZIZ": -0.5}
    bk_num_zero = bk.encode_fermion_product("+-", [0, 0], 1.0)
    bk_num_zero_scaled = bk.encode_fermion_product("+-", [0, 0], scaler)
    assert bk_num_zero == bk_expected
    scaler_expected = {k: scaler * v for k, v in bk_expected.items()}
    assert set(bk_num_zero_scaled.keys()) == set(scaler_expected.keys())
    assert all(
        np.isclose(scaler_expected[k], bk_num_zero_scaled[k])
        for k in bk_num_zero_scaled.keys()
    )


def test_maxnto_even_modes_raises():
    # MaxNTO requires n_modes - 1 to be odd.
    with pytest.raises(ValueError):
        MaxNTO(5)


def test_apply_mode_enumeration_requires_permutation(jw_four):
    with pytest.raises(ValueError):
        jw_four.apply_mode_enumeration([0, 1, 2, 2])
    swapped = jw_four.apply_mode_enumeration([3, 2, 1, 0])
    assert swapped != jw_four
    assert swapped.apply_mode_enumeration([3, 2, 1, 0]) == jw_four


def test_encode_rejects_mode_mismatch(jw_four):
    fham = FermionHamiltonian(terms={"+-": np.eye(6)})
    with pytest.raises(ValueError) as excinfo:
        jw_four.encode(fham)
    assert "modes" in str(excinfo.value)


def test_encoding_equality(jw_four):
    assert jw_four == MajoranaEncoding.jordan_wigner(4)
    assert jw_four != MajoranaEncoding.jordan_wigner(5)
    assert jw_four != MajoranaEncoding.bravyi_kitaev(4)
    assert jw_four != "MajoranaEncoding"


def test_encoding_pickle_roundtrip(jw_four):
    assert pickle.loads(pickle.dumps(jw_four)) == jw_four


def test_qubit_hamiltonian_mapping_behaviour():
    qham = QubitHamiltonian({"XX": 0.5, "YY": 0.5j})
    assert len(qham) == 2
    assert qham["XX"] == 0.5
    assert "YY" in qham
    assert set(qham) == {"XX", "YY"}
    assert dict(qham) == {"XX": 0.5, "YY": 0.5j}
    assert qham == {"XX": 0.5, "YY": 0.5j}
    assert qham.get("ZZ") is None
    assert qham.get("ZZ", 0.0) == 0.0
    assert qham.n_qubits == 2

    with pytest.raises(KeyError):
        qham["ZZ"]
    with pytest.raises(ValueError):
        qham["ZZZ"] = 1.0
    with pytest.raises(ValueError):
        qham["AB"] = 1.0

    qham["ZZ"] = 1.0
    assert len(qham) == 3
    del qham["ZZ"]
    assert len(qham) == 2

    with pytest.raises(ValueError):
        QubitHamiltonian().n_qubits


def test_qubit_hamiltonian_pickle_roundtrip():
    qham = QubitHamiltonian({"XX": 0.5, "YY": 0.5j})
    assert pickle.loads(pickle.dumps(qham)) == qham


def test_fermion_hamiltonian_pickle_roundtrip():
    fham = FermionHamiltonian(
        terms={"+-": np.eye(4), "++--": np.zeros((4, 4, 4, 4))},
        constant_energy=0.25,
    )
    rebuilt = pickle.loads(pickle.dumps(fham))
    assert rebuilt == fham
    assert rebuilt.n_modes == 4
    assert rebuilt.constant_energy == 0.25


def test_encode_accepts_majorana_sparse(jw_four):
    fham = FermionHamiltonian(terms={"+-": np.eye(4)})
    msparse = fham.to_majorana_sparse()
    assert isinstance(msparse, MajoranaSparse)
    assert jw_four.encode(msparse) == jw_four.encode(fham)


@pytest.mark.parametrize("factory", FACTORIES)
def test_encode_majorana_sparse_matches_fermion_path(factory):
    n_modes = 4
    encoding = factory(n_modes)
    fham = FermionHamiltonian(
        terms={"+-": np.random.rand(n_modes, n_modes)},
    )
    assert encoding.encode(fham.to_majorana_sparse()) == encoding.encode(fham)


def test_encode_majorana_sparse_preserves_constant(jw_four):
    """The MajoranaSparse constant must reach the identity Pauli string."""
    terms = {"+-": np.eye(4)}
    plain = FermionHamiltonian(terms=terms)
    with_constant = FermionHamiltonian(terms=terms, constant_energy=0.75)

    msparse = with_constant.to_majorana_sparse()
    assert msparse.constant == pytest.approx(0.75)

    qham = jw_four.encode(msparse)
    assert qham == jw_four.encode(with_constant)
    # The identity coefficient also collects the 1/2-per-mode from each a†a, so
    # compare against the same Hamiltonian without a constant energy.
    identity_shift = qham["IIII"] - jw_four.encode(plain.to_majorana_sparse())["IIII"]
    assert identity_shift == pytest.approx(0.75)


def test_encode_majorana_sparse_rejects_out_of_range_indices(jw_four):
    """Out-of-range Majorana indices must raise, not panic in a rayon worker."""
    six_mode = FermionHamiltonian(terms={"+-": np.eye(6)})
    msparse = six_mode.to_majorana_sparse()
    assert max(max(term) for term in msparse.indices) >= 2 * jw_four.n_modes
    with pytest.raises(ValueError) as excinfo:
        jw_four.encode(msparse)
    assert "modes" in str(excinfo.value)


def test_encode_rejects_unsupported_type(jw_four):
    with pytest.raises(TypeError):
        jw_four.encode("not an operator")


def test_encode_majorana_product_returns_pauli_coefficient_pair(jw_four):
    """A Majorana product is always a single Pauli term, returned as a tuple."""
    result = jw_four.encode_majorana_product([0])
    assert isinstance(result, tuple) and len(result) == 2
    pauli, coeff = result
    assert isinstance(pauli, str)
    assert isinstance(coeff, complex)


def test_encode_majorana_product_jordan_wigner_convention(jw_four):
    """Under JW the first two Majoranas are X and Y on qubit 0."""
    assert jw_four.encode_majorana_product([0]) == ("XIII", 1 + 0j)
    assert jw_four.encode_majorana_product([1]) == ("YIII", 1 + 0j)
    # X * Y = iZ
    assert jw_four.encode_majorana_product([0, 1]) == ("ZIII", 1j)


def test_encode_majorana_product_anticommutes(jw_four):
    """Swapping two distinct Majoranas flips the sign; squaring gives identity."""
    forward_pauli, forward_coeff = jw_four.encode_majorana_product([0, 1])
    reversed_pauli, reversed_coeff = jw_four.encode_majorana_product([1, 0])
    assert forward_pauli == reversed_pauli
    assert reversed_coeff == pytest.approx(-forward_coeff)

    assert jw_four.encode_majorana_product([0, 0]) == ("IIII", 1 + 0j)
    assert jw_four.encode_majorana_product([]) == ("IIII", 1 + 0j)


@pytest.mark.parametrize("factory", FACTORIES)
def test_encode_majorana_product_matches_number_operator(factory):
    """n_i = 1/2 - (i/2) * y_2i * y_2i+1, independently of the encoding.

    Cross-checks encode_majorana_product against the number_operator path.
    """
    n_modes = 4
    encoding = factory(n_modes)
    identity = "I" * encoding.n_qubits
    for mode in range(n_modes):
        pauli, coeff = encoding.encode_majorana_product(
            [2 * mode, 2 * mode + 1], 0.5j
        )
        assert encoding.number_operator(mode).to_dict() == {
            identity: 0.5 + 0j,
            pauli: coeff,
        }


def test_encode_majorana_product_scales_coefficient(jw_four):
    base_pauli, base_coeff = jw_four.encode_majorana_product([0, 2])
    scaled_pauli, scaled_coeff = jw_four.encode_majorana_product([0, 2], 2.5 - 1j)
    assert base_pauli == scaled_pauli
    assert scaled_coeff == pytest.approx(base_coeff * (2.5 - 1j))


@pytest.mark.parametrize("bad_index", [-1, 8, 100])
def test_encode_majorana_product_rejects_out_of_range(jw_four, bad_index):
    """Indices run over 2 * n_modes Majoranas, not n_modes."""
    assert jw_four.n_modes == 4
    # The last valid index is 7; nothing beyond it may reach the symplectic rows.
    jw_four.encode_majorana_product([7])
    with pytest.raises(ValueError):
        jw_four.encode_majorana_product([0, bad_index])
