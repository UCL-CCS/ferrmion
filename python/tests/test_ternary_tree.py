from copy import deepcopy
from typing import Callable
from ferrmion import FermionHamiltonian
import numpy as np
import pytest
from ferrmion.encode.ternary_tree import (
    TernaryTree,
    TTNode,
    JW,
    JordanWigner,
    BK,
    BravyiKitaev,
    JKMN,
    ParityEncoding,
    PE,
    TTFlatpack,
)
from ferrmion.utils import symplectic_hash, symplectic_unhash
from ferrmion.hamiltonians import molecular_hamiltonian
from ferrmion.core import MajoranaEncoding
from .conftest import diagonalise_pauli_hamiltonian
from hypothesis import given,strategies as st
from hypothesis.extra.numpy import arrays
import logging
logger = logging.getLogger(__name__)

try:
    import symmer
    from symmer import PauliwordOp, QuantumState
except ImportError:
    logger.warning("Could not import symmer.")
    symmer = None

@pytest.fixture
def six_mode_tree():
    return TernaryTree(n_modes=6, root_node=TTNode())


@pytest.fixture(scope="module")
def bonsai_paper_tree():
    tt = TernaryTree(n_modes=11)
    tt = tt.add_node("x")
    tt = tt.add_node("y")
    tt = tt.add_node("z")
    tt = tt.add_node("xx")
    tt = tt.add_node("xy")
    tt = tt.add_node("yx")
    tt = tt.add_node("yy")
    tt = tt.add_node("yz")
    tt = tt.add_node("zz")
    tt = tt.add_node("yzz")
    tt.enumeration_scheme = tt.default_enumeration_scheme()
    return tt


def test_standard_encoding_functions(six_mode_tree):
    # Test function aliases
    assert JW(6) == JordanWigner(6)
    assert BK(6) == BravyiKitaev(6)

    # Test TT function aliases
    assert six_mode_tree.JW() == JW(6)
    assert six_mode_tree.JordanWigner() == JordanWigner(6)
    assert six_mode_tree.BK() == BK(6)
    assert six_mode_tree.BravyiKitaev() == BravyiKitaev(6)
    assert six_mode_tree.JKMN() == JKMN(6)
    assert six_mode_tree.ParityEncoding() == ParityEncoding(6)

    # Test inequality by type
    assert JW(6) != BK(6)
    assert JW(6) != JKMN(6)
    assert JW(6) != ParityEncoding(6)
    assert BK(6) != JKMN(6)
    assert BK(6) != ParityEncoding(6)
    assert JKMN(6) != ParityEncoding(6)

    # Test inequality
    assert JW(6) != JW(5)
    assert JW(6) != JW
    assert JW(6) != "JW(6)"

    jw_tree = TernaryTree.jordan_wigner(6)
    swapped_scheme = {**jw_tree.enumeration_scheme}
    swapped_scheme["z"], swapped_scheme["zz"] = (
        swapped_scheme["zz"],
        swapped_scheme["z"],
    )
    different_tree = TernaryTree.jordan_wigner(6)
    different_tree.enumeration_scheme = swapped_scheme
    assert JW(6) != different_tree.build_encoding()


def test_default_enumeration_scheme(six_mode_tree):
    assert six_mode_tree.default_enumeration_scheme() == {"": (0, 0)}
    jkmn = TernaryTree.jkmn(6)
    assert jkmn.default_enumeration_scheme() == {
        "": (0, 0),
        "x": (1, 1),
        "y": (2, 2),
        "z": (3, 3),
        "xx": (4, 4),
        "xy": (5, 5),
    }


def test_invalid_enumeration_scheme(six_mode_tree):
    jkmn = TernaryTree.jkmn(6)
    # Not enough qubit labels
    with pytest.raises(ValueError) as exc:
        jkmn.enumeration_scheme = {
            "": (0, 0),
            "x": (1, 1),
            "y": (2, 2),
            "z": (3, 3),
            "xx": (4, 4),
            "xy": (5, 4),
        }
    assert "Expected 6 qubit labels" in str(exc.value)

    # Not enough mode labels
    with pytest.raises(ValueError) as exc:
        jkmn.enumeration_scheme = {
            "": (0, 0),
            "x": (1, 1),
            "y": (2, 2),
            "z": (3, 3),
            "xx": (5, 4),
            "xy": (5, 5),
        }
    assert "Invalid mode labels" in str(exc.value)

    # Mode label not in range
    with pytest.raises(ValueError) as exc:
        jkmn.enumeration_scheme = {
            "": (6, 0),
            "x": (1, 1),
            "y": (2, 2),
            "z": (3, 3),
            "xx": (4, 4),
            "xy": (5, 5),
        }
    assert "Invalid mode labels" in str(exc.value)


def test_valid_enumeration_scheme(six_mode_tree):
    jkmn = TernaryTree.jkmn(6)
    # We allow any qubit labels
    jkmn.enumeration_scheme = {
        "": (3, 10),
        "x": (2, 50),
        "y": (0, 30),
        "z": (1, 40),
        "xx": (4, 20),
        "xy": (5, 0),
    }


    jkmn.enumeration_scheme = {
        "": (3, 1),
        "x": (2, 5),
        "y": (0, 3),
        "z": (1, 4),
        "xx": (4, 2),
        "xy": (5, 0),
    }


def test_bravyi_kitaev(six_mode_tree):
    tt = TernaryTree.bravyi_kitaev(6)
    assert tt.root_node.branch_strings == {
        "xxzy",
        "xxzx",
        "xxzz",
        "xzx",
        "xzy",
        "xzz",
        "xxy",
        "xxxx",
        "y",
        "xy",
        "z",
        "xxxz",
        "xxxy",
    }

    assert tt.root_node.child_strings == ["", "x", "xx", "xz", "xxx", "xxz"]

    assert tt.as_dict() == {
        "x": {
            "x": {
                "x": {"x": None, "y": None, "z": None},
                "y": None,
                "z": {"x": None, "y": None, "z": None},
            },
            "y": None,
            "z": {"x": None, "y": None, "z": None},
        },
        "y": None,
        "z": None,
    }

    assert tt.default_enumeration_scheme() == {
        "": (0, 0),
        "x": (1, 1),
        "xx": (2, 2),
        "xz": (3, 3),
        "xxx": (4, 4),
        "xxz": (5, 5),
    }

    assert tt.string_pairs == {
        "": ("xzz", "y"),
        "x": ("xxzz", "xy"),
        "xx": ("xxxz", "xxy"),
        "xz": ("xzx", "xzy"),
        "xxx": ("xxxx", "xxxy"),
        "xxz": ("xxzx", "xxzy"),
    }

    assert tt.branch_pauli_map == {
        "xxzy": "XXZIIY",
        "xxzx": "XXZIIX",
        "xxzz": "XXZIIZ",
        "xzx": "XZIXII",
        "xzy": "XZIYII",
        "xzz": "XZIZII",
        "xxy": "XXYIII",
        "xxxx": "XXXIXI",
        "y": "YIIIII",
        "xy": "XYIIII",
        "xxxz": "XXXIZI",
        "xxxy": "XXXIYI",
        "z": "ZIIIII",
    }

    assert tt.n_qubits == len(tt.root_node.child_strings)
    assert np.all(
        tt.build_encoding().symplectic_matrix
        == np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            ],
            dtype=np.int8,
        )
    )

    for line in tt.build_encoding().symplectic_matrix:
        assert np.all(line == symplectic_unhash(symplectic_hash(line), len(line)))


def tests_bonsai_paper_tree(bonsai_paper_tree):
    tt = bonsai_paper_tree
    assert tt.root_node.branch_strings == {
        "xyz",
        "zzy",
        "yyx",
        "yxz",
        "yzx",
        "yyy",
        "yzzx",
        "xyx",
        "xxx",
        "xxz",
        "yxx",
        "yzy",
        "xyy",
        "xxy",
        "yzzz",
        "yyz",
        "yxy",
        "zx",
        "zzz",
        "xz",
        "yzzy",
        "zzx",
        "zy",
    }

    assert tt.root_node.child_strings == [
        "",
        "x",
        "y",
        "z",
        "xx",
        "xy",
        "yx",
        "yy",
        "yz",
        "zz",
        "yzz",
    ]

    assert tt.as_dict() == {
        "x": {
            "x": {"x": None, "y": None, "z": None},
            "y": {"x": None, "y": None, "z": None},
            "z": None,
        },
        "y": {
            "x": {"x": None, "y": None, "z": None},
            "y": {"x": None, "y": None, "z": None},
            "z": {"x": None, "y": None, "z": {"x": None, "y": None, "z": None}},
        },
        "z": {"x": None, "y": None, "z": {"x": None, "y": None, "z": None}},
    }

    assert tt.default_enumeration_scheme() == {
        "": (0, 0),
        "x": (1, 1),
        "y": (2, 2),
        "z": (3, 3),
        "xx": (4, 4),
        "xy": (5, 5),
        "yx": (6, 6),
        "yy": (7, 7),
        "yz": (8, 8),
        "zz": (9, 9),
        "yzz": (10, 10),
    }

    assert tt.string_pairs == {
        "": ("xz", "yzzz"),
        "x": ("xxz", "xyz"),
        "y": ("yyz", "yxz"),
        "z": ("zx", "zy"),
        "xx": ("xxx", "xxy"),
        "xy": ("xyy", "xyx"),
        "yx": ("yxy", "yxx"),
        "yy": ("yyx", "yyy"),
        "yz": ("yzy", "yzx"),
        "zz": ("zzx", "zzy"),
        "yzz": ("yzzy", "yzzx"),
    }

    assert tt.branch_pauli_map == {
        "xyz": "XYIIIZIIIII",
        "zzy": "ZIIZIIIIIYI",
        "yyx": "YIYIIIIXIII",
        "yxz": "YIXIIIZIIII",
        "yzx": "YIZIIIIIXII",
        "yyy": "YIYIIIIYIII",
        "yzzx": "YIZIIIIIZIX",
        "xyx": "XYIIIXIIIII",
        "xxx": "XXIIXIIIIII",
        "xxz": "XXIIZIIIIII",
        "yxx": "YIXIIIXIIII",
        "yzy": "YIZIIIIIYII",
        "xyy": "XYIIIYIIIII",
        "xxy": "XXIIYIIIIII",
        "yzzz": "YIZIIIIIZIZ",
        "yyz": "YIYIIIIZIII",
        "yxy": "YIXIIIYIIII",
        "zx": "ZIIXIIIIIII",
        "xz": "XZIIIIIIIII",
        "yzzy": "YIZIIIIIZIY",
        "zzx": "ZIIZIIIIIXI",
        "zy": "ZIIYIIIIIII",
        "zzz": "ZIIZIIIIIZI",
    }

    assert tt.n_qubits == len(tt.root_node.child_strings)

    for line in tt.build_encoding().symplectic_matrix:
        assert np.all(line == symplectic_unhash(symplectic_hash(line), len(line)))

def test_default_mode_op_map(water_tt):
    assert np.all(water_tt.default_mode_op_map == [*range(water_tt.n_qubits)])

@pytest.mark.parametrize("optimisation", ["naive", "anneal", "topphatt"])
@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.bravyi_kitaev,
        TernaryTree.parity,
        TernaryTree.jkmn,
    ],
)
def test_encode_num_terms_equal_expected(encoding: Callable[[int], TernaryTree], optimisation:str, mol_data_sets: dict):
    ones = mol_data_sets["ones"]
    twos = mol_data_sets["twos"]
    e_nuc = mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]
    fham = FermionHamiltonian(terms = {"+-":ones,"++--":twos}, constant_energy=e_nuc)
    initial_ones = deepcopy(ones)
    initial_twos = deepcopy(twos)

    match optimisation:
        case "naive":
            qham = encoding(fham.n_modes).encode(fham)
        case "anneal":
            qham = encoding(fham.n_modes).encode_annealed(fham)[0]
        case "topphatt":
            qham = encoding(fham.n_modes).encode_topphatt(fham)[0]

    assert len(qham) == mol_data_sets["num_terms"]

@pytest.mark.parametrize("optimisation", ["naive", "anneal", "topphatt"])
@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.bravyi_kitaev,
        TernaryTree.parity,
        TernaryTree.jkmn,
    ],
)
def test_encode_h2_eigvals_equal_expected(encoding: Callable[[int], TernaryTree], optimisation:str, h2_mol_data_sets: dict):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]
    fham = FermionHamiltonian(terms = {"+-":ones,"++--":twos}, constant_energy=e_nuc)
    initial_ones = deepcopy(ones)
    initial_twos = deepcopy(twos)

    match optimisation:
        case "naive":
            qham = encoding(fham.n_modes).encode(fham)
        case "anneal":
            qham = encoding(fham.n_modes).encode_annealed(fham)[0]
        case "topphatt":
            qham = encoding(fham.n_modes).encode_topphatt(fham)[0]
    diag  = diagonalise_pauli_hamiltonian(qham, 2*n_modes)

    assert np.all(initial_ones == ones)
    assert np.all(initial_twos == twos)
    assert np.allclose(np.sort(diag), np.sort(h2_mol_data_sets["eigvals"]))

@pytest.mark.parametrize("optimisation", ["naive", "topphatt"])
@pytest.mark.parametrize("encoding", [TernaryTree.jordan_wigner])
def test_encode_jw_water_eigvals_equal_expected(encoding: Callable[[int], TernaryTree], optimisation:str,  water_data: dict):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    n_modes = ones.shape[0]

    fham = FermionHamiltonian(terms = {"+-":ones,"++--":twos}, constant_energy=e_nuc)

    match optimisation:
        case "naive":
            qham = encoding(fham.n_modes).encode(fham)
        # Takes too long for tests!
        # case "anneal":
            # qham = encoding(fham.n_modes).encode_annealed(fham)
        case "topphatt":
            qham = encoding(fham.n_modes).encode_topphatt(fham)[0]
    assert np.isclose(qham["I"*14], -46.465600781952176)
    diag = diagonalise_pauli_hamiltonian(qham, 2)

    assert np.allclose(np.sort(diag), np.sort(water_data["eigvals"])[:2])

@given(arrays(dtype=np.bool, shape=st.integers(1, 9)))
def test_naive_jw_hf_state_unchanged(fermionic_hf_state):
    tree = TernaryTree.jordan_wigner(len(fermionic_hf_state))
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    print(f"fermionic HF {fermionic_hf_state}")
    qubit_hf_state = tree.hartree_fock_state(
        fermionic_hf_state=fermionic_hf_state,
        mode_op_map=[*range(len(fermionic_hf_state))],
    )
    assert np.all(qubit_hf_state == fermionic_hf_state)

@given(mode_op_map=st.permutations([*range(10)]), n_electrons=st.integers(min_value=1, max_value=10))
def test_enumerated_jw_hf_state_match_reordered_naive(mode_op_map, n_electrons):
    fermionic_hf_state = np.array([True] * n_electrons + [False] * (10-n_electrons), dtype=np.bool)

    tree = TernaryTree.jordan_wigner(len(fermionic_hf_state))
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    print(f"\nfermionic HF {fermionic_hf_state}")
    print(f"Enumeration {mode_op_map}")
    naive_qubit_hf_state = tree.hartree_fock_state(
        fermionic_hf_state=fermionic_hf_state,
        mode_op_map=[*range(len(fermionic_hf_state))],
    )

    print(f"naive {naive_qubit_hf_state}")
    enumerated_qubit_hf_state = tree.hartree_fock_state(
        fermionic_hf_state=fermionic_hf_state,
        mode_op_map=mode_op_map,
    )
    print(f"enumerated {enumerated_qubit_hf_state}")
    expected_emnumerated = np.array([False] * 10, dtype=np.bool)
    expected_emnumerated[mode_op_map[:n_electrons]] = True
    print(f"expected {enumerated_qubit_hf_state}")

    assert np.all(naive_qubit_hf_state == fermionic_hf_state)
    assert np.all(enumerated_qubit_hf_state == expected_emnumerated)

@pytest.mark.skipif(symmer is None, reason="Dependency group test not installed.")
@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.parity,
        TernaryTree.bravyi_kitaev,
        TernaryTree.jkmn,
    ],
)
def test_naive_water_hf_energy_correct(encoding, water_data):
    fermionic_hf_state = np.array([True]*10 + [False] * 4, dtype=np.bool)

    tree = encoding(len(fermionic_hf_state))
    tree.enumeration_scheme = tree.default_enumeration_scheme()

    fham = molecular_hamiltonian(water_data["ones"], water_data["twos"], constant_energy=water_data["constant_energy"])
    qham = tree.encode(fham)
    enumerated_qubit_hf_state = tree.hartree_fock_state(
        fermionic_hf_state=fermionic_hf_state,
    )
    assert np.isclose(PauliwordOp.from_dictionary(qham).expval(QuantumState([int(v) for v in enumerated_qubit_hf_state])), water_data["e_hf"])

@given(arrays(dtype=np.bool, shape=st.integers(1, 9)))
def test_naive_parity_hf_state(fermionic_hf_state):
    tree = TernaryTree.parity(len(fermionic_hf_state))
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    qubit_hf_state = tree.hartree_fock_state(
        fermionic_hf_state=fermionic_hf_state,
        mode_op_map=[*range(len(fermionic_hf_state))],
    )

    print(f"fermionic HF\t {fermionic_hf_state}")
    print(f"qubit HF\t {qubit_hf_state}")
    # The convention for Parity is that X is applied to indices
    # *higher* than the qubit being changed.
    # We have to change these around as we have an x-tail for lesser indices.
    expected_parity = np.cumsum(fermionic_hf_state[::-1]) % 2
    expected_parity = np.array(expected_parity, dtype=np.bool)[::-1]
    print(f"expected parity\t {expected_parity}")
    print(f"Result\t {np.all(qubit_hf_state == expected_parity)}")

    assert np.all(qubit_hf_state == expected_parity)

@given(arrays(dtype=np.bool, shape=st.integers(1, 9)))
def test_naive_jkmn_hf_state_runs(fermionic_hf_state):
    tree = TernaryTree.jkmn(len(fermionic_hf_state))
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    qubit_hf_state = tree.hartree_fock_state(
        fermionic_hf_state=fermionic_hf_state,
        mode_op_map=[*range(len(fermionic_hf_state))],
    )

@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.parity,
        TernaryTree.bravyi_kitaev,
        TernaryTree.jkmn,
    ],
)
def test_benchmark_encode_naive(benchmark,encoding, mol_data_sets):
    ones = mol_data_sets["ones"]
    twos = mol_data_sets["twos"]
    e_nuc = mol_data_sets["constant_energy"]
    fham = FermionHamiltonian(terms={"+-": ones, "++--": twos}, constant_energy=e_nuc)
    encoding = encoding(fham.n_modes)
    benchmark(lambda: encoding.encode(fham))

@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.parity,
        TernaryTree.bravyi_kitaev,
        TernaryTree.jkmn,
    ],
)
def test_benchmark_encode_topphatt(benchmark,encoding, mol_data_sets):
    ones = mol_data_sets["ones"]
    twos = mol_data_sets["twos"]
    e_nuc = mol_data_sets["constant_energy"]
    fham = FermionHamiltonian(terms={"+-": ones, "++--": twos}, constant_energy=e_nuc)
    encoding:TernaryTree= encoding(fham.n_modes)
    benchmark(lambda: encoding.encode_topphatt(fham))




# @pytest.mark.parametrize("encoding", [TernaryTree.jordan_wigner, TernaryTree.parity, TernaryTree.bravyi_kitaev, TernaryTree.jkmn])
# def test_benchmark_encode_annealed(benchmark,encoding, h2_mol_data_sets):
#     ones = h2_mol_data_sets["ones"]
#     twos = h2_mol_data_sets["twos"]
#     e_nuc = h2_mol_data_sets["constant_energy"]
#     fham = FermionHamiltonian(terms={"+-": ones, "++--": twos}, constant_energy=e_nuc)
#     encoding:TernaryTree = encoding(fham.n_modes)
#     benchmark(lambda: encoding.encode_annealed(fham))


@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.parity,
        TernaryTree.bravyi_kitaev,
        TernaryTree.jkmn,
    ],
)
@pytest.mark.parametrize("n_modes", [32,64,128])
def test_benchmark_hartree_fock_state(benchmark, encoding,n_modes):
    fermionic_hf_state = np.array([True]*n_modes, dtype=np.bool)
    tree:TernaryTree = encoding(len(fermionic_hf_state))
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    benchmark(lambda: tree.hartree_fock_state(fermionic_hf_state=fermionic_hf_state))


@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.parity,
        TernaryTree.bravyi_kitaev,
        TernaryTree.jkmn,
    ],
)
@pytest.mark.parametrize("n_modes", [32, 64, 128])
def test_benchmark_decode_zbasis_ensemble(benchmark, encoding, n_modes):
    fermionic_hf_state = np.ones(n_modes, dtype=np.bool)
    tree: TernaryTree = encoding(n_modes)
    tree.enumeration_scheme = tree.default_enumeration_scheme()
    # Build an ensemble of 10000 identical states.
    rng = np.random.default_rng(seed=17042026)
    states = rng.choice([False, True], size=(10000, n_modes), p=[0.5, 0.5])
    benchmark(lambda: tree.decode(states))

@st.composite
def tt_flatpack_strategy(draw, n_nodes_strategy):
    n_nodes = draw(n_nodes_strategy)
    if n_nodes == 0:
        return []
    nodes = draw(st.permutations([*range(n_nodes)]))
    flatpack = {n: {'x':None, 'y':None, 'z':None} for n in nodes}
    parents = [nodes[0]]
    for child in nodes[1:]:
        parent = draw(st.sampled_from(parents))
        assert isinstance(parent, int)
        parents.append(child)

        edges = [e for e, v in flatpack[parent].items() if v == None]
        flatpack[parent][draw(st.sampled_from(edges))] = child
        if len(edges) == 1:
            parents.remove(parent)
    # flatpack = [(k , tuple(val for val in v.values())) for k, v in flatpack.items()]
    to_add = [nodes[0]]
    output = []
    for node in to_add:
        output.append((node, tuple(v for v in flatpack[node].values())))
        for child in flatpack[node].values():
            if child is not None:
                to_add.append(child)
    return output


@given(flatpack=tt_flatpack_strategy(st.integers(1, 20)))
def test_validate_tt_flatpack_strategy(flatpack):
    used_qubit_indices = [flatpack[0][0]]
    for item in flatpack:
        assert item[0] in used_qubit_indices
        children = item[1]
        assert len(children) == 3
        for child in children:
            assert isinstance(child,  int | None)
            if isinstance(child, int):
                used_qubit_indices.append(child)


@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.parity,
        TernaryTree.bravyi_kitaev,
        TernaryTree.jkmn,
    ],
)
@pytest.mark.parametrize("n_modes", [1, 5, 10])
def test_encoding_flatpack_validate(encoding, n_modes):
    tree = encoding(n_modes)
    flatpack = tree.flatpack()
    used_qubit_indices = [flatpack[0][0]]
    for item in flatpack:
        assert item[0] in used_qubit_indices
        children = item[1]
        assert len(children) == 3
        for child in children:
            assert isinstance(child,  int | None)
            if isinstance(child, int):
                used_qubit_indices.append(child)

@given(flatpack=tt_flatpack_strategy(st.integers(1, 20)))
def test_from_flatpack_roundtrip(flatpack):
    reconstructed = TernaryTree.from_flatpack(flatpack)
    assert sorted(reconstructed.flatpack()) == sorted(flatpack)


@given(flatpack=tt_flatpack_strategy(st.integers(1, 20)))
def test_from_flatpack_properties(flatpack):
    reconstructed = TernaryTree.from_flatpack(flatpack)

    # Check n_modes equals the number of nodes
    assert reconstructed.n_modes == len(flatpack)

    # Check n_qubits equals the number of nodes
    assert reconstructed.n_qubits == len(flatpack)

    # Check enumeration_scheme has correct length
    assert len(reconstructed.enumeration_scheme) == len(flatpack)

    # Check all qubit indices are unique and match flatpack ids
    qubit_indices = {mode_qubit[1] for mode_qubit in reconstructed.enumeration_scheme.values()}
    flatpack_ids = {item[0] for item in flatpack}
    assert qubit_indices == flatpack_ids


@given(flatpack=tt_flatpack_strategy(st.integers(1, 20)))
def test_symplectic_matrix_roundtrip(flatpack):
    original_tree = TernaryTree.from_flatpack(flatpack)
    print(f"Original flatpack {flatpack}")
    flatpack = original_tree.flatpack()
    reconstructed_tree = TernaryTree.from_flatpack(flatpack)
    print(f"Reconstructed flatpack {flatpack}")

    original_encoding = original_tree.build_encoding()
    reconstructed_encoding = reconstructed_tree.build_encoding()

    assert np.array_equal(original_encoding.ipowers, reconstructed_encoding.ipowers)
    assert np.array_equal(
        original_encoding.symplectic_matrix, reconstructed_encoding.symplectic_matrix
    )

@given(flatpack=tt_flatpack_strategy(st.integers(1, 20)))
def test_core_python_symplectics_from_flatpack_equal(flatpack):
    tree_encoding = TernaryTree.from_flatpack(flatpack).build_encoding()
    direct_encoding = MajoranaEncoding.from_flatpack(flatpack)

    assert np.array_equal(tree_encoding.ipowers, direct_encoding.ipowers)
    assert np.array_equal(
        tree_encoding.symplectic_matrix, direct_encoding.symplectic_matrix
    )
