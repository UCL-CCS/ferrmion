import numpy as np
import pytest
import scipy as sp
from ferrmion.encode.ternary_tree import TernaryTree, TTNode, JW, JordanWigner, BK, BravyiKitaev, JKMN, ParityEncoding
from ferrmion.utils import symplectic_hash, symplectic_unhash
from openfermion import QubitOperator, get_sparse_operator
from openfermion.ops import InteractionOperator
from openfermion.transforms import jordan_wigner
from ferrmion.hamiltonians import molecular_hamiltonian


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

    jw_different_enumeration = JW(6)
    jw_different_enumeration.enumeration_scheme["z"] = JW(6).enumeration_scheme["zz"]
    jw_different_enumeration.enumeration_scheme["zz"] = JW(6).enumeration_scheme["z"]
    assert JW(6) != jw_different_enumeration

def test_default_enumeration_scheme(six_mode_tree):
    assert six_mode_tree.default_enumeration_scheme() == {'':(0,0)}
    jkmn = six_mode_tree.JKMN()
    assert jkmn.default_enumeration_scheme() == {'': (0, 0), 'x': (1, 1), 'y': (2, 2), 'z': (3, 3), 'xx': (4, 4), 'xy': (5, 5)}

def test_invalid_enumeration_scheme(six_mode_tree):
    jkmn = six_mode_tree.JKMN()
    # Not enough qubit labels
    with pytest.raises(ValueError) as exc:
        jkmn.enumeration_scheme = {'': (0, 0), 'x': (1, 1), 'y': (2, 2), 'z': (3, 3), 'xx': (4, 4), 'xy': (5, 4)}
    assert "Invalid qubit labels" in str(exc.value)

    # Not enough mode labels
    with pytest.raises(ValueError) as exc:
        jkmn.enumeration_scheme = {'': (0, 0), 'x': (1, 1), 'y': (2, 2), 'z': (3, 3), 'xx': (5, 4), 'xy': (5, 5)}
    assert "Invalid mode labels" in str(exc.value)

    # Qubit label not in range
    with pytest.raises(ValueError) as exc:
        jkmn.enumeration_scheme = {'': (0, 6), 'x': (1, 1), 'y': (2, 2), 'z': (3, 3), 'xx': (4, 4), 'xy': (5, 5)}
    assert "Invalid qubit labels" in str(exc.value)

    # Mode label not in range
    with pytest.raises(ValueError) as exc:
        jkmn.enumeration_scheme = {'': (6, 0), 'x': (1, 1), 'y': (2, 2), 'z': (3, 3), 'xx': (4, 4), 'xy': (5, 5)}
    assert "Invalid mode labels" in str(exc.value)

def test_valid_enumeration_scheme(six_mode_tree):
    jkmn = six_mode_tree.JKMN()
    jkmn.enumeration_scheme = {'': (3, 1), 'x': (2, 5), 'y': (0, 3), 'z': (1, 4), 'xx': (4, 2), 'xy': (5, 0)}
    assert np.all(jkmn._build_symplectic_matrix()[1] == np.array([[False,  True, False,  True, False, False, False,  True, False,
         True, False, False],
       [False,  True, False,  True, False, False, False,  True, False,
        False, False, False],
       [False, False, False, False,  True, False, False,  True, False,
        False, False, False],
       [False, False, False, False,  True, False, False,  True, False,
        False,  True, False],
       [False,  True, False, False, False,  True, False, False,  True,
        False, False, False],
       [False,  True, False, False, False,  True,  True, False, False,
        False, False,  True],
       [False,  True, False, False, False, False, False, False, False,
        False, False,  True],
       [False,  True, False, False, False, False, False,  True, False,
         True, False, False],
       [False,  True,  True, False, False,  True, False, False, False,
        False, False, False],
       [False,  True,  True, False, False,  True, False, False,  True,
        False, False, False],
       [ True,  True, False, False, False,  True,  True, False, False,
        False, False,  True],
       [ True,  True, False, False, False,  True, False, False, False,
        False, False,  True]]))


def test_bravyi_kitaev(six_mode_tree):
    tt = six_mode_tree.BK()
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
        "": (0,0),
        "x": (1,1),
        "xx": (2,2),
        "xz": (3,3),
        "xxx": (4,4),
        "xxz": (5,5),
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
        tt._build_symplectic_matrix()[1]
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

    for line in tt._build_symplectic_matrix()[1]:
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
        "": (0,0),
        "x": (1,1),
        "y": (2,2),
        "z": (3,3),
        "xx": (4,4),
        "xy": (5,5),
        "yx": (6,6),
        "yy": (7,7),
        "yz": (8,8),
        "zz": (9,9),
        "yzz": (10,10),
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
    assert np.all(
        tt._build_symplectic_matrix()[1]
        == np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.int8,
        )
    )

    for line in tt._build_symplectic_matrix()[1]:
        assert np.all(line == symplectic_unhash(symplectic_hash(line), len(line)))

def test_eigenvalues_with_openfermion(water_eigenvalues, water_integrals):
    # qham_zeros = InteractionOperator(0, tt.one_e_coeffs, np.zeros(tt.two_e_coeffs.shape))
    # ofop_zeros = jordan_wigner(qham_zeros)
    one_e_ints, two_e_ints = water_integrals
    qham = InteractionOperator(
        0, one_e_ints, 0.5*two_e_ints
    )
    # print(qham)
    ofop = jordan_wigner(qham)
    # print(f"diff {ofop-ofop_zeros}")
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ofop), k=6, which="SA")

    assert np.allclose(sorted(diag), sorted(water_eigenvalues))


def test_eigenvalues_across_encodings(water_eigenvalues, water_tt, water_integrals):
    one_e_ints, two_e_ints = water_integrals

    qham2 = molecular_hamiltonian(water_tt.JKMN(), one_e_ints, 0.5*two_e_ints, 0)
    ofop2 = QubitOperator()
    for k, v in qham2.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop2 += QubitOperator(term=string, coefficient=v)
    diag2, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ofop2), k=6, which="SA")

    assert np.allclose(sorted(water_eigenvalues), sorted(diag2))


def test_default_mode_op_map(water_tt):
    assert np.all(water_tt.default_mode_op_map == [*range(water_tt.n_qubits)])
