from ferrmion.ternary_tree import TTNode, TernaryTree
from ferrmion.utils import symplectic_hash, symplectic_unhash
import numpy as np
import pytest
from openfermion import get_sparse_operator, QubitOperator
from openfermion.ops import InteractionOperator
from openfermion.transforms import jordan_wigner
import scipy as sp


@pytest.fixture
def one_e_ints():
    return np.random.random((6, 6))


@pytest.fixture
def two_e_ints():
    return np.random.random((6, 6, 6, 6))


@pytest.fixture
def fermion_modes():
    return {i for i in range(6)}


@pytest.fixture
def six_mode_tree(one_e_ints, two_e_ints):
    tt = TernaryTree(one_e_ints, two_e_ints, root_node=TTNode())
    tt.enumeration_scheme = tt.default_enumeration_scheme()
    return tt


def test_ttnode():
    root = TTNode()
    child = root.add_child("x", 0)
    child = child.add_child("y", 1)
    child = child.add_child("z", 2)
    assert child.parent.parent.parent == root
    assert child.parent.parent == root.x
    assert child.parent == root.x.y
    assert root.label is None
    assert root.x.label == 0
    assert root.as_dict() == {
        "x": {
            "x": {},
            "y": {"x": {}, "y": {}, "z": {"x": {}, "y": {}, "z": {}}},
            "z": {},
        },
        "y": {},
        "z": {},
    }
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


# def test_jordan_wigner(six_mode_tree):
#     jw = six_mode_tree.JW()
#     assert jw.root.branch_strings == {
#         "x",
#         "y",
#         "zx",
#         "zy",
#         "zzx",
#         "zzy",
#         "zzzx",
#         "zzzy",
#         "zzzz",
#     }

#     assert jw.root.child_strings == ['', 'z', 'zz', 'zzz']
#     assert len(jw.qubits) == len(jw.root.child_strings)

#     assert jw.string_pairs == {'': ('x', 'y'),
#         'z': ('zx', 'zy'),
#         'zz': ('zzx', 'zzy'),
#         'zzz': ('zzzx', 'zzzy')
#     }
#     assert jw.branch_operator_map == {
#         "zx": "ZXII",
#         "zzx": "ZZXI",
#         "zzy": "ZZYI",
#         "zzzx": "ZZZX",
#         "zy": "ZYII",
#         "y": "YIII",
#         "x": "XIII",
#         "zzzy": "ZZZY",
#     }
#     assert jw._build_symplectic_matrix().shape == (2 * len(jw.qubits), 2 * len(jw.qubits))
#     assert np.all(
#         jw._build_symplectic_matrix()
#         == np.array(
#             [
#                 [1, 0, 0, 0, 0, 0, 0, 0],
#                 [1, 0, 0, 0, 1, 0, 0, 0],
#                 [0, 1, 0, 0, 1, 0, 0, 0],
#                 [0, 1, 0, 0, 1, 1, 0, 0],
#                 [0, 0, 1, 0, 1, 1, 0, 0],
#                 [0, 0, 1, 0, 1, 1, 1, 0],
#                 [0, 0, 0, 1, 1, 1, 1, 0],
#                 [0, 0, 0, 1, 1, 1, 1, 1],
#             ]
#         )
#     )
#     assert jw.as_dict() == {'x': {},
#         'y': {},
#         'z': {'x': {},
#         'y': {},
#         'z': {'x': {}, 'y': {}, 'z': {
#             'x': {}, 'y': {}, 'z': {}}
#             }}
#     }
#     assert jw.default_enumeration_scheme() == {'': {'mode': 0, 'qubit': 0},
#         'z': {'mode': 1, 'qubit': 1},
#         'zz': {'mode': 2, 'qubit': 2},
#         'zzz': {'mode': 3, 'qubit': 3}
#     }
#     for line in jw._build_symplectic_matrix():
#         assert np.all(line == symplectic_unhash(symplectic_hash(line), len(line)))


# def test_parity(six_mode_tree):
#     assert six_mode_tree.ParityEncoding().as_dict() == {
#         "x": {
#             "x": {"x": {"x": {}, "y": {}, "z": {}}, "y": {}, "z": {}},
#             "y": {},
#             "z": {},
#         },
#         "y": {},
#         "z": {},
#     }
#     assert six_mode_tree.ParityEncoding().pauli_strings == {
#         "xxxy": "XXXY",
#         "xy": "XYII",
#         "y": "YIII",
#         "xz": "XZII",
#         "xxxx": "XXXX",
#         "xxxz": "XXXZ",
#         "xxz": "XXZI",
#         "xxy": "XXYI",
#     }

#     assert six_mode_tree.ParityEncoding().root.branch_strings == {
#         "xxxy",
#         "xy",
#         "y",
#         "xz",
#         "xxxx",
#         "xxxz",
#         "xxz",
#         "z",
#         "xxy",
#     }


def test_bravyi_kitaev(six_mode_tree):
    tt = six_mode_tree.BK()
    assert tt.root.branch_strings == {
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

    assert tt.root.child_strings == ["", "x", "xx", "xz", "xxx", "xxz"]

    assert tt.as_dict() == {
        "x": {
            "x": {
                "x": {"x": {}, "y": {}, "z": {}},
                "y": {},
                "z": {"x": {}, "y": {}, "z": {}},
            },
            "y": {},
            "z": {"x": {}, "y": {}, "z": {}},
        },
        "y": {},
        "z": {},
    }

    assert tt.default_enumeration_scheme() == {
        "": {"mode": 0, "qubit": 0},
        "x": {"mode": 1, "qubit": 1},
        "xx": {"mode": 2, "qubit": 2},
        "xz": {"mode": 3, "qubit": 3},
        "xxx": {"mode": 4, "qubit": 4},
        "xxz": {"mode": 5, "qubit": 5},
    }

    assert tt.string_pairs == {
        "": ("xzz", "y"),
        "x": ("xxzz", "xy"),
        "xx": ("xxxz", "xxy"),
        "xz": ("xzx", "xzy"),
        "xxx": ("xxxx", "xxxy"),
        "xxz": ("xxzx", "xxzy"),
    }

    assert tt.branch_operator_map == {
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
        'z': 'ZIIIII',
    }

    assert tt.n_qubits == len(tt.root.child_strings)
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


# def test_JKMN(six_mode_tree):
#     assert six_mode_tree.JKMN().as_dict() == {
#         "x": {"x": {}, "y": {}, "z": {}},
#         "y": {"x": {}, "y": {}, "z": {}},
#         "z": {"x": {}, "y": {}, "z": {}},
#     }
#     assert six_mode_tree.JKMN().root.branch_strings == {
#         "yz",
#         "yy",
#         "xz",
#         "yx",
#         "xy",
#         "zy",
#         "zx",
#         "zz",
#         "xx",
#     }

#     assert six_mode_tree.JKMN().pauli_strings == {
#         "yz": "YIZI",
#         "yy": "YIYI",
#         "xz": "XZII",
#         "yx": "YIXI",
#         "xy": "XYII",
#         "zy": "ZIIY",
#         "zx": "ZIIX",
#         "xx": "XXII",
#     }

#     assert six_mode_tree.JKMN().string_pairs == {
#         "xz": "yz",
#         "yz": "xz",
#         "xx": "xy",
#         "xy": "xx",
#         "zx": "zy",
#         "zy": "zx",
#         "yx": "yy",
#         "yy": "yx",
#     }


def tests_bonsai_paper_tree():
    tt = TernaryTree(
        np.zeros((11, 11)), np.zeros((11, 11, 11, 11)),
    )
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

    assert tt.root.branch_strings == {
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

    assert tt.root.child_strings == [
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
            "x": {"x": {}, "y": {}, "z": {}},
            "y": {"x": {}, "y": {}, "z": {}},
            "z": {},
        },
        "y": {
            "x": {"x": {}, "y": {}, "z": {}},
            "y": {"x": {}, "y": {}, "z": {}},
            "z": {"x": {}, "y": {}, "z": {"x": {}, "y": {}, "z": {}}},
        },
        "z": {"x": {}, "y": {}, "z": {"x": {}, "y": {}, "z": {}}},
    }

    assert tt.default_enumeration_scheme() == {
        "": {"mode": 0, "qubit": 0},
        "x": {"mode": 1, "qubit": 1},
        "y": {"mode": 2, "qubit": 2},
        "z": {"mode": 3, "qubit": 3},
        "xx": {"mode": 4, "qubit": 4},
        "xy": {"mode": 5, "qubit": 5},
        "yx": {"mode": 6, "qubit": 6},
        "yy": {"mode": 7, "qubit": 7},
        "yz": {"mode": 8, "qubit": 8},
        "zz": {"mode": 9, "qubit": 9},
        "yzz": {"mode": 10, "qubit": 10},
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

    assert tt.branch_operator_map == {
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
        'zzz': 'ZIIZIIIIIZI',
    }

    assert tt.n_qubits == len(tt.root.child_strings)
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


def test_eigenvalues_with_openfermion(six_mode_tree):
    # qham_zeros = InteractionOperator(0, tt.one_e_coeffs, np.zeros(tt.two_e_coeffs.shape))
    # ofop_zeros = jordan_wigner(qham_zeros)
    qham = InteractionOperator(
        0, six_mode_tree.one_e_coeffs, six_mode_tree.two_e_coeffs
    )
    # print(qham)
    ofop = jordan_wigner(qham)
    # print(f"diff {ofop-ofop_zeros}")
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ofop), k=6, which="SA")

    qham2 = six_mode_tree.JW().to_qubit_hamiltonian()
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

    assert np.allclose(diag, diag2)


def test_eigencalues_across_encodings(six_mode_tree):
    qham = six_mode_tree.JW().to_qubit_hamiltonian()
    ofop = QubitOperator()
    for k, v in qham.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop += QubitOperator(term=string, coefficient=v)
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ofop), k=6, which="SA")

    qham2 = six_mode_tree.JKMN().to_qubit_hamiltonian()
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

    assert np.allclose(sorted(diag), sorted(diag2))

def test_defaul_mode_op_map(six_mode_tree):
    assert six_mode_tree.default_mode_op_map == {i:i for i in range(six_mode_tree.n_qubits)}