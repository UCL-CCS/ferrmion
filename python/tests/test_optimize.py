"""Tests for functions in the optimize submodule."""

from ferrmion.optimize.rett import reduced_entanglement_ternary_tree
from ferrmion.optimize.huffman import huffman_ternary_tree
from ferrmion.optimize.enumeration import minimise_mi_distance, distance_squared, pauli_weighted_norm
from ferrmion.optimize.bonsai import bonsai_algorithm
from ferrmion.encode import TernaryTree
from ferrmion.core import molecular_hamiltonian_template, fill_template
import rustworkx as rx
import numpy as np
import numpy as np
from pytest import fixture

@fixture
def n2mi():
    return np.array([
        [8.95411720e-05, 8.00313442e-07, 3.57202788e-05, 5.64570141e-06, 3.39388829e-06, 3.67820340e-06, 3.71248133e-06, 8.39012020e-06, 7.60808045e-06, 4.00424690e-07],
        [8.00313442e-07, 1.88782428e-04, 4.61021178e-06, 1.43974455e-04, 3.97230035e-06, 4.34316525e-06, 4.25262029e-06, 1.40315431e-05, 1.31042878e-05, 4.73322942e-07],
        [3.57202788e-05, 4.61021178e-06, 5.97013599e-02, 1.53473952e-03, 1.14739409e-03, 1.67868542e-02, 2.03055787e-03, 1.57647091e-02, 2.20614397e-03, 2.26381776e-02],
        [5.64570141e-06, 1.43974455e-04, 1.53473952e-03, 9.22768981e-02, 8.06189477e-03, 8.02634734e-03, 3.59937207e-03, 3.44315829e-02, 3.39878461e-02, 1.58240762e-03],
        [3.39388829e-06, 3.97230035e-06, 1.14739409e-03, 8.06189477e-03, 4.94601967e-01, 1.27072663e-01, 2.87313949e-03, 9.00062433e-02, 2.49380494e-01, 2.23093410e-03],
        [3.67820340e-06, 4.34316525e-06, 1.67868542e-02, 8.02634734e-03, 1.27072663e-01, 5.27514044e-01, 8.66763070e-03, 2.58110275e-01, 8.95621909e-02, 2.18442824e-02],
        [3.71248133e-06, 4.25262029e-06, 2.03055787e-03, 3.59937207e-03, 2.87313949e-03, 8.66763070e-03, 5.41448119e-02, 1.25017005e-02, 8.08334819e-03, 1.85391136e-02],
        [8.39012020e-06, 1.40315431e-05, 1.57647091e-02, 3.44315829e-02, 9.00062433e-02, 2.58110275e-01, 1.25017005e-02, 5.62235070e-01, 1.28998497e-01, 2.20469210e-02],
        [7.60808045e-06, 1.31042878e-05, 2.20614397e-03, 3.39878461e-02, 2.49380494e-01, 8.95621909e-02, 8.08334819e-03, 1.28998497e-01, 5.30526412e-01, 2.39364166e-03],
        [4.00424690e-07, 4.73322942e-07, 2.26381776e-02, 1.58240762e-03, 2.23093410e-03, 2.18442824e-02, 1.85391136e-02, 2.20469210e-02, 2.39364166e-03, 9.04878107e-02]
        ])

def test_minimise_mi_distance(n2mi):
    unpaired = minimise_mi_distance(n2mi, pair_spins=False, spinless_mi=False)
    assert set(unpaired).symmetric_difference({*range(n2mi.shape[0])}) == set()

    paired = minimise_mi_distance(n2mi, pair_spins=True, spinless_mi=False)
    assert np.all((paired[1::2] - paired[0::2]) == 1)
    assert set(paired).symmetric_difference({*range(n2mi.shape[0])}) == set()


def test_distance_squared(n2mi):
    permutation = [*range(n2mi.shape[0])]
    forwards = distance_squared(n2mi, permutation)
    permutation.reverse()
    backwards = distance_squared(n2mi, permutation)

    assert len(forwards) == len(backwards) == 1
    assert forwards[0] == backwards[0] == np.float64(21.85687074218722)
    assert distance_squared(n2mi , [0,9,1,8,2,7,3,6,4,5]) == np.float64(37.944815941146125)
    assert distance_squared(n2mi , [*range(n2mi.shape[0]-1)]) == [np.inf]
    assert distance_squared(n2mi , [*range(n2mi.shape[0]+1)]) == [np.inf]
    assert distance_squared(n2mi , [*range(1, n2mi.shape[0]+1)]) == [np.inf]


def test_rett(n2mi):
    np.random.seed(1017)
    rett = reduced_entanglement_ternary_tree(n2mi, squash=True)
    assert rett.branch_operator_map == {'zzzzx': 'ZZZZXIIIII', 'zzzx': 'ZZZXIIIIII', 'zzzzzzx': 'ZZZZZZXIII', 'zzx': 'ZZXIIIIIII', 'zzy': 'ZZYIIIIIII', 'zzzzzzzzx': 'ZZZZZZZZXI', 'zzzzy': 'ZZZZYIIIII', 'zzzzzzzzzy': 'ZZZZZZZZZY', 'y': 'YIIIIIIIII', 'zzzzzzzx': 'ZZZZZZZXII', 'zx': 'ZXIIIIIIII', 'zzzzzzy': 'ZZZZZZYIII', 'zy': 'ZYIIIIIIII', 'zzzy': 'ZZZYIIIIII', 'zzzzzzzzzz': 'ZZZZZZZZZZ', 'zzzzzzzy': 'ZZZZZZZYII', 'zzzzzy': 'ZZZZZYIIII', 'zzzzzzzzzx': 'ZZZZZZZZZX', 'zzzzzx': 'ZZZZZXIIII', 'zzzzzzzzy': 'ZZZZZZZZYI', 'x': 'XIIIIIIIII'}

def test_huffman(water_integrals):
    ones, twos = water_integrals
    tree = huffman_ternary_tree(ones, twos)
    tree_dict = {'x': {'x': {'x': {}, 'y': {}, 'z': {}},
                       'y': {'x': {}, 'y': {}, 'z': {}},
                       'z': {'x': {}, 'y': {}, 'z': {}}},
                       'y': {'x': {'x': {}, 'y': {}, 'z': {}},
                             'y': {'x': {},
                                   'y': {'x': {}, 'y': {}, 'z': {}},
                                   'z': {'x': {}, 'y': {}, 'z': {}}},
                                   'z': {'x': {'x': {}, 'y': {}, 'z': {}},
                                         'y': {'x': {}, 'y': {}, 'z': {}},
                                         'z': {'x': {}, 'y': {}, 'z': {}}}},
                                         'z': {}}
    assert tree.as_dict() == tree_dict
    assert tree.string_pairs == {
        '': ('xzz', 'yzzz'),
        'x': ('xxz', 'xyz'),
        'y': ('yyzz', 'yxz'),
        'xx': ('xxx', 'xxy'),
        'xy': ('xyy', 'xyx'),
        'xz': ('xzx', 'xzy'),
        'yx': ('yxy', 'yxx'),
        'yy': ('yyx', 'yyyz'),
        'yz': ('yzyz', 'yzxz'),
        'yyy': ('yyyy', 'yyyx'),
        'yyz': ('yyzx', 'yyzy'),
        'yzx': ('yzxy', 'yzxx'),
        'yzy': ('yzyx', 'yzyy'),
        'yzz': ('yzzy', 'yzzx')
        }

def test_pauli_weighted_norm(water_integrals):
    jw = TernaryTree(14).JW()
    ipowers, symplectics = jw._build_symplectic_matrix()
    ones, twos = water_integrals
    jw_pauli_ham = molecular_hamiltonian_template(ipowers, symplectics, True)
    jw_filled_template = fill_template(jw_pauli_ham, 0., ones, twos, jw.default_mode_op_map)
    jw_norm = pauli_weighted_norm(jw_filled_template)

    assert np.allclose(jw_norm, [np.float64(272.4190655251233)])

    pe = TernaryTree(14).ParityEncoding()
    ipowers, symplectics = pe._build_symplectic_matrix()
    pe_template = molecular_hamiltonian_template(ipowers, symplectics, True)
    pe_filled_template = fill_template(pe_template, 0, ones, twos, pe.default_mode_op_map)
    pe_norm = pauli_weighted_norm(pe_filled_template)
    assert np.allclose(pe_norm, [np.float64(354.23056347814577)])


def test_bonsai():

    graph=rx.PyGraph()
    graph.add_nodes_from(range(37))
    graph.add_edges_from_no_data([(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),
                                (4,7),(4,8),(5,9),(5,10),(6,11),(6,12),
                                (7,13),(8,14),(9,15),(10,16),(11,17),(12,18),
                                (13,19),(13,20),(14,21),(14,22),(15,23),(15,24),
                                (16,25),(16,26),(17,27),(17,28),(18,29),(18,30),
                                (22,31),(26,32),(30,33),(31,34),(32,35),(33,36),
                                ])

    bonsai_homo = bonsai_algorithm(graph=graph, homogenous=True)
    assert bonsai_homo.as_dict() == {'x': {'x': {'x': {'x': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {'x': {'x': {'x': {}, 'y': {}, 'z': {}}, 'y': {}, 'z': {}},
        'y': {},
        'z': {}},
        'z': {}},
        'y': {},
        'z': {}},
    'y': {'x': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {'x': {}, 'y': {}, 'z': {}},
        'z': {}},
        'y': {},
        'z': {}},
    'z': {}},
    'y': {},
    'z': {}},
    'y': {'x': {'x': {'x': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {'x': {}, 'y': {}, 'z': {}},
        'z': {}},
        'y': {},
        'z': {}},
    'y': {'x': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {'x': {'x': {'x': {}, 'y': {}, 'z': {}}, 'y': {}, 'z': {}},
        'y': {},
        'z': {}},
        'z': {}},
        'y': {},
        'z': {}},
    'z': {}},
    'y': {},
    'z': {}},
    'z': {'x': {'x': {'x': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {'x': {}, 'y': {}, 'z': {}},
        'z': {}},
        'y': {},
        'z': {}},
    'y': {'x': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {'x': {'x': {'x': {}, 'y': {}, 'z': {}}, 'y': {}, 'z': {}},
        'y': {},
        'z': {}},
        'z': {}},
        'y': {},
        'z': {}},
    'z': {}},
    'y': {},
    'z': {}}}

    bonsai_hetero = bonsai_algorithm(graph=graph, homogenous=False)
    assert bonsai_hetero.as_dict() == {'x': {'x': {},
    'y': {},
    'z': {'x': {'x': {},
        'y': {},
        'z': {'x': {'x': {},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {'x': {}, 'y': {}, 'z': {}}}},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {}}}},
    'y': {},
    'z': {'x': {},
        'y': {},
        'z': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {}}}}}},
    'y': {'x': {},
    'y': {},
    'z': {'x': {'x': {},
        'y': {},
        'z': {'x': {'x': {},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {'x': {}, 'y': {}, 'z': {}}}},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {}}}},
    'y': {},
    'z': {'x': {},
        'y': {},
        'z': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {}}}}}},
    'z': {'x': {},
    'y': {},
    'z': {'x': {'x': {},
        'y': {},
        'z': {'x': {'x': {}, 'y': {}, 'z': {}},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {}}}},
    'y': {},
    'z': {'x': {},
        'y': {},
        'z': {'x': {'x': {},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {'x': {}, 'y': {}, 'z': {}}}},
        'y': {},
        'z': {'x': {}, 'y': {}, 'z': {}}}}}}}

    assert bonsai_hetero.root_node.child_qubit_labels == {'': 0,
    'x': 2,
    'y': 3,
    'z': 1,
    'xz': 5,
    'yz': 6,
    'zz': 4,
    'xzx': 10,
    'xzz': 9,
    'yzx': 12,
    'yzz': 11,
    'zzx': 7,
    'zzz': 8,
    'xzxz': 16,
    'xzzz': 15,
    'yzxz': 18,
    'yzzz': 17,
    'zzxz': 13,
    'zzzz': 14,
    'xzxzx': 26,
    'xzxzz': 25,
    'xzzzx': 23,
    'xzzzz': 24,
    'yzxzx': 30,
    'yzxzz': 29,
    'yzzzx': 28,
    'yzzzz': 27,
    'zzxzx': 20,
    'zzxzz': 19,
    'zzzzx': 22,
    'zzzzz': 21,
    'xzxzxz': 32,
    'yzxzxz': 33,
    'zzzzxz': 31,
    'xzxzxzz': 35,
    'yzxzxzz': 36,
    'zzzzxzz': 34}
