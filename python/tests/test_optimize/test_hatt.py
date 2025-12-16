"""Tests for functions in the optimize submodule."""

from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree, _qubit_term_weight

def test_reduce_hamiltonian():
    assert _qubit_term_weight([0], [0,1,2]) == 1
    assert _qubit_term_weight([1], [0,1,2]) == 1
    assert _qubit_term_weight([2], [0,1,2]) == 1

    assert _qubit_term_weight([0, 0], [0,1,2]) == 0
    assert _qubit_term_weight([0, 1], [0,1,2]) == 1
    assert _qubit_term_weight([0, 1, 1], [0,1,2]) == 1
    assert _qubit_term_weight([0, 1, 1, 1], [0,1,2]) == 1

    assert _qubit_term_weight([0, 1,2], [0,1,2]) == 0
    assert _qubit_term_weight([0, 1,2,2], [0,1,2]) == 1

    assert _qubit_term_weight([0, 0], [0,1,2]) == 0
    assert _qubit_term_weight([0, 0,0], [0,1,2]) == 1
    assert _qubit_term_weight([0, 0,0,0], [0,1,2]) == 0

    assert _qubit_term_weight([0, 3,5,6], [0,1,2]) == 1

def test_hatt():
    majorana_ham = {(0, 1): 0.5j, (2, 3): -0.5j, (4, 5): -0.5j, (2, 3, 4, 5): 0.5}
    n_modes = 3
    hatt = hamiltonian_adaptive_ternary_tree(majorana_ham, n_modes)
    assert hatt.as_dict() == {
        "x": {"x": 2, "y": 3, "z": 4},
        "y": 5,
        "z": {"x": 0, "y": 1, "z": 6},
    }
    assert hatt.enumeration_scheme == {"": (0, 2), "x": (1, 1), "z": (2, 0)}
    assert hatt.root_node.branch_majorana_map == {
        "y": 5,
        "xx": 2,
        "xy": 3,
        "xz": 4,
        "zx": 0,
        "zy": 1,
        "zz": 6,
    }
