"""Tests for TOPP-HATT Algorithm."""
import json
import ferrmion as fr
from ferrmion.optimize.cost_functions import pauli_weight, coefficient_pauli_weight
from ferrmion.encode.ternary_tree import TernaryTree
import numpy as np
import pytest
from ferrmion.optimize.huffman import huffman_ternary_tree
from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh


@pytest.mark.parametrize(
    "encoding",
    [
        TernaryTree.jordan_wigner,
        TernaryTree.bravyi_kitaev,
        TernaryTree.parity,
        TernaryTree.jkmn,
    ],
)
def test_topphatt_preserves_topology(encoding, water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    tree: TernaryTree = encoding(14)
    print(tree.flatpack())
    _ = tree.encode_topphatt(fham)
    assert tree.root_node.child_strings == encoding(14).root_node.child_strings
    assert tree.root_node.branch_strings == encoding(14).root_node.branch_strings

def test_topphatt_huffman(water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    test_tree = huffman_ternary_tree(water_data["ones"], water_data["twos"])
    initial_children = test_tree.root_node.child_strings
    initial_branches = test_tree.root_node.branch_strings
    _ = test_tree.encode_topphatt(fham)
    assert test_tree.root_node.child_strings == initial_children
    assert test_tree.root_node.branch_strings == initial_branches

def test_topphatt_hatt(water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    test_tree = hamiltonian_adaptive_ternary_tree(fham, n_modes=14)
    initial_children = test_tree.root_node.child_strings
    initial_branches = test_tree.root_node.branch_strings
    _ = test_tree.encode_topphatt(fham)
    assert test_tree.root_node.child_strings == initial_children
    assert test_tree.root_node.branch_strings == initial_branches


@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN"])
@pytest.mark.parametrize("parallelize", [True, False])
def test_topphatt_standard_h2_eigvals_equal_expected(encoding, parallelize, h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    match encoding:
        case "JW":
            tree = TernaryTree.jordan_wigner(fham.n_modes)
        case "BK":
            tree = TernaryTree.bravyi_kitaev(fham.n_modes)
        case "PE":
            tree = TernaryTree.parity(fham.n_modes)
        case "JKMN":
            tree = TernaryTree.jkmn(fham.n_modes)
    qham, _ = tree.encode_topphatt(fham, parallelize)

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
@pytest.mark.parametrize("parallelize", [True, False])
def test_topphatt_standard_h2o_weights_not_increased(encoding, parallelize, water_data, topphatt_weight_snapshot):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    match encoding:
        case "JW":
            tree = TernaryTree.jordan_wigner(fham.n_modes)
        case "BK":
            tree = TernaryTree.bravyi_kitaev(fham.n_modes)
        case "PE":
            tree = TernaryTree.parity(fham.n_modes)
        case "JKMN":
            tree = TernaryTree.jkmn(fham.n_modes)
        case "HATT":
            tree = hamiltonian_adaptive_ternary_tree(fham, fham.n_modes)
        case "Huffman":
            tree = huffman_ternary_tree(ones, twos)
    qham_naive = tree.encode(fham)
    qham, _ = tree.encode_topphatt(fham, parallelize)

    assert np.isclose(float(pauli_weight(qham)[0]/pauli_weight(qham_naive)[0]),topphatt_weight_snapshot[encoding]["pauli_weight"], atol=0.01)
    assert np.isclose(float(coefficient_pauli_weight(qham)[0]/coefficient_pauli_weight(qham_naive)[0]), topphatt_weight_snapshot[encoding]["coefficient_pauli_weight"], atol=0.01)


@pytest.mark.parametrize("encoding", ["JW", "BK", "PE", "JKMN"])
def test_topphatt_dense_transpose_h2o_weights_match_snapshot(
    encoding, water_data, topphatt_weight_snapshot
):
    """Regression-pin the dense_transpose backend's output-tree quality on H2O sto-3g.

    Compares the dense_transpose topphatt Pauli weight and coefficient Pauli weight
    (each normalised to the naive encode) against a stored snapshot. The
    dense_transpose backend deduplicates whole terms on the same multiset rule as the
    default index_list backend, so it produces an identical encoding; this test
    locks in the resulting weights so a regression is caught.
    """
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    match encoding:
        case "JW":
            tree = TernaryTree.jordan_wigner(fham.n_modes)
        case "BK":
            tree = TernaryTree.bravyi_kitaev(fham.n_modes)
        case "PE":
            tree = TernaryTree.parity(fham.n_modes)
        case "JKMN":
            tree = TernaryTree.jkmn(fham.n_modes)
    qham_naive = tree.encode(fham)
    qham, _ = tree.encode_topphatt(fham, parallelize=False, backend="dense_transpose")

    snapshot = topphatt_weight_snapshot[encoding]
    assert np.isclose(
        float(pauli_weight(qham)[0] / pauli_weight(qham_naive)[0]),
        snapshot["pauli_weight"],
        atol=0.01,
    )
    assert np.isclose(
        float(coefficient_pauli_weight(qham)[0] / coefficient_pauli_weight(qham_naive)[0]),
        snapshot["coefficient_pauli_weight"],
        atol=0.01,
    )


@pytest.mark.parametrize("heuristic", ["min_weight", "x_first", "z_first", "random"])
def test_topphatt_heuristics_run(heuristic, h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    tree = TernaryTree.jkmn(fham.n_modes)
    qham, _ = tree.encode_topphatt(fham, parallelize=False, heuristic=heuristic, seed=42)
    assert len(qham) > 0


def test_topphatt_random_seed_is_reproducible(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)

    qham_a, enc_a = TernaryTree.jkmn(fham.n_modes).encode_topphatt(
        fham, parallelize=False, heuristic="random", seed=7
    )
    qham_b, enc_b = TernaryTree.jkmn(fham.n_modes).encode_topphatt(
        fham, parallelize=False, heuristic="random", seed=7
    )
    assert qham_a == qham_b
    assert enc_a == enc_b


def test_topphatt_unknown_heuristic_raises(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    with pytest.raises(ValueError):
        TernaryTree.jkmn(fham.n_modes).encode_topphatt(fham, heuristic="not_a_strategy")


def test_topphatt_heuristic_distribution_h2o_jkmn(water_data):
    """Compare Pauli weights produced by each heuristic on JKMN(14) + H2O sto-3g.

    Runs MinWeight / XFirst / ZFirst once each and Random across 10 seeds,
    then surfaces the comparison via stdout and checks that random selection
    does not systematically beat the best deterministic strategy in
    expectation.
    """
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    fham = fr.molecular_hamiltonian(ones, twos, e_nuc)
    n_modes = fham.n_modes

    def run(heuristic: str, seed: int | None = None) -> float:
        qham, _ = TernaryTree.jkmn(n_modes).encode_topphatt(
            fham, parallelize=False, heuristic=heuristic, seed=seed
        )
        return float(pauli_weight(qham)[0])

    weight_min_weight = run("min_weight")
    weight_x_first = run("x_first")
    weight_z_first = run("z_first")
    random_weights = np.array([run("random", seed=s) for s in range(10)])

    deterministic_min = min(weight_min_weight, weight_x_first, weight_z_first)

    print(
        "JKMN H2O sto-3g — min_weight=%.3f x_first=%.3f z_first=%.3f "
        "random: min=%.3f max=%.3f mean=%.3f std=%.3f n_unique=%d"
        % (
            weight_min_weight,
            weight_x_first,
            weight_z_first,
            random_weights.min(),
            random_weights.max(),
            random_weights.mean(),
            random_weights.std(),
            np.unique(random_weights).size,
        )
    )

    # All weights are positive and finite — every heuristic produced a usable encoding.
    assert weight_min_weight > 0 and np.isfinite(weight_min_weight)
    assert weight_x_first > 0 and np.isfinite(weight_x_first)
    assert weight_z_first > 0 and np.isfinite(weight_z_first)
    assert np.all(np.isfinite(random_weights)) and np.all(random_weights > 0)

    # Random seeds should explore more than one outcome on a branched tree.
    assert np.unique(random_weights).size >= 2
    assert random_weights.std() > 0

    # The best deterministic heuristic on this dataset should match or beat the
    # mean of uniformly-random node selection.
    assert deterministic_min <= random_weights.mean()
