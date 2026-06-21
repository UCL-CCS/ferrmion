"""Tests for ``QubitHamiltonian.coloring_groups``."""

import numpy as np
import pytest
from ferrmion import QubitHamiltonian, molecular_hamiltonian
from ferrmion.encode.ternary_tree import JordanWigner, BravyiKitaev

try:
    from qiskit.quantum_info import SparsePauliOp, Pauli
except ImportError:
    SparsePauliOp = None
    Pauli = None

requires_qiskit = pytest.mark.skipif(
    SparsePauliOp is None,
    reason="Extra dependency `ferrmion[qiskit]` not installed.",
)


def _partition_covers_all(groups, n_terms):
    """Every term index appears in exactly one group."""
    flat = sorted(i for group in groups for i in group)
    return flat == list(range(n_terms))


def _groups_to_labels(qham, groups):
    """Map index groups back to their Pauli-string labels."""
    keys = qham.keys()
    return [[keys[i] for i in group] for group in groups]


def _all_pairs_commute(labels):
    """True if every pair of Pauli labels mutually commutes (via Qiskit)."""
    return all(
        Pauli(labels[i]).commutes(Pauli(labels[j]))
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    )


def _qubit_wise_commute(a, b):
    """True if two Pauli labels commute qubit-wise (equal or identity per qubit)."""
    return all(x == y or x == "I" or y == "I" for x, y in zip(a, b))


def _encoded_h2(data, encoding_cls):
    """Encode an H2 dataset into a ``QubitHamiltonian``."""
    ones = np.array(data["ones"])
    twos = np.array(data["twos"])
    fham = molecular_hamiltonian(ones, twos, data.get("constant_energy", 0.0))
    return encoding_cls(ones.shape[0]).encode(fham)


# A fixed, non-trivial Pauli set with a known-optimal commuting structure.
_FIXED_LABELS = ["XYZI", "ZZXX", "IXYZ", "YYII", "ZIIX", "IIZZ", "XXXX", "ZZZZ"]


def test_coloring_groups_partitions_all_terms():
    qham = QubitHamiltonian(
        {"XYZI": 1.0, "ZZXX": 1.0, "IXYZ": 1.0, "YYII": 1.0, "ZIIX": 1.0}
    )
    for conflict in ("support", "commutation"):
        n_groups, groups = qham.coloring_groups(conflict)
        assert n_groups == len(groups)
        assert _partition_covers_all(groups, len(qham))


def test_support_separates_overlapping_terms():
    # ZZII and XXII share qubits 0,1, so they cannot run in parallel.
    qham = QubitHamiltonian({"ZZII": 1.0, "XXII": 1.0})
    n_groups, groups = qham.coloring_groups("support")
    assert n_groups == 2
    assert _partition_covers_all(groups, 2)


def test_commutation_groups_overlapping_but_commuting_terms():
    # ZZII and XXII overlap but commute, so they belong to one commuting group.
    qham = QubitHamiltonian({"ZZII": 1.0, "XXII": 1.0})
    n_groups, groups = qham.coloring_groups("commutation")
    assert n_groups == 1
    assert sorted(groups[0]) == [0, 1]


def test_indices_map_back_to_terms():
    qham = QubitHamiltonian({"XIII": 1.0, "IIXI": 1.0})
    keys = qham.keys()
    n_groups, groups = qham.coloring_groups("support")
    # Disjoint supports -> a single parallel group containing both terms.
    assert n_groups == 1
    recovered = sorted(keys[i] for i in groups[0])
    assert recovered == ["IIXI", "XIII"]


def test_default_conflict_is_support():
    qham = QubitHamiltonian({"ZZII": 1.0, "XXII": 1.0})
    assert qham.coloring_groups() == qham.coloring_groups("support")


def test_unknown_conflict_raises():
    qham = QubitHamiltonian({"XIII": 1.0})
    with pytest.raises(ValueError, match="unknown conflict"):
        qham.coloring_groups("bogus")


# --- Comparison against Qiskit's grouping --------------------------------------
#
# ferrmion's ``"commutation"`` grouping and Qiskit's
# ``SparsePauliOp.group_commuting(qubit_wise=False)`` solve the same problem:
# both build a graph whose edges join *non-commuting* Paulis and greedily colour
# it. The partition into commuting groups is the hard correctness guarantee
# (verified independently with Qiskit's own commutation check); the group *count*
# is a heuristic estimate, so we assert ferrmion is never worse than Qiskit rather
# than demanding an exact match.


@requires_qiskit
@pytest.mark.parametrize(
    "labels",
    [
        ["ZZII", "XXII"],  # overlap but commute -> one group
        ["XIII", "ZIII"],  # anticommute -> two groups
        _FIXED_LABELS,
    ],
)
def test_commutation_groups_are_valid_vs_qiskit(labels):
    qham = QubitHamiltonian({label: 1.0 for label in labels})
    _, groups = qham.coloring_groups("commutation")

    assert _partition_covers_all(groups, len(qham))
    # Every ferrmion group must be internally mutually-commuting, per Qiskit.
    for group_labels in _groups_to_labels(qham, groups):
        assert _all_pairs_commute(group_labels)


@requires_qiskit
@pytest.mark.parametrize(
    "labels, expected",
    [
        (["ZZII", "XXII"], 1),
        (["XIII", "ZIII"], 2),
    ],
)
def test_commutation_group_count_matches_qiskit_small(labels, expected):
    qham = QubitHamiltonian({label: 1.0 for label in labels})
    n_groups, _ = qham.coloring_groups("commutation")
    qiskit_groups = SparsePauliOp(labels, coeffs=[1.0] * len(labels)).group_commuting(
        qubit_wise=False
    )
    # On these small inputs the optimum is unambiguous: both agree exactly.
    assert n_groups == len(qiskit_groups) == expected


@requires_qiskit
@pytest.mark.parametrize("encoding_cls", [JordanWigner, BravyiKitaev])
def test_commutation_group_count_matches_qiskit_h2(encoding_cls, h2_mol_data_sets):
    qham = _encoded_h2(h2_mol_data_sets, encoding_cls)

    n_groups, groups = qham.coloring_groups("commutation")

    # Valid partition into commuting groups (checked against Qiskit).
    assert _partition_covers_all(groups, len(qham))
    for group_labels in _groups_to_labels(qham, groups):
        assert _all_pairs_commute(group_labels)

    # Qiskit's `group_commuting` is order-sensitive (its greedy colouring depends
    # on term order), whereas ferrmion colours in a canonical sorted order. Feed
    # Qiskit that same canonical order for an apples-to-apples comparison of the two
    # greedy-colouring implementations on an identical graph.
    labels = sorted(qham.keys())
    coeffs = [qham[label] for label in labels]
    qiskit_groups = SparsePauliOp(labels, coeffs=coeffs).group_commuting(
        qubit_wise=False
    )
    assert n_groups == len(qiskit_groups)
    # Sanity: Qiskit's own groups are internally commuting too.
    for group in qiskit_groups:
        assert _all_pairs_commute(list(group.paulis.to_labels()))


@requires_qiskit
@pytest.mark.parametrize("encoding_cls", [JordanWigner, BravyiKitaev])
def test_support_groups_are_qubit_wise_commuting_h2(encoding_cls, h2_mol_data_sets):
    qham = _encoded_h2(h2_mol_data_sets, encoding_cls)
    labels = qham.keys()
    coeffs = [qham[label] for label in labels]

    n_groups, groups = qham.coloring_groups("support")

    # Disjoint support implies qubit-wise commutation, so every "support" group is
    # a valid qubit-wise-commuting group.
    for group_labels in _groups_to_labels(qham, groups):
        for i in range(len(group_labels)):
            for j in range(i + 1, len(group_labels)):
                assert _qubit_wise_commute(group_labels[i], group_labels[j])

    # Disjoint support is strictly stronger than qubit-wise commutation, so it
    # never yields fewer groups than Qiskit's qubit-wise grouping.
    qwc_groups = SparsePauliOp(labels, coeffs=coeffs).group_commuting(qubit_wise=True)
    assert n_groups >= len(qwc_groups)
