"""Tests for the Clifford heuristic encoding optimisation."""
from qiskit.primitives.containers.bit_array import _WEIGHT_LOOKUP

import numpy as np
import pytest
from ferrmion import QubitHamiltonian, TernaryTree, molecular_hamiltonian
from ferrmion.encode.ternary_tree import JordanWigner, BravyiKitaev, ParityEncoding, JKMN
from ferrmion.optimize.cost_functions import coefficient_pauli_weight, pauli_weight
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh
from ..conftest import ENERGY_TOLERANCE, WEIGHT_TOLERANCE


def _qham_to_ofop(qham: dict) -> QubitOperator:
    ofop = QubitOperator()
    for k, v in qham.items():
        string = " ".join(
            f"{char.upper()}{pos}" if char != "I" else ""
            for pos, char in enumerate(k)
        )
        ofop += QubitOperator(term=string, coefficient=v)
    return ofop


@pytest.mark.parametrize("encoding_cls", [JordanWigner, ParityEncoding, BravyiKitaev, JKMN])
def test_clifford_heuristic_does_not_increase_pauli_weight(encoding_cls, h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    baseline_qham = encoding_cls(n_modes).encode(fham)
    baseline_weight = pauli_weight(baseline_qham)[0]

    opt_qham = baseline_qham.clifford_heuristic(
        temperature=float(n_modes),
        coefficient_weighted=False,
        seed=42,
    )
    opt_weight = pauli_weight(opt_qham)[0]

    assert opt_weight <= baseline_weight + WEIGHT_TOLERANCE


@pytest.mark.parametrize("encoding_cls", [JordanWigner, ParityEncoding, BravyiKitaev, JKMN])
def test_clifford_heuristic_does_not_increase_coeff_pauli_weight(encoding_cls, water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    baseline_qham = encoding_cls(n_modes).encode(fham)
    baseline_weight = coefficient_pauli_weight(baseline_qham)[0]

    opt_qham = baseline_qham.clifford_heuristic(
        temperature=2 * float(n_modes),
        coefficient_weighted=True,
        seed=42,
    )
    opt_weight = coefficient_pauli_weight(opt_qham)[0]

    assert opt_weight <= baseline_weight + WEIGHT_TOLERANCE


def test_clifford_heuristic_seed_is_reproducible(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    qham = JKMN(n_modes).encode(fham)

    def run(seed):
        return qham.clifford_heuristic(
            temperature=float(n_modes),
            coefficient_weighted=False,
            seed=seed,
        )

    assert run(42) == run(42)


def test_clifford_heuristic_preserves_constant_energy(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    n_modes = ones.shape[0]
    constant_energy = 3.14

    fham_base = molecular_hamiltonian(ones, twos, 0.0)
    fham_const = molecular_hamiltonian(ones, twos, constant_energy)

    enc = JKMN(n_modes)
    qham_base = enc.encode(fham_base).clifford_heuristic(
        temperature=float(n_modes), coefficient_weighted=False, seed=42,
    )
    qham_const = JKMN(n_modes).encode(fham_const).clifford_heuristic(
        temperature=float(n_modes), coefficient_weighted=False, seed=42,
    )

    identity_key = "I" * qham_const.n_qubits
    base_identity = qham_base.get(identity_key, complex(0)).real
    assert identity_key in qham_const
    assert abs(qham_const[identity_key].real - base_identity - constant_energy) < ENERGY_TOLERANCE


def test_clifford_heuristic_preserves_eigenvalues(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    qham = JKMN(n_modes).encode(fham).clifford_heuristic(seed=42)

    assert len(qham) > 0
    assert isinstance(qham, QubitHamiltonian)

    ofop = _qham_to_ofop(qham)
    diag, _ = eigsh(get_sparse_operator(ofop), k=2 * n_modes, which="SA")
    assert np.allclose(np.sort(diag), np.sort(h2_mol_data_sets["eigvals"]))


def test_randomised_subsystem_descent_does_not_increase_pauli_weight(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    baseline_qham = JordanWigner(n_modes).encode(fham)
    baseline_weight = pauli_weight(baseline_qham)[0]

    opt_qham = baseline_qham.randomised_subsystem_descent(
        iterations=4,
        subsystem_dimension=max(2, baseline_qham.n_qubits // 2),
        temperature=float(n_modes),
        coefficient_weighted=False,
        sampler="uniform",
        seed=42,
    )

    assert isinstance(opt_qham, QubitHamiltonian)
    assert opt_qham.n_qubits == baseline_qham.n_qubits
    assert pauli_weight(opt_qham)[0] <= baseline_weight


@pytest.mark.parametrize("sampler", ["full_system", "uniform", "hamming"])
def test_randomised_subsystem_descent_samplers(sampler, h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    qham = JKMN(n_modes).encode(fham)

    opt = qham.randomised_subsystem_descent(
        iterations=2,
        subsystem_dimension=2,
        temperature=float(n_modes),
        sampler=sampler,
        seed=7,
    )
    assert isinstance(opt, QubitHamiltonian)
    assert opt.n_qubits == qham.n_qubits


def test_randomised_subsystem_descent_unknown_sampler_raises(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    n_modes = ones.shape[0]
    fham = molecular_hamiltonian(ones, twos, 0.0)
    qham = JordanWigner(n_modes).encode(fham)

    with pytest.raises(ValueError):
        qham.randomised_subsystem_descent(
            iterations=1, subsystem_dimension=2, sampler="bogus", seed=0,
        )


def test_clifford_heuristic_vp_does_not_increase_pauli_weight(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    baseline_qham = JKMN(n_modes).encode(fham)
    baseline_weight = pauli_weight(baseline_qham)[0]

    opt_qham = baseline_qham.clifford_heuristic(
        temperature=float(n_modes),
        coefficient_weighted=False,
        seed=42,
        clifford_subset="vp",
    )

    assert pauli_weight(opt_qham)[0] <= baseline_weight


def test_clifford_heuristic_vp_seed_is_reproducible(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    qham = JKMN(n_modes).encode(fham)

    def run():
        return qham.clifford_heuristic(
            temperature=float(n_modes),
            coefficient_weighted=False,
            seed=42,
            clifford_subset="vp",
        )

    assert run() == run()


def test_clifford_heuristic_vp_preserves_eigenvalues(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    qham = JKMN(n_modes).encode(fham).clifford_heuristic(
        seed=42, clifford_subset="vp",
    )

    assert isinstance(qham, QubitHamiltonian)
    ofop = _qham_to_ofop(qham)
    diag, _ = eigsh(get_sparse_operator(ofop), k=2 * n_modes, which="SA")
    assert np.allclose(np.sort(diag), np.sort(h2_mol_data_sets["eigvals"]))


def test_clifford_heuristic_unknown_subset_raises(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    n_modes = ones.shape[0]
    fham = molecular_hamiltonian(ones, twos, 0.0)
    qham = JordanWigner(n_modes).encode(fham)

    with pytest.raises(ValueError):
        qham.clifford_heuristic(seed=0, clifford_subset="bogus")
