"""Tests for the Clifford heuristic encoding optimisation."""

import numpy as np
import pytest
from ferrmion import TernaryTree, molecular_hamiltonian
from ferrmion.core import clifford_heuristic_encoding, encode
from ferrmion.encode.ternary_tree import JordanWigner, BravyiKitaev, ParityEncoding, JKMN
from ferrmion.optimize.cost_functions import coefficient_pauli_weight, pauli_weight
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh


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

    enc = encoding_cls(n_modes)
    ipow, sym = enc._build_symplectic_matrix()
    vacuum = enc.vacuum_state.astype(bool)

    baseline_qham = encode(
        ipowers=ipow,
        symplectics=sym,
        vacuum_state=vacuum,
        signatures=["+-", "++--"],
        coeffs=[ones, twos],
        constant_energy=e_nuc,
    )
    baseline_weight = pauli_weight(baseline_qham)[0]

    opt_qham = clifford_heuristic_encoding(
        ipowers=ipow,
        symplectics=sym,
        signatures=["+-", "++--"],
        coeffs=[ones, twos],
        temperature=float(n_modes),
        coefficient_weighted=False,
        seed=42,
    )
    opt_weight = pauli_weight(opt_qham)[0]

    assert opt_weight <= baseline_weight


@pytest.mark.parametrize("encoding_cls", [JordanWigner, ParityEncoding, BravyiKitaev, JKMN])
def test_clifford_heuristic_does_not_increase_coeff_pauli_weight(encoding_cls, water_data):
    ones = water_data["ones"]
    twos = water_data["twos"]
    e_nuc = water_data["constant_energy"]
    n_modes = ones.shape[0]

    enc = encoding_cls(n_modes)
    ipow, sym = enc._build_symplectic_matrix()
    vacuum = enc.vacuum_state.astype(bool)

    baseline_qham = encode(
        ipowers=ipow,
        symplectics=sym,
        vacuum_state=vacuum,
        signatures=["+-", "++--"],
        coeffs=[ones, twos],
        constant_energy=e_nuc,
    )
    baseline_weight = coefficient_pauli_weight(baseline_qham)[0]

    opt_qham = clifford_heuristic_encoding(
        ipowers=ipow,
        symplectics=sym,
        signatures=["+-", "++--"],
        coeffs=[ones, twos],
        temperature=2*float(n_modes),
        coefficient_weighted=True,
        seed=42,
    )
    opt_weight = coefficient_pauli_weight(opt_qham)[0]

    assert opt_weight <= baseline_weight


def test_clifford_heuristic_seed_is_reproducible(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    n_modes = ones.shape[0]

    enc = JKMN(n_modes)
    ipow, sym = enc._build_symplectic_matrix()

    def run(seed):
        return clifford_heuristic_encoding(
            ipowers=ipow,
            symplectics=sym,
            signatures=["+-", "++--"],
            coeffs=[ones, twos],
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

    enc = JKMN(n_modes)
    ipow, sym = enc._build_symplectic_matrix()

    common_kwargs = dict(
        ipowers=ipow,
        symplectics=sym,
        signatures=["+-", "++--"],
        coeffs=[ones, twos],
        temperature=float(n_modes),
        coefficient_weighted=False,
        seed=42,
    )

    qham_base = clifford_heuristic_encoding(**common_kwargs)
    qham = clifford_heuristic_encoding(**common_kwargs, constant_energy=constant_energy)

    identity_key = "I" * (sym.shape[1] // 2)
    base_identity = qham_base.get(identity_key, complex(0)).real
    assert identity_key in qham
    assert abs(qham[identity_key].real - base_identity - constant_energy) < 1e-10


def test_encode_clifford_heuristic_method_preserves_eigenvalues(h2_mol_data_sets):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    fham = molecular_hamiltonian(ones, twos, e_nuc)
    qham = JKMN(n_modes).encode_clifford_heuristic(fham, seed=42)

    assert len(qham) > 0

    ofop = _qham_to_ofop(qham)
    diag, _ = eigsh(get_sparse_operator(ofop), k=2 * n_modes, which="SA")
    assert np.allclose(np.sort(diag), np.sort(h2_mol_data_sets["eigvals"]))
