"""Tests for Simulated Annealing Optimisation."""
from typing import Callable
from ferrmion import TernaryTree

from ferrmion.optimize.topphatt import topphatt
from ferrmion.utils import fermionic_to_sparse_majorana
from ferrmion.encode.ternary_tree import (
    JordanWigner,
    BravyiKitaev,
    ParityEncoding,
    JKMN,
)
import numpy as np
import pytest
from ferrmion.core import anneal_enumerations, encode
from ferrmion.optimize.huffman import huffman_ternary_tree
from ferrmion.optimize.hatt import hamiltonian_adaptive_ternary_tree, fast_hatt
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse.linalg import eigsh



@pytest.mark.parametrize("encoding", [JordanWigner, ParityEncoding, BravyiKitaev, JKMN])
@pytest.mark.parametrize("coeff_weight", [True, False])
def test_core_topphatt_standard_h2_eigvals_equal_expected(encoding, coeff_weight, h2_mol_data_sets: dict):
    ones = h2_mol_data_sets["ones"]
    twos = h2_mol_data_sets["twos"]
    e_nuc = h2_mol_data_sets["constant_energy"]
    n_modes = ones.shape[0]

    ipow, sym = encoding(n_modes)._build_symplectic_matrix()
    anneal_enumerations(ipow, sym, ["+-","++--"], [ones, twos], n_modes, np.array([*range(n_modes)], dtype=np.uint), coeff_weight)
    qham = encode(ipow, sym, ["+-","++--"],[ones, twos], e_nuc)


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
    diag, _ = eigsh(get_sparse_operator(ofop), k=2*n_modes, which="SA")
    print(diag)
    assert np.allclose(np.sort(diag), np.sort(h2_mol_data_sets["eigvals"]))
