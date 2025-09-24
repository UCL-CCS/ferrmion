"""Shared Fixtures for tests."""

import pickle
from pytest import fixture
from ferrmion.encode import TernaryTree
from pathlib import Path
from openfermion import InteractionOperator, jordan_wigner, get_sparse_operator
from scipy.sparse.linalg import eigsh

@fixture(scope="module")
def water_integrals():
    folder = Path(__file__).parent
    with open(folder.joinpath("./data/water_1e.pkl"), 'rb') as file:
        ones = pickle.load(file)

    with open(folder.joinpath("./data/water_2e.pkl"), 'rb') as file:
        twos = pickle.load(file)
    return (ones, twos)

@fixture(scope="module")
def water_tt(water_integrals) -> TernaryTree:
    return TernaryTree.from_hamiltonian_coefficients(water_integrals)

# @fixture(scope="module")
# def water_MaxNTO(water_integrals) -> MaxNTO:
#     return MaxNTO(*water_integrals)

@fixture(scope="module")
def water_eigenvalues(water_integrals) -> list[float]:
    qham = InteractionOperator(
        0, water_integrals[0], 0.5*water_integrals[1]
    )
    # print(qham)
    ofop = jordan_wigner(qham)
    # print(f"diff {ofop-ofop_zeros}")
    diag, _ = eigsh(get_sparse_operator(ofop), k=6, which="SA")
    return diag
