"""Shared Fixtures for tests."""

import json
import numpy as np
from pytest import fixture
from ferrmion.encode import TernaryTree
from pathlib import Path
from openfermion import InteractionOperator, jordan_wigner, get_sparse_operator, QubitOperator
from scipy.sparse.linalg import eigsh
from ferrmion.utils import fermionic_to_sparse_majorana
from ferrmion.hamiltonians import FermionHamiltonian


@fixture(scope="module")
def water_data():
    folder = Path(__file__).parent
    with open(folder.joinpath("./data/h2o_sto-3g.json"), "rb") as file:
        data = json.load(file)
    data["ones"] = np.array(data["ones"])
    data["twos"] = np.array(data["twos"])
    return data

@fixture(scope="module", params=["h2_sto-3g","h2_6-31g"])
def h2_mol_data_sets(request):
    folder = Path(__file__).parent
    with open(folder.joinpath(f"./data/{request.param}.json"), "rb") as file:
        data = json.load(file)
    data["ones"] = np.array(data["ones"])
    data["twos"] = np.array(data["twos"])
    return data

@fixture(scope="module")
def water_tt(water_data) -> TernaryTree:
    return TernaryTree(14)

# @fixture(scope="module")
# def water_MaxNTO(water_data) -> MaxNTO:
#     return MaxNTO(*water_data)


@fixture(scope="module")
def water_eigenvalues(water_data) -> list[float]:
    return water_data["eigvals"]


@fixture(scope="module")
def water_fham(water_data) -> FermionHamiltonian:
    ones, twos = water_data
    return FermionHamiltonian(terms={"+-":water_data["ones"], "++--":water_data["twos"]})

@fixture(scope="module")
def water_sparse_majorana(water_data) -> dict:
    return fermionic_to_sparse_majorana(((water_data["ones"], "+-"), (water_data["twos"], "++--")))

def diagonalise_pauli_hamiltonian(qham, neigvals:int):
    ofop = QubitOperator()
    for k, v in qham.items():
        string = " ".join(
            [
                f"{char.upper()}{pos}" if char != "I" else ""
                for pos, char in enumerate(k)
            ]
        )
        ofop+= QubitOperator(term=string, coefficient=v)
    diag, _ = eigsh(get_sparse_operator(ofop), k=neigvals, which="SA")
    return diag
