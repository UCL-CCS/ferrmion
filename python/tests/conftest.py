"""Shared Fixtures for tests."""

import pickle
from pytest import fixture
from ferrmion.encode import TernaryTree, KNTO
from pathlib import Path

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
    return TernaryTree(*water_integrals)

# @fixture(scope="module")
# def water_knto(water_integrals) -> KNTO:
#     return KNTO(*water_integrals)
