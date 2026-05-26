"""Utility functions for tutorial and demo notebooks."""
import ferrmion as fr
import numpy as np
from pathlib import Path
import json

def get_water_data():
    folder = Path.cwd().joinpath(Path("../../../python/tests/data/"))

    # Takes a little too long to run for readthedocs
    # with open(folder.joinpath("h2o_6-31g.json"), 'r') as file:
    #     data = json.load(file)

    with open(folder.joinpath("h2o_6-31g.json")) as file:
        data = json.load(file)

    ones = np.array(data["ones"])
    twos = np.array(data["twos"])
    enuc = data["constant_energy"]
    return ones, twos, enuc

def pauli_weights(pauli_hamiltonian: dict[str, float]) -> tuple[float, float, int]:
    unscaled_terms = []
    scaled_terms = []
    for k, v in pauli_hamiltonian.items():
        assert v != 0
        pw = len(k) - k.count("I")
        unscaled_terms.append(pw)
        scaled_terms.append(pw * np.abs(v))

    return (
        int(np.sum(unscaled_terms)),
        float(np.sum(scaled_terms)),
        len(pauli_hamiltonian),
    )



def get_naive_result(tree: fr.TernaryTree, fham: fr.FermionHamiltonian):
    print("Getting Naive result...")
    sdmeans_naive = {}
    ipow, sym = tree._build_symplectic_matrix()
    result = pauli_weights(tree.encode(fham))
    print("Naive result:", result)

    sdmeans_naive = {}
    sdmeans_naive["unscaled"] = result[0]
    sdmeans_naive["scaled"] = result[1]
    sdmeans_naive["length"] = result[2]
    return sdmeans_naive


def get_permuation_results(
    tree: fr.FermionQubitEncoding, fham: fr.FermionHamiltonian, limit: int
):
    print(f"Getting {limit} random results...")
    sdmeans = {}
    sdmeans = {"unscaled": [], "scaled": []}
    weights = tree._batch_pauli_weights(fham, limit)
    sdmeans["unscaled"] = [w for w in weights[0]]
    sdmeans["scaled"] = [w for w in weights[1]]
    print("Permutation Results")

    return sdmeans


def get_annealed_result(tree, fham, coef_weight, n_seeds=10):
    print(f"Getting Annealed results ({n_seeds} seeds, coef_weight={coef_weight})...")
    sdmeans_annealed = {"unscaled": [], "scaled": [], "length": []}
    for seed in range(n_seeds):
        # To avoid re-annealing between weighted and unweighted (and across seeds)
        encoding = fr.MajoranaStringEncoding(*tree._build_symplectic_matrix())
        result = pauli_weights(
            encoding.encode_annealed(fham, coefficient_weighted=coef_weight, seed=seed)
        )
        sdmeans_annealed["unscaled"].append(result[0])
        sdmeans_annealed["scaled"].append(result[1])
        sdmeans_annealed["length"].append(result[2])
    return sdmeans_annealed


def get_topphatt_result(tree, fham, n_random=10):
    print("Running rust TOPP-HATT...")

    results = {}
    for heuristic in ("min_weight", "z_first", "x_first"):
        print(f"Getting rust TOPP-HATT result (heuristic={heuristic})...")
        result = pauli_weights(tree.encode_topphatt(fham, heuristic=heuristic))
        print(f"TOPP-HATT {heuristic} result:", result)
        results[heuristic] = {
            "unscaled": result[0],
            "scaled": result[1],
            "length": result[2],
        }

    print(f"Getting {n_random} rust TOPP-HATT random results...")
    random_results = {"unscaled": [], "scaled": [], "length": []}
    for seed in range(n_random):
        result = pauli_weights(
            tree.encode_topphatt(fham, heuristic="random", seed=seed)
        )
        random_results["unscaled"].append(result[0])
        random_results["scaled"].append(result[1])
        random_results["length"].append(result[2])
    results["random"] = random_results

    return results

def get_clifford_heuristic_result(tree,fham,coef_weight, temperature=1, n_seeds=10):
    print(f"Getting Clifford Heuristic results ({n_seeds} seeds, coef_weight={coef_weight})...")
    sdmeans = {"unscaled": [], "scaled": [], "length": []}
    for seed in range(n_seeds):
        # To avoid re-annealing between weighted and unweighted (and across seeds)
        result = pauli_weights(
            tree.encode_clifford_heuristic(fham, coefficient_weighted=coef_weight, seed=seed * np.random.randint(1,101), temperature=temperature)
        )
        sdmeans["unscaled"].append(result[0])
        sdmeans["scaled"].append(result[1])
        sdmeans["length"].append(result[2])
    return sdmeans
