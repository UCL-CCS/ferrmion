"""Optimize the Enumeration of an encoding using the mutual information."""

import logging
from functools import partial

import numpy as np
from numpy.typing import NDArray

from .evolutionary import lambda_plus_mu

logger = logging.getLogger(__name__)


def distance_squared(
    mutual_information: NDArray, permutation: list | NDArray
) -> list[float]:
    """Cost function weighting by the mutualinformation and mode index difference squared.

    Args:
        mutual_information (NDArray): The mutual information of modes.
        permutation (list | NDArray): A permutation of mode labels from 0 to M.

    Returns:
        list[float]: Calculated cost as a length 1 list. (Needed for DEAP.)
    distance matrix(6) = array([
        [ 0,  1,  4,  9, 16, 25],
        [ 1,  0,  1,  4,  9, 16],
        [ 4,  1,  0,  1,  4,  9],
        [ 9,  4,  1,  0,  1,  4],
        [16,  9,  4,  1,  0,  1],
        [25, 16,  9,  4,  1,  0]
        ])
    """
    n_mode = mutual_information.shape[0]
    if set(permutation) != set(range(n_mode)):
        logger.warning(
            "Not all modes included in permutation, returning infinite cost."
        )
        return [np.inf]
    distance_matrix = np.array(
        [[np.abs(i - j) ** 2 for j in range(n_mode)] for i in range(n_mode)]
    )
    return [
        np.sum(mutual_information[permutation, :][:, permutation] * distance_matrix)
    ]


def minimise_mi_distance(
    mutual_information: NDArray,
    pop_size: int = 500,
    ngen: int = 50,
    pair_spins: bool = False,
) -> NDArray:
    """Place modes with high mutual information near eachother.

    This uses the lambda_plus_mu evolutionary algorithm.

    Args:
        mutual_information (NDArray): A 2d array of mode mutual information values.
        pop_size (int): The size of the initial population.
        ngen (int): The number of generations to evolve.
        pair_spins (bool): Pair the alpha and beta spins so that they remain adjacent in the mode ordering.

    Returns:
        NDArray: The best mode ordering found.
    """
    logger.debug("Minimising disance between high MI modes.")
    if pair_spins:
        logger.debug("Pairing spins.")
        squash_rows = mutual_information[::2] + mutual_information[1::2]
        mutual_information = squash_rows[:, ::2] + squash_rows[:, 1::2]

    evaluate = partial(distance_squared, mutual_information)
    n_modes: int = mutual_information.shape[0]
    best = lambda_plus_mu(n_modes, evaluate, pop_size, ngen)

    if pair_spins:
        best = np.stack((2 * best, (2 * best) + 1)).T.flatten()
    logger.debug("Found best ordering %s", best)
    return best
