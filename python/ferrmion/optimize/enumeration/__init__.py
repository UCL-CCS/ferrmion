"""Init for enumeration optimizations."""

import numpy as np

from ferrmion.core import anneal_enumerations as core_anneal_enumerations

from .evolutionary import lambda_plus_mu


def anneal_enumerations(
    template: dict,
    one_e_coeffs: np.ndarray,
    two_e_coeffs: np.ndarray,
    temperature: float | None = None,
    initial_guess=np.typing.ArrayLike | None,
    coefficient_weighted: bool = False,
):
    """Optimise mode enumeration using simulated annealing.

    Args:
        template (dict): Encoding template dictionary.
        one_e_coeffs (np.ndarray): One-electron coefficient matrix.
        two_e_coeffs (np.ndarray): Two-electron coefficient matrix.
        temperature (float | None): Initial annealing temperature. Defaults to ``n_modes``.
        initial_guess (ArrayLike | None): Starting permutation. Defaults to identity.
        coefficient_weighted (bool): If True, minimise coefficient-weighted Pauli weight.
    """
    n_modes = one_e_coeffs.shape[0]
    if temperature is None:
        temperature = float(n_modes)
    if initial_guess is None:
        initial_guess = np.array([*range(n_modes)], dtype=np.uint)
    return core_anneal_enumerations(
        template=template,
        one_e_coeffs=one_e_coeffs,
        two_e_coeffs=two_e_coeffs,
        temperature=temperature,
        initial_guess=initial_guess,
        coefficient_weighted=coefficient_weighted,
    )


__all__ = ["lambda_plus_mu", "anneal_enumerations"]
