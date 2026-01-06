import numpy as np
import numpy.typing as npt
from ferrmion.core import anneal_enumerations
from ferrmion.encode import FermionQubitEncoding
from ferrmion.hamiltonians import FermionHamiltonian


def anneal_pauli_weight(
    encoding: FermionQubitEncoding,
    hamiltonian: FermionHamiltonian,
    initial_guess: list[int] | None = None,
    temperature: int | None = None,
) -> tuple[float, npt.NDArray[np.uint]]:
    ipow, sym = encoding._build_symplectic_matrix()
    sigs, coeffs = hamiltonian.signatures_and_coefficients

    if initial_guess is None:
        initial_guess_array = np.arange(hamiltonian.n_modes, dtype=np.uint)
    elif isinstance(initial_guess, list)
        initial_guess_array = np.array([*initial_guess], dtype=np.uint)

    if temperature is None:
        temperature = hamiltonian.n_modes

    return anneal_enumerations(
        ipowers=ipow,
        symplectics=sym,
        signatures=sigs,
        coeffs=coeffs,
        temperature=temperature,
        initial_guess=initial_guess_array,
        coefficient_weighted=False,
    )


def anneal_coefficient_pauli_weight(
    encoding: FermionQubitEncoding,
    hamiltonian: FermionHamiltonian,
    initial_guess: list[int] | None = None,
    temperature: int | None = None,
) -> tuple[float, npt.NDArray[np.uint]]:
    ipow, sym = encoding._build_symplectic_matrix()
    sigs, coeffs = hamiltonian.signatures_and_coefficients

    if initial_guess is None:
        initial_guess_array = np.arange(hamiltonian.n_modes, dtype=np.uint)
    elif isinstance(initial_guess, list)
        initial_guess_array = np.array([*initial_guess], dtype=np.uint)

    if temperature is None:
        temperature = hamiltonian.n_modes

    return anneal_enumerations(
        ipowers=ipow,
        symplectics=sym,
        signatures=sigs,
        coeffs=coeffs,
        temperature=temperature,
        initial_guess=initial_guess_array,
        coefficient_weighted=True,
    )
