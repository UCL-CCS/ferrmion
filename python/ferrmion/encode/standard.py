"""Functions for running naive and optimised encodings."""

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ferrmion.core import (
    anneal_enumerations,
    encode,
    encode_standard,
    standard_symplectic_matrix,
    topphatt_standard,
)

from ..hamiltonians import FermionHamiltonian, QubitHamiltonian


class CoreEncodingTypes(Enum):
    JordanWigner = "JW"
    BravyiKitaev = "BK"
    ParityEncoding = "PE"
    JKMN = "JKMN"


def jordan_wigner(fham: FermionHamiltonian) -> QubitHamiltonian:
    """Naive Jordan-Wigner Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    sigs, coeffs = fham.signatures_and_coefficients
    return encode_standard(
        encoding="JW",
        n_modes=fham.n_modes,
        n_qubits=fham.n_modes,
        signatures=sigs,
        coeffs=coeffs,
        constant_energy=fham.constant_energy,
    )


def bravyi_kitaev(fham: FermionHamiltonian) -> QubitHamiltonian:
    """Naive Bravyi-Kitaev Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    sigs, coeffs = fham.signatures_and_coefficients
    return encode_standard(
        encoding="BK",
        n_modes=fham.n_modes,
        n_qubits=fham.n_modes,
        signatures=sigs,
        coeffs=coeffs,
        constant_energy=fham.constant_energy,
    )


def parity(fham: FermionHamiltonian) -> QubitHamiltonian:
    """Naive Parity Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    sigs, coeffs = fham.signatures_and_coefficients
    return encode_standard(
        encoding="PE",
        n_modes=fham.n_modes,
        n_qubits=fham.n_modes,
        signatures=sigs,
        coeffs=coeffs,
        constant_energy=fham.constant_energy,
    )


def jkmn(fham: FermionHamiltonian) -> QubitHamiltonian:
    """Naive Jiang-Kalev-Mruczkiewicz-Neven Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    sigs, coeffs = fham.signatures_and_coefficients
    return encode_standard(
        encoding="JKMN",
        n_modes=fham.n_modes,
        n_qubits=fham.n_modes,
        signatures=sigs,
        coeffs=coeffs,
        constant_energy=fham.constant_energy,
    )


def jordan_wigner_topphatt(fham: FermionHamiltonian) -> QubitHamiltonian:
    """TOPP-HATT optimised Jordan-Wigner Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return _standard_topphatt("JW", fham)


def bravyi_kitaev_topphatt(fham: FermionHamiltonian) -> QubitHamiltonian:
    """TOPP-HATT optimised Bravyi-Kitaev Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return _standard_topphatt("BK", fham)


def parity_topphatt(fham: FermionHamiltonian) -> QubitHamiltonian:
    """TOPP-HATT optimised Parity Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return _standard_topphatt("PE", fham)


def jkmn_topphatt(fham: FermionHamiltonian) -> QubitHamiltonian:
    """TOPP-HATT optimised Jiang-Kalev-Mruczkiewicz-Neven Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return _standard_topphatt("JKMN", fham)


def _standard_topphatt(encoding: str, fham: FermionHamiltonian) -> QubitHamiltonian:
    sigs, coeffs = fham.signatures_and_coefficients
    ipow, sym = topphatt_standard(
        encoding=encoding,
        n_modes=fham.n_modes,
        n_qubits=fham.n_modes,
        signatures=sigs,
        coeffs=coeffs,
    )
    return encode(
        ipowers=ipow,
        symplectics=sym,
        signatures=sigs,
        coeffs=coeffs,
        constant_energy=fham.constant_energy,
    )


def jordan_wigner_annealed(
    fham: FermionHamiltonian,
    temperature: int | None = None,
    initial_guess: list[int] | None = None,
    coefficient_weighted: bool = True,
) -> QubitHamiltonian:
    """TOPP-HATT optimised Jordan-Wigner Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.
        temperature (Optional[int]): Initial annealing temperature.
        initial_guess (Optional[list[int]]): Initial mode enumeration.
        coefficient_weighted (bool): True to minimise coefficient Pauli-weight.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return _standard_annealed(
        "JW", fham, temperature, initial_guess, coefficient_weighted
    )


def bravyi_kitaev_annealed(
    fham: FermionHamiltonian,
    temperature: int | None = None,
    initial_guess: list[int] | None = None,
    coefficient_weighted: bool = True,
) -> QubitHamiltonian:
    """TOPP-HATT optimised Bravyi-Kitaev Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.
        temperature (Optional[int]): Initial annealing temperature.
        initial_guess (Optional[list[int]]): Initial mode enumeration.
        coefficient_weighted (bool): True to minimise coefficient Pauli-weight.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return _standard_annealed(
        "BK", fham, temperature, initial_guess, coefficient_weighted
    )


def parity_annealed(
    fham: FermionHamiltonian,
    temperature: int | None = None,
    initial_guess: list[int] | None = None,
    coefficient_weighted: bool = True,
) -> QubitHamiltonian:
    """TOPP-HATT optimised Parity Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.
        temperature (Optional[int]): Initial annealing temperature.
        initial_guess (Optional[list[int]]): Initial mode enumeration.
        coefficient_weighted (bool): True to minimise coefficient Pauli-weight.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return _standard_annealed(
        "PE", fham, temperature, initial_guess, coefficient_weighted
    )


def jkmn_annealed(
    fham: FermionHamiltonian,
    temperature: int | None = None,
    initial_guess: list[int] | None = None,
    coefficient_weighted: bool = True,
) -> QubitHamiltonian:
    """TOPP-HATT optimised Jiang-Kalev-Mruczkiewicz-Neven Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.
        temperature (Optional[int]): Initial annealing temperature.
        initial_guess (Optional[list[int]]): Initial mode enumeration.
        coefficient_weighted (bool): True to minimise coefficient Pauli-weight.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return _standard_annealed(
        "JKMN", fham, temperature, initial_guess, coefficient_weighted
    )


def _standard_annealed(
    encoding: str,
    fham: FermionHamiltonian,
    temperature: int | None,
    initial_guess: list[int] | None,
    coefficient_weighted: bool,
):
    sigs, coeffs = fham.signatures_and_coefficients
    ipow, sym = standard_symplectic_matrix(encoding, fham.n_modes)

    if temperature is None:
        temperature = fham.n_modes

    if isinstance(initial_guess, list):
        initial_guess: NDArray[np.uint] = np.array(initial_guess, dtype=np.uint)
    else:
        initial_guess: NDArray[np.uint] = np.array(
            [*range(fham.n_modes)], dtype=np.uint
        )

    anneal_enumerations(
        ipowers=ipow,
        symplectics=sym,
        signatures=sigs,
        coeffs=coeffs,
        temperature=temperature,
        initial_guess=initial_guess,
        coefficient_weighted=coefficient_weighted,
    )

    return encode(
        ipowers=ipow,
        symplectics=sym,
        signatures=sigs,
        coeffs=coeffs,
        constant_energy=fham.constant_energy,
    )
