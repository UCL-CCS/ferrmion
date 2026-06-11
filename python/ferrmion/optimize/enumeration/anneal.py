"""Optimisation with simulated annealing."""

from ferrmion.core import FermionHamiltonian, MajoranaEncoding


def anneal_pauli_weight(
    encoding: MajoranaEncoding,
    hamiltonian: FermionHamiltonian,
    initial_guess: list[int] | None = None,
    temperature: int | None = None,
) -> tuple[float, MajoranaEncoding]:
    """Optimise over mode enumeration with Pauli-weight as cost function.

    Args:
        encoding (MajoranaEncoding): Encoding to optimise.
        hamiltonian (FermionHamiltonian): A hamiltonian of fermionic operators.
        initial_guess (list[int] | None): Optional inital enumeration for simulated annealing.
        temperature (int | None): Optional annealing temperature.

    Returns:
        tuple[float, MajoranaEncoding]: Best cost found, optimised encoding.
    """
    return encoding.anneal_enumeration(
        hamiltonian,
        temperature=temperature,
        initial_guess=initial_guess,
        coefficient_weighted=False,
    )


def anneal_coefficient_pauli_weight(
    encoding: MajoranaEncoding,
    hamiltonian: FermionHamiltonian,
    initial_guess: list[int] | None = None,
    temperature: int | None = None,
) -> tuple[float, MajoranaEncoding]:
    """Optimise over mode enumeration with coefficient Pauli-weight as cost function.

    Args:
        encoding (MajoranaEncoding): Encoding to optimise.
        hamiltonian (FermionHamiltonian): A hamiltonian of fermionic operators.
        initial_guess (list[int] | None): Optional inital enumeration for simulated annealing.
        temperature (int | None): Optional annealing temperature.

    Returns:
        tuple[float, MajoranaEncoding]: Best cost found, optimised encoding.
    """
    return encoding.anneal_enumeration(
        hamiltonian,
        temperature=temperature,
        initial_guess=initial_guess,
        coefficient_weighted=True,
    )
