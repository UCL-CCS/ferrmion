from ferrmion.core import anneal_enumerations
from ferrmion.encode import FermionQubitEncoding


def anneal_pauli_weight(
    encoding: FermionQubitEncoding,
    hamlitonian: FermionHamiltonian,
    initial_guess: list[int] | None,
    temperature: int,
):
    pass


def anneal_coefficient_pauli_weight(
    encoding: FermionQubitEncoding,
    hamlitonian: FermionHamiltonian,
    initial_guess: list[int] | None,
    temperature: int,
):
    pass
