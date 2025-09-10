"""Initialisation for Hamiltonians."""

from ferrmion.core import (
    fill_template,
    hubbard_hamiltonian_template,
    molecular_hamiltonian_template,
)

from .hubbard import (
    cube_adjacency_matrix,
    hubbard_hamiltonian,
    linear_adjacency_matrix,
    square_adjacency_matrix,
)
from .molecular import molecular_hamiltonian

__all__ = [
    "fill_template",
    "molecular_hamiltonian_template",
    "molecular_hamiltonian",
    "hubbard_hamiltonian",
    "hubbard_hamiltonian_template",
    "linear_hubbard_hamiltonian",
    "square_hubbard_hamiltonian",
    "linear_adjacency_matrix",
    "square_adjacency_matrix",
    "cube_adjacency_matrix",
]
