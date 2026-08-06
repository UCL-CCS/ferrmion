import numpy as np
import numpy.typing as npt

from .encode.ternary_tree import TTFlatpack

# Rust-backed classes and functions exposed to Python

class QubitHamiltonian:
    """Mapping from Pauli strings to complex coefficients."""

    def __init__(self, data: dict[str, complex] | None = None) -> None: ...
    @property
    def n_qubits(self) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: str) -> complex: ...
    def __setitem__(self, key: str, value: complex) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self): ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[complex]: ...
    def items(self) -> list[tuple[str, complex]]: ...
    def get(self, key: str, default=None): ...
    def to_dict(self) -> dict[str, complex]: ...
    def pauli_weight(self) -> int: ...
    def coeff_pauli_weight(self) -> float: ...
    def clifford_heuristic(
        self,
        temperature: float | None = None,
        coefficient_weighted: bool = False,
        seed: int | None = None,
        clifford_subset: str = "chs",
    ) -> "QubitHamiltonian": ...
    def randomised_subsystem_descent(
        self,
        iterations: int,
        subsystem_dimension: int,
        temperature: float | None = None,
        coefficient_weighted: bool = False,
        sampler: str = "hamming",
        seed: int | None = None,
        clifford_subset: str = "chs",
    ) -> "QubitHamiltonian": ...

class FermionHamiltonian:
    """Builder for fermionic Hamiltonians."""

    def __init__(
        self,
        *,
        terms: dict[str, npt.NDArray[np.float64]] | None = None,
        constant_energy: float = 0.0,
    ) -> None: ...
    @property
    def n_modes(self) -> int: ...
    constant_energy: float
    @property
    def terms(self) -> dict[str, npt.NDArray[np.float64]]: ...
    @property
    def signatures_and_coefficients(
        self,
    ) -> tuple[list[str], list[npt.NDArray[np.float64]]]: ...
    def creation(self) -> "FermionHamiltonian": ...
    def annihilation(self) -> "FermionHamiltonian": ...
    def with_coefficients(
        self, coefficients: npt.NDArray[np.float64]
    ) -> "FermionHamiltonian": ...
    def add_constant(self, constant_energy: float) -> "FermionHamiltonian": ...
    def to_sparse_majorana(self) -> dict[tuple[int, ...], complex]: ...
    def to_majorana_sparse(self) -> "MajoranaSparse": ...

class MajoranaSparse:
    """A sparse Majorana-operator representation of a Hamiltonian."""

    @property
    def indices(self) -> list[list[int]]: ...
    @property
    def coefficients(self) -> npt.NDArray[np.complex128]: ...
    @property
    def constant(self) -> float: ...

class MajoranaEncoding:
    """A fermion-to-qubit encoding defined by its Majorana operators."""

    def __init__(
        self,
        ipowers: npt.NDArray[np.uint8],
        symplectics: npt.NDArray[np.bool],
        vacuum_state: npt.NDArray[np.bool] | None = None,
    ) -> None: ...
    @staticmethod
    def jordan_wigner(
        n_modes: int, n_qubits: int | None = None
    ) -> "MajoranaEncoding": ...
    @staticmethod
    def bravyi_kitaev(
        n_modes: int, n_qubits: int | None = None
    ) -> "MajoranaEncoding": ...
    @staticmethod
    def parity(n_modes: int, n_qubits: int | None = None) -> "MajoranaEncoding": ...
    @staticmethod
    def jkmn(n_modes: int, n_qubits: int | None = None) -> "MajoranaEncoding": ...
    @staticmethod
    def maxnto(n_modes: int) -> "MajoranaEncoding": ...
    @staticmethod
    def from_flatpack(
        flatpack: TTFlatpack, n_qubits: int | None = None
    ) -> "MajoranaEncoding": ...
    @staticmethod
    def from_json(data: dict) -> "MajoranaEncoding": ...
    def to_json(self) -> dict: ...
    @property
    def n_modes(self) -> int: ...
    @property
    def n_qubits(self) -> int: ...
    @property
    def ipowers(self) -> npt.NDArray[np.uint8]: ...
    @property
    def symplectic_matrix(self) -> npt.NDArray[np.bool]: ...
    @property
    def vacuum_state(self) -> npt.NDArray[np.bool]: ...
    def encode(
        self, operator: FermionHamiltonian | MajoranaSparse
    ) -> QubitHamiltonian: ...
    def encode_annealed(
        self,
        fham: FermionHamiltonian,
        temperature: float | None = None,
        initial_guess: list[int] | None = None,
        coefficient_weighted: bool = True,
        seed: int | None = None,
    ) -> QubitHamiltonian: ...
    def anneal_enumeration(
        self,
        fham: FermionHamiltonian,
        temperature: float | None = None,
        initial_guess: list[int] | None = None,
        coefficient_weighted: bool = False,
        seed: int | None = None,
    ) -> float: ...
    def decode(self, states: npt.NDArray[np.bool]) -> npt.NDArray[np.bool]: ...
    def hartree_fock_state(
        self,
        fermionic_hf_state: npt.NDArray[np.bool],
        mode_op_map: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.bool]: ...
    def number_operator(self, mode: int, coeff: complex = 1.0) -> QubitHamiltonian: ...
    def edge_operator(
        self,
        edge_indices: tuple[int, int],
        coeff: complex = 1.0,
        with_conjugate: bool = False,
    ) -> QubitHamiltonian: ...
    def interaction_operator(
        self,
        mode_indices: tuple[int, int, int, int],
        coeff: complex = 1.0,
        physicist_notation: bool = True,
        with_conjugate: bool = False,
    ) -> QubitHamiltonian: ...
    def encode_fermion_product(
        self,
        signature: str,
        mode_indices: list[int],
        coeff: complex = 1.0,
        with_conjugate: bool = False,
    ) -> QubitHamiltonian: ...
    def encode_majorana_product(
        self,
        majorana_indices: list[int],
        coeff: complex = 1.0,
    ) -> tuple[str, complex]: ...
    def batch_pauli_weights(
        self,
        fham: FermionHamiltonian,
        permutations: npt.NDArray[np.uint],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def apply_mode_enumeration(self, mode_op_map: list[int]) -> "MajoranaEncoding": ...

def symplectic_product(
    left: npt.NDArray[np.bool], right: npt.NDArray[np.bool]
) -> tuple[int, npt.NDArray[np.bool]]: ...
def symplectic_to_pauli(
    symplectic: npt.NDArray[np.bool], ipower: int = 0
) -> tuple[str, int]: ...
def pauli_to_symplectic(
    pauli: str, ipower: int
) -> tuple[npt.NDArray[np.bool], int]: ...
def symplectic_to_sparse(
    symplectic: npt.NDArray[np.bool],
    ipower: int,
) -> tuple[str, npt.NDArray[np.uintp], complex]: ...
def hatt(
    fham: FermionHamiltonian,
    n_modes: int | None = None,
) -> tuple[TTFlatpack, int]: ...
def topphatt(
    flatpack: TTFlatpack,
    n_qubits: int,
    hamiltonian: MajoranaSparse,
    parallelize: bool = True,
    heuristic: str = "min_weight",
    seed: int | None = None,
    backend: str = "dense_transpose",
) -> MajoranaEncoding: ...
def encode_topphatt(
    flatpack: TTFlatpack,
    n_qubits: int,
    fham: FermionHamiltonian,
    parallelize: bool = True,
    heuristic: str = "min_weight",
    seed: int | None = None,
    backend: str = "dense_transpose",
) -> tuple[QubitHamiltonian, MajoranaEncoding]: ...
