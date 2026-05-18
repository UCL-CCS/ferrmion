import numpy as np
import numpy.typing as npt

from .encode.ternary_tree import TTFlatpack

class FermionProduct:
    def __init__(
        self,
        action: list[str],
        indices: list[int],
        coefficient: complex,
    ) -> None: ...
    @property
    def action(self) -> list[str]: ...
    @property
    def indices(self) -> list[int]: ...
    @property
    def coefficient(self) -> complex: ...
    def to_sparse_majorana(self) -> SparseMajorana: ...

class SparseFermion:
    def __init__(
        self,
        action: list[str],
        indices: npt.NDArray[np.int64],
        coefficients: npt.NDArray[np.complex128],
    ) -> None: ...
    @property
    def action(self) -> list[str]: ...
    @property
    def indices(self) -> npt.NDArray[np.uintp]: ...
    @property
    def coefficients(self) -> npt.NDArray[np.complex128]: ...
    def to_sparse_majorana(self) -> SparseMajorana: ...

class MatrixFermion:
    def __init__(
        self,
        action: list[str],
        coefficients: npt.NDArray[np.float64],
    ) -> None: ...
    @property
    def action(self) -> list[str]: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]: ...
    def to_sparse(self) -> SparseFermion: ...
    def to_sparse_majorana(self) -> SparseMajorana: ...

class MajoranaProduct:
    def __init__(self, indices: list[int], coefficient: complex) -> None: ...
    @property
    def indices(self) -> list[int]: ...
    @property
    def coefficient(self) -> complex: ...
    def to_sparse_majorana(self) -> SparseMajorana: ...
    def encode(
        self,
        ipowers: npt.NDArray[np.uint8],
        symplectics: npt.NDArray[np.bool],
    ) -> dict[str, complex]: ...

class SparseMajorana:
    def __init__(
        self,
        indices: list[list[int]],
        coefficients: list[complex],
        constant: float,
    ) -> None: ...
    @property
    def indices(self) -> list[list[int]]: ...
    @property
    def coefficients(self) -> list[complex]: ...
    @property
    def constant(self) -> float: ...
    @classmethod
    def from_signatures_and_coeffs(
        cls,
        signatures: list[str],
        coeffs: list[np.ndarray],
        constant_energy: float,
    ) -> SparseMajorana: ...

# Rust-accelerated functions exposed to Python

def symplectic_product(
    left: npt.NDArray[np.bool], right: npt.NDArray[np.bool]
) -> tuple[int, npt.NDArray[np.bool]]: ...
def hartree_fock_state(
    fermionic_hf_state: npt.NDArray[np.bool],
    mode_op_map: npt.NDArray[np.uint],
    ipowers: npt.NDArray[np.uint8],
    symplectic_matrix: npt.NDArray[np.bool],
    vacuum_state: npt.NDArray[np.bool],
) -> npt.NDArray[np.bool]: ...
def symplectic_to_pauli(
    symplectic: npt.NDArray[np.bool], ipower: int
) -> tuple[str, int]: ...
def pauli_to_symplectic(
    pauli: str, ipower: int
) -> tuple[npt.NDArray[np.bool], int]: ...
def symplectic_to_sparse(
    symplectic: npt.NDArray[np.bool],
    ipower: int,
) -> tuple[str, npt.NDArray[np.uintp], complex]: ...
def anneal_enumerations(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool],
    signatures: list[str],
    coeffs: list[np.ndarray],
    temperature: float,
    initial_guess: npt.NDArray[np.uint],
    coefficient_weighted: bool,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool]]: ...
def batch_pauli_weights(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool],
    vacuum_state: npt.NDArray[np.bool],
    signatures: list[str],
    coeffs: list[np.ndarray],
    permutations: npt.NDArray[np.uint],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def topphatt(
    flatpack: list[tuple[np.uint, tuple[np.uint, np.uint, np.uint]]],
    n_qubits: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
    parallelize: bool,
    heuristic: str = "min_weight",
    seed: int | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool], npt.NDArray[np.bool]]: ...
def hatt(
    n_modes: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
) -> tuple[TTFlatpack, int]: ...
def flatpack_symplectic_matrix(
    flatpack: TTFlatpack,
    n_qubits: None | int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool], npt.NDArray[np.bool]]: ...
def encode_fermion_product(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool],
    signatures: str,
    indices: list[int],
    coefficient: complex,
) -> dict: ...
def encode(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool],
    vacuum_state: npt.NDArray[np.bool],
    signatures: list[str],
    coeffs: list[np.ndarray],
    constant_energy: float,
) -> dict: ...
def encode_topphatt(
    flatpack: list[tuple[np.uint, tuple[np.uint, np.uint, np.uint]]],
    n_qubits: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
    constant_energy: float,
    parallelize: bool,
    heuristic: str = "min_weight",
    seed: int | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool], dict, npt.NDArray[np.bool]]: ...
def fermionic_to_sparse_majorana(
    signatures: list[str],
    coeffs: list[np.ndarray],
    constant_energy: float,
) -> dict: ...
def decode(
    states: npt.NDArray[np.bool],
    ipowers: npt.NDArray[np.uint8],
    symplectic_matrix: npt.NDArray[np.bool],
    vacuum_state: npt.NDArray[np.bool],
) -> npt.NDArray[np.bool]: ...
def maxnto_symplectic_matrix(
    n_modes: int,
) -> npt.NDArray[np.bool]: ...
