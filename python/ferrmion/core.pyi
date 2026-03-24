import numpy as np
import numpy.typing as npt

from .encode.ternary_tree import TTFlatpack

# Rust-accelerated functions exposed to Python

def symplectic_product(
    left: npt.NDArray[np.bool], right: npt.NDArray[np.bool]
) -> tuple[int, npt.NDArray[np.bool]]: ...
def ternary_tree_hartree_fock_state(
    fermionic_hf_state: npt.NDArray[np.bool],
    mode_op_map: npt.NDArray[np.uint],
    ipowers: npt.NDArray[np.uint8],
    symplectic_matrix: npt.NDArray[np.bool],
    vacuum_state: npt.NDArray[np.bool],
) -> npt.NDArray[np.bool]: ...
def symplectic_to_pauli(symplectic: npt.NDArray[np.bool], ipower: int) -> tuple[str, int]: ...
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
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool]]: ...
def topphatt(
    flatpack: list[tuple[np.uint, tuple[np.uint, np.uint, np.uint]]],
    n_qubits: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
    parallelize: bool,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool], npt.NDArray[np.bool]]: ...
def topphatt_standard(
    encoding: str,
    n_modes: int,
    n_qubits: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
    parallelize: bool,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool], npt.NDArray[np.bool]]: ...
def flatpack_symplectic_matrix(
    flatpack: TTFlatpack,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool], npt.NDArray[np.bool]]: ...
def standard_symplectic_matrix(
    encoding: str, n_modes: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool], npt.NDArray[np.bool]]: ...
def encode_standard(
    encoding: str,
    n_modes: int,
    n_qubits: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
    constant_energy: float,
) -> dict: ...
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
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool], dict, npt.NDArray[np.bool]]: ...
def fermionic_to_sparse_majorana(
    signatures: list[str],
    coeffs: list[np.ndarray],
    constant_energy: float,
) -> dict: ...
