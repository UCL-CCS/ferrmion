import numpy as np
import numpy.typing as npt

# Rust-accelerated functions exposed to Python

def symplectic_product(
    left: npt.NDArray[bool], right: npt.NDArray[bool]
) -> tuple[int, npt.NDArray[bool]]: ...
def hartree_fock_state(
    vacuum_state: npt.NDArray[np.float64],
    fermionic_hf_state: npt.NDArray[bool],
    mode_op_map: list[int],
    ipowers: npt.NDArray[np.uint8],
    symplectic_matrix: npt.NDArray[bool],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[bool]]: ...
def symplectic_to_pauli(symplectic: npt.NDArray[bool], int) -> tuple[str, int]: ...
def pauli_to_symplectic(pauli: str, ipower: int) -> tuple[npt.NDArray[bool], int]: ...
def symplectic_product_map(
    symplectics: npt.NDArray[bool],
    ipowers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[bool]]: ...
def symplectic_to_sparse(
    symplectic: npt.NDArray[bool],
    ipower: int,
) -> tuple[str, npt.NDArray[np.uintp], np.complex]: ...
def molecular_hamiltonian_template(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[bool],
    physicist_notation: bool,
) -> dict: ...
def hubbard_hamiltonian_template(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool],
) -> dict: ...
def template_weight_distribution(
    template: dict,
    constant_energy: float,
    one_e_coeffs: npt.NDArray[np.float64],
    two_e_coeffs: npt.NDArray[np.float64],
    n_permutations: int,
) -> dict: ...
def fill_template(
    template: dict,
    constant_energy: float,
    one_e_coeffs: npt.NDArray[np.float64],
    two_e_coeffs: npt.NDArray[np.float64],
    mode_op_map: npt.NDArray[np.uint],
) -> dict: ...
def anneal_enumerations(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool],
    signatures: list[str],
    coeffs: list[np.ndarray],
    temperature: float,
    initial_guess: npt.NDArray[np.uint],
    coefficient_weighted: bool,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool]]: ...
def icount_to_sign(icount: int) -> np.complex64: ...
def topphatt(
    flatpack: list[tuple[np.uint, tuple[np.uint, np.uint, np.uint]]],
    n_qubits: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool]]: ...
def topphatt_standard(
    encoding: str,
    n_modes: int,
    n_qubits: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool]]: ...
def standard_symplectic_matrix(
    encoding: str, n_modes: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool]]: ...
def encode_standard(
    encoding: str,
    n_modes: int,
    n_qubits: int,
    signatures: list[str],
    coeffs: list[np.ndarray],
    constant_energy: float,
) -> dict: ...
def encode(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool],
    signatures: list[str],
    coeffs: list[np.ndarray],
    constant_energy: float,
) -> dict: ...
