import numpy as np
import numpy.typing as npt

# Rust-accelerated functions exposed to Python

def symplectic_product(
    left: npt.NDArray[np.bool_], right: npt.NDArray[np.bool_]
) -> tuple[int, npt.NDArray[np.bool_]]: ...
def hartree_fock_state(
    vacuum_state: npt.NDArray[np.float64],
    fermionic_hf_state: npt.NDArray[np.bool_],
    mode_op_map: dict[int, int],
    symplectic_matrix: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.bool_]]: ...
def symplectic_to_pauli(symplectic: npt.NDArray[np.bool_]) -> tuple[int, str]: ...
def pauli_to_symplectic(pauli: str) -> tuple[int, npt.NDArray[np.bool_]]: ...
def symplectic_product_map(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]: ...
def symplectic_to_sparse(
    symplectic: npt.NDArray[np.bool_],
) -> tuple[int, str, npt.NDArray[np.uintp]]: ...
def molecular_hamiltonian_template(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool_],
    physicist_notation: bool,
) -> dict: ...
def hubbard_hamiltonian_template(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[np.bool_],
) -> dict: ...
def fill_template(
    template: dict,
    constant_energy: float,
    one_e_coeffs: npt.NDArray[np.float64],
    two_e_coeffs: npt.NDArray[np.float64],
    mode_op_map: npt.NDArray[np.uint],
) -> dict: ...
def anneal_enumerations(
    template: dict,
    constant_energy: float,
    one_e_coeffs: npt.NDArray[np.float64],
    two_e_coeffs: npt.NDArray[np.float64],
    temp: float,
    initial_guess: npt.NDArray[np.uint],
) -> dict: ...
def icount_to_sign(icount: int) -> np.complex64: ...
