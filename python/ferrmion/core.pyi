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
    symplectic_matrix: npt.NDArray[bool],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[bool]]: ...
def symplectic_to_pauli(symplectic: npt.NDArray[bool]) -> tuple[int, str]: ...
def pauli_to_symplectic(pauli: str) -> tuple[int, npt.NDArray[bool]]: ...
def symplectic_product_map(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[bool],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[bool]]: ...
def symplectic_to_sparse(
    symplectic: npt.NDArray[bool],
) -> tuple[int, str, npt.NDArray[np.uintp]]: ...
def molecular_hamiltonian_template(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[bool],
    physicist_notation: bool,
) -> dict: ...
def hubbard_hamiltonian_template(
    ipowers: npt.NDArray[np.uint8],
    symplectics: npt.NDArray[bool],
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
    template: dict,
    one_e_coeffs: npt.NDArray[np.float64],
    two_e_coeffs: npt.NDArray[np.float64],
    temperature: float,
    initial_guess: npt.NDArray[np.uint],
) -> dict: ...
def icount_to_sign(icount: int) -> np.complex64: ...
