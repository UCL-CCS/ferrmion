"""Utility functions"""

import numpy as np
import datetime
import json
import logging

logger = logging.getLogger(__name__)


def symplectic_product(left, right) -> tuple[int, np.ndarray[np.uint8]]:
    """Calculate the product of two symplectic vectors."""
    term = np.bitwise_xor(left, right)
    # print(symplectics[i], symplectics[j], term)

    n_qubits = len(left) // 2

    # any time S1 @ S2 involves Z on the left of an X we gain a -1
    zx_count = np.sum(np.bitwise_and(left[n_qubits:], right[:n_qubits]))

    # XZ products are stored as XZ and not as Y

    ipower = (2 * zx_count) % 4
    return ipower, term


def icount_to_sign(icount: int) -> np.complex64:
    """Convert a power of i to a complex value."""
    vals = {0: 1, 1: 1j, 2: -1, 3: -1j}
    return vals[icount % 4]


# Not in use, currently the code is set up using tostring
def symplectic_hash(symp: np.ndarray[np.uint8]) -> bytes:
    return np.packbits(symp).tobytes()


def symplectic_unhash(symp: bytes, length: int) -> np.ndarray[np.uint8]:
    unpacked = np.unpackbits(np.frombuffer(symp, dtype=np.uint8))
    if len(unpacked) < length:
        unpacked = np.pad(
            unpacked, length - len(unpacked), "constant", constant_values=0
        )
    return unpacked[:length]


def symplectic_to_pauli(symplectic: np.ndarray) -> str:
    """Convert a symplectic vector into a Pauli String

    NOTE: symplectic XZ does represent XZ and not Y
        So Y=-iXZ needs an imaginary cofactor

    Args:
        symplectic [np.ndarray[np.uint8]] : symplectic vector [X terms, Y terms]
    """
    logger.debug(f"Converting symplectic to Pauli\n{symplectic}")
    half_length = len(symplectic) // 2
    xlist = ["X" if line == 1 else "" for line in symplectic[:half_length]]
    zlist = ["Z" if line == 1 else "" for line in symplectic[half_length:]]
    two_p = {"": "I", "X": "X", "Y": "Y", "Z": "Z", "XZ": "Y"}

    pauli_list = [two_p[x + z] for x, z in zip(xlist, zlist)]
    pauli_string = "".join(pauli_list)
    y_count = pauli_string.count("Y")
    ipower = (3 * y_count) % 4
    logger.debug(f"{ipower=}, {pauli_string=}")
    return ipower, pauli_string


def pauli_to_symplectic(pauli: str) -> tuple[int, np.ndarray[np.uint8, np.uint8]]:
    pauli_array = np.array(list(pauli))
    x_map = {
        "I": 0,
        "X": 1,
        "Y": 1,
        "Z": 0,
    }
    z_map = {
        "I": 0,
        "X": 0,
        "Y": 1,
        "Z": 1,
    }
    # each y is turned into a iY=XZ
    y_count = np.count_nonzero(pauli_array == "Y") % 4
    # logger.debug(f{y_count=})
    x_array = np.array([x_map[term] for term in pauli], dtype=np.uint8)
    z_array = np.array([z_map[term] for term in pauli], dtype=np.uint8)
    return y_count, np.hstack((x_array, z_array), dtype=np.uint8)


def xz_swap(symplectic, index=None) -> np.ndarray[np.uint8]:
    """Swap X and Z Pauli operators in a symplectic matrix."""
    if index is None:
        x_block, z_block = np.hsplit(symplectic, 2)
        new_x_block = np.copy(x_block)
        new_x_block[np.where(z_block - x_block == 1)] = 1
        new_x_block[np.where(x_block - z_block == 1)] = 0
        new_x_block[np.where(x_block + z_block == 2)] = 1

        new_z_block = np.copy(z_block)
        new_z_block[np.where(z_block - x_block == 1)] = 0
        new_z_block[np.where(x_block - z_block == 1)] = 1
        new_z_block[np.where(x_block + z_block == 2)] = 1
        return np.hstack((new_x_block, new_z_block))


def xy_swap(symplectic, index=None) -> np.ndarray[np.uint8]:
    """Swap X and Z Pauli operators in a symplectic matrix."""
    if index is None:
        x_block, z_block = np.hsplit(symplectic, 2)
        is_y = np.where(x_block + z_block == 2)
        is_x = np.where(x_block - z_block == 1)

        new_x_block = np.copy(x_block)
        new_x_block[is_x] = 1
        new_x_block[is_y] = 1

        new_z_block = np.copy(z_block)
        new_z_block[is_x] = 1
        new_z_block[is_y] = 0
        return np.hstack((new_x_block, new_z_block))


def yz_swap(symplectic, index=None) -> np.ndarray[np.uint8]:
    """Swap X and Z Pauli operators in a symplectic matrix."""
    if index is None:
        x_block, z_block = np.hsplit(symplectic, 2)
        is_y = np.where(x_block + z_block == 2)
        is_z = np.where(z_block - x_block == 1)

        new_x_block = np.copy(x_block)
        new_x_block[is_y] = 0
        new_x_block[is_z] = 1

        new_z_block = np.copy(z_block)
        new_z_block[is_y] = 1
        new_z_block[is_z] = 1
        return np.hstack((new_x_block, new_z_block)) 


def qubit_swap(symplectic, index_pair) -> np.ndarray[np.uint8]:
    """Swap the position of two qubits in a symplectic matrix."""
    half_length = symplectic.shape[1] // 2
    i1, i2 = index_pair
    x1 = np.copy(symplectic[:, i1])
    x2 = np.copy(symplectic[:, i2])
    z1 = np.copy(symplectic[:, half_length + i1])
    z2 = np.copy(symplectic[:, half_length + i2])
    symplectic[:, i2] = x1
    symplectic[:, i1] = x2
    symplectic[:, half_length + i1] = z2
    symplectic[:, half_length + i2] = z1
    return symplectic


def check_trivial_overlap(symplectic) -> tuple[bool, np.ndarray[int]]:
    """Check the Non-trivial Overlap of a symplectic matrix."""
    x_length = int(len(symplectic[0]) / 2)

    symp_x = symplectic[:, :x_length]
    symp_z = symplectic[:, x_length:]
    symp_i = np.abs(symp_x - 1) * np.abs(symp_z - 1)
    symp_y = symp_x * symp_z

    symp_x = symp_x - symp_y
    symp_z = symp_z - symp_y

    i_trivial = symp_i @ symp_i.T
    same_p_trivial = symp_x @ symp_x.T + symp_y @ symp_y.T + symp_z @ symp_z.T
    one_i_trivial = (
        symp_x @ symp_i.T
        + symp_i @ symp_x.T
        + symp_y @ symp_i.T
        + symp_i @ symp_y.T
        + symp_z @ symp_i.T
        + symp_i @ symp_z.T
    )
    all_trivial = i_trivial + same_p_trivial + one_i_trivial

    nto = all_trivial.shape[0] / 2 - all_trivial

    satisfied = np.all((nto + np.eye(nto.shape[0])) % 2 == 1)

    if not satisfied:
        print("Not valid")
    else:
        print("Valid")

    return satisfied, nto


def find_pauli_weight(symplectic_hamiltonian) -> float:
    """Find the average Pauli weight of a symplectic hamiltonian"""
    half_length = symplectic_hamiltonian.shape[-1] // 2
    has_pauli = np.bitwise_or(
        symplectic_hamiltonian[:, :half_length], symplectic_hamiltonian[:, half_length:]
    )
    return np.mean(np.sum(has_pauli, axis=1))


def save_pauli_ham(pauli_hamiltonian: dict[str, float], filename: str = None) -> None:
    if filename is None:
        filename = "pauli_hamiltonian_" + str(datetime.datetime.now()) + ".json"

    with open(filename, "w") as f:
        f.write(json.dumps(pauli_hamiltonian))
    logger.debug(f"Saved Pauli Hamiltonian to {filename}")
