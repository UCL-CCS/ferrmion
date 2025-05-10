"""Utility functions."""

import datetime
import json
import logging
import logging.config

import numpy as np

logger = logging.getLogger(__name__)


def icount_to_sign(icount: int) -> np.complex64:
    """Convert a power of i to a complex value.

    Args:
        icount (int): The power of i.

    Returns:
        np.complex64: The complex value.
    """
    vals = {0: 1, 1: 1j, 2: -1, 3: -1j}
    return vals[icount % 4]


def symplectic_hash(symp: np.ndarray[np.bool]) -> bytes:
    """Convert a symplectic vector into a hashable form.

    Args:
        symp (np.ndarray[np.bool]): The symplectic vector.

    Returns:
        bytes: The hashed form of the symplectic vector.
    """
    return np.packbits(symp).tobytes()


def symplectic_unhash(symp: bytes, length: int) -> np.ndarray[np.bool]:
    """Convert a hashed symplectic vector back to its original form.

    Args:
        symp (bytes): The hashed form of the symplectic vector.
        length (int): The length of the original symplectic vector.

    Returns:
        np.ndarray[np.bool]: The original symplectic vector.
    """
    unpacked = np.unpackbits(np.frombuffer(symp, dtype=np.uint8))
    if len(unpacked) < length:
        unpacked = np.pad(
            unpacked, length - len(unpacked), "constant", constant_values=0
        )
    return np.array(unpacked[:length], dtype=np.bool)


def symplectic_to_pauli(symplectic: np.ndarray) -> tuple[int, str]:
    """Convert a symplectic vector into a Pauli String.

    Args:
        symplectic (np.ndarray[np.uint8]) : symplectic vector [X terms, Y terms]

    Returns:
        tuple[int, str]: The imaginary cofactor and Pauli string.

    NOTE: symplectic XZ does represent XZ and not Y
        So Y=-iXZ needs an imaginary cofactor
    """
    half_length = len(symplectic) // 2
    xlist = ["X" if line == 1 else "" for line in symplectic[:half_length]]
    zlist = ["Z" if line == 1 else "" for line in symplectic[half_length:]]
    two_p = {"": "I", "X": "X", "Y": "Y", "Z": "Z", "XZ": "Y"}

    pauli_list = [two_p[x + z] for x, z in zip(xlist, zlist)]
    pauli_string = "".join(pauli_list)
    y_count = pauli_string.count("Y")
    ipower = (3 * y_count) % 4
    return ipower, pauli_string


def pauli_to_symplectic(pauli: str) -> tuple[int, np.ndarray[np.uint8, np.uint8]]:
    """Convert a Pauli operator to symplectic form.

    Args:
        pauli (str): The Pauli operator string.

    Returns:
        tuple[int, np.ndarray[np.uint8, np.uint8]]: The imaginary cofactor and symplectic matrix.
    """
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


def xz_swap(symplectic) -> np.ndarray[np.uint8]:
    """Swap X and Z Pauli operators in a symplectic matrix.

    Args:
        symplectic (np.ndarray): The symplectic matrix.

    Returns:
        np.ndarray[np.uint8]: The symplectic matrix with X and Z swapped.
    """
    logger.debug(f"Swapping X and Z in symplectic matrix\n{symplectic=}")
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


def xy_swap(symplectic) -> np.ndarray[np.uint8]:
    """Swap X and Y Pauli operators in a symplectic matrix.

    Args:
        symplectic (np.ndarray): The symplectic matrix.

    Returns:
        np.ndarray[np.uint8]: The symplectic matrix with X and Y swapped.
    """
    logger.debug(f"Swapping X and Y in symplectic matrix\n{symplectic=}")
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


def yz_swap(symplectic) -> np.ndarray[np.uint8]:
    """Swap Y and Z Pauli operators in a symplectic matrix.

    Args:
        symplectic (np.ndarray): The symplectic matrix.

    Returns:
        np.ndarray[np.uint8]: The symplectic matrix with Y and Z swapped.
    """
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
    """Swap the position of two qubits in a symplectic matrix.

    Args:
        symplectic (np.ndarray): The symplectic matrix.
        index_pair (tuple[int]): The indices of the qubits to swap.

    Returns:
        np.ndarray[np.uint8]: The symplectic matrix with the qubits swapped.
    """
    logger.debug(f"Swapping qubits {index_pair} in symplectic matrix\n{symplectic=}")
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
    """Check the Non-trivial Overlap of a symplectic matrix.

    Args:
        symplectic (np.ndarray): The symplectic matrix.

    Returns:
        tuple[bool, np.ndarray[int]]: A boolean indicating if the overlap is trivial and the overlap matrix.
    """
    logger.debug(f"Checking trivial overlap\n{symplectic=}")
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

    logger.debug(f"Trivial overlap satisfied: {satisfied}")
    logger.debug(f"Trivial overlap matrix:\n{nto}")
    return satisfied, nto


def find_pauli_weight(symplectic_hamiltonian) -> float:
    """Find the average Pauli weight of a symplectic hamiltonian.

    Args:
        symplectic_hamiltonian (np.ndarray): The symplectic Hamiltonian.

    Returns:
        float: The average Pauli weight.
    """
    logger.debug("Finding Pauli weight of symplectic Hamiltonian")
    half_length = symplectic_hamiltonian.shape[-1] // 2
    has_pauli = np.bitwise_or(
        symplectic_hamiltonian[:, :half_length], symplectic_hamiltonian[:, half_length:]
    )
    return np.mean(np.sum(has_pauli, axis=1))


def save_pauli_ham(pauli_hamiltonian: dict[str, float], filename: str = None) -> None:
    """Save the Pauli Hamiltonian to a JSON file.

    Args:
        pauli_hamiltonian (dict[str, float]): The Pauli Hamiltonian.
        filename (str, optional): The filename to save the Hamiltonian to. Defaults to None.
    """
    logger.debug("Saving Pauli Hamiltonian to JSON file")
    if filename is None:
        filename = "pauli_hamiltonian_" + str(datetime.datetime.now()) + ".json"

    with open(filename, "w") as f:
        f.write(json.dumps(pauli_hamiltonian))
    logger.debug(f"Saved Pauli Hamiltonian to {filename}")


def setup_logs() -> None:
    """Initialise logging."""
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s: %(name)s: %(lineno)d: %(levelname)s: %(message)s"
            },
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": ".ferrmion.log",
                "mode": "w",
                "encoding": "utf-8",
            },
            "stream_handler": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {"handlers": ["file_handler", "stream_handler"], "level": "DEBUG"}
        },
    }

    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(__name__)
    logger.debug("Logging initialised.")
