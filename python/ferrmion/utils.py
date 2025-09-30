"""Utility functions."""

import datetime
import json
import logging
import logging.config

import numpy as np
from numpy.typing import NDArray

from ferrmion.core import symplectic_product

logger = logging.getLogger(__name__)


def icount_to_sign(icount: int) -> np.complex64:
    """Convert a power of i to a complex value.

    Args:
        icount (int): The power of i.

    Returns:
        np.complex64: The complex value.

    Example:
        >>> from ferrmion.utils import icount_to_sign
        >>> icount_to_sign(1)
        1j
        >>> icount_to_sign(2)
        -1
    """
    vals = {0: 1, 1: 1j, 2: -1, 3: -1j}
    return vals[icount % 4]


def symplectic_hash(symp: NDArray[bool]) -> bytes:
    """Convert a symplectic vector into a hashable form.

    Args:
        symp (NDArray[bool]): The symplectic vector.

    Returns:
        bytes: The hashed form of the symplectic vector.

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import symplectic_hash
        >>> symp = np.array([True, False, True, False], dtype=bool)
        >>> h = symplectic_hash(symp)
        >>> isinstance(h, bytes)
        True
    """
    return np.packbits(symp).tobytes()


def symplectic_unhash(symp: bytes, length: int) -> NDArray[bool]:
    """Convert a hashed symplectic vector back to its original form.

    Args:
        symp (bytes): The hashed form of the symplectic vector.
        length (int): The length of the original symplectic vector.

    Returns:
        NDArray[bool]: The original symplectic vector.

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import symplectic_hash, symplectic_unhash
        >>> arr = np.array([True, False, True, False], dtype=bool)
        >>> h = symplectic_hash(arr)
        >>> arr2 = symplectic_unhash(h, 4)
        >>> np.all(arr == arr2)
        True
    """
    unpacked = np.unpackbits(np.frombuffer(symp, dtype=np.uint8))
    if len(unpacked) < length:
        unpacked = np.pad(
            unpacked, length - len(unpacked), "constant", constant_values=0
        )
    return np.array(unpacked[:length], dtype=bool)


def symplectic_to_pauli(ipower: int, symplectic: NDArray[bool]) -> tuple[int, str]:
    """Convert a symplectic vector into a Pauli String.

    Args:
        ipower (NDArray[np.uint]): power of i coefficient
        symplectic (NDArray[np.uint8]) : symplectic vector [X terms, Y terms]

    Returns:
        tuple[int, str]: The imaginary cofactor and Pauli string.

    NOTE: symplectic XZ does represent XZ and not Y
        So Y=-iXZ needs an imaginary cofactor

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import symplectic_to_pauli
        >>> arr = np.array([1, 0, 0, 1], dtype=bool)
        >>> ipower, pauli = symplectic_to_pauli(arr)
        >>> isinstance(pauli, str)
        True
    """
    left, right = np.hsplit(symplectic, 2)
    total = left + 2 * right

    def to_pauli(x):
        match x:
            case 0:
                return "I"
            case 1:
                return "X"
            case 2:
                return "Z"
            case 3:
                return "Y"

    to_paulis = np.vectorize(to_pauli)
    pauli_list = to_paulis(total)

    pauli_string = "".join(pauli_list)
    y_count = pauli_string.count("Y")
    ipower += 3 * y_count
    ipower %= 4
    return ipower, pauli_string


def symplectic_to_sparse(
    ipower: int,
    symplectic: NDArray[bool],
) -> tuple[int, str, NDArray[int]]:
    """Convert a symplectic vector into a Pauli String (sparse form).

    Args:
        ipower (NDArray[np.uint]): power of i coefficient
        symplectic (NDArray[np.uint8]) : symplectic vector [X terms, Y terms]

    Returns:
        tuple[int, str, NDArray[int]]: The imaginary cofactor, Pauli string, and indices of non-identity terms.

    NOTE: symplectic XZ does represent XZ and not Y
        So Y=-iXZ needs an imaginary cofactor

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import symplectic_to_sparse
        >>> arr = np.array([1, 0, 0, 1], dtype=bool)
        >>> ipower, pauli, idx = symplectic_to_sparse(arr)
        >>> isinstance(pauli, str)
        True
        >>> isinstance(idx, np.ndarray)
        True
    """
    xhalf, zhalf = np.hsplit(symplectic, 2)
    total = xhalf + 2 * zhalf

    def to_pauli(x):
        match x:
            case 0:
                return ""
            case 1:
                return "X"
            case 2:
                return "Z"
            case 3:
                return "Y"

    to_paulis = np.vectorize(to_pauli)
    pauli_list = to_paulis(total)

    pauli_string = "".join(pauli_list)
    indices = np.where(total != 0)[0]
    y_count = pauli_string.count("Y")
    ipower += 3 * y_count
    ipower %= 4
    return ipower, pauli_string, indices


def pauli_to_symplectic(ipower: int, pauli: str) -> tuple[int, NDArray[bool]]:
    """Convert a Pauli operator to symplectic form.

    Args:
        ipower (NDArray[np.uint]): power of i coefficient
        pauli (str): The Pauli operator string.

    Returns:
        tuple[int, NDArray[np.uint8, np.uint8]]: The imaginary cofactor and symplectic matrix.

    Example:
        >>> from ferrmion.utils import pauli_to_symplectic
        >>> ipower, symp = pauli_to_symplectic('XIZY')

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
    ipower += y_count
    ipower %= 4
    # logger.debug(f{y_count=})
    x_array = np.array([x_map[term] for term in pauli], dtype=bool)
    z_array = np.array([z_map[term] for term in pauli], dtype=bool)
    return ipower, np.hstack((x_array, z_array), dtype=bool)


def xz_swap(symplectic) -> NDArray[bool]:
    """Swap X and Z Pauli operators in a symplectic matrix.

    Args:
        symplectic (NDArray): The symplectic matrix.

    Returns:
        NDArray[np.uint8]: The symplectic matrix with X and Z swapped.

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import xz_swap
        >>> arr = np.zeros((2, 4), dtype=bool)
        >>> swapped = xz_swap(arr)

    """
    logger.debug(f"Swapping X and Z in symplectic matrix\n{symplectic=}")
    symplectic = np.array(symplectic, dtype=np.uint8)
    x_block, z_block = np.hsplit(symplectic, 2)
    is_z = np.where(np.logical_and(z_block, np.logical_not(x_block)))
    is_x = np.where(np.logical_and(x_block, np.logical_not(z_block)))

    new_x_block = np.copy(x_block)
    new_x_block[is_z] = True
    new_x_block[is_x] = False

    new_z_block = np.copy(z_block)
    new_z_block[is_x] = True
    new_z_block[is_z] = False
    return np.hstack((new_x_block, new_z_block))


def xy_swap(symplectic) -> NDArray[np.uint8]:
    """Swap X and Y Pauli operators in a symplectic matrix.

    Args:
        symplectic (NDArray): The symplectic matrix.

    Returns:
        NDArray[np.uint8]: The symplectic matrix with X and Y swapped.

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import xy_swap
        >>> arr = np.zeros((2, 4), dtype=bool)
        >>> swapped = xy_swap(arr)

    """
    logger.debug(f"Swapping X and Y in symplectic matrix\n{symplectic=}")
    symplectic = np.array(symplectic, dtype=np.uint8)
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


def yz_swap(symplectic) -> NDArray[np.uint8]:
    """Swap Y and Z Pauli operators in a symplectic matrix.

    Args:
        symplectic (NDArray): The symplectic matrix.

    Returns:
        NDArray[np.uint8]: The symplectic matrix with Y and Z swapped.

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import yz_swap
        >>> arr = np.zeros((2, 4), dtype=bool)
        >>> swapped = yz_swap(arr)

    """
    symplectic = np.array(symplectic, dtype=np.uint8)
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


def qubit_swap(symplectic, index_pair) -> NDArray[np.uint8]:
    """Swap the position of two qubits in a symplectic matrix.

    Args:
        symplectic (NDArray): The symplectic matrix.
        index_pair (tuple[int]): The indices of the qubits to swap.

    Returns:
        NDArray[np.uint8]: The symplectic matrix with the qubits swapped.

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import qubit_swap
        >>> arr = np.zeros((2, 4), dtype=bool)
        >>> swapped = qubit_swap(arr, (0, 1))

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


def check_trivial_overlap(symplectic) -> tuple[bool, NDArray[np.uint]]:
    """Check the Non-trivial Overlap of a symplectic matrix.

    Args:
        symplectic (NDArray): The symplectic matrix.

    Returns:
        tuple[bool, NDArray[int]]: A boolean indicating if the overlap is trivial and the overlap matrix.

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import check_trivial_overlap
        >>> arr = np.eye((4, 4), dtype=bool)
        >>> satisfied, nto = check_trivial_overlap(arr)
        >>> isinstance(satisfied, bool)
        True
    """
    symplectic = np.array(symplectic, dtype=np.uint8)
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

    nto: NDArray[np.uint] = all_trivial.shape[0] / 2 - all_trivial

    satisfied: bool = np.all((nto + np.eye(nto.shape[0])) % 2 == 1)

    logger.debug(f"Trivial overlap satisfied: {satisfied}")
    logger.debug(f"Trivial overlap matrix:\n{nto}")
    return satisfied, nto


def two_operator_product(creation: tuple[bool, bool], left, right) -> NDArray:
    """Calculate the product of two operators in symplectic form.

    Args:
        creation (tuple[bool, bool]): A tuple of two booleans indicating if the operators are creation operators.
        left (NDArray): The left operator in symplectic form.
        right (NDArray): The right operator in symplectic form.

    Returns:
        NDArray: The product of the two operators in symplectic form.

    Example:
        >>> left = np.array([[1, 0], [0, 1]])
        >>> right = np.array([[0, 1], [1, 0]])
        >>> creation = (True, False)
        >>> two_operator_product(creation, left, right)
            array([
                [0, 0],
                [1, 0]
                [0, 1],
                [0, 0]])
    """
    logger.debug("Calculating two operator product.")
    # (a+ib)(c+id) -> ac, iad, ibc, -bd
    first_term = symplectic_product(left[:, 0], right[:, 0])
    second_term = symplectic_product(left[:, 0], right[:, 1])
    third_term = symplectic_product(left[:, 1], right[:, 0])
    fourth_term = symplectic_product(left[:, 1], right[:, 1])

    # left creation -> -iad, +bd
    # right creation -> -ibc, +bd
    # both creation -> -iad, -ibc, -bd
    if creation[0] is True:
        second_term[0] += 2
        fourth_term[0] += 2
    if creation[1] is True:
        third_term[0] += 2
        fourth_term[0] += 2

    return np.vstack((first_term, second_term, third_term, fourth_term))


def find_pauli_weight(symplectic_hamiltonian: NDArray[bool]) -> np.floating:
    """Find the average Pauli weight of a symplectic hamiltonian.

    Args:
        symplectic_hamiltonian (NDArray): The symplectic Hamiltonian.

    Returns:
        float: The average Pauli weight.

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import find_pauli_weight
        >>> arr = np.eye((2, 4), dtype=bool)
        >>> find_pauli_weight(arr)
        1.0
    """
    logger.debug("Finding Pauli weight of symplectic Hamiltonian")
    half_length = symplectic_hamiltonian.shape[-1] // 2
    has_pauli = np.bitwise_or(
        symplectic_hamiltonian[:, :half_length], symplectic_hamiltonian[:, half_length:]
    )
    return np.mean(np.sum(has_pauli, axis=1))


def save_pauli_ham(
    pauli_hamiltonian: dict[str, float], filename: str | None = None
) -> None:
    """Save the Pauli Hamiltonian to a JSON file.

    Args:
        pauli_hamiltonian (dict[str, float]): The Pauli Hamiltonian.
        filename (str, optional): The filename (without extension) to save the Hamiltonian to. Defaults to None.

    Example:
        >>> from ferrmion.utils import save_pauli_ham
        >>> save_pauli_ham({'X': 1.0, 'Z': -1.0}, filename='test_ham')
    """
    logger.debug("Saving Pauli Hamiltonian to JSON file")
    if filename is None:
        filename = "pauli_hamiltonian_" + str(datetime.datetime.now())

    with open(f"{filename}.json", "w") as f:
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
