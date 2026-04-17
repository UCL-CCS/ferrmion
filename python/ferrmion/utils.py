"""Utility functions."""

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


def symplectic_to_pauli(
    symplectic: NDArray[bool],
    ipower: int = 0,
) -> tuple[str, int]:
    """Convert a symplectic vector into a Pauli String.

    Args:
        symplectic (NDArray[np.uint8]) : symplectic vector [X terms, Y terms]
        ipower (NDArray[np.uint]): power of i coefficient

    Returns:
        tuple[str, int]: The Pauli string and imaginary cofactor.

    NOTE: symplectic XZ does represent XZ and not Y
        So Y=-iXZ needs an imaginary cofactor

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import symplectic_to_pauli
        >>> arr = np.array([1, 0, 0, 1], dtype=bool)
        >>> pauli, ipower = symplectic_to_pauli(arr)
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
    return pauli_string, ipower


def symplectic_to_sparse(
    symplectic: NDArray[bool],
    ipower: int = 0,
) -> tuple[str, NDArray[int], np.complex64]:
    """Convert a symplectic vector into a Pauli String (sparse form).

    Args:
        symplectic (NDArray[np.uint8]) : symplectic vector [X terms, Y terms]
        ipower (NDArray[np.uint]): power of i coefficient

    Returns:
        tuple[str, NDArray[int]]: The Pauli string, indices of non-identity terms and imaginary coefficient.

    NOTE: symplectic XZ does represent XZ and not Y
        So Y=-iXZ needs an imaginary cofactor

    Example:
        >>> import numpy as np
        >>> from ferrmion.utils import symplectic_to_sparse
        >>> arr = np.array([1, 0, 0, 1], dtype=bool)
        >>> pauli, idx, coeff = symplectic_to_sparse(arr)
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
    return pauli_string, indices, icount_to_sign(ipower)


def pauli_to_symplectic(
    pauli: str,
    ipower: int = 0,
) -> tuple[NDArray[bool], int]:
    """Convert a Pauli operator to symplectic form.

    Args:
        pauli (str): The Pauli operator string.
        ipower (NDArray[np.uint]): power of i coefficient

    Returns:
        tuple[int, NDArray[np.uint8, np.uint8]]: The imaginary cofactor and symplectic matrix.

    Example:
        >>> from ferrmion.utils import pauli_to_symplectic
        >>> symp, ipower = pauli_to_symplectic('XIZY')

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
    return np.hstack((x_array, z_array), dtype=bool), ipower


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
