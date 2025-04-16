"""Init for fermion qubit encodings"""

from .base import FermionQubitEncoding
from .utils import (
    pauli_to_symplectic,
    symplectic_to_pauli,
    symplectic_product,
    symplectic_hash,
    symplectic_unhash,
    icount_to_sign,
)
from .ternary_tree import TernaryTree
from .ternary_tree_node import TTNode, node_sorter
from .knto import KNTO, knto_symplectic_matrix

def setup_logs() -> None:
    """Initialise logging."""
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s: %(name)s: %(levelname)s: %(message)s"},
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": ".nbed.log",
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


setup_logs()

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "FermionQubitEncoding",
    "TernaryTree",
    "TTNode",
    "node_sorter",
    "pauli_to_symplectic",
    "symplectic_to_pauli",
    "symplectic_hash",
    "symplectic_unhash",
    "symplectic_product",
    "icount_to_sign",
    "KNTO",
    "knto_symplectic_matrix",
]
