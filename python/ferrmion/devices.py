"""Classes which represent physical devices or objects."""

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Qubit(BaseModel):
    """A qubit object which represents a physical qubit.

    Args:
        label (int): The qubit label.
        gate_error (float): The gate error rate.
        t1 (float): The T1 time.
        t2 (float): The T2 time.
    """

    label: int
    gate_error: float
    t1: float
    t2: float


class Toplogy:
    """A topology object which represents a physical device.

    Attributes:
        qubits (set[Qubit]): A set of qubits.
        connections (dict): A dictionary of qubit connections.

    Methods:
        add_connection(control, target, error): Add a connection between two qubits.
    """

    def __init__(self, qubits: set[Qubit]):
        """Initialize the topology object.

        Args:
            qubits (set[Qubit]): A set of qubits.
        """
        self.qubits = qubits
        self.connections = {q.root_path: {} for q in qubits}

    def add_connection(self, control: Qubit, target: Qubit, error: float):
        """Add a connection between two qubits.

        Args:
            control (Qubit): The control qubit.
            target (Qubit): The target qubit.
            error (float): The error rate of the connection.
        """
        # check if the control is in the set, then check if a value is set for the target error
        self.connections[control.root_path][target.root_path] = error
