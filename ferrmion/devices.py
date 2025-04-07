"""Classes which represent physical devices or objects."""

import logging
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class Qubit(BaseModel):
    label: int
    gate_error: float
    t1: float
    t2: float


class Toplogy:
    def __init__(self, qubits: set[Qubit]):
        self.qubits = qubits
        self.connections = {q.label: {} for q in qubits}

    def add_connection(self, control, target, error):
        # check if the control is in the set, then check if a value is set for the target error
        self.connections[control.label][target.label] = error
