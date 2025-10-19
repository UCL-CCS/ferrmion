"""Init for Qiskit Interop.

These functions require ferrmion is installed with optional extra dependencies
for example, using pip: `pip install ferrmion[qiskit]`
"""

try:
    from qiskit_mapper import QiskitAdapter

    __all__ = ["QiskitAdapter"]
except ImportError:
    __all__ = []
