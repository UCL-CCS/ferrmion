"""Tests for Interop Functions."""

from ferrmion.encode import JordanWigner, BravyiKitaev
import pytest

try:
    import qiskit
    import qiskit_nature
    from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
    from qiskit_nature.second_q.operators import FermionicOp
    from qiskit.quantum_info import SparsePauliOp
    from ferrmion.interop.qiskit_mapper import QiskitAdapter
except ImportError:
    qiskit = None
    qiskit_nature = None

@pytest.mark.skipif(qiskit is None or qiskit_nature is None, reason="Extra depdency `ferrmion[qiskit]` not installed.")
def test_qiskit_adapter_jw():
    fop = FermionicOp(
        {
            "+_0 -_0": 1.0,
            "+_1 -_1": -1.0,
            "-_1 +_0": 2.
        },
        num_spin_orbitals=2,
    )
    expected = SparsePauliOp(['IZ', 'ZI', 'XY', 'XX', 'YY', 'YX'],
              coeffs=[-0.5+0.j ,  0.5+0.j ,  0. +0.5j, -0.5+0.j , -0.5+0.j ,  0. -0.5j])

    mapper = JordanWignerMapper()
    inbuilt = mapper.map(fop)
    assert inbuilt == expected
    assert expected.equiv(QiskitAdapter(JordanWigner(2)).map(fop))
