# Quantum SDK Interop

Various software development kits implement some form of Fermion-qubit encoding. To interface with these, it is easiest to transform the outputs of ferrmion to the native format of the SDK.

The two main formats that you are likely to need are dictionary format Pauli-string Hamiltonians, which are obtained from `ferrmion.hamiltonian` functions, or the `fill_template` function where a template has been used:

```python
from ferrmion.hamiltonians import molecular_hamiltonian, hubbard_hamiltonian
...

ferrmion_qham: dict[str, float] = molecular_hamiltonian(encoding, ones, twos, constant)
```

and even fermionic operators, obtained from `.number_opertor`, `.edge_operator` and `ferrmion.encode.base.double_fermionic_operator`

```python
from ferrmion.encode.base import double_fermionic_operator
from ferrmion.encode import JordanWigner
encoding = JordanWigner(4)

n_0 = encoding.number_operator(0)
edge_0_2 = encoding.edge_operator((0,2))
create_0_create_1 = double_fermionic_operator(encoding, (0,1), "++")
```

## Qiskit

Operators defined in `ferrmion` can be used in qiskit by creating a `SparsePauliOp`

```python
from qiskit.circuit.library import SparsePauliOp
from ferrmion.hamiltonians import molecular_hamiltonian

qham = molecular_hamiltonian(encoding, ones, twos, constant)
qiskit_op = SparsePauliOp.from_list([(k, v) for k,v in qham.items()])
```

## Symmer

The main operator type in Symmer which is relevant is the `PauliWordOp`. This can be generated straightforwardly from `ferrmion` by creating a dict mapping pauli operators to coefficients.

```python
from symmer import PauliWordOp
from ferrmion.hamiltonians import molecular_hamiltonian

qham = molecular_hamiltonian(encoding, ones, twos, constant)
pwop = PauliWordOp.from_dict(qham)
```
