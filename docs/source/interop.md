# Quantum SDK Interop

Various software development kits implement some form of Fermion-qubit encoding. To interface with these, it is easiest to transform the outputs of ferrmion to the native format of the SDK.

The two main formats that you are likely to need are dictionary format Pauli-string Hamiltonians, which are obtained from `ferrmion.hamiltonian` functions, or the `fill_template` function where a template has been used:

```python
from ferrmion.hamiltonians import molecular_hamiltonian, hubbard_hamiltonian

ferrmion_qham: dict[str, float] = molecular_hamiltonian(encoding, ones, twos, constant)
```

and even fermionic operators, obtained from `.number_opertor`, `.edge_operator` and `ferrmion.encode.base._double_fermionic_op

## Qiskit

Operators defined in `ferrmion` can be used in qiskit by creating a `SparsePauliOp`

```python
from symmer import PauliWordOp
from ferrmion.hamiltonians import molecular_hamiltonian
qham = {"I":1, "X":0.5}
pwop = PauliWordOp.from_dict(qham)

ferrmion_qham = molecular_hamiltonian(encoding, ones, twos, constant)
pwop = PauliWordOp.from_dict(ferrmion_qham)
```

## Symmer

The main operator type in Symmer which is relevant is the `PauliWordOp`. This can be generated straightforwardly from `ferrmion` by creating a dict mapping pauli operators to coefficients.

```python
from symmer import PauliWordOp
qham = {"I":1, "X":0.5}
pwop = PauliWordOp.from_dict(qham)

ferrmion_qham = to_qubit_hamiltonian(encoding, hashed_hamiltonian)
pwop = PauliWordOp.from_dict(ferrmion_qham)
```

## ffsim

A method is prodived in `ffsim` to create a Unitary Cluster Jastrow operator, but strictly for the Jordan-Wigner encoding. `ferrmion.interop` contains slightly altered methods for allowing the use of arbitrary encodings.
<!-- TODO link to notebook -->
