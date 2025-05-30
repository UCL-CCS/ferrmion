# Quantum SDK Interop

Various software development kits implement some form of Fermion-qubit encoding. To interface with these, it is easiest to transform the outputs of ferrmion to the native format of the SDK.

## Symmer

The main operator type in Symmer which is relevant is the `PauliWordOp`. This can be generated straightforwardly from `ferrmion` by creating a dict mapping pauli operators to coefficients.

```python
from symmer import PauliWordOp
qham = {"I":1, "X":0.5}
pwop = PauliWordOp.from_dict(qham)

ferrmion_qham = to_qubit_hamiltonian(encoding, hashed_hamiltonian)
pwop = PauliWordOp.from_dict(ferrmion_qham)
```

# Inbuilt Methods

Where the interaction with another code base is more complex, or the desired functionality isn't made easy by an SDK, functions to smooth this over are included in `ferrmion.interop`.

## ffsim

A method is prodived in `ffsim` to create a Unitary Cluster Jastrow operator, but strictly for the Jordan-Wigner encoding. `ferrmion.interop` contains slightly altered methods for allowing the use of arbitrary encodings.

```{eval-rst}
.. automodule:: ferrmion.interop
   :members:
   :undoc-members:
   :show-inheritance:
```
