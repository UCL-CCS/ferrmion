% ferrmion documentation master file, created by
% sphinx-quickstart on Sat May 10 22:28:38 2025.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# ferrmion

Quantum simulation of fermionic Hamiltonians requires encoding to enforce commutation relations.

While it's common to use one of the most basic encodings, such as Jordan-Wigner or Bravyi-Kitaev, encoded circuits and the quality of results is _strongly_ dependent on which encoding is used.

`ferrmion` provides tools for generating encodings, and functions to _optimize for the specific Hamiltonian and Hardware being used_.

Take a look at the [examples gallery](Examples) to get started.

```{toctree}
:caption: 'Contents:'
:maxdepth: 1

examples
encode
optimize
hamiltonians
utils
devices
interop
development
```
