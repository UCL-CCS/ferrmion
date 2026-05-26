# Optimisation methods for fermion-qubit encodings.

This folder is split according to the type which is optimised, and then further by the optimisation method used.

## `encoding/`
- `anneal.rs`: Simulated annealing

## `ternarytree/`
- `hatt.rs`: Hamiltonian adaptive ternary tree
- `topphatt.rs`: Hamiltonian adaptive ternary tree with top-p sampling
  - This has some dependencies on `hatt.rs`.

## `hamiltonian/`
- `cliffordheuristic.rs`: Clifford Heuristic optimisation of encoded qubit Hamiltonians.
