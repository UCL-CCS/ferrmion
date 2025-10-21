# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]
### Added
- `interop.QiskitAdapter` which takes a `FermionQubitEncoding` as input, returning a `qiskit_nature.QubitMapper` which can be used in the normal way with `mapper.map(<fermionic operator>)`

### Changed
- Conversion functions `symplectic_to_pauli`, `pauli_to_symplectic` now take in ipower as second argument, returning updated ipower.
- `symplectic_to_sparse` output has been reodered so that it can be directly input the `SparsePauliOp`.
- `anneal_enerumations` takes a flag "coefficient_weighted" to switch between pauli weight and coeffient pauli weight.
- `optimize.enumeration.cost_functions` move to `optimize.cost_functions`.

## [0.2.0]
### Added
- `hamiltonian_adaptive_ternary_tree` in `optimize.hatt`, with explainer notebook in Examples.
- `TTNode.branch_majorana_map` returns dict from branch strings to indices of majporana operators.

#### Utils
- `fermionic_to_sparse_majorana` converts hamiltonian formatt for use in `hatt`

#### TTNode
- `z_descendant` and `z_ancestor` functions to find farthest relative on the all-z branch

### Removed
- ruff removed from dependencies

### Changed
#### TTNode
- `__str__` function showing value of `root_path`
_ `prefix_root_path` renamed `update_root_path` as not every change is prefixed
- `leaf_majorana_indices` attribute added
- `add_child` will replace an existing child with warning output rather than raise exception
- `add_child` will remove `TTNode.leaf_majorana_indices` item to attach a child

#### TernaryTree
- `.branch_operator_map` renamed `.branch_pauli_map`
- `string_pairing_algorithm` seperaed out, returning map from branches to majorana operator indices (see `TTNode.branch_majorana_map`)
- `_build_symplectic_matrix` uses majorana operator indices from `branch_majorana_map` rather than enumeration scheme to define operator ordering.

## [0.1.1]
### Changed
- Updates to release pipeline to support mac and windows.

## [0.1.0]

### Added
- Majorana-String Encodings in `encode.base`
    - `encode.TernaryTree` tree with helper functions for:
        Jordan Wigner
        Parity Encoding
        Bravyi-Kitaev
        JKMN
    - `encode.maxnto` MaxNTO Encoding
- Ternary Tree Optimizations
    - Bonsai Algorithm ternary trees `encode.optimize.bonsai`
    - Huffman encoded ternary tree `encode.optimize.huffman`
    - Reduced Entanglement ternary tree `encode.optimize.rett`
- Numerical encoding optimization `optimize`
    - `anneal_enumerations` simulated annealing to reduce coefficient-pauli-weight.
    - `lambda_plus_mu` evolutionary algorithm for approximate enumeration optimization.
    - `pauli_weighted_norm` and `minimise_mi_distance` cost functions.
- Fermionic Hamiltonians
    - `hubbard_hamiltonian` and `hubbard_hamiltonian_template` in `.hamiltonians.hubbard`
    - `molcular_hamiltonian` and `molecular_hamiltonian_template` functions in `.hamiltonians.molecular` with support for physicist or chemist notation.
- Utils
    - basic unit tests in test
    - Python logging setup in `utils.setup_logs`
    - `.pre-commit-config.yaml`
- Sphinx docs set up in `docs/source/` using autodoc, myst with `.readthedocs.yaml` for hosting.
    - Example notebooks for
        - General and standad Ternary Trees
        - Reduced entanglement ternary tree
        - Huffman encoded ternary tree
        - Bonsai Algorithm
        - Defining and minimising pauli-weight
        - Encoding the Molecular Hamiltonian
        - Encoding the Hubbard Hamiltonian
- Rust functions in submodule `core`
    - `encoding`
    - `hamiltonians`
    - `lib`
    - `optimize`
    - `utils`

### Removed

### Changed

### Fixed
