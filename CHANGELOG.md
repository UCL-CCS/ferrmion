# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
