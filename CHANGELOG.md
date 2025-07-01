# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `FermionQubitEncoding` base class in `base`
    - `hartree_fock_state`
- `TernaryTree` class in `ternary_tree`
- `KNTO` class in `knto`
- basic unit tests in test
- Python logging setup in `utils.setup_logs`
- `.pre-commit-config.yaml`
- Sphinx docs set up in `docs/source/` using autodoc, myst with `.readthedocs.yaml` for hosting.
- `optimize` section
    - `lambda_plus_mu` evolutionary algorithm for approximate enumeration optimization.
    - `reduced_entanglement_tree` builder, takin in MI matrix
    - `pauli_weighted_norm` and `minimis_mi_distance` cost functions
- `molcular_hamiltonian` and `molecular_hamiltonian_template` functions in `.hamiltonians.molecular`
- Rust functions in submodule `core` for:
    - `symplectic_product`
    - `hartree_fock_state`
    - `pauli_to_symplectic`
    - `symplectic_to_pauli`
    - `symplectic_to_sparse`
    - `symplectic_product`
    - `symplectic_product_map`
    - `molecular_hamiltonian_template`
    - `fill_template`
    - `icount_to_sign`

### Removed

### Changed

### Fixed
