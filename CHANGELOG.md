# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `FermionQubitEncoding` base class in `base`
- `TernaryTree` class in `ternary_tree`
- `KNTO` class in `knto`
- basic unit test for the above classs in `tests/`
- Logging setup in `utils.setup_logs`
- `hartree_fock_state` function in base encoding
- Rust functions for `symplectic_product` and `hartree_fock_state`

### Removed

### Changed
- One and Two electron Hamiltonians are first found as templates with terms and coefficient labels, which can then be changed without recalculating terms.
- Restructured project as a maturin mixed rust/python project.

### Fixed 