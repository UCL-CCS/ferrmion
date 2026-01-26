# Ferrmion Copilot Instructions

## Architecture Overview
Ferrmion is a Rust-based library with Python bindings for optimizing fermion-to-qubit encodings in quantum computing. Core components:
- **Encoding**: Ternary tree (JW, Parity, BK, JKMN) and MaxNTO schemes in `src/encoding.rs` and `python/ferrmion/encode/`
- **Hamiltonians**: FermionHamiltonian class in `python/ferrmion/hamiltonians.py` for molecular/Hubbard models
- **Optimization**: Evolutionary algorithms (deap), simulated annealing in `src/optimise/`
- **Data Flow**: FermionHamiltonian → FermionQubitEncoding → QubitHamiltonian with symplectic representations

Rust handles performance-critical ops (e.g., `src/operators.rs`), Python provides high-level API and interop.

## Developer Workflows
- **Version Control**: Use `jj` (Jujutsu) instead of Git; `jj describe -m "msg"` for commits
- **Build**: `uv run maturin develop -r` for local Python package, `cargo build` for Rust
- **Test**: `uv run pytest` with hypothesis for testing
- **Lint**: `pre-commit run --all` (Ruff for Python, Clippy for Rust)
- **Debug**: Enable logging with `fr.utils.setup_logs()`; use notebooks in `development/` for prototyping

## Project Conventions
- **Symplectic Representation**: Use boolean arrays for Pauli operators
- **Testing**: Extensive property-based testing with `proptest` in Rust (`src/` tests) and `hypothesis` in Python (`python/tests/`); aim for dual coverage on core Rust functions and Python API where possible; fixtures for encodings like `six_mode_tree`
- **Documentation**: Use Google-style docstrings (enforced by Ruff); maintain CHANGELOG.md for all changes; keep README.md up-to-date; ensure examples in `docs/notebooks` are runnable and accurate
- **Dependencies**: Core via Cargo.toml (PyO3, ndarray); Python via uv/pyproject.toml (deap, rustworkx)
- **Interop**: Qiskit via `FermionQubitEncoding.to_qiskit()`; install with `pip install ferrmion[qiskit]`
- **Performance**: Prefer Rust for loops/arrays; Python for orchestration and external libs

## Key Files
- `src/lib.rs`: PyO3 bindings entry
- `python/ferrmion/__init__.py`: Main API imports
- `python/ferrmion/encode/`: Encoding classes (e.g., `TernaryTree`)
- `development/`: Jupyter notebooks for examples and validation against OpenFermion</content>
<parameter name="filePath">/Users/michaelwilliamsdelabastida/Code/ferrmion/.github/copilot-instructions.md
