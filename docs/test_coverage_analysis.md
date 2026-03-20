# Test Coverage Analysis

This document identifies gaps in the current test suite and proposes areas for improvement.

---

## Current State

The project has a reasonable test suite across 15 Python test files and inline Rust tests, but several modules and code paths have no coverage or very limited coverage.

---

## Priority 1 — Core Functionality With No Tests

### 1. Hubbard model (`hamiltonians.py`)

The following public functions are **completely untested**:

- `linear_adjacency_matrix(length, periodic)`
- `square_lattice_adjacency_matrix(shape, periodic)`
- `cube_lattice_adjacency_matrix(shape, periodic)`
- `hubbard_coefficients(n_modes, adjacency_matrix, onsite_term, hopping_term, spinless)`
- `hubbard_hamiltonian(adjacency_matrix, onsite_term, hopping_term, spinless)`

These form a major user-facing entry point for lattice models. A new `test_hubbard.py` file should cover:
- Correct construction of adjacency matrices (1D, 2D, 3D) with and without periodic boundary conditions
- Verification that `hubbard_hamiltonian` produces the correct number of terms
- Eigenvalue comparison against a known analytical result (e.g., 2-site Hubbard)

### 2. Utility functions (`utils.py`)

Several public functions are never called in any test:

| Function | Purpose |
|---|---|
| `xz_swap(symplectic)` | Swap X and Z components |
| `xy_swap(symplectic)` | Swap X and Y components |
| `yz_swap(symplectic)` | Swap Y and Z components |
| `qubit_swap(symplectic, index_pair)` | Swap two qubit positions |
| `check_trivial_overlap(symplectic)` | Check for non-trivial overlap |
| `two_operator_product(creation, left, right)` | Product of two operators |
| `find_pauli_weight(symplectic_hamiltonian)` | Average Pauli weight |
| `save_pauli_ham(pauli_hamiltonian, filename)` | JSON serialization |

These should be covered in an extended `test_utils.py`. The swap functions in particular could benefit from property-based tests using `hypothesis` (the library is already a dev dependency but is currently unused on the Python side): e.g., swapping twice should be idempotent.

### 3. Evolutionary algorithm (`optimize/enumeration/evolutionary.py`)

`lambda_plus_mu` has no direct tests. The function is non-deterministic (it uses `random`), so tests should:
- Seed the RNG for determinism
- Verify the returned array is a valid permutation of `range(n_modes)`
- Verify the returned logbook contains expected keys (`avg`, `std`, `min`, `max`)
- Verify that the function finds a known optimum for a trivially small problem

---

## Priority 2 — Modules With Only Superficial Coverage

### 4. Annealing entry points (`encode/standard.py`)

The four "annealed" convenience wrappers are untested:

- `jordan_wigner_annealed`
- `bravyi_kitaev_annealed`
- `parity_annealed`
- `jkmn_annealed`

The existing `test_anneal.py` only exercises the lower-level `anneal_pauli_weight` and `anneal_coefficient_pauli_weight`. The top-level wrappers should be tested to ensure they wire up the correct encoding and return a valid Hamiltonian dictionary.

### 5. Qiskit interoperability (`interop/qiskit_mapper.py`)

`test_interop.py` contains exactly one test (`test_qiskit_adapter_jw`). Coverage should be extended to:
- All four standard encodings (JW, BK, PE, JKMN)
- Verify that the mapped operator produces the same eigenvalues as the direct encoding
- Round-trip test: encode → Qiskit → diagonalise, compare against the reference eigenvalues already used in `test_hamiltonians.py`

### 6. MaxNTO encoding (`encode/maxnto.py`)

`test_maxnto.py` has a single integration-style test. Missing:
- Direct test for the standalone `maxnto_symplectic_matrix(n_modes)` function
- Test for `MaxNTO._valid_qubit_number` guard (should raise on invalid qubit counts)
- Eigenvalue correctness check for the MaxNTO encoding (mirroring the checks done for JW/BK/PE in `test_hamiltonians.py`)

### 7. RETT (`optimize/rett.py`)

`test_rett.py` has one test that only exercises `squash=True` with default parameters. Missing:
- `squash=False` path
- Non-default `cutoff` values
- Non-default `max_branches` values
- Verify that the resulting tree has ≤ `max_branches` branches per node

---

## Priority 3 — Completely Untested Modules

### 8. Devices (`devices.py`)

`Qubit` and `Topology` have no tests at all. While `Qubit` is abstract, `Topology` is concrete. Tests should cover:
- `Topology.__init__` creates empty connections
- `Topology.add_connection` stores the error correctly and is retrievable
- Note: the current implementation references `q.root_path` on `Qubit` instances in `__init__`, but `root_path` is not declared on the `Qubit` ABC — this is likely a latent bug that tests would expose.

### 9. Visualisation (`visualise/`)

`draw_tt` and `symplectic_matshow` are untested. These are harder to test meaningfully without a display, but at a minimum:
- Call `draw_tt` with `type="standard"`, `"spaced"`, and `"linear"` on a small tree — verify no exception is raised
- Verify `draw_tt` raises `ValueError` for an unknown `type` string (this branch exists in the code at line 67 of `graph.py`)
- Confirm `draw_tt` accepts all three input types (`TTNode`, `TernaryTree`, `rx.PyDiGraph`)
- Call `symplectic_matshow` and confirm it returns without error

Tests can use `matplotlib`'s non-interactive backend (`matplotlib.use("Agg")`) to avoid needing a display.

---

## Priority 4 — Test Quality and Missing Edge Cases

### 10. Property-based testing with `hypothesis`

`hypothesis[numpy]` is already a dev dependency but is not used anywhere in the Python test suite. The Rust side uses `proptest` effectively for encoding and operator invariants. The Python side should adopt the same approach for:

- **Symplectic/Pauli round-trips**: `pauli_to_symplectic(symplectic_to_pauli(s)) == s` for arbitrary symplectic arrays
- **Swap idempotency**: `xz_swap(xz_swap(s)) == s` for arbitrary inputs
- **Hash round-trips**: `symplectic_unhash(symplectic_hash(s), n) == s`
- **TernaryTree flatpack round-trip**: `TernaryTree.from_flatpack(tree.flatpack()) == tree`

### 11. Error handling and invalid input tests

No tests currently verify that functions raise appropriate errors for bad inputs:
- Encoding functions called with mismatched `n_modes`
- `bonsai_algorithm` with an empty or disconnected graph
- `MaxNTO` with an invalid number of qubits
- `reduced_entanglement_ternary_tree` with a non-square mutual information matrix

### 12. Import error in `test_topphatt.py`

Line 2 of `test_topphatt.py` contains `from autoray import e`. `autoray` is not a declared dependency of the project and `e` is unused. This should be removed before it causes test collection failures in environments where `autoray` is not installed.

---

## Rust-Side Gaps

### 13. `hamiltonians.rs` — no test module

The `PauliWeight` and `CoefficientPauliWeight` implementations for `QubitHamiltonian` have no inline `#[cfg(test)]` tests. Unit tests should verify:
- An empty Hamiltonian has weight 0
- A single-term Hamiltonian gives the expected weight
- Weight is invariant under reordering of terms

### 14. `ternarytree.rs` — limited tests

The `Edge` and `YParity` types and their trait implementations (`as_char`, `as_u8`, `Not`) are not explicitly tested. These are straightforward but worth covering given they underpin the whole encoding.

### 15. `encoding.rs` — error path coverage

`MajoranaEncodingError` variants are never exercised directly. Tests should construct invalid inputs and assert that the correct error variant is returned.
