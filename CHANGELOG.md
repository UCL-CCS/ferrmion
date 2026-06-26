# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Prototype bit-vector Majorana-term backend for TOPP-HATT. The inner weight
  evaluation and Hamiltonian reduction are now expressed behind a
  `MajoranaTermStore` trait (in `ferrmion-core`), with two implementations:
  `ArrayVecTermStore` (the production index-list representation, unchanged
  behaviour) and `BitTermStore<W>` (one fixed-width word per term, bit `i` =
  Majorana `i` present with odd parity). The bit backend is generic over a
  `BitWord` (`u64` → ≤ 31 modes, `u128` → ≤ 63 modes, `bnum` `U256` → ≤ 127
  modes), with aliases `BitTermStore64` / `BitTermStore128` / `BitTermStore256`.
  `topphatt` is now a thin wrapper over the generic `topphatt_impl`.
  `BitSlicedTermStore` adds a transposed ("bit-sliced") layout: one `u64`
  bit-vector per Majorana index whose bits index terms, so scoring a selection
  reads only the three relevant vectors and computes the Pauli weight with
  word-parallel bit ops; it has no mode ceiling. A criterion benchmark
  (`topphatt_bitvec`) sweeps the index-list, `u64`, `u128`, `u256` and bit-sliced
  backends over a grid of mode count and Majorana term degree (plus large-mode
  points only the index-list and bit-sliced backends can reach), with additional
  `term_store_build` (conversion-only) and `topphatt_end_to_end`
  (preparation-inclusive) groups confirming the format-conversion cost is
  negligible (microseconds) next to the optimizer. On dense inputs
  the bit backends can pick different (but valid) encodings because they
  deduplicate terms by parity-set rather than by multiset (the bit-sliced backend
  does no deduplication at all); all `BitTermStore` word widths run the identical
  algorithm and always agree with each other.
- `MajoranaSparse`, a Rust-backed class exposed from `ferrmion.core` representing
  a sparse Majorana-operator Hamiltonian, with `indices`, `coefficients` and
  `constant` properties. Obtainable via `FermionHamiltonian.to_majorana_sparse()`.
- `QubitHamiltonian`, `FermionHamiltonian` and `MajoranaEncoding` are now Rust-backed
  classes exposed directly from `ferrmion.core` via PyO3, replacing the duplicated
  Python implementations.
  - `QubitHamiltonian` supports the mapping protocol (`q[key]`, `len(q)`, `dict(q)`,
    `q.items()`, equality with plain dicts), `pauli_weight`/`coeff_pauli_weight`,
    pickling, and the Clifford optimisation methods.
  - `FermionHamiltonian` keeps the fluent builder API
    (`creation()/annihilation()/with_coefficients()/add_constant()`), exposes
    `terms`, `signatures_and_coefficients` and `to_sparse_majorana()`, and pickles.
  - `MajoranaEncoding` provides factory staticmethods (`jordan_wigner`,
    `bravyi_kitaev`, `parity`, `jkmn`, `maxnto`, `from_flatpack`, `from_json`),
    properties (`n_modes`, `n_qubits`, `ipowers`, `symplectic_matrix`,
    `vacuum_state`) and methods (`encode`, `encode_annealed`, `anneal_enumeration`,
    `decode`, `hartree_fock_state`, `number_operator`, `edge_operator`,
    `interaction_operator`, `encode_fermion_product`, `batch_pauli_weights`,
    `apply_mode_enumeration`, `to_json`), with parallel Rust sections releasing
    the GIL.
- `FermionHamiltonian` type in `ferrmion-core`, aggregating `FermionMatrix` terms
  with a constant energy offset and converting to `MajoranaSparse` through the
  existing `FermionMatrix -> FermionSparse -> MajoranaSparse` chain.
- `TernaryTree.build_encoding()` returning a `MajoranaEncoding` for the tree, and
  tree-building classmethods `TernaryTree.jordan_wigner/bravyi_kitaev/parity/jkmn`.
- `SymplecticMatrix::to_concatenated()` returning the `[x_block | z_block]` layout.

### Changed
- `QubitHamiltonian` is no longer a `dict` subclass; it implements the mapping
  protocol instead (`json.dumps(qham)` no longer works; use `qham.to_dict()`).
- The public encoding factories (`JordanWigner`, `BravyiKitaev`, `ParityEncoding`,
  `JKMN`, `MaxNTO`) now return a `MajoranaEncoding` rather than an encoding class
  instance; they also accept an optional `n_qubits` argument.
- `encode_annealed` and `encode_topphatt` now return a
  `(QubitHamiltonian, MajoranaEncoding)` tuple instead of mutating the encoding
  in place; the `*_annealed`/`*_topphatt` convenience functions still return only
  the `QubitHamiltonian`.
- `to_json` now also includes the `vacuum_state`.
- `ferrmion.core.hatt`, `topphatt` and `encode_topphatt` take a
  `FermionHamiltonian` instead of loose signature/coefficient lists; `topphatt`
  returns a `MajoranaEncoding` and `encode_topphatt` returns
  `(QubitHamiltonian, MajoranaEncoding)`.
- `ferrmion.core.topphatt` now takes a `MajoranaSparse` instead of a
  `FermionHamiltonian`; use `fham.to_majorana_sparse()` to convert.
- Constructing a `MajoranaEncoding` from explicit symplectic data without a
  vacuum state now determines the vacuum automatically (GF(2) solve) instead of
  assuming the all-zero state.
- `FermionHamiltonian.terms` returns coefficient tensors with antisymmetry-
  disallowed entries zeroed (encoded output is unchanged; those terms were
  always filtered during encoding).
- The optimize wrappers `anneal_pauli_weight`/`anneal_coefficient_pauli_weight`
  take a `MajoranaEncoding` and return `(best_cost, MajoranaEncoding)`.
- TOPP-HATT's parallel weight search now uses a `rayon` parallel iterator with a
  lock-free atomic bound and an associative, deterministic reduction, replacing the
  manual scoped-thread pool with its `Mutex`/`RwLock`. The branch-and-bound
  early-exit and reproducible tie-breaking are preserved, and the `num_cpus`
  dependency is dropped in favour of rayon's global thread pool.

### Removed
- Python classes `FermionQubitEncoding`, `MajoranaStringEncoding` and the
  `MaxNTO` class (use `MajoranaEncoding(...)` and `MajoranaEncoding.maxnto(...)`;
  a `MaxNTO(n_modes)` function alias remains in `ferrmion.encode`).
- Superseded `ferrmion.core` functions: `encode`, `encode_fermion_product`,
  `encode_standard`, `decode`, `hartree_fock_state`, `batch_pauli_weights`,
  `anneal_enumerations`, `clifford_heuristic`, `randomised_subsystem_descent`,
  `flatpack_symplectic_matrix`, `maxnto_symplectic_matrix` and
  `fermionic_to_sparse_majorana` — all replaced by methods on the new classes.

## [0.9.0] - 2026-06-02
### Added
- `NodeOrderHeuristic`s for TOPP-HATT.
  - `min_weight` (default): evaluate every active node and keep the one with the lowest weight.
  - `x_first`: Pick the x-most node on the tree.
  - `z_first`: Pick the z-most node on the tree.
  - `random`: Pick a random node on the tree. Allows for a random seed to be passed.
- `SymplecticHamiltonian` added to `core`.
- Clifford heuristic optimisation of qubit hamiltonians.
- `QubitHamiltonian.clifford_heuristic` and `QubitHamiltonian.randomised_subsystem_descent`
  methods for Clifford-circuit optimisation directly on encoded qubit Hamiltonians.
- `randomised_subsystem_descent` Rust algorithm and `SubsystemSampler` strategy enum
  (`full_system | uniform | hamming`) in `ferrmion_core::optimise::hamiltonian`.

### Changed
- `encode_annealed` now takes a random `seed` argument.
- `MajoranaEncoding.try_encode` applies operators in order of decreasing index to match expected output for encodings as linear combinations of fock states.
- `QubitHamiltonian` is now a `dict[str, complex]` subclass with optimisation methods,
  replacing the prior type alias.
- `QubitHamiltonian.clifford_heuristic`: Optimisation of qubit hamiltonian by transformation with clifford opertors.

### Removed
- `FermionQubitEncoding.encode_clifford_heuristic` — use
  `enc.encode(fham).clifford_heuristic(...)` instead.

## [0.8.0] - 2026-04-15
### Added
- `decode` function for encodings to transform a set of Z-basis measurements into fock states.

### Changed
- `flatpack_symplectic_matrix` now returns the vacuum state as well as the ipower and symplectic matrix.
- `encode` and `core.topphatt` now take the vacuum state as input.

### Removed
- `encode.standard` removed, users should use the encodings classes directly.
- `core.topphatt_standard` removed, use `encode.TernaryTree.topphatt` and `encode.TernaryTree.encode_topphatt` instead.
- `TernaryTree.from_hamiltonian_coefficients`, users can construct a `FermionHamiltonain` and use `TernaryTree(fham.n_modes)`.

## [0.7.3]
### Added
- Water 6-31G benchmark data for performance testing.
- CodSpeed continuous performance tracking in CI.
- Performance benchmarks and improved CI/CD workflows.

### Changed
- Encoding avoids intermediate representations, reducing memory overhead.
- Optimized Rust encoding performance: replaced `BTreeSet` with sort+dedup in `reduce_hamiltonian`, used `sort_unstable` for fixed-size arrays.
- TOPP-HATT performance improvement: batched thread work to reduce lock contention

### Removed
- `encode_anneal` benchmarks were unpredictable, even with fixed rng.

## [0.7.2]
### Added
- `SymplecticOperator`, owned variant, and `SymplecticMatrix` for representing symplectic operations.
- `ZBasisState` and supporting states infrastructure in `states` module.

### Changed
- Encodings now use symplectic operators internally.

## [0.7.1]
### Added
- `encode_topphatt` function in core to allow re-use of hamiltonian.

### Changed
- `topphatt` checks the number of hamiltonian terms remaining before spawning threads

## [0.7.0]
### Added
- `zero_disallowed_terms` deletes coefficients of degree 4 fermionic operators with invalid combinations of indices.

### Changed
- core `topphatt` implementation runs concurrently for Hamiltonians with more than 1,000 terms
- 'signature' of fermionic operators changed to 'action'

### Removed
- Python implementation of `topphatt`

## [0.6.3]
### Added
  - `MajoranaStringEncoding` class for input-defined encodings.

## [0.6.2]
### Changed
  - Altered log level in `set_qubit_indices` to `debug`.

## [0.6.1]
### Added
  - base `FermionQubitEncoding` uses `_encode_fermion_product` to encode number, edge and interaction operators.
  - `TernaryTree.from_flatpack` to allow re-use of flattened trees.
  - `docs/publications/` folder to keep track of notebooks used to produce published results.

### Changed
  - `FermionQubitEncoding` functions now using `_encode_fermion_product`:
    - `.number_operator()`
    - `.edge_operator()`
    - `.interaction_operator()`

## [0.6.0]
### Added
- Benchmarking tests of `TernaryTree.encode...()` functions.
- `core.flatpack_symplectic_matrix`
- `core.fermionic_to_sparse_majorana`

### Changed
- Required python version changed to `>=3.12`
- `TernaryTree._build_symplectic_matrix` now calls to `core.flatpack_symplectic_matrix`
- `hartree_fock_state` function narrowed to `ternary_tree_hartree_fock_state` to simplify implementation.

### Fixed
- `MajoranaSparse::majorise` reintroduced, resulting in smaller hamiltonians and faster optimisation.

## [0.5.3]
### Added
- Initial property based testing with `hypothesis` for python and `proptest` for rust.
- Inline documentation for `core.operators`.

### Fixed
- `TernaryTree.encode_annealed` now updates encoding ipowers and coeffients after running.

## [0.5.2]
### Fixed
- Removed `norm()` from constant energy, giving incorrect sign.

## [0.5.0]
### Added
- `FermionHamiltonian` class in `hamiltonians.py` for building general hamiltonians with matrix coefficients.
  - `creation` and `annihilation` functions for building term coefficients
  - `with_coefficients` to add coefficients and end term building.
- `encode.FermionQubitEncoding`
  - `encode`, `encode_annealed` which accept a `FermionHamiltonian`
- `encode.TernaryTree` now has methods:
  - `topphatt` which returns a new encoding optimised using TOPP-HATT
  - `encode_topphatt`
- `encode.standard` with wrappers on the `core` functions for enocoding Jordan-Wigner, Bravyi-Kitaev, Parity and JKMN. Each of naive, topphatt and annealed.
- `PauliWeight` and `CoefficientPauliWeight` traits in `core`.
- `optimise.enumeration.anneal` has wrapper functions `anneal_pauli_weight` and `anneal_coefficient_pauli_weight` to simplify interface.

### Removed
- There are now no functions relating to hamiltonian templates. The SparseMajorana type in core is used instead.

### Fixed
- Energy of hamiltonians was incorrect owing to a bug with `MajoranaSparse.majorise`. For now this isn't used.

### Changed
- Example notebooks relating to hamiltonians now condensed into `hamiltonians`
- Annealing uses `SparseMajorana` rather than `HamiltonianTemplate`.

## [0.4.1]
### Fixed
- `Encode<&MajoranaSparse>` now correctly handles constant term.

## [0.4.0]
### Added
- Topology-preserving Hamiltonian Adaptive Ternary Tree (TOPP-HATT) in `src/topphatt`.
- `MajoranaEncodingOwned` in `src/encoding`
- `TernaryTree` in `src/ternarytree`
- `FermionMatrix`, `FermionSparse`, `FermionProduct`, `MajoranaProduct`, `MajoranaSparse` in `src/types`
- `max_nodes` option in `bonsai_algorithm` to build trees without using all the nodes of a device.
- New functions exposed to python api of `core`: `topphatt`, `topphatt_standard`, `encode`, `encode_standard`, `standard_symplectic_matrix`.
- `TernaryTree.to_flatpack` to serialise TT structure.

### Changed
- `TernaryTree.default_enumeration_scheme` allows arbitrary qubit labels but enforces mode labels from `range(n_modes)`.
- `TernaryTree.n_qubits` allowed as init input, with default to `n_modes`, this allows building operators where the qubit labels are not `(0,...,n_qubits)`.

### Fixed
- `bonai_algorithm` now deterministic in choice of qubits

## [0.3.0]
### Added
- `interop.QiskitAdapter` which takes a `FermionQubitEncoding` as input, returning a `qiskit_nature.QubitMapper` which can be used in the normal way with `mapper.map(<fermionic operator>)`
- `encode.base.majorana_product` function to calculate general majorana operator products from an encoding, also added as attribute `FermionQubitEncoding.majorana_product`.

### Changed
- Conversion functions `symplectic_to_pauli`, `pauli_to_symplectic` now take in ipower as second argument, returning updated ipower.
- `symplectic_to_sparse` output has been reodered so that it can be directly input the `SparsePauliOp`.
- `anneal_enumerations` takes a flag "coefficient_weighted" to switch between pauli weight and coeffient pauli weight.
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
