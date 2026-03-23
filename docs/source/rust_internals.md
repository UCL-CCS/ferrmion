# Rust internals

This section is aimed at contributors who want to extend the Rust core,
understand algorithmic details, or add new PyO3 bindings.

The internal Rust API is documented via `cargo doc` and served alongside
this documentation. It covers types and functions that are not part of the
public Python API, including:

| Module | Purpose |
|--------|---------|
| `operators` | Symplectic operators, ladder operators, Majorana sparse representation |
| `ternarytree` | Ternary tree data structure used for all encodings |
| `encoding` | Trait-based encoding abstraction and Majorana encoding implementation |
| `hamiltonians` | Qubit Hamiltonian type and template generation |
| `optimise` | Annealing and TOPPHATT optimisation algorithms |
| `states` | Z-basis qubit state representation |
| `utils` | Shared utility functions (symplectic arithmetic, phase accounting) |

**[Browse the Rust API docs →](rustdoc/ferrmion/index.html)**

```{note}
The Rust API docs are generated with `cargo doc --no-deps --document-private-items`
and are available at `rustdoc/ferrmion/index.html` relative to this page.
```
