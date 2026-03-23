# Rust internals

This section documents the internal Rust API of `ferrmion`. It is primarily
useful for contributors who want to extend the Rust core, understand algorithmic
details, or add new PyO3 bindings.

The Rust crate is organised into the following public modules:

| Module | Purpose |
|--------|---------|
| `operators` | Symplectic operators, ladder operators, Majorana sparse representation |
| `ternarytree` | Ternary tree data structure used for all encodings |
| `encoding` | Trait-based encoding abstraction and Majorana encoding implementation |
| `hamiltonians` | Qubit Hamiltonian type and template generation |
| `optimise` | Annealing and TOPPHATT optimisation algorithms |
| `states` | Z-basis qubit state representation |
| `utils` | Shared utility functions (symplectic arithmetic, phase accounting) |

---

## operators

```{rust:module} ferrmion::operators
```

---

## ternarytree

```{rust:module} ferrmion::ternarytree
```

---

## encoding

```{rust:module} ferrmion::encoding
```

---

## optimise

```{rust:module} ferrmion::optimise
```

```{rust:module} ferrmion::optimise::anneal
```

```{rust:module} ferrmion::optimise::topphatt
```
