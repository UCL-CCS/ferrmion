# Core (Rust extension)

The `ferrmion.core` module contains the Rust-accelerated functions that power
the encoding and optimisation pipelines. All functions are compiled via PyO3 and
available directly on the `ferrmion` namespace through re-exports in
`ferrmion/__init__.py`.

```{eval-rst}
.. automodule:: ferrmion.core
   :members:
   :undoc-members:
   :show-inheritance:
```
