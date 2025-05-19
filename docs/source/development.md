# Development

## Dependencies
`uv` is used for depedency management, with `ruff` for linting.

Additional dependency groups are needed for:
- dev
- docs

## Documentation
Documentation is generated using `spinx`, with the majority written as docstrings in google format. `MyST` is used to enable markdown in documentation sources.

## pyo3
Rust acceleration is enabled through `pyo3`.

Rust functions are found in `src/` and used `ndarray::numpy` bindings.
