# Development

## Dependencies
`uv` is used for depedency management, with `ruff` for linting.

Additional dependency groups are needed for:
- dev
- docs

## Documentation
Documentation is generated using `spinx`, with the majority written as docstrings in google format. `MyST` is used to enable markdown in documentation sources.

## Rust Acceleration
Rust acceleration is enabled through `pyo3`.

Rust functions are found in `src/` and used `ndarray::numpy` bindings.

## Testing

Testing of python functions is handld by `pytest` with tests contained in `python/tests/`.

For rust functions, test are included in-file, these can be run with `cargo test`.

Where functions exist in both rust and pure python, agreement between these is tested within `python/tests/`
