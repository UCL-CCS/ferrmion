//! Fast, reliable and easy optimisation of fermion-qubit encodings.
//!
//! To simulate fermionic Hamiltonians with gate-based quantum computers,
//! it is necessary to encode the fermionic operators to qubit operators
//! which obey fermionic commutation relations.
//!
//! This crate contains the PyO3 interop layer which exposes the
//! `ferrmion-core` types and algorithms to Python. The core Hamiltonian and
//! encoding types are exposed directly as Python classes
//! ([`PyQubitHamiltonian`], [`PyFermionHamiltonian`], [`PyMajoranaEncoding`]),
//! with a small number of free functions for symplectic utilities and
//! tree-construction algorithms.

mod encoding;
mod error;
mod functions;
mod hamiltonians;

pub use encoding::PyMajoranaEncoding;
pub use hamiltonians::{PyFermionHamiltonian, PyQubitHamiltonian};

use log::debug;
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "core")]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    debug!("Initializing Python module 'core'");

    m.add_class::<hamiltonians::PyQubitHamiltonian>()?;
    m.add_class::<hamiltonians::PyFermionHamiltonian>()?;
    m.add_class::<encoding::PyMajoranaEncoding>()?;

    m.add_function(wrap_pyfunction!(functions::symplectic_product, m)?)?;
    m.add_function(wrap_pyfunction!(functions::symplectic_to_pauli, m)?)?;
    m.add_function(wrap_pyfunction!(functions::pauli_to_symplectic, m)?)?;
    m.add_function(wrap_pyfunction!(functions::symplectic_to_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(functions::hatt_py, m)?)?;
    m.add_function(wrap_pyfunction!(functions::topphatt_py, m)?)?;
    m.add_function(wrap_pyfunction!(functions::encode_topphatt_py, m)?)?;
    Ok(())
}
