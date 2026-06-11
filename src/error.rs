//! Bridging of `ferrmion_core` errors to Python exceptions.

use ferrmion_core::encode::majorana::MajoranaEncodingError;
use ferrmion_core::encode::maxnto::MaxNTOError;
use ferrmion_core::encode::ternarytree::TernaryTreeError;
use ferrmion_core::hamiltonians::FermionHamiltonianError;
use ferrmion_core::operators::FermionProductError;
use ferrmion_core::optimise::{CliffordHeuristicError, HattError, ToppHattError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Local error type bridging `ferrmion_core` errors to `PyErr`.
///
/// The orphan rule prevents `impl From<ForeignError> for PyErr` when both
/// types come from external crates. This local type acts as a bridge:
/// `impl From<CoreError> for PyErr` is allowed (local → foreign), and
/// `impl From<ForeignError> for CoreError` is allowed (foreign → local).
#[derive(Debug)]
pub(crate) enum CoreError {
    Value(String),
    Runtime(String),
    Py(PyErr),
}

impl From<CoreError> for PyErr {
    fn from(e: CoreError) -> PyErr {
        match e {
            CoreError::Value(s) => PyValueError::new_err(s),
            CoreError::Runtime(s) => PyRuntimeError::new_err(s),
            CoreError::Py(e) => e,
        }
    }
}

impl From<PyErr> for CoreError {
    fn from(e: PyErr) -> Self {
        CoreError::Py(e)
    }
}

impl From<MajoranaEncodingError> for CoreError {
    fn from(e: MajoranaEncodingError) -> Self {
        CoreError::Value(e.to_string())
    }
}

impl From<TernaryTreeError> for CoreError {
    fn from(e: TernaryTreeError) -> Self {
        CoreError::Value(e.to_string())
    }
}

impl From<ToppHattError> for CoreError {
    fn from(e: ToppHattError) -> Self {
        CoreError::Runtime(e.to_string())
    }
}

impl From<HattError> for CoreError {
    fn from(e: HattError) -> Self {
        CoreError::Runtime(e.to_string())
    }
}

impl From<FermionProductError> for CoreError {
    fn from(_: FermionProductError) -> Self {
        CoreError::Value(
            "Invalid FermionProduct: operators and indices must have equal length".to_string(),
        )
    }
}

impl From<FermionHamiltonianError> for CoreError {
    fn from(e: FermionHamiltonianError) -> Self {
        CoreError::Value(e.to_string())
    }
}

impl From<MaxNTOError> for CoreError {
    fn from(e: MaxNTOError) -> Self {
        CoreError::Value(e.to_string())
    }
}

impl From<CliffordHeuristicError> for CoreError {
    fn from(e: CliffordHeuristicError) -> Self {
        CoreError::Value(e.to_string())
    }
}
