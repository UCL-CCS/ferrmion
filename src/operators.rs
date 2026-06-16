//! `pyclass` wrapper for the core [`MajoranaSparse`] type.

use ferrmion_core::operators::MajoranaSparse;
use numpy::{Complex64, IntoPyArray, PyArray1};
use pyo3::prelude::*;

/// A sparse representation of a Hamiltonian as a sum of Majorana operator
/// products, backed by the Rust [`MajoranaSparse`] type.
///
/// Instances are obtained via `FermionHamiltonian.to_majorana_sparse()`.
#[pyclass(name = "MajoranaSparse", module = "ferrmion.core")]
#[derive(Clone, Debug)]
pub struct PyMajoranaSparse(pub MajoranaSparse);

#[pymethods]
impl PyMajoranaSparse {
    /// The Majorana indices of each term, as a list of lists.
    #[getter]
    fn indices(&self) -> Vec<Vec<u16>> {
        self.0
            .indices
            .iter()
            .map(|term| term.as_slice().to_vec())
            .collect()
    }

    /// The complex coefficient of each term.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.0.coefficients.clone().into_pyarray(py)
    }

    /// The constant (identity) term of the Hamiltonian.
    #[getter]
    fn constant(&self) -> f64 {
        self.0.constant
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .map(|o| self.0 == o.0)
            .unwrap_or(false)
    }

    fn __repr__(&self) -> String {
        format!(
            "MajoranaSparse({} terms, constant {})",
            self.0.indices.len(),
            self.0.constant
        )
    }
}
