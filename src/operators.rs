//! `pyclass` wrapper for the core [`MajoranaSparse`] type.

use crate::functions::simplified_majorana_terms;
use ferrmion_core::operators::MajoranaSparse;
use numpy::{Complex64, IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// A sparse representation of a Hamiltonian as a sum of Majorana operator
/// products, backed by the Rust [`MajoranaSparse`] type.
///
/// Instances are obtained via `FermionHamiltonian.to_majorana_sparse()`.
#[pyclass(name = "MajoranaSparse", module = "ferrmion.core")]
#[derive(Clone, Debug)]
pub struct PyMajoranaSparse {
    pub inner: MajoranaSparse,
    /// Mode count of the `FermionHamiltonian` this was converted from.
    ///
    /// The core `MajoranaSparse` carries no mode count of its own, so this is
    /// tracked here to let Python-facing methods that need a mode count
    /// (e.g. defaulting `hatt`'s `n_modes` or `encode_annealed`'s
    /// `temperature`) work without requiring the caller to pass it explicitly.
    pub n_modes: usize,
}

#[pymethods]
impl PyMajoranaSparse {
    /// The Majorana indices of each term, as a list of lists.
    #[getter]
    fn indices(&self) -> Vec<Vec<u16>> {
        self.inner
            .indices
            .iter()
            .map(|term| term.as_slice().to_vec())
            .collect()
    }

    /// The complex coefficient of each term.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.inner.coefficients.clone().into_pyarray(py)
    }

    /// The constant (identity) term of the Hamiltonian.
    #[getter]
    fn constant(&self) -> f64 {
        self.inner.constant
    }

    /// The mode count of the `FermionHamiltonian` this was converted from.
    #[getter]
    fn n_modes(&self) -> usize {
        self.n_modes
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|o| self.inner == o.inner && self.n_modes == o.n_modes)
    }

    fn __repr__(&self) -> String {
        format!(
            "MajoranaSparse({} terms, constant {}, n_modes {})",
            self.inner.indices.len(),
            self.inner.constant,
            self.n_modes
        )
    }

    /// Convert to a dictionary mapping Majorana index tuples to coefficients.
    /// Simplified by removing paired adjacent indices.
    fn to_dict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let simplified = simplified_majorana_terms(self.inner.clone());
        let output = PyDict::new(py);
        for (key, val) in simplified {
            let key_tuple = PyTuple::new(py, key.as_slice()).unwrap();
            let _ = output.set_item(key_tuple, val);
        }
        output
    }
}
