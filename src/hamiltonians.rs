//! `pyclass` wrappers for the core Hamiltonian types.

use crate::error::CoreError;
use crate::operators::PyMajoranaSparse;
use ferrmion_core::hamiltonians::{FermionHamiltonian, QubitHamiltonian, SymplecticHamiltonian};
use ferrmion_core::operators::{CoefficientPauliWeight, PauliWeight};
use ferrmion_core::optimise::{
    clifford_heuristic_optimisation, randomised_subsystem_descent, AnnealingParameters,
    CliffordSubset, SubsystemSampler,
};
use numpy::{Complex64, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyIterator, PyList, PyTuple, PyType};
use std::collections::HashMap;
use std::str::FromStr;

fn validate_pauli_key(key: &str, n_qubits: Option<usize>) -> Result<(), CoreError> {
    if let Some(n) = n_qubits {
        if key.len() != n {
            return Err(CoreError::Value(format!(
                "Pauli string '{key}' has length {}, expected {n}.",
                key.len()
            )));
        }
    }
    if !key.chars().all(|c| matches!(c, 'I' | 'X' | 'Y' | 'Z')) {
        return Err(CoreError::Value(format!(
            "Pauli string '{key}' may only contain characters I, X, Y, Z."
        )));
    }
    Ok(())
}

/// Mapping from Pauli strings to complex coefficients, backed by the Rust
/// [`QubitHamiltonian`] type.
///
/// Supports the standard mapping operations (`q[key]`, `q.items()`, `len(q)`,
/// `q.get(...)`, `dict(q)`) and adds Clifford-based optimisation methods that
/// return a new `QubitHamiltonian`.
#[pyclass(name = "QubitHamiltonian", module = "ferrmion.core")]
#[derive(Clone, Debug, Default)]
pub struct PyQubitHamiltonian(pub QubitHamiltonian);

#[pymethods]
impl PyQubitHamiltonian {
    #[new]
    #[pyo3(signature = (data = None))]
    fn new(data: Option<HashMap<String, Complex64>>) -> Result<Self, CoreError> {
        let mut inner = QubitHamiltonian::default();
        if let Some(data) = data {
            let mut n_qubits: Option<usize> = None;
            for (key, value) in data {
                validate_pauli_key(&key, n_qubits)?;
                n_qubits = Some(key.len());
                inner.0.insert(key, value);
            }
        }
        Ok(Self(inner))
    }

    /// Number of qubits, inferred from the length of any Pauli key.
    #[getter]
    fn n_qubits(&self) -> Result<usize, CoreError> {
        self.0 .0.keys().next().map(String::len).ok_or_else(|| {
            CoreError::Value("QubitHamiltonian is empty; cannot infer n_qubits".to_string())
        })
    }

    fn __len__(&self) -> usize {
        self.0 .0.len()
    }

    fn __getitem__(&self, key: &str) -> PyResult<Complex64> {
        self.0
             .0
            .get(key)
            .copied()
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }

    fn __setitem__(&mut self, key: String, value: Complex64) -> Result<(), CoreError> {
        let n_qubits = self.0 .0.keys().next().map(String::len);
        validate_pauli_key(&key, n_qubits)?;
        self.0 .0.insert(key, value);
        Ok(())
    }

    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.0
             .0
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0 .0.contains_key(key)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyIterator>> {
        let keys = PyList::new(py, self.0 .0.keys())?;
        Ok(keys.try_iter()?.unbind())
    }

    fn keys(&self) -> Vec<String> {
        self.0 .0.keys().cloned().collect()
    }

    fn values(&self) -> Vec<Complex64> {
        self.0 .0.values().copied().collect()
    }

    fn items(&self) -> Vec<(String, Complex64)> {
        self.0 .0.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    #[pyo3(signature = (key, default = None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &str,
        default: Option<PyObject>,
    ) -> PyResult<Option<PyObject>> {
        match self.0 .0.get(key) {
            Some(v) => Ok(Some(v.into_pyobject(py)?.into_any().unbind())),
            None => Ok(default),
        }
    }

    /// Return the Hamiltonian as a plain ``dict``.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.items().into_py_dict(py)
    }

    /// Total number of non-identity Pauli operators across all terms.
    fn pauli_weight(&self) -> usize {
        self.0.pauli_weight()
    }

    /// Pauli weight of each term multiplied by its coefficient magnitude.
    fn coeff_pauli_weight(&self) -> f64 {
        self.0.coeff_pauli_weight()
    }

    /// Optimise this Hamiltonian via Clifford-heuristic simulated annealing.
    ///
    /// Args:
    ///     temperature: Initial annealing temperature. Defaults to ``n_qubits``.
    ///     `coefficient_weighted`: If ``True``, minimise coefficient-weighted Pauli weight.
    ///     seed: Seed for the RNG. Defaults to ``1017`` when omitted.
    ///     `clifford_subset`: Gate families to sample from. One of ``"all"``, ``"c"``,
    ///         ``"ch"``, ``"cs"``, ``"chs"`` (default), or ``"vp"``.
    ///
    /// Returns:
    ///     `QubitHamiltonian`: The optimised Hamiltonian.
    #[pyo3(signature = (temperature = None, coefficient_weighted = false, seed = None, clifford_subset = "chs".to_string()))]
    fn clifford_heuristic(
        &self,
        py: Python<'_>,
        temperature: Option<f64>,
        coefficient_weighted: bool,
        seed: Option<u64>,
        clifford_subset: String,
    ) -> Result<Self, CoreError> {
        let n_qubits = self.n_qubits()?;
        let temperature = temperature.unwrap_or(n_qubits as f64);
        let clifford_subset = CliffordSubset::from_str(&clifford_subset.to_lowercase())?;
        let mut sym_ham = SymplecticHamiltonian::from_qubit_hamiltonian(&self.0, n_qubits);

        let result = py
            .allow_threads(|| {
                clifford_heuristic_optimisation(
                    &mut sym_ham,
                    temperature,
                    coefficient_weighted,
                    seed.unwrap_or(1017),
                    None,
                    Some(clifford_subset),
                )
            })
            .map_err(|e| CoreError::Runtime(e.to_string()))?;

        sym_ham.operators.apply_clifford_chain(&result.chain);
        Ok(Self(sym_ham.to_qubit_hamiltonian()))
    }

    /// Iteratively optimise by Clifford descent on randomly sampled subsystems.
    ///
    /// Args:
    ///     iterations: Number of subsystem-local Clifford descents to perform.
    ///     `subsystem_dimension`: Number of qubits in each sampled subsystem.
    ///     temperature: Annealing temperature for each descent. Defaults to ``n_qubits``.
    ///     `coefficient_weighted`: If ``True``, minimise coefficient-weighted Pauli weight.
    ///     sampler: Subsystem sampling strategy: ``"full_system"``, ``"uniform"``,
    ///         or ``"hamming"`` (default).
    ///     seed: Seed for the RNG. Defaults to ``1017`` when omitted.
    ///     `clifford_subset`: Gate families to sample from.
    ///
    /// Returns:
    ///     `QubitHamiltonian`: The optimised Hamiltonian.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (iterations, subsystem_dimension, temperature = None, coefficient_weighted = false, sampler = "hamming".to_string(), seed = None, clifford_subset = "chs".to_string()))]
    fn randomised_subsystem_descent(
        &self,
        py: Python<'_>,
        iterations: usize,
        subsystem_dimension: usize,
        temperature: Option<f64>,
        coefficient_weighted: bool,
        sampler: String,
        seed: Option<usize>,
        clifford_subset: String,
    ) -> Result<Self, CoreError> {
        let sampler = match sampler.as_str() {
            "full_system" => SubsystemSampler::FullSystem,
            "uniform" => SubsystemSampler::Uniform,
            "hamming" => SubsystemSampler::Hamming,
            other => {
                return Err(CoreError::Value(format!(
                    "unknown sampler '{other}'; expected one of full_system, uniform, hamming"
                )))
            }
        };
        let n_qubits = self.n_qubits()?;
        let temperature = temperature.unwrap_or(n_qubits as f64);
        let clifford_subset = CliffordSubset::from_str(&clifford_subset.to_lowercase())?;
        let sym_ham = SymplecticHamiltonian::from_qubit_hamiltonian(&self.0, n_qubits);
        let opt = py.allow_threads(|| {
            randomised_subsystem_descent(
                sym_ham,
                AnnealingParameters::new(temperature, iterations, seed.unwrap_or(1017)),
                coefficient_weighted,
                sampler,
                subsystem_dimension,
                Some(clifford_subset),
            )
        });
        Ok(Self(opt.to_qubit_hamiltonian()))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(other) = other.extract::<PyRef<'_, Self>>() {
            self.0 == other.0
        } else if let Ok(map) = other.extract::<HashMap<String, Complex64>>() {
            self.0 .0.len() == map.len() && map.iter().all(|(k, v)| self.0 .0.get(k) == Some(v))
        } else {
            false
        }
    }

    fn __repr__(&self) -> String {
        let mut items: Vec<_> = self.0 .0.iter().collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        let body = items
            .iter()
            .map(|(k, v)| format!("'{k}': {v}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("QubitHamiltonian({{{body}}})")
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, (Bound<'py, PyDict>,))> {
        let dict = slf.borrow().to_dict(slf.py())?;
        Ok((slf.get_type(), (dict,)))
    }
}

/// Builder for fermionic Hamiltonians, backed by the Rust
/// [`FermionHamiltonian`] type.
///
/// Terms map ladder-operator signatures (e.g. ``"+-"``, ``"++--"``) to dense
/// float64 coefficient tensors with one square dimension per signature
/// character.
#[pyclass(name = "FermionHamiltonian", module = "ferrmion.core")]
#[derive(Clone, Debug, Default)]
pub struct PyFermionHamiltonian {
    pub inner: FermionHamiltonian,
    next_term: String,
}

#[pymethods]
impl PyFermionHamiltonian {
    #[new]
    #[pyo3(signature = (*, terms = None, constant_energy = 0.0))]
    fn new(terms: Option<&Bound<'_, PyDict>>, constant_energy: f64) -> Result<Self, CoreError> {
        let mut inner = FermionHamiltonian::new(constant_energy);
        if let Some(terms) = terms {
            for (key, value) in terms.iter() {
                let signature: String = key.extract()?;
                let coeffs: PyReadonlyArrayDyn<f64> = value.extract()?;
                inner.set_term(&signature, coeffs.as_array().to_owned())?;
            }
        }
        Ok(Self {
            inner,
            next_term: String::new(),
        })
    }

    /// Append a creation operator to the term being built.
    fn creation(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.next_term.push('+');
        slf
    }

    /// Append an annihilation operator to the term being built.
    fn annihilation(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.next_term.push('-');
        slf
    }

    /// Set the coefficients for the accumulated operator signature.
    fn with_coefficients<'py>(
        mut slf: PyRefMut<'py, Self>,
        coefficients: PyReadonlyArrayDyn<'py, f64>,
    ) -> Result<PyRefMut<'py, Self>, CoreError> {
        let signature = std::mem::take(&mut slf.next_term);
        match slf
            .inner
            .set_term(&signature, coefficients.as_array().to_owned())
        {
            Ok(()) => Ok(slf),
            Err(e) => {
                slf.next_term = signature;
                Err(e.into())
            }
        }
    }

    /// Add a constant term to the Hamiltonian.
    fn add_constant(mut slf: PyRefMut<'_, Self>, constant_energy: f64) -> PyRefMut<'_, Self> {
        slf.inner.add_constant(constant_energy);
        slf
    }

    /// Number of fermionic modes, or 0 if no terms have been set.
    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn constant_energy(&self) -> f64 {
        self.inner.constant_energy
    }

    #[setter]
    fn set_constant_energy(&mut self, constant_energy: f64) {
        self.inner.constant_energy = constant_energy;
    }

    /// The terms as a ``dict`` mapping signatures to coefficient arrays.
    #[getter]
    fn terms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for term in self.inner.iter() {
            dict.set_item(
                term.signature(),
                term.coefficients().to_owned().into_pyarray(py),
            )?;
        }
        Ok(dict)
    }

    /// The signature and coefficients of all terms, as parallel lists.
    #[getter]
    fn signatures_and_coefficients<'py>(
        &self,
        py: Python<'py>,
    ) -> (Vec<String>, Vec<Bound<'py, PyArrayDyn<f64>>>) {
        let mut sigs = Vec::new();
        let mut coeffs = Vec::new();
        for term in self.inner.iter() {
            sigs.push(term.signature());
            coeffs.push(term.coefficients().to_owned().into_pyarray(py));
        }
        (sigs, coeffs)
    }

    /// Convert to a sparse Majorana representation.
    ///
    /// Returns:
    ///     `MajoranaSparse`: The sparse Majorana representation of this Hamiltonian.
    fn to_majorana_sparse(&self) -> PyMajoranaSparse {
        PyMajoranaSparse(self.inner.to_majorana_sparse())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|o| self.inner == o.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "FermionHamiltonian({}, {} modes, constant {})",
            self.inner.signatures().join(", "),
            self.inner.n_modes(),
            self.inner.constant_energy
        )
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyDict>, f64, String)> {
        Ok((
            self.terms(py)?,
            self.inner.constant_energy,
            self.next_term.clone(),
        ))
    }

    fn __setstate__(&mut self, state: (Bound<'_, PyDict>, f64, String)) -> Result<(), CoreError> {
        let (terms, constant_energy, next_term) = state;
        let mut inner = FermionHamiltonian::new(constant_energy);
        for (key, value) in terms.iter() {
            let signature: String = key.extract()?;
            let coeffs: PyReadonlyArrayDyn<f64> = value.extract()?;
            inner.set_term(&signature, coeffs.as_array().to_owned())?;
        }
        self.inner = inner;
        self.next_term = next_term;
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(
        Bound<'py, PyType>,
        Bound<'py, PyTuple>,
        (Bound<'py, PyDict>, f64, String),
    )> {
        let py = slf.py();
        let state = slf.borrow().__getstate__(py)?;
        Ok((slf.get_type(), PyTuple::empty(py), state))
    }
}
