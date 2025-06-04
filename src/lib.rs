use numpy::{Complex64, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::types::{PyDict, PyInt, PyString};
use pyo3::{prelude::*, pymodule, Bound};
use std::collections::HashMap;

mod utils;
use crate::utils::*;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "core")]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "symplectic_product")]
    fn wrap_symplectic_product_py<'py>(
        py: Python<'py>,
        left: PyReadonlyArray1<bool>,
        right: PyReadonlyArray1<bool>,
    ) -> (usize, Bound<'py, PyArray1<bool>>) {
        let left = left.as_array();
        let right = right.as_array();
        let (ipower, product) = symplectic_product(left, right);
        let pyproduct = PyArray1::from_owned_array(py, product);
        (ipower, pyproduct)
    }

    #[pyfn(m)]
    #[pyo3(name = "hartree_fock_state")]
    fn wrap_hartree_fock_state_py<'py>(
        py: Python<'py>,
        vacuum_state: PyReadonlyArray1<f64>,
        fermionic_hf_state: PyReadonlyArray1<bool>,
        mode_op_map: Bound<'py, PyDict>,
        symplectic_matrix: PyReadonlyArray2<bool>,
    ) -> (Bound<'py, PyArray1<Complex64>>, Bound<'py, PyArray2<bool>>) {
        let vacuum_state = vacuum_state.as_array();
        let fermionic_hf_state = fermionic_hf_state.as_array();
        let rust_mode_op_map: HashMap<usize, usize> = mode_op_map.extract().unwrap();
        let symplectic_matrix = symplectic_matrix.as_array();
        let (coeffs, states) = hartree_fock_state(
            vacuum_state,
            fermionic_hf_state,
            rust_mode_op_map,
            symplectic_matrix,
        );
        (
            PyArray1::from_owned_array(py, coeffs),
            PyArray2::from_owned_array(py, states),
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "symplectic_to_pauli")]
    fn wrap_symplectic_to_pauli<'py>(
        py: Python<'py>,
        symplectic: PyReadonlyArray1<bool>,
    ) -> (Bound<'py, PyInt>, Bound<'py, PyString>) {
        let symplectic = symplectic.as_array();
        let (ipower, pauli) = symplectic_to_pauli(symplectic);
        (PyInt::new(py, ipower), PyString::new(py, &pauli))
    }

    #[pyfn(m)]
    #[pyo3(name = "puali_to_symplectic")]
    fn wrap_pauli_to_symplectic(
        py: Python<'_>,
        pauli: String,
    ) -> (Bound<'_, PyInt>, Bound<'_, PyArray1<bool>>) {
        // let pauli = pauli.extract();
        let (ipower, symplectic) = pauli_to_symplectic(pauli);
        (
            PyInt::new(py, ipower),
            PyArray1::from_owned_array(py, symplectic),
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "symplectic_product_map")]
    fn wray_symplectic_product_map<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<usize>,
        symplectics: PyReadonlyArray2<bool>,
    ) -> (Bound<'py, PyArray2<usize>>, Bound<'py, PyArray3<bool>>) {
        let ipowers = ipowers.as_array();
        let symplectics = symplectics.as_array();
        let (power_map, product_map) = symplectic_product_map(ipowers, symplectics);
        (
            PyArray2::from_owned_array(py, power_map),
            PyArray3::from_owned_array(py, product_map),
        )
    }

    Ok(())
}
