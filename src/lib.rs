use log::info;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray4,
};
use pyo3::types::{IntoPyDict, PyDict, PyInt, PyString};
use pyo3::{prelude::*, pymodule, Bound};

mod utils;
use crate::optimise::template_weight;
use crate::utils::*;
mod hamiltonians;
use crate::hamiltonians::{fill_template, hubbard, molecular, Notation, QubitHamiltonianTemplate};
mod encoding;
use crate::encoding::{hartree_fock_state, symplectic_product_map};
mod optimise;
use crate::optimise::anneal_enumerations;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "core")]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    info!("Initializing Python module 'core'");

    #[pyfn(m)]
    #[pyo3(name = "symplectic_product")]
    fn wrap_symplectic_product_py<'py>(
        py: Python<'py>,
        left: PyReadonlyArray1<bool>,
        right: PyReadonlyArray1<bool>,
    ) -> (usize, Bound<'py, PyArray1<bool>>) {
        /*
        Computes the symplectic product between two numpy boolean arrays.

        # Simple example
        ```python
        import ferrmion
        import numpy as np
        a = np.array([True, False, True, False])
        b = np.array([False, True, False, True])
        ipower, product = ferrmion.symplectic_product(a, b)
        ```
        */
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
        mode_op_map: PyReadonlyArray1<usize>,
        symplectic_matrix: PyReadonlyArray2<bool>,
    ) -> (Bound<'py, PyArray1<Complex64>>, Bound<'py, PyArray2<bool>>) {
        /*
        Computes the Hartree-Fock state from Python using numpy arrays.

        # Simple example
        ```python
        import ferrmion
        import numpy as np
        vacuum = np.zeros(6)
        hf = np.array([True, True, False, False, False, False])
        mode_op_map = np.array([0,1,2,3,4,5])
        symplectic = np.eye(6, 12, dtype=bool)
        coeffs, states = ferrmion.hartree_fock_state(vacuum, hf, mode_op_map, symplectic)
        ```
        */
        let vacuum_state = vacuum_state.as_array();
        let fermionic_hf_state = fermionic_hf_state.as_array();
        let mode_op_map = mode_op_map.as_array();
        let symplectic_matrix = symplectic_matrix.as_array();
        let (coeffs, states) = hartree_fock_state(
            vacuum_state,
            fermionic_hf_state,
            mode_op_map,
            symplectic_matrix,
        )
        .unwrap();
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
    #[pyo3(name = "pauli_to_symplectic")]
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
    fn wrap_symplectic_product_map<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
    ) -> (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray3<bool>>) {
        let ipowers = ipowers.as_array();
        let symplectics = symplectics.as_array();
        let (power_map, product_map) = symplectic_product_map(ipowers, symplectics);
        (
            PyArray2::from_owned_array(py, power_map),
            PyArray3::from_owned_array(py, product_map),
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "symplectic_to_sparse")]
    fn wrap_symplectic_to_sparse<'py>(
        py: Python<'py>,
        symplectic: PyReadonlyArray1<bool>,
    ) -> (
        Bound<'py, PyInt>,
        Bound<'py, PyString>,
        Bound<'py, PyArray1<usize>>,
    ) {
        let symplectic = symplectic.as_array();
        let (ipower, pauli_string, position_vec) = symplectic_to_sparse(symplectic);
        (
            PyInt::new(py, ipower),
            PyString::new(py, &pauli_string),
            PyArray1::from_owned_array(py, position_vec),
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "molecular_hamiltonian_template")]
    fn wrap_molecular_hamiltonian<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        physicist_notation: bool,
    ) -> Bound<'py, PyDict> {
        let ipowers = ipowers.as_array();
        let symplectics = symplectics.as_array();
        let hamiltonian: QubitHamiltonianTemplate = match physicist_notation {
            true => molecular(ipowers, symplectics, Notation::Physicist),
            false => molecular(ipowers, symplectics, Notation::Chemist),
        };
        hamiltonian
            .into_py_dict(py)
            .expect("Cannot parse Hamiltonian Template dict.")
    }

    #[pyfn(m)]
    #[pyo3(name = "hubbard_hamiltonian_template")]
    fn wrap_hubbard_hamiltonian<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
    ) -> Bound<'py, PyDict> {
        let ipowers = ipowers.as_array();
        let symplectics = symplectics.as_array();
        let hamiltonian = hubbard(ipowers, symplectics);
        hamiltonian
            .into_py_dict(py)
            .expect("Cannot parse Hamiltonian Template dict.")
    }

    #[pyfn(m)]
    #[pyo3(name = "fill_template")]
    fn wrap_fill_template<'py>(
        py: Python<'py>,
        template: &Bound<'py, PyDict>,
        constant_energy: f64,
        one_e_coeffs: PyReadonlyArray2<f64>,
        two_e_coeffs: PyReadonlyArray4<f64>,
        mode_op_map: PyReadonlyArray1<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // let constant_energy = constant_energy.extract(py)?;
        let mode_op_map = mode_op_map.as_array();
        let template = template.extract::<QubitHamiltonianTemplate>()?;
        let one_e_coeffs = one_e_coeffs.as_array();
        let two_e_coeffs = two_e_coeffs.as_array();
        let hamiltonian = fill_template(
            &template,
            constant_energy,
            one_e_coeffs,
            two_e_coeffs,
            mode_op_map,
        );
        Ok(hamiltonian
            .into_py_dict(py)
            .expect("Cannot parse Hamiltonian dict."))
    }

    #[pyfn(m)]
    #[pyo3(name = "template_weight_distribution")]
    fn wrap_template_weight<'py>(
        py: Python<'py>,
        template: &Bound<'py, PyDict>,
        constant_energy: f64,
        one_e_coeffs: PyReadonlyArray2<f64>,
        two_e_coeffs: PyReadonlyArray4<f64>,
        n_permutations: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // let constant_energy = constant_energy.extract(py)?;
        let template = template.extract::<QubitHamiltonianTemplate>()?;
        let one_e_coeffs = one_e_coeffs.as_array();
        let two_e_coeffs = two_e_coeffs.as_array();
        let weight = template_weight(
            &template,
            constant_energy,
            one_e_coeffs,
            two_e_coeffs,
            n_permutations,
        );
        Ok(weight.into_pyarray(py))
    }
    #[pyfn(m)]
    #[pyo3(name = "anneal_enumerations")]
    fn wrap_anneal_enumerations<'py>(
        py: Python<'py>,
        template: &Bound<'py, PyDict>,
        one_e_coeffs: PyReadonlyArray2<f64>,
        two_e_coeffs: PyReadonlyArray4<f64>,
        temperature: f64,
        initial_guess: PyReadonlyArray1<usize>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<usize>>)> {
        let one_e_coeffs = one_e_coeffs.as_array();
        let two_e_coeffs = two_e_coeffs.as_array();
        let template = template.extract::<QubitHamiltonianTemplate>()?;
        let initial_guess = initial_guess.as_array();
        let result = anneal_enumerations(
            template,
            one_e_coeffs,
            two_e_coeffs,
            temperature,
            initial_guess,
        );
        let (cost, permutation) = result.expect("Annealing output error.");
        Ok((cost, permutation.into_pyarray(py)))
    }
    Ok(())
}
