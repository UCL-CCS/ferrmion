#![warn(missing_docs)]
//! Fast, reliable and easy optimisation of fermion-qubit encodings.
//!
//! To simulate fermionic Hamiltonians with gate-based quantum computers,
//! it is necessary to encode the fermionic operators to qubit operators
//! which obey commutation fermionic relations.
//!
//! This file contains the PyO3 interop layer which wraps rust functions and exposes
//! these to a python API.

use ::core::panic;
use log::{debug, info};
use numpy::ndarray::Array1;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::types::{IntoPyDict, PyComplex, PyDict, PyInt, PyString};
use pyo3::{prelude::*, pymodule, Bound};
pub mod operators;
mod utils;
use crate::operators::MajoranaSparse;
use crate::optimise::topphatt;
use crate::utils::*;
mod hamiltonians;
use crate::hamiltonians::QubitHamiltonian;
mod encoding;
use crate::encoding::{Encode, MajoranaEncoding};
mod optimise;
use crate::optimise::anneal_enumerations;
pub mod ternarytree;
use crate::ternarytree::{TTFlatPack, TernaryTree};

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
        let (product, ipower) = MajoranaEncoding::symplectic_product(left, right, 0);
        let pyproduct = PyArray1::from_owned_array(py, product);
        (ipower as usize, pyproduct)
    }

    #[pyfn(m)]
    #[pyo3(name = "hartree_fock_state")]
    fn wrap_hartree_fock_state_py<'py>(
        py: Python<'py>,
        vacuum_state: PyReadonlyArray1<f64>,
        fermionic_hf_state: PyReadonlyArray1<bool>,
        mode_op_map: PyReadonlyArray1<usize>,
        ipowers: PyReadonlyArray1<u8>,
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
        let ipowers = ipowers.as_array().to_owned();
        let symplectic_matrix = symplectic_matrix.as_array().to_owned();
        let encoding = MajoranaEncoding::new(ipowers, symplectic_matrix);
        let (coeffs, states) =
            encoding.hartree_fock_state(vacuum_state, fermionic_hf_state, mode_op_map);
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
        ipower: u8,
    ) -> (Bound<'py, PyString>, Bound<'py, PyInt>) {
        let symplectic = symplectic.as_array();
        let (pauli, ipower) = MajoranaEncoding::symplectic_to_pauli(symplectic, ipower);
        (PyString::new(py, &pauli), PyInt::new(py, ipower))
    }

    #[pyfn(m)]
    #[pyo3(name = "pauli_to_symplectic")]
    fn wrap_pauli_to_symplectic(
        py: Python<'_>,
        pauli: String,
        ipower: usize,
    ) -> (Bound<'_, PyArray1<bool>>, Bound<'_, PyInt>) {
        // let pauli = pauli.extract();
        let (symplectic, ipower) = pauli_to_symplectic(pauli, ipower);
        (
            PyArray1::from_owned_array(py, symplectic),
            PyInt::new(py, ipower),
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "symplectic_product_map")]
    fn wrap_symplectic_product_map<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
    ) -> (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray3<bool>>) {
        let encoding = MajoranaEncoding::new(
            ipowers.as_array().to_owned(),
            symplectics.as_array().to_owned(),
        );

        let (power_map, product_map) = encoding.symplectic_product_map();
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
        ipower: usize,
    ) -> (
        Bound<'py, PyString>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyComplex>,
    ) {
        let symplectic = symplectic.as_array();
        let (pauli_string, position_vec, coeff) = symplectic_to_sparse(symplectic, ipower);
        (
            PyString::new(py, &pauli_string),
            PyArray1::from_owned_array(py, position_vec),
            PyComplex::from_complex_bound(py, coeff),
        )
    }

    // #[pyfn(m)]
    // #[pyo3(name = "molecular_hamiltonian_template")]
    // fn wrap_molecular_hamiltonian_template<'py>(
    //     py: Python<'py>,
    //     ipowers: PyReadonlyArray1<u8>,
    //     symplectics: PyReadonlyArray2<bool>,
    //     physicist_notation: bool,
    // ) -> Bound<'py, PyDict> {
    //     let encoding = MajoranaEncoding::new(ipowers.as_array(), symplectics.as_array());
    //     let hamiltonian: QubitHamiltonianTemplate = match physicist_notation {
    //         true => molecular(encoding, Notation::Physicist),
    //         false => molecular(encoding, Notation::Chemist),
    //     };
    //     hamiltonian
    //         .into_py_dict(py)
    //         .expect("Cannot parse Hamiltonian Template dict.")
    // }

    // #[pyfn(m)]
    // #[pyo3(name = "hubbard_hamiltonian_template")]
    // fn wrap_hubbard_hamiltonian_template<'py>(
    //     py: Python<'py>,
    //     ipowers: PyReadonlyArray1<u8>,
    //     symplectics: PyReadonlyArray2<bool>,
    // ) -> Bound<'py, PyDict> {
    //     let encoding = MajoranaEncoding::new(
    //         ipowers.as_array().to_owned(),
    //         symplectics.as_array().to_owned(),
    //     );

    //     let hamiltonian = hubbard(encoding);
    //     hamiltonian
    //         .into_py_dict(py)
    //         .expect("Cannot parse Hamiltonian Template dict.")
    // }

    // #[pyfn(m)]
    // #[pyo3(name = "pauli_weight_distribution")]
    // fn wrap_pauli_weight_distribution<'py>(
    //     py: Python<'py>,
    //     constant_energy: f64,
    //     one_e_coeffs: PyReadonlyArray2<f64>,
    //     two_e_coeffs: PyReadonlyArray4<f64>,
    //     n_permutations: usize,
    // ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    //     // let constant_energy = constant_energy.extract(py)?;
    //     let one_e_coeffs = one_e_coeffs.as_array();
    //     let two_e_coeffs = two_e_coeffs.as_array();

    //     Ok((weight.0.into_pyarray(py), weight.1.into_pyarray(py)))
    // }

    #[pyfn(m)]
    #[pyo3(name = "anneal_enumerations")]
    fn wrap_anneal_enumerations<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        temperature: f64,
        initial_guess: PyReadonlyArray1<usize>,
        coefficient_weighted: bool,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray2<bool>>)> {
        let initial_guess = initial_guess.as_array();

        let msparse = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            0.,
        );
        let encoding = MajoranaEncoding::new(
            ipowers.as_array().to_owned(),
            symplectics.as_array().to_owned(),
        );
        let best_mode_enumeration: Array1<usize>;
        (_, best_mode_enumeration) = anneal_enumerations(
            msparse,
            encoding,
            temperature,
            initial_guess,
            coefficient_weighted,
        )
        .expect("Annealing should have succeeded.");

        let encoding = MajoranaEncoding::new(
            ipowers.as_array().to_owned(),
            symplectics.as_array().to_owned(),
        )
        .apply_mode_enumeration(best_mode_enumeration.to_vec());

        Ok((
            encoding.ipowers.into_pyarray(py),
            encoding.symplectics.into_pyarray(py),
        ))
    }

    #[pyfn(m)]
    #[pyo3(name = "standard_symplectic_matrix")]
    fn wrap_standard_symplectic_matrix(
        py: Python<'_>,
        encoding: String,
        n_modes: usize,
    ) -> PyResult<(Bound<'_, PyArray1<u8>>, Bound<'_, PyArray2<bool>>)> {
        // ) -> PyResult<()> {
        debug!("Starting TOPPHATT");
        // let flatpack: TTFlatPack = node_map.extract::<TTFlatPack>()?;

        let tree: TernaryTree = match encoding.as_str() {
            "Jordan-Wigner" | "JW" => TernaryTree::naive_jordan_wigner(n_modes),
            "Bravyi-Kitaev" | "BK" => TernaryTree::naive_bravyi_kitaev(n_modes),
            "Parity" | "PE" => TernaryTree::naive_parity(n_modes),
            "JKMN" => TernaryTree::naive_jkmn(n_modes),
            _ => panic!("Encoding must be one of JW, PE, BK or JKMN."),
        };
        debug!("Got Tree");
        let encoding = tree.build_encoding(n_modes).unwrap();
        debug!("Got encoding");

        debug!("Got qham");
        Ok((
            encoding.ipowers.into_pyarray(py),
            encoding.symplectics.into_pyarray(py),
        ))
    }
    #[pyfn(m)]
    #[pyo3(name = "encode")]
    fn wrap_encode<'py>(
        py: Python<'py>,
        ipowers: PyReadonlyArray1<u8>,
        symplectics: PyReadonlyArray2<bool>,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        constant_energy: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        // ) -> PyResult<()> {
        assert_eq!(
            signatures.len(),
            coeffs.len(),
            "Signatures and coefficients should be same length"
        );
        let ipowers = ipowers.as_array().to_owned();
        let symplectics = symplectics.as_array().to_owned();
        let n_qubits = symplectics.ncols() / 2;
        let n_modes = symplectics.nrows() / 2;

        assert!(
            n_qubits >= n_modes,
            "Must have at least as many qubits as modes."
        );

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            constant_energy,
        );
        let encoding = MajoranaEncoding::new(ipowers, symplectics);
        debug!("Got encoding");
        let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        debug!("Got Hamiltonian");

        debug!("Got qham");
        Ok(qham
            .into_py_dict(py)
            .expect("Should be able to convert QubitHamiltonian to PyDict."))
        // Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "encode_standard")]
    fn wrap_encode_standard<'py>(
        py: Python<'py>,
        encoding: String,
        n_modes: usize,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
        constant_energy: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        // ) -> PyResult<()> {
        assert_eq!(
            signatures.len(),
            coeffs.len(),
            "Signatures and coefficients should be same length"
        );
        assert!(
            n_qubits >= n_modes,
            "Must have at least as many qubits as modes."
        );

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            constant_energy,
        );
        debug!("Got MSparse");
        debug!("Got Hamiltonian");
        let tree: TernaryTree = match encoding.as_str() {
            "Jordan-Wigner" | "JW" => TernaryTree::naive_jordan_wigner(n_modes),
            "Bravyi-Kitaev" | "BK" => TernaryTree::naive_bravyi_kitaev(n_modes),
            "Parity" | "PE" => TernaryTree::naive_parity(n_modes),
            "JKMN" => TernaryTree::naive_jkmn(n_modes),
            _ => panic!("Encoding must be one of JW, PE, BK or JKMN."),
        };
        debug!("Got Tree");
        debug!("Hamiltonian {:?}", hamiltonian);
        debug!("Hamiltonian {:?}", hamiltonian);
        let encoding = tree.build_encoding(n_qubits).unwrap();
        debug!("Got encoding {:?}", encoding);
        debug!("Got encoding {:?}", encoding);

        let qham: QubitHamiltonian = encoding.encode(&hamiltonian);

        debug!("Got qham");
        debug!("Got qham {:?}", qham);
        Ok(qham
            .into_py_dict(py)
            .expect("Should be able to convert QubitHamiltonian to PyDict."))
        // Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "topphatt")]
    fn wrap_topphatt<'py>(
        py: Python<'py>,
        flatpack: Vec<(usize, (Option<usize>, Option<usize>, Option<usize>))>,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray2<bool>>)> {
        // ) -> PyResult<()> {
        debug!("Starting TOPPHATT");
        // let flatpack: TTFlatPack = node_map.extract::<TTFlatPack>()?;
        let flatpack: TTFlatPack = flatpack;
        debug!("Got flatpack");

        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            0.,
        );

        debug!("Got MSparse");
        debug!("Got Hamiltonian");
        let mut tree: TernaryTree = TernaryTree::from_flatpack_naive(flatpack)
            .expect("Ternary tree should build from flatpack");
        debug!("Got Tree");
        debug!("Hamiltonian {:?}", hamiltonian);
        tree = topphatt(hamiltonian.clone(), tree).expect("TOPPHATT should have failed by now.");

        let encoding = tree.build_encoding(n_qubits).unwrap();
        debug!("Got encoding");
        Ok((
            encoding.ipowers.into_pyarray(py),
            encoding.symplectics.into_pyarray(py),
        ))
        // let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        // debug!("Got qham");
        // Ok(qham
        //     .into_py_dict(py)
        //     .expect("Should be able to convert QubitHamiltonian to PyDict."))
        // Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "topphatt_standard")]
    fn wrap_topphatt_standard<'py>(
        py: Python<'py>,
        encoding: String,
        n_modes: usize,
        n_qubits: usize,
        signatures: Vec<String>,
        coeffs: Vec<PyReadonlyArrayDyn<f64>>,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray2<bool>>)> {
        // ) -> PyResult<Bound<'py, PyDict>> {
        assert_eq!(
            signatures.len(),
            coeffs.len(),
            "Signatures and coefficients should be same length"
        );
        assert!(
            n_qubits >= n_modes,
            "Must have at least as many qubits as modes."
        );

        debug!("Starting TOPPHATT");
        // let flatpack: TTFlatPack = node_map.extract::<TTFlatPack>()?;
        let hamiltonian = MajoranaSparse::from_signatures_and_coeffs(
            signatures,
            coeffs.iter().map(|v| v.as_array()).collect(),
            0.,
        );
        debug!("Got MSparse");
        debug!("Got Hamiltonian");
        let mut tree: TernaryTree = match encoding.as_str() {
            "Jordan-Wigner" | "JW" => TernaryTree::naive_jordan_wigner(n_modes),
            "Bravyi-Kitaev" | "BK" => TernaryTree::naive_bravyi_kitaev(n_modes),
            "Parity" | "PE" => TernaryTree::naive_parity(n_modes),
            "JKMN" => TernaryTree::naive_jkmn(n_modes),
            _ => panic!("Encoding must be one of JW, PE, BK or JKMN."),
        };
        debug!("Got Tree");
        debug!("Hamiltonian {:?}", hamiltonian);
        tree = topphatt(hamiltonian.clone(), tree).expect("TOPPHATT should have failed by now.");
        let encoding = tree.build_encoding(n_qubits).unwrap();
        debug!("Got encoding");
        Ok((
            encoding.ipowers.into_pyarray(py),
            encoding.symplectics.into_pyarray(py),
        ))
        // let qham: QubitHamiltonian = encoding.encode(&hamiltonian);
        // debug!("Got qham");
        // Ok(qham
        //     .into_py_dict(py)
        //     .expect("Should be able to convert QubitHamiltonian to PyDict."))
        // Ok(())
    }
    Ok(())
}
