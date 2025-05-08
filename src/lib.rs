use std::collections::HashMap;
use std::os::macos::raw::stat;

use ndarray::{concatenate, Axis, Zip};
use pyo3::conversion::FromPyObjectBound;
use pyo3::types::{PyDict, PyComplex, PyInt};
use pyo3::{prelude::*, pymodule, Bound};
// use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, Complex64};
use numpy::ndarray::{Array, Array1, s, arr1, arr2, ArrayView1,ArrayView2, Array2};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn vector_kron(left:&Array1<Complex64>, right:&Array1<Complex64>) -> Array1<Complex64> {
    concatenate![Axis(0), left.mapv(|l| l*right[0]), left.mapv(|l| l*right[1])]
}

// super ugly function, should definitely work on writing nice rust
fn rust_hartree_fock_state(
    vaccum_state: ArrayView1<f64>,
    fermionic_hf_state: ArrayView1<bool>,
    mode_op_map: HashMap<usize, usize>,
    symplectic_matrix: ArrayView2<bool>
) -> (Array1<Complex64>, Array2<bool>)
 {
    let mut current_state = vec![Array1::from(
        vec![Complex64::new(1.,0.), Complex64::new(0.,0.)]); vaccum_state.len_of(Axis(0))];

    let mut matrices: HashMap<(bool, bool), Array2<Complex64>> = HashMap::new();
    matrices.insert((false,false),arr2(
        &[[Complex64::new(1.,0.),Complex64::new(0.,0.)],
            [Complex64::new(0.,0.),Complex64::new(1.,0.)]]
        ));
    matrices.insert((true, false),arr2(
        &[[Complex64::new(0.,0.),Complex64::new(1.,0.)],
            [Complex64::new(1.,0.),Complex64::new(0.,0.)]]
        ));
    matrices.insert((false, true),arr2(
        &[[Complex64::new(1.,0.),Complex64::new(0.,0.)],
            [Complex64::new(0.,0.),Complex64::new(1.,0.)]]
        ));
    matrices.insert((true, true),arr2(
        &[[Complex64::new(0.,0.),Complex64::new(0.,-1.)],
            [Complex64::new(0.,1.),Complex64::new(0.,0.)]]
        ));

    let half_length = symplectic_matrix.len_of(ndarray::Axis(1))/2;

    for (mode, occ) in fermionic_hf_state.into_iter().enumerate() {
        if !occ {continue;}
        let left_index = match mode_op_map.get(&(2* mode)) {
            Some(val)=> val.to_owned(),
            None => {
                eprintln!("Mode op map points to invalid operator number");
                break;
            },
        };
        let right_index = match mode_op_map.get(&(2* mode +1)) {
            Some(val)=> val.to_owned(),
            None => {
                eprintln!("Mode op map points to invalid operator number");
                break;
            },
        };

        let (left_x, left_z) = symplectic_matrix
            .index_axis(ndarray::Axis(0), left_index)
            .split_at(Axis(0), half_length);
        let (right_x, right_z) = symplectic_matrix
            .index_axis(ndarray::Axis(0), right_index)
            .split_at(Axis(0), half_length);
        
        // split the left and righ operators into x and z sections
        Zip::from(&mut current_state)
            .and(&left_x)
            .and(&left_z)
            .and(&right_x)
            .and(&right_z)
            .for_each(|s, &lx, &lz, &rx, &rz| {
                // Create an operator to act on the state with
                let left_op = matrices.get(&(lx, lz)).unwrap();
                let right_op = matrices.get(&(rx, rz)).unwrap();
                let total_op = left_op - right_op.map(|op| op * Complex64::new(0.,1.));
                *s = arr1(&[
                    &total_op[[0,0]]*&s[0]
                        +&total_op[[0,1]]*&s[1],
                    &total_op[[1,0]]*&s[0]
                        +&total_op[[1,1]]*&s[1]
                    ]);
            });
    };

    // this is going to break if the vaccum state isnt constant between qubits
    let mut vector_state: Array1<Complex64> = Array1::from_vec(vec![
        Complex64::new(1.-vaccum_state[0],0.),
        Complex64::new(vaccum_state[0],0.)]);
    for state in &current_state {
        vector_state = vector_kron(&vector_state, &state);
    }
    let norm = vector_state.mapv(|s| s*s.conj()).sum().sqrt();
    let mut coeffs = vector_state.mapv(|s| s/ norm);

    let mut zero_coeffs = Vec::new(); 
    let mut hf_components: Vec<bool> = Vec::new();
    // convert vector state to computational basis state
    for index in 0..coeffs.len() {
        let coeff = coeffs[index];
        if !(coeff == Complex64::new(0.,0.)) {
            let binary = format!(
                "{:0<width$}", 
                format!("{index:b}"), 
                width=(half_length)
            );
            for val in binary.chars() {
                println!("{}",val);
                hf_components.push(val.to_digit(10).unwrap() == 1)
                };
        } else {
            zero_coeffs.push(index);
        }
    };
    for index in zero_coeffs.iter().rev() {
        coeffs.remove_index(Axis(0), *index);
    }
    coeffs = coeffs.mapv(|c| c/coeffs[0]);
    let hf_components = Array2::from_shape_vec((coeffs.len(),vaccum_state.len()), hf_components).unwrap();
    (coeffs, hf_components)
}

#[test]
fn test_hartree_fock() {
    let vaccum_state: ArrayView1<f64> = ArrayView1::from(&[0.,0.,0.,0.,0.,0.]);
    let fermionic_hf_state: ArrayView1<bool> = ArrayView1::from(&[true, true, true, false, false, false]);
    let mut mode_op_map: HashMap<usize, usize> = HashMap::new();
    mode_op_map.insert(0, 0);
    mode_op_map.insert(1, 1);
    mode_op_map.insert(2, 2);
    mode_op_map.insert(3, 3);
    mode_op_map.insert(4, 4);
    mode_op_map.insert(5, 5);
    mode_op_map.insert(6, 6);
    let symplectic_matrix: ArrayView2<bool> = ArrayView2::from(
        &[[true , false, false, false, false, false, false, false, false, false, false, false],
            [true , false, false, false, false, false, true , false, false, false, false, false],
            [false, true , false, false, false, false, true , false, false, false, false, false],
            [false, true , false, false, false, false, true , true , false, false, false, false],
            [false, false, true , false, false, false, true , true , false, false, false, false],
            [false, false, true , false, false, false, true , true , true , false, false, false],
            [false, false, false, true , false, false, true , true , true , false, false, false],
            [false, false, false, true , false, false, true , true , true , true , false, false],
            [false, false, false, false, true , false, true , true , true , true , false, false],
            [false, false, false, false, true , false, true , true , true , true , true , false],
            [false, false, false, false, false, true , true , true , true , true , true , false],
            [false, false, false, false, false, true , true , true , true , true , true , true ]]
    );
    let result = rust_hartree_fock_state(vaccum_state, fermionic_hf_state, mode_op_map, symplectic_matrix);
    let c1 = Complex64::new(1.,0.);
    println!("{:?}", result.0);
    println!("{:?}", result.1);
    assert!(result.0 == arr1(&[c1]));
    assert!(result.1 == arr2(&[[true, true, true, false, false, false]]));
}

fn rust_symplectic_product(left: ArrayView1<bool>, right:ArrayView1<bool>) -> (usize, Array1<bool>) {
    // bitwise or between two vectors
    let product = &left ^ &right;

    // bitwise sum of left z and right x
    let half_length: usize = left.len()/2;
    
    let mut zx_count: usize = 0;
    let left_z = left.slice(s![half_length..]);
    let right_x = right.slice(s![..half_length]);
    for index in 0..half_length {
        if &left_z[index] & &right_x[index] {
            zx_count += 1; 
        }; 
    }

    let ipower: usize = (2*zx_count) % 4;   

    (ipower, product)
}

#[test]
fn test_symplectic_product() {
    let xxx: Array1<bool> = arr1(&[true, true, true, false, false, false]);
    let zzz: Array1<bool> = arr1(&[false, false, false, true, true, true]);
    let product_result  = rust_symplectic_product(xxx.view(), zzz.view());
    let expected = (0 as usize, arr1(&[true, true, true, true, true, true]));
    assert_eq!(product_result, expected);
}


/// A Python module implemented in Rust.
#[pymodule]
fn ferrmion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    #[pyfn(m)]
    #[pyo3(name="rust_symplectic_product")]
    fn rust_symplectic_product_py<'py>(
        py: Python<'py>,
        left:PyReadonlyArray1<bool>,
        right:PyReadonlyArray1<bool>
    ) -> (usize, Bound<'py, PyArray1<bool>>) {
        let left = left.as_array();
        let right = right.as_array();
        let (ipower , product) = rust_symplectic_product(left, right);
        let pyproduct = PyArray1::from_owned_array(py, product);
        (ipower, pyproduct)
    }

    #[pyfn(m)]
    #[pyo3(name="rust_hartree_fock_state")]
    fn rust_hartree_fock_state_py<'py>(
        py: Python<'py>,
        vaccum_state: PyReadonlyArray1<f64>,
        fermionic_hf_state: PyReadonlyArray1<bool>,
        mode_op_map: Bound<'py, PyDict>,
        symplectic_matrix: PyReadonlyArray2<bool>
    ) -> (Bound<'py, PyArray1<Complex64>>, Bound<'py, PyArray2<bool>>) {
        let vaccum_state = vaccum_state.as_array();
        let fermionic_hf_state = fermionic_hf_state.as_array();
        let rust_mode_op_map: HashMap<usize, usize> = mode_op_map.extract().unwrap();
        let symplectic_matrix = symplectic_matrix.as_array();
        let (coeffs, states) = rust_hartree_fock_state(vaccum_state, fermionic_hf_state, rust_mode_op_map, symplectic_matrix);
        
        (PyArray1::from_owned_array(py, coeffs), PyArray2::from_owned_array(py, states))
    }
    Ok(())
}
