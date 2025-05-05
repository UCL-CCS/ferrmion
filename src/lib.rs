use std::collections::HashMap;

use ndarray::{azip, concatenate, Axis};
use pyo3::{prelude::*, pymodule, Bound};
use numpy::{Complex64, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use numpy::ndarray::{Array, Array1, s, stack, arr1, arr2, ArrayView1, Array2, ArrayViewD};
use numpy::ndarray::linalg::kron;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn vector_kron(left:&Array1<Complex64>, right:&Array1<Complex64>) -> Array1<Complex64> {
    concatenate![Axis(1), left.mapv(|l| l*right[0]), left.mapv(|l| l*right[1])]
}

fn rust_hartree_fock_state(
    vaccum_state: ArrayView1<f64>,
    fermionic_hf_state: ArrayView1<bool>,
    mode_op_map: HashMap<usize, usize>,
    symplectic_matrix: ArrayViewD<bool>
) ->()// (Array1<Complex64>, ArrayD<bool>)
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

    let half_length = symplectic_matrix.len_of(ndarray::Axis(1));

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
        let left = symplectic_matrix.index_axis(ndarray::Axis(0), left_index);
        let right = symplectic_matrix.index_axis(ndarray::Axis(0), right_index);
        
        // would probably look nices as from_shape_fn
        let vec = vec![matrices.get(&(false, false)).unwrap().to_owned(); vaccum_state.len_of(Axis(0))];
        let mut operators: Array1<Array2<Complex64>> = Array::from_vec(vec);

        // can try this as a Zip like in ffsim
        for (pos,mut state) in current_state.iter().enumerate() {
            let left_op = matrices.get(&(left[pos], left[pos + half_length]))?;
            let right_op = matrices.get(&(right[pos], right[pos + half_length]))?;
            let total_op = left_op - &(Complex64::new(0.,1.) * right_op);
            // it seems ndarray does not have a good solution for complex matrix dot product?
            // https://github.com/rust-ndarray/ndarray/issues/272
            state = &arr1(&[
                &total_op[[0,0]]*&state[0]
                    +&total_op[[0,1]]*&state[1],
                &total_op[[1,0]]*&state[0]
                    +&total_op[[1,1]]*&state[1]
                ]);
        }
        
        let mut comp_basis_state: Array1<Complex64> = Array1::from_vec(vec![
            Complex64::new(vaccum_state[0],0.),
            Complex64::new(1.-vaccum_state[0],0.)]);
            
        for state in current_state.slice(s![1..]) {
            comp_basis_state = vector_kron(&comp_basis_state, &state);
        }
    };

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
    Ok(())
}
