use ndarray::OwnedRepr;
use pyo3::{prelude::*, pymodule, Bound};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use numpy::ndarray::{Array1, s, arr1, ArrayView1};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
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
