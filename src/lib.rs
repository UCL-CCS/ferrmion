use pyo3::{prelude::*, types::PyTuple};
use ndarray::{Array1, arr1, s};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn rust_symplectic_product(left: Array1<bool>, right:Array1<bool>) -> (usize, Array1<bool>){
    // bitwise or between two vectors
    let product = &left ^ &right;

    // bitwise sumof left z and right x
    let half_length: usize = left.len()/2;
    
    let mut zx_count: usize = 0;
    let left_z = left.slice(s![half_length..]);
    let right_x = right.slice(s![..half_length]);
    for index in 0..half_length {
        zx_count = if &left_z[index] & &right_x[index] {zx_count + 1 } else {zx_count}; 
    }

    let ipower: usize = (2*zx_count) % 4;   

    (ipower, product)
}

#[test]
fn test_symplectic_product() {
    let xxx: Array1<bool> = arr1(&[true, true, true, false, false, false]);
    let zzz: Array1<bool> = arr1(&[false, false, false, true, true, true]);
    let product_result  = rust_symplectic_product(xxx, zzz);
    let expected = (0 as usize, arr1(&[true, true, true, true, true, true]));
    assert_eq!(product_result, expected);
}


/// A Python module implemented in Rust.
#[pymodule]
fn ferrmion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
