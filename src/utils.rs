/*
Utility functions for core functionality.
*/
use ndarray::{concatenate, Axis, Zip};
use numpy::ndarray::{s, Array1, ArrayView1};
use numpy::Complex64;

pub fn icount_to_sign(icount: usize) -> Complex64 {
    match icount % 4 {
        0 => Complex64::new(1., 0.),
        1 => Complex64::new(0., 1.),
        2 => Complex64::new(-1., 0.),
        3 => Complex64::new(0., -1.),
        _ => unreachable!(),
    }
}

pub fn vector_kron(left: &Array1<Complex64>, right: &Array1<Complex64>) -> Array1<Complex64> {
    concatenate![
        Axis(0),
        left.mapv(|l| l * right[0]),
        left.mapv(|l| l * right[1])
    ]
}

pub fn symplectic_to_pauli(symplectic: ArrayView1<bool>) -> (usize, String) {
    let block_width = symplectic.len_of(Axis(0)) / 2;
    let mut ipower: usize = 0;
    let (x_block, z_block) = symplectic.split_at(Axis(0), block_width);
    let mut pauli_string = String::new();
    Zip::from(x_block)
        .and(z_block)
        .for_each(|&x, &z| match (&x, &z) {
            (&true, &true) => {
                pauli_string.push('Y');
                ipower += 1;
            }
            (&true, &false) => pauli_string.push('X'),
            (&false, &true) => pauli_string.push('Z'),
            (&false, &false) => pauli_string.push('I'),
        });
    ((3 * ipower) % 4, pauli_string)
}

#[test]
fn test_symplectic_to_pauli() {
    // YXZI
    let symplectic: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 1]>> =
        ndarray::arr1(&[true, true, false, false, true, false, true, false]);
    assert_eq!(
        symplectic_to_pauli(symplectic.view()),
        (3, String::from("YXZI"))
    );
}

pub fn symplectic_to_sparse(symplectic: ArrayView1<bool>) -> (u8, String, Array1<usize>) {
    let ipower: usize = 0;
    let (x_block, z_block) = symplectic.split_at(Axis(0), symplectic.len_of(Axis(0)) / 2);
    let mut pauli_string = String::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut index: usize = 0;
    Zip::from(x_block).and(z_block).for_each(|&x, &z| {
        let pauli = match (&x, &z) {
            (&false, &false) => 'I',
            (&true, &true) => 'Y',
            (&true, &false) => 'X',
            (&false, &true) => 'Z',
        };
        match pauli {
            'I' => {}
            _ => {
                pauli_string.push(pauli);
                indices.push(index);
            }
        };
        index += 1;
    });
    let ipower = ipower % 4;
    (ipower as u8, pauli_string, ndarray::arr1(&indices))
}

#[test]
fn test_symplectic_to_sparse() {
    let symplectic: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 1]>> =
        ndarray::arr1(&[true, true, false, false, true, false, true, false]);
    assert_eq!(
        symplectic_to_sparse(symplectic.view()),
        (0, "YXZ".to_string(), ndarray::arr1(&[0, 1, 2]))
    );
}

fn _valid_pauli_string(pauli: &str) -> bool {
    pauli.chars().all(|c| matches!(c, 'X' | 'Y' | 'Z' | 'I'))
}

#[test]
fn test_valid_pauli_string() {
    assert!(_valid_pauli_string("XYZI"));
    assert!(_valid_pauli_string("X"));
    assert!(_valid_pauli_string("Y"));
    assert!(_valid_pauli_string("Z"));
    assert!(_valid_pauli_string("I"));
    assert!(!_valid_pauli_string("XYZA"));
}

pub fn pauli_to_symplectic(pauli: String) -> (usize, Array1<bool>) {
    let string_len = pauli.len();
    assert!(_valid_pauli_string(&pauli));

    let mut ipower: usize = 0;
    let mut x_block: Array1<bool> = Array1::from_elem(string_len, false);
    let mut z_block: Array1<bool> = Array1::from_elem(string_len, false);
    Zip::from(&Array1::from_iter(pauli.chars()))
        .and(&mut x_block)
        .and(&mut z_block)
        .for_each(|c, x, z| match c {
            'X' => *x = true,
            'Z' => *z = true,
            'Y' => {
                *x = true;
                *z = true;
                ipower += 1;
            }
            _ => {
                *x = false;
                *z = false;
            }
        });
    (ipower % 4, concatenate![Axis(0), x_block, z_block])
}

#[test]
fn test_pauli_to_symplectic() {
    let valid_string = String::from("IXZY");
    let valid_symplectic = ndarray::arr1(&[false, true, false, true, false, false, true, true]);
    assert_eq!(pauli_to_symplectic(valid_string), (1, valid_symplectic));

    let all_y = String::from("YYY");
    assert_eq!(
        pauli_to_symplectic(all_y),
        (3, ndarray::arr1(&[true, true, true, true, true, true]))
    )
}

pub fn symplectic_product(
    left: ArrayView1<bool>,
    right: ArrayView1<bool>,
) -> (usize, Array1<bool>) {
    // bitwise or between two vectors
    let product = &left ^ &right;

    // bitwise sum of left z and right x
    let half_length: usize = left.len() / 2;

    let mut zx_count: usize = 0;
    let left_z = left.slice(s![half_length..]);
    let right_x = right.slice(s![..half_length]);
    for index in 0..half_length {
        if left_z[index] & right_x[index] {
            zx_count += 1;
        };
    }

    let ipower: usize = (2 * zx_count) % 4;

    (ipower, product)
}

#[test]
fn test_symplectic_product() {
    let xxx: Array1<bool> = ndarray::arr1(&[true, true, true, false, false, false]);
    let zzz: Array1<bool> = ndarray::arr1(&[false, false, false, true, true, true]);
    let product_result = symplectic_product(xxx.view(), zzz.view());
    let expected = (0, ndarray::arr1(&[true, true, true, true, true, true]));
    assert_eq!(product_result, expected);
}
