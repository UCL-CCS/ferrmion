use ndarray::{azip, concatenate, Axis, Zip};
use num_complex::c64;
use numpy::ndarray::{arr2, s, Array1, Array2, Array3, ArrayView1, ArrayView2};
use numpy::Complex64;
use std::collections::HashMap;
use std::vec;

fn vector_kron(left: &Array1<Complex64>, right: &Array1<Complex64>) -> Array1<Complex64> {
    concatenate![
        Axis(0),
        left.mapv(|l| l * right[0]),
        left.mapv(|l| l * right[1])
    ]
}

// super ugly function, should definitely work on writing nice rust
pub fn hartree_fock_state(
    vacuum_state: ArrayView1<f64>,
    fermionic_hf_state: ArrayView1<bool>,
    mode_op_map: HashMap<usize, usize>,
    symplectic_matrix: ArrayView2<bool>,
) -> (Array1<Complex64>, Array2<bool>) {
    let mut current_state =
        vec![Array1::from(vec![c64(1., 0.), c64(0., 0.)]); vacuum_state.len_of(Axis(0))];

    let mut matrices = HashMap::new();
    matrices.insert(
        (false, false),
        arr2(&[[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(1., 0.)]]),
    );
    matrices.insert(
        (true, false),
        arr2(&[[c64(0., 0.), c64(1., 0.)], [c64(1., 0.), c64(0., 0.)]]),
    );
    matrices.insert(
        (false, true),
        arr2(&[[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(1., 0.)]]),
    );
    matrices.insert(
        (true, true),
        arr2(&[[c64(0., 0.), c64(0., -1.)], [c64(0., 1.), c64(0., 0.)]]),
    );

    let half_length = symplectic_matrix.len_of(ndarray::Axis(1)) / 2;
    let (x_block, z_block) = symplectic_matrix.split_at(Axis(1), half_length);

    for (mode, occ) in fermionic_hf_state.into_iter().enumerate() {
        if !occ {
            continue;
        }
        let mode_index = mode_op_map.get(&mode).unwrap();

        let left_x = x_block.index_axis(ndarray::Axis(0), 2 * mode_index);
        let right_x = x_block.index_axis(ndarray::Axis(0), 2 * mode_index + 1);
        let left_z = z_block.index_axis(ndarray::Axis(0), 2 * mode_index);
        let right_z = z_block.index_axis(ndarray::Axis(0), 2 * mode_index + 1);

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
                let total_op = left_op - right_op.map(|op| op * c64(0., 1.));
                *s = total_op.dot(s);
            });
    }

    let mut vector_state: Array1<Complex64> = Zip::from(&current_state)
        .fold(Array1::from_elem(1, c64(1., 0.)), |acc, c| {
            vector_kron(&acc, c)
        });

    let mut zero_coeffs = Vec::new();
    let mut hf_components: Vec<bool> = Vec::new();
    // According to ndarray docs, when we don't know the final size
    // of a multidimensional array we want to build iteratively
    // the best thing to do is create a flat array and then reshape
    for index in 0..vector_state.len() {
        let coeff = vector_state[index];
        if !(coeff == c64(0., 0.)) {
            let binary = format!("{:0<width$}", format!("{index:b}"), width = (half_length));
            for val in binary.chars() {
                println!("{}", val);
                hf_components.push(val.to_digit(10).unwrap() == 1)
            }
        } else {
            zero_coeffs.push(index);
        }
    }
    for index in zero_coeffs.iter().rev() {
        vector_state.remove_index(Axis(0), *index);
    }

    let coeffs = vector_state.mapv(|c| c / (vector_state[0]));

    let hf_components =
        Array2::from_shape_vec((coeffs.len(), vacuum_state.len()), hf_components).unwrap();
    (coeffs, hf_components)
}

#[test]
fn test_hartree_fock() {
    let vacuum_state: ArrayView1<f64> = ArrayView1::from(&[0., 0., 0., 0., 0., 0.]);
    let fermionic_hf_state: ArrayView1<bool> =
        ArrayView1::from(&[true, true, true, false, false, false]);
    let mut mode_op_map: HashMap<usize, usize> = HashMap::new();
    mode_op_map.insert(0, 0);
    mode_op_map.insert(1, 1);
    mode_op_map.insert(2, 2);
    mode_op_map.insert(3, 3);
    mode_op_map.insert(4, 4);
    mode_op_map.insert(5, 5);
    mode_op_map.insert(6, 6);
    let symplectic_matrix: ArrayView2<bool> = ArrayView2::from(&[
        [
            true, false, false, false, false, false, false, false, false, false, false, false,
        ],
        [
            true, false, false, false, false, false, true, false, false, false, false, false,
        ],
        [
            false, true, false, false, false, false, true, false, false, false, false, false,
        ],
        [
            false, true, false, false, false, false, true, true, false, false, false, false,
        ],
        [
            false, false, true, false, false, false, true, true, false, false, false, false,
        ],
        [
            false, false, true, false, false, false, true, true, true, false, false, false,
        ],
        [
            false, false, false, true, false, false, true, true, true, false, false, false,
        ],
        [
            false, false, false, true, false, false, true, true, true, true, false, false,
        ],
        [
            false, false, false, false, true, false, true, true, true, true, false, false,
        ],
        [
            false, false, false, false, true, false, true, true, true, true, true, false,
        ],
        [
            false, false, false, false, false, true, true, true, true, true, true, false,
        ],
        [
            false, false, false, false, false, true, true, true, true, true, true, true,
        ],
    ]);
    let result = hartree_fock_state(
        vacuum_state,
        fermionic_hf_state,
        mode_op_map.clone(),
        symplectic_matrix,
    );
    let c1 = c64(1., 0.);
    assert!(result.0 == ndarray::arr1(&[c1]));
    assert!(result.1 == arr2(&[[true, true, true, false, false, false]]));

    let result2 = hartree_fock_state(
        vacuum_state,
        ArrayView1::from(&[true, true, true, true, false, false]),
        mode_op_map.clone(),
        symplectic_matrix,
    );
    assert!(result2.0 == ndarray::arr1(&[c1]));
    assert!(result2.1 == arr2(&[[true, true, true, true, false, false]]));
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
    let symplectic = ndarray::arr1(&[true, true, false, false, true, false, true, false]);
    assert_eq!(
        symplectic_to_pauli(symplectic.view()),
        (3, String::from("YXZI"))
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

// This caches symplectic products so that we don't have to calculate them
// four times for each pair of fermionic operators
// it does require memory scaling On^3 so if that becomes an issue we can be more clever.
pub fn symplectic_product_map(
    ipowers: ArrayView1<usize>,
    symplectics: ArrayView2<bool>,
) -> (Array2<usize>, Array3<bool>) {
    let n_majoranas = symplectics.nrows();
    assert_eq!(n_majoranas, ipowers.len());

    let mut product_powers: Array2<usize> = Array2::zeros((n_majoranas, n_majoranas));
    let mut product_map: Array3<bool> =
        Array3::from_elem((n_majoranas, n_majoranas, n_majoranas), false);
    azip!((index (l, r), pow in &mut product_powers) {
        let left = symplectics.slice(s![l,..]);
        let right = symplectics.slice(s![r,..]);
        let (imaginary, term) = symplectic_product(left, right);

        *pow += &((imaginary + ipowers[[l]] + ipowers[[r]]) % 4);
        product_map.slice_mut(s![l,r,..]).assign(&term);
    });

    // how to do a zip over 2d array ?

    (product_powers, product_map)
}

#[test]
fn test_symplectic_product_map() {
    let ipowers = ndarray::arr1(&[0, 1]);
    let symplectics = ndarray::arr2(&[[true, true, false, false], [true, false, true, false]]);
    let (iproducts, symplectic_products) =
        symplectic_product_map(ipowers.view(), symplectics.view());
    println!("{}", iproducts);
    // assert_eq!(iproducts, ndarray::arr2(&[[0,0],[0,0]]));
}
