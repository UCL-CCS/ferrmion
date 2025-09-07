/*
Functions relating to the FermionQubitEncoding base class.
*/
use anyhow::Result;
use log::debug;
use ndarray::{arr2, Axis, Zip};
use num_complex::c64;
use numpy::ndarray::{azip, s, Array1, Array2, Array3, ArrayView1, ArrayView2};
use numpy::Complex64;
use std::collections::HashMap;

use crate::utils::{symplectic_product, vector_kron};

// pub fn edge_operator(encoding:FermionQubitEncoding, edge_indices:(usize,usize)) -> Array1<(String, Array1<usize>, Complex64)> {
//     let (icount, symplectic_products) =
// }

// This caches symplectic products so that we don't have to calculate them
// four times for each pair of fermionic operators
// it does require memory scaling On^3 so if that becomes an issue we can be more clever.
pub fn symplectic_product_map(
    ipowers: ArrayView1<u8>,
    symplectics: ArrayView2<bool>,
) -> (Array2<u8>, Array3<bool>) {
    debug!("Calculating symplectic product map");

    let n_majoranas = symplectics.nrows();
    assert_eq!(n_majoranas, ipowers.len());

    let mut product_powers: Array2<u8> = Array2::zeros((n_majoranas, n_majoranas));
    let mut product_map: Array3<bool> =
        Array3::from_elem((n_majoranas, n_majoranas, symplectics.ncols()), false);
    azip!((index (l, r), pow in &mut product_powers) {
        let left = symplectics.slice(s![l,..]);
        let right = symplectics.slice(s![r,..]);
        let (imaginary, term) = symplectic_product(left, right);

        *pow += &((imaginary as u8 + ipowers[[l]] + ipowers[[r]]) % 4);
        product_map.slice_mut(s![l,r,..]).assign(&term);
    });

    // how to do a zip over 2d array ?

    debug!("Found symplectic product map.");
    (product_powers, product_map)
}

#[test]
fn test_symplectic_product_map() {
    let ipowers = ndarray::arr1(&[0, 1]);
    let symplectics = ndarray::arr2(&[[true, true, false, false], [true, false, true, false]]);
    let (iproducts, symplectic_products) =
        symplectic_product_map(ipowers.view(), symplectics.view());
    println!("{}", iproducts);
    println!("{}", symplectic_products);
    assert_eq!(iproducts, ndarray::arr2(&[[0, 1], [3, 0]]));
    assert_eq!(
        symplectic_products.view(),
        ndarray::arr3(&[
            [[false, false, false, false], [false, true, true, false]],
            [[false, true, true, false], [false, false, false, false]]
        ])
        .view()
    );
}

// super ugly function, should definitely work on writing nice rust
pub fn hartree_fock_state(
    vacuum_state: ArrayView1<f64>,
    fermionic_hf_state: ArrayView1<bool>,
    mode_op_map: ArrayView1<usize>,
    symplectic_matrix: ArrayView2<bool>,
) -> Result<(Array1<Complex64>, Array2<bool>)> {
    debug!("Calculating Hartree-fock state");

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
        let mode_index = mode_op_map[[mode]];

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

    let hf_components: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 2]>> =
        Array2::from_shape_vec((coeffs.len(), vacuum_state.len()), hf_components)?;
    debug!(
        "Found Hartree-Fock state: coeffs={:?}, hf_components={:#?}",
        coeffs, hf_components
    );
    Ok((coeffs, hf_components))
}

#[test]
fn test_hartree_fock() {
    let vacuum_state: ArrayView1<f64> = ArrayView1::from(&[0., 0., 0., 0., 0., 0.]);
    let fermionic_hf_state: ArrayView1<bool> =
        ArrayView1::from(&[true, true, true, false, false, false]);
    let mode_op_map: ArrayView1<usize> = ArrayView1::from(&[0, 1, 2, 3, 4, 5, 6]);
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
        mode_op_map,
        symplectic_matrix,
    )
    .unwrap();
    let c1 = c64(1., 0.);
    assert!(result.0 == ndarray::arr1(&[c1]));
    assert!(result.1 == arr2(&[[true, true, true, false, false, false]]));

    let result2 = hartree_fock_state(
        vacuum_state,
        ArrayView1::from(&[true, true, true, true, false, false]),
        mode_op_map.clone(),
        symplectic_matrix,
    )
    .unwrap();
    assert!(result2.0 == ndarray::arr1(&[c1]));
    assert!(result2.1 == arr2(&[[true, true, true, true, false, false]]));
}
