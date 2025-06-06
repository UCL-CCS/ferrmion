use ndarray::Axis;
// use ndarray::{azip, concatenate, Axis, Zip};
// use num_complex::{c64, Complex};
use itertools::izip;
use numpy::ndarray::{s, ArrayView1, ArrayView2, ArrayView4};
use numpy::Complex64;
use std::collections::HashMap;

use crate::utils::{
    icount_to_sign, symplectic_product, symplectic_product_map, symplectic_to_pauli,
};

pub fn molecular(
    ipowers: ArrayView1<usize>,
    symplectics: ArrayView2<bool>,
) -> HashMap<String, HashMap<String, Complex64>> {
    assert_eq!(ipowers.len(), symplectics.nrows());

    let (iproducts, sym_products) = symplectic_product_map(ipowers, symplectics);

    let mut hamiltonian: HashMap<String, HashMap<String, Complex64>> = HashMap::new();
    // assume 8-fold symmetry
    let n_modes = symplectics.nrows() / 2;
    for m in 0..n_modes {
        for n in 0..n_modes {
            for (l, r) in std::iter::zip(0..1, 0..1) {
                let term = sym_products.slice(s![2 * m + l, 2 * n + r, ..]);
                let (im_term_pauli, pauli_string) = symplectic_to_pauli(term);
                let weight = 0.25
                    * icount_to_sign(
                        iproducts[[2 * m + l, 2 * n + r]] + im_term_pauli + ((l - r) % 4),
                    );
                let components = hamiltonian.entry(pauli_string).or_default();
                components.insert(format!("{},{}", m, n), weight);
            }
            // 2e terms cancel
            if m == n {
                continue;
            }
            for p in 0..n_modes {
                for q in 0..n_modes {
                    if p == q {
                        for (l1, l2, r1, r2) in izip!(0..1, 0..1, 0..1, 0..1) {
                            let left = sym_products.slice(s![2 * m + l1, 2 * n + l2, ..]);
                            let right = sym_products.slice(s![2 * p + r1, 2 * q + r2, ..]);
                            let product = symplectic_product(left, right);
                            let (im_term_pauli, pauli_string) =
                                symplectic_to_pauli(product.1.view());
                            let weight = 0.0625
                                * icount_to_sign(
                                    product.0
                                        + im_term_pauli
                                        + 3 * (l1 + l2)
                                        + (r1 + r2)
                                        + iproducts[[2 * m + l1, 2 * n + l2]]
                                        + iproducts[[2 * p + r1, 2 * q + r2]],
                                );

                            let components = hamiltonian.entry(pauli_string).or_default();
                            components.insert(format!("{},{},{},{}", m, n, p, q), weight);
                        }
                    }
                }
            }
        }
    }
    hamiltonian
}

#[test]
fn test_molecular() {
    let ipowers = ndarray::arr1(&[0, 1, 2, 3]);
    let symplectics = ndarray::arr2(&[
        [true, false, false, false],
        [true, false, true, false],
        [false, true, true, false],
        [false, true, true, true],
    ]);
    let ham = molecular(ipowers.view(), symplectics.view());
    let mut expected: HashMap<String, HashMap<String, Complex64>> = HashMap::new();
    expected.insert(String::from("IIII"), {
        let mut value = HashMap::new();
        value.insert(String::from("01"), Complex64::new(1., 0.));
        value
    });
    println!("{:#?}", ham);
    assert!(ham.keys().all(|k| expected.contains_key(k)));
}

#[allow(dead_code)]
pub fn fill_template(
    template: HashMap<String, HashMap<String, Complex64>>,
    mode_op_map: HashMap<usize, usize>,
    one_e_terms: ArrayView2<f64>,
    two_e_terms: ArrayView4<f64>,
) -> HashMap<String, Complex64> {
    assert!(one_e_terms
        .shape()
        .iter()
        .all(|&s| s == two_e_terms.len_of(Axis(0))));
    assert!(two_e_terms
        .shape()
        .iter()
        .all(|&s| s == one_e_terms.len_of(Axis(0))));
    assert!((0..one_e_terms.len_of(Axis(0))).all(|v| { mode_op_map.contains_key(&v) }));
    assert!(mode_op_map
        .values()
        .all(|v| { mode_op_map.contains_key(v) }));
    // assert_eq!(HashSet::from(mode_op_map.keys()), HashSet::from(0..one_e_terms.len_of(Axis(0))));
    // assert_eq!(HashSet::from(mode_op_map.values()), (HashSet::from(0..one_e_terms.len_of(Axis(0)))));
    let mut hamiltonian: HashMap<String, Complex64> = HashMap::new();

    for (pauli_term, components) in template {
        hamiltonian.insert(pauli_term, {
            let mut val = Complex64::new(0., 0.);
            for (indices, factor) in components {
                let ni: Vec<usize> = indices
                    .split(' ')
                    .map(|c| {
                        let parsed_index = c.parse::<usize>().unwrap();
                        let mapped_index: usize = *mode_op_map.get(&parsed_index).unwrap();
                        mapped_index
                    })
                    .collect();
                let coeff = match ni.len() {
                    2 => one_e_terms[[ni[0], ni[1]]],
                    4 => two_e_terms[[ni[0], ni[1], ni[2], ni[3]]],
                    _ => panic!(),
                };
                val += factor * Complex64::new(coeff, 0.);
            }
            val
        });
    }

    hamiltonian
}
