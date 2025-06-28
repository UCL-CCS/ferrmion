use ndarray::{Axis, Zip};
// use ndarray::{azip, concatenate, Axis, Zip};
use itertools::iproduct;
use numpy::ndarray::{s, ArrayView1, ArrayView2, ArrayView4};
use numpy::Complex64;
use pyo3::{FromPyObject, IntoPyObject};
use std::collections::HashMap;

use crate::utils::{
    icount_to_sign, symplectic_product, symplectic_product_map, symplectic_to_pauli,
};

fn y_count(symplectic: ArrayView1<bool>) -> usize {
    let (x_part, z_part) = symplectic.split_at(Axis(0), symplectic.len_of(Axis(0)) / 2);
    Zip::from(x_part)
        .and(z_part)
        .fold(0, |acc, x, z| acc + (x & z) as usize)
}

#[derive(Eq, PartialEq, Hash, IntoPyObject, FromPyObject, Debug)]
pub enum IntegralIndex {
    OneE(usize, usize),
    TwoE(usize, usize, usize, usize),
}

pub fn molecular(
    ipowers: ArrayView1<u8>,
    symplectics: ArrayView2<bool>,
) -> HashMap<String, HashMap<IntegralIndex, Complex64>> {
    assert_eq!(ipowers.len(), symplectics.nrows());

    let (iproducts, sym_products) = symplectic_product_map(ipowers, symplectics);

    let mut hamiltonian: HashMap<String, HashMap<IntegralIndex, Complex64>> = HashMap::new();
    // assume 8-fold symmetry
    let n_modes = symplectics.nrows() / 2;
    for m in 0..n_modes {
        for n in 0..n_modes {
            for (l, r) in iproduct!(0..2, 0..2) {
                let term = sym_products.slice(s![2 * m + l, 2 * n + r, ..]);
                let (im_term_pauli, pauli_string) = symplectic_to_pauli(term);
                let weight = Complex64::new(0.25, 0.)
                    * icount_to_sign(
                        iproducts[[2 * m + l, 2 * n + r]] as usize
                            + im_term_pauli
                            + (r + 3 * l)
                            + 3 * y_count(term),
                    );
                let components = hamiltonian.entry(pauli_string).or_default();
                components
                    .entry(IntegralIndex::OneE(m, n))
                    .and_modify(|e| *e += weight)
                    .or_insert(weight);
            }
            // 2e terms cancel
            if m == n {
                continue;
            }
            for p in 0..n_modes {
                for q in 0..n_modes {
                    if (p == q) {
                        //| (m == p && n == q) | (m == q && n == p) {
                        continue;
                    }
                    for (l1, l2, r1, r2) in iproduct!(0..2, 0..2, 0..2, 0..2) {
                        let left = sym_products.slice(s![2 * m + l1, 2 * n + l2, ..]);
                        let right = sym_products.slice(s![2 * p + r1, 2 * q + r2, ..]);
                        let (iproduct, product_term) = symplectic_product(left, right);
                        let (im_term_pauli, pauli_string) =
                            symplectic_to_pauli(product_term.view());
                        let weight = Complex64::new(0.0625, 0.)
                            * icount_to_sign(
                                iproduct
                                    + im_term_pauli
                                    + 3 * (l1 + l2)
                                    + (r1 + r2)
                                    + iproducts[[2 * m + l1, 2 * n + l2]] as usize
                                    + iproducts[[2 * p + r1, 2 * q + r2]] as usize
                                    + 3 * y_count(product_term.view()),
                            );

                        let components = hamiltonian.entry(pauli_string).or_default();
                        components
                            .entry(IntegralIndex::TwoE(m, n, p, q))
                            .and_modify(|e| *e += weight)
                            .or_insert(weight);
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

    expected.insert(String::from("YX"), {
        let mut value = HashMap::new();
        value.insert(String::from("0,1,0,0"), Complex64::new(0., 0.0625));
        value.insert(String::from("1,0"), Complex64::new(0., -0.25));
        value.insert(String::from("1,0,0,0"), Complex64::new(0., -0.0625));
        value.insert(String::from("0,1"), Complex64::new(0., 0.25));
        value.insert(String::from("0,1,1,1"), Complex64::new(0., 0.0625));
        value.insert(String::from("1,0,1,1"), Complex64::new(0., -0.0625));
        value
    });
    expected.insert(String::from("II"), {
        let mut value = HashMap::new();
        value.insert(String::from("0,0"), Complex64::new(0.25, 0.));
        value.insert(String::from("1,1"), Complex64::new(0.25, 0.));
        value
    });
    println!("{:#?}", ham);
    assert!(ham.keys().all(|k| expected.contains_key(k)));
}

#[allow(dead_code)]
pub fn fill_template(
    template: HashMap<String, HashMap<IntegralIndex, Complex64>>,
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
        let mut val = Complex64::new(0., 0.);
        let err_str = "Mode op map does not contain integral index.";
        for (indices, factor) in components {
            let coeff = match indices {
                IntegralIndex::OneE(m, n) => {
                    one_e_terms[[
                        *mode_op_map.get(&m).expect(err_str),
                        *mode_op_map.get(&n).expect(err_str),
                    ]]
                }
                IntegralIndex::TwoE(p, q, r, s) => {
                    two_e_terms[[
                        *mode_op_map.get(&p).expect(err_str),
                        *mode_op_map.get(&q).expect(err_str),
                        *mode_op_map.get(&r).expect(err_str),
                        *mode_op_map.get(&s).expect(err_str),
                    ]]
                }
            };
            val += factor * Complex64::new(coeff, 0.);
        }
        if val.norm() > 1e-12 {
            hamiltonian.insert(pauli_term, val);
        };
    }

    hamiltonian
}
