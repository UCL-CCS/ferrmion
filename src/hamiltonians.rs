use log::debug;
use ndarray::Axis;
// use ndarray::{azip, concatenate, Axis, Zip};
use ahash::RandomState;
use itertools::iproduct;
use numpy::ndarray::{s, ArrayView1, ArrayView2, ArrayView4};
use numpy::Complex64;
use pyo3::{FromPyObject, IntoPyObject};
use std::collections::HashMap;

use crate::encoding::symplectic_product_map;
use crate::utils::{icount_to_sign, symplectic_product, symplectic_to_pauli};

#[derive(Eq, PartialEq, Hash, IntoPyObject, FromPyObject, Debug)]
pub enum IntegralIndex {
    //TwoE terms are more common, and pyo3 tries from top to bottom
    //So putting them first in the Enum
    TwoE(usize, usize, usize, usize),
    OneE(usize, usize),
}

pub type QubitHamiltonianTemplate =
    HashMap<String, HashMap<IntegralIndex, Complex64, RandomState>, RandomState>;

pub type QubitHamiltonian<'template> = HashMap<&'template String, Complex64, RandomState>;

pub enum Notation {
    Physicist,
    Chemist,
}

pub fn molecular(
    ipowers: ArrayView1<u8>,
    symplectics: ArrayView2<bool>,
    notation: Notation,
) -> QubitHamiltonianTemplate {
    debug!(
        "Creating molecular hamiltonian template with\n ipowers={:?}, symplectics shape={:?}",
        ipowers,
        symplectics.shape()
    );

    assert_eq!(ipowers.len(), symplectics.nrows());

    let (iproducts, sym_products) = symplectic_product_map(ipowers, symplectics);

    let mut hamiltonian: QubitHamiltonianTemplate =
        QubitHamiltonianTemplate::with_hasher(RandomState::new());
    // assume 8-fold symmetry
    let n_modes = symplectics.nrows() / 2;
    hamiltonian.insert(
        "I".repeat(n_modes).to_string(),
        HashMap::with_hasher(RandomState::new()),
    );
    for m in 0..n_modes {
        for n in 0..n_modes {
            // ipowers can be updated to account for +/- operators
            for (l, r) in iproduct!(0..2, 0..2) {
                let term = sym_products.slice(s![2 * m + l, 2 * n + r, ..]);
                let (im_term_pauli, pauli_string) = symplectic_to_pauli(term);
                let weight = Complex64::new(0.25, 0.)
                    * icount_to_sign(
                        iproducts[[2 * m + l, 2 * n + r]] as usize + im_term_pauli + (r + 3 * l),
                    );
                let components = hamiltonian.entry(pauli_string).or_default();
                components
                    .entry(IntegralIndex::OneE(m, n))
                    .and_modify(|e| *e += weight)
                    .or_insert(weight);
            }
            //if m == n {
            // continue;
            //}
            for p in 0..n_modes {
                for q in 0..n_modes {
                    for (l1, l2, r1, r2) in iproduct!(0..2, 0..2, 0..2, 0..2) {
                        let left = sym_products.slice(s![2 * m + l1, 2 * n + l2, ..]);
                        let right = sym_products.slice(s![2 * p + r1, 2 * q + r2, ..]);
                        let (iproduct, product_term) = symplectic_product(left, right);
                        let (im_term_pauli, pauli_string) =
                            symplectic_to_pauli(product_term.view());
                        let term_ipowers = match notation {
                            Notation::Physicist => 3 * (l1 + l2) + r1 + r2,
                            Notation::Chemist => 3 * (l1 + r1) + l2 + r2,
                        };
                        let weight = Complex64::new(0.0625, 0.)
                            * icount_to_sign(
                                iproduct
                                    + im_term_pauli
                                    + term_ipowers
                                    + iproducts[[2 * m + l1, 2 * n + l2]] as usize
                                    + iproducts[[2 * p + r1, 2 * q + r2]] as usize,
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
    debug!("Molecular Hamiltonian template created.");
    hamiltonian
}

pub fn hubbard(ipowers: ArrayView1<u8>, symplectics: ArrayView2<bool>) -> QubitHamiltonianTemplate {
    debug!(
        "Creating molecular hamiltonian template with\n ipowers={:?}, symplectics shape={:?}",
        ipowers,
        symplectics.shape()
    );

    assert_eq!(ipowers.len(), symplectics.nrows());

    let (iproducts, sym_products) = symplectic_product_map(ipowers, symplectics);

    let s = RandomState::new();
    let mut hamiltonian: QubitHamiltonianTemplate = QubitHamiltonianTemplate::with_hasher(s);
    // assume 8-fold symmetry
    let n_modes = symplectics.nrows() / 2;
    hamiltonian.insert(
        "I".repeat(n_modes).to_string(),
        HashMap::with_hasher(RandomState::new()),
    );
    for m in 0..n_modes {
        for n in 0..n_modes {
            // ipowers can be updated to account for +/- operators
            for (l, r) in iproduct!(0..2, 0..2) {
                let term = sym_products.slice(s![2 * m + l, 2 * n + r, ..]);
                let (im_term_pauli, pauli_string) = symplectic_to_pauli(term);
                let weight = Complex64::new(0.25, 0.)
                    * icount_to_sign(
                        iproducts[[2 * m + l, 2 * n + r]] as usize + im_term_pauli + (r + 3 * l),
                    );
                let components = hamiltonian.entry(pauli_string).or_default();
                components
                    .entry(IntegralIndex::OneE(m, n))
                    .and_modify(|e| *e += weight)
                    .or_insert(weight);
            }
            if m == n {
                let p = m;
                let q = m;
                for (l1, l2, r1, r2) in iproduct!(0..2, 0..2, 0..2, 0..2) {
                    let left = sym_products.slice(s![2 * m + l1, 2 * n + l2, ..]);
                    let right = sym_products.slice(s![2 * p + r1, 2 * q + r2, ..]);
                    let (iproduct, product_term) = symplectic_product(left, right);
                    let (im_term_pauli, pauli_string) = symplectic_to_pauli(product_term.view());
                    let term_ipowers = 3 * (l1 + r1) + l2 + r2;
                    let weight = Complex64::new(0.0625, 0.)
                        * icount_to_sign(
                            iproduct
                                + im_term_pauli
                                + term_ipowers
                                + iproducts[[2 * m + l1, 2 * n + l2]] as usize
                                + iproducts[[2 * p + r1, 2 * q + r2]] as usize,
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
    debug!("Hubbard Hamiltonian template created.");
    hamiltonian
}

// pub fn add_one_e_term_to_template(term_signature: &str, ipowers: ArrayView<u8>, symplectics: ArrayView<bool>, hamiltonian_template: &mut QubitHamiltonianTemplate) {
//     assert!(term_signature.chars().all(|c| matches!(c, '+'|'-')));
//     assert!(term_signature.len() == 2);
//     assert!(symplectics.ndim() +1 == term_signature.len());
//     let (mut i_products, sym_products) = symplectic_product_map(ipowers, symplectics);
//     if term_signature[0] == "+" {
//         i_products.slice_mut(s![..;2,..]) + 2;
//     }
//     if term_signature[1] == "+" {
//         i_products.slice_mut(s![..,..;2]) + 2;
//     }
// }

// #[allow(dead_code)]
// pub fn molecular_iter(
//     ipowers: ArrayView1<u8>,
//     symplectics: ArrayView2<bool>,
// ) -> HashMap<String, HashMap<IntegralIndex, Complex64>> {
//     let (iproducts, sym_products) = symplectic_product_map(ipowers, symplectics);
//     let mut hamiltonian: QubitHamiltonianTemplate = QubitHamiltonianTemplate::new();
//     let n_modes = symplectics.len_of(Axis(0)) / 2;
//     Zip::from(sym_products.exact_chunks((n_modes, n_modes, 1))).for_each(|i| println!("{}", i));
//     hamiltonian
// }

// #[test]
// fn test_molecular() {
//     let ipowers = ndarray::arr1(&[0, 1, 2, 3]);
//     let symplectics = ndarray::arr2(&[
//         [true, false, false, false],
//         [true, false, true, false],
//         [false, true, true, false],
//         [false, true, true, true],
//     ]);
//     let (iproducts, sym_products) = symplectic_product_map(ipowers.view(), symplectics.view());
//     let mut hamiltonian: QubitHamiltonianTemplate = QubitHamiltonianTemplate::new();
//     let n_modes = symplectics.len_of(Axis(0)) / 2;
//     Zip::from(sym_products.exact_chunks((n_modes, n_modes, 1))).for_each(|i| println!("{}", i));
// }

pub fn fill_template<'template>(
    template: &'template QubitHamiltonianTemplate,
    constant_energy: f64,
    one_e_coeffs: ArrayView2<f64>,
    two_e_coeffs: ArrayView4<f64>,
    mode_op_map: ArrayView1<usize>,
) -> QubitHamiltonian<'template> {
    debug!("Filling template with mode-operator map {:#?}", mode_op_map);
    assert!(one_e_coeffs
        .shape()
        .iter()
        .all(|&s| s == two_e_coeffs.len_of(Axis(0))));
    assert!(two_e_coeffs
        .shape()
        .iter()
        .all(|&s| s == one_e_coeffs.len_of(Axis(0))));
    assert!(one_e_coeffs.len_of(Axis(0)) == mode_op_map.len());
    // assert_eq!(HashSet::from(mode_op_map.keys()), HashSet::from(0..one_e_coeffs.len_of(Axis(0))));
    // assert_eq!(HashSet::from(mode_op_map.values()), (HashSet::from(0..one_e_coeffs.len_of(Axis(0)))));
    let s = RandomState::new();
    let mut hamiltonian: QubitHamiltonian<'template> =
        QubitHamiltonian::with_capacity_and_hasher(template.keys().len(), s);
    if let Some((identity_key, _)) =
        template.get_key_value(&"I".repeat(mode_op_map.len()).to_string())
    {
        hamiltonian.insert(identity_key, Complex64::new(constant_energy, 0.));
    };
    for (pauli_term, components) in template {
        let val = components
            .iter()
            .fold(Complex64::new(0., 0.), |acc, (indices, factor)| {
                let coeff = match indices {
                    IntegralIndex::TwoE(p, q, r, s) => {
                        two_e_coeffs[[
                            mode_op_map[[*p]],
                            mode_op_map[[*q]],
                            mode_op_map[[*r]],
                            mode_op_map[[*s]],
                        ]]
                    }
                    IntegralIndex::OneE(m, n) => {
                        one_e_coeffs[[mode_op_map[[*m]], mode_op_map[[*n]]]]
                    }
                };
                acc + factor * Complex64::new(coeff, 0.)
            });
        if val.norm() > 1e-12 {
            hamiltonian.insert(pauli_term, val);
        };
    }

    debug!(
        "Template filled: hamiltonian.keys()={:?}",
        hamiltonian.keys()
    );
    hamiltonian
}
