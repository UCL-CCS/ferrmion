use ahash::RandomState;
use itertools::iproduct;
use log::debug;
use numpy::ndarray::{s, ArrayView1, ArrayView2, ArrayView4};
use numpy::Complex64;
use pyo3::{FromPyObject, IntoPyObject};
use std::collections::HashMap;

use crate::encoding::MajoranaEncoding;
use crate::utils::icount_to_sign;

pub type QubitHamiltonian = HashMap<String, Complex64, RandomState>;

pub type QubitHamiltonianTemplate =
    HashMap<String, HashMap<IntegralIndex, Complex64, RandomState>, RandomState>;
pub type FilledTemplate<'template> = HashMap<&'template String, Complex64, RandomState>;

pub enum Notation {
    Physicist,
    Chemist,
}

#[derive(Eq, PartialEq, Hash, IntoPyObject, FromPyObject, Debug)]
pub enum IntegralIndex {
    //TwoE terms are more common, and pyo3 tries from top to bottom
    //So putting them first in the Enum
    TwoE(usize, usize, usize, usize),
    OneE(usize, usize),
}

pub fn molecular(encoding: MajoranaEncoding, notation: Notation) -> QubitHamiltonianTemplate {
    debug!(
        "Creating molecular hamiltonian template with\n ipowers={:?}, symplectics shape={:?}",
        encoding.ipowers,
        encoding.symplectics.shape()
    );

    let (iproducts, sym_products) = encoding.symplectic_product_map();

    let mut hamiltonian: QubitHamiltonianTemplate =
        QubitHamiltonianTemplate::with_hasher(RandomState::new());
    // assume 8-fold symmetry
    let n_modes = encoding.n_modes;
    hamiltonian.insert(
        "I".repeat(encoding.n_qubits).to_string(),
        HashMap::with_hasher(RandomState::new()),
    );
    for m in 0..n_modes {
        for n in 0..n_modes {
            // ipowers can be updated to account for +/- operators
            for (l, r) in iproduct!(0..2, 0..2) {
                let term = sym_products.slice(s![2 * m + l, 2 * n + r, ..]);
                let (pauli_string, im_term_pauli) = MajoranaEncoding::symplectic_to_pauli(term, 0);
                let weight = Complex64::new(0.25, 0.)
                    * icount_to_sign(
                        iproducts[[2 * m + l, 2 * n + r]] as usize
                            + im_term_pauli as usize
                            + (r + 3 * l),
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
                        let (product_term, iproduct) =
                            MajoranaEncoding::symplectic_product(left, right, 0);
                        let (pauli_string, im_term_pauli) =
                            MajoranaEncoding::symplectic_to_pauli(product_term.view(), 0);
                        let term_ipowers = match notation {
                            Notation::Physicist => 3 * (l1 + l2) + r1 + r2,
                            Notation::Chemist => 3 * (l1 + r1) + l2 + r2,
                        };
                        let weight = Complex64::new(0.0625, 0.)
                            * icount_to_sign(
                                iproduct as usize
                                    + im_term_pauli as usize
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

pub fn hubbard(encoding: MajoranaEncoding) -> QubitHamiltonianTemplate {
    debug!(
        "Creating molecular hamiltonian template with\n ipowers={:?}, symplectics shape={:?}",
        encoding.ipowers,
        encoding.symplectics.shape()
    );

    let (iproducts, sym_products) = encoding.symplectic_product_map();

    let s = RandomState::new();
    let mut hamiltonian: QubitHamiltonianTemplate = QubitHamiltonianTemplate::with_hasher(s);
    // assume 8-fold symmetry
    let n_modes = encoding.n_modes;
    hamiltonian.insert(
        "I".repeat(n_modes).to_string(),
        HashMap::with_hasher(RandomState::new()),
    );
    for m in 0..n_modes {
        for n in 0..n_modes {
            // ipowers can be updated to account for +/- operators
            for (l, r) in iproduct!(0..2, 0..2) {
                let term = sym_products.slice(s![2 * m + l, 2 * n + r, ..]);
                let (pauli_string, im_term_pauli) = MajoranaEncoding::symplectic_to_pauli(term, 0);
                let weight = Complex64::new(0.25, 0.)
                    * icount_to_sign(
                        iproducts[[2 * m + l, 2 * n + r]] as usize
                            + im_term_pauli as usize
                            + (r + 3 * l),
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
                    let (product_term, iproduct) =
                        MajoranaEncoding::symplectic_product(left, right, 0);
                    let (pauli_string, im_term_pauli) =
                        MajoranaEncoding::symplectic_to_pauli(product_term.view(), 0);
                    let term_ipowers = 3 * (l1 + r1) + l2 + r2;
                    let weight = Complex64::new(0.0625, 0.)
                        * icount_to_sign(
                            iproduct as usize
                                + im_term_pauli as usize
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

pub fn fill_template<'template>(
    template: &'template QubitHamiltonianTemplate,
    constant_energy: f64,
    one_e_coeffs: ArrayView2<f64>,
    two_e_coeffs: ArrayView4<f64>,
    mode_op_map: ArrayView1<usize>,
) -> FilledTemplate<'template> {
    debug!("Filling template with mode-operator map {:#?}", mode_op_map);
    // assert_eq!(HashSet::from(mode_op_map.keys()), HashSet::from(0..one_e_coeffs.len_of(Axis(0))));
    // assert_eq!(HashSet::from(mode_op_map.values()), (HashSet::from(0..one_e_coeffs.len_of(Axis(0)))));
    let s = RandomState::new();
    let mut hamiltonian: FilledTemplate<'template> =
        FilledTemplate::with_capacity_and_hasher(template.keys().len(), s);
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
        // if val.norm() > 1e-12 {
        hamiltonian.insert(pauli_term, val);
        // };
    }

    debug!(
        "Template filled: hamiltonian.keys()={:?}",
        hamiltonian.keys()
    );
    hamiltonian
}

#[cfg(test)]
mod tests {
    use crate::{encoding::MajoranaEncoding, hamiltonians::molecular};

    #[test]
    fn test_template_padding() {
        let ipowers = ndarray::arr1(&[0, 1, 2, 3]);
        let symplectics = ndarray::arr2(&[
            [true, false, false, false],
            [true, false, true, false],
            [false, true, true, false],
            [false, true, true, true],
        ]);
        let encoding = MajoranaEncoding::new(ipowers.view(), symplectics.view());
        let template = molecular(encoding, super::Notation::Physicist);

        let symplectics = ndarray::arr2(&[
            [true, false, false, false, false, false],
            [true, false, false, true, false, false],
            [false, true, false, true, false, false],
            [false, true, false, true, true, false],
        ]);
        let padded_encoding = MajoranaEncoding::new(ipowers.view(), symplectics.view());
        let padded_template = molecular(padded_encoding, super::Notation::Physicist);
        for short_key in template.keys() {
            assert!(padded_template.contains_key(&format!("{short_key}I")))
        }
    }
}
