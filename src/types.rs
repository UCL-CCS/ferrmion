use ndarray::{Axis, Zip};
/*
Shared Types.
*/
use crate::utils::vector_kron;
use itertools::Itertools;
use numpy::ndarray::{arr1, Array1, Array2};
use numpy::Complex64;
use std::collections::BTreeMap;
use std::iter::{repeat_n, zip};
use std::{result::Result, str::FromStr};

#[allow(dead_code)]
#[derive(Debug, Default)]
pub enum Pauli {
    #[default]
    I,
    X,
    Y,
    Z,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum LadderOperator {
    Creation,
    Annihilation,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ParseLadderError;

impl FromStr for LadderOperator {
    type Err = ParseLadderError;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        if string == "+" {
            Ok(LadderOperator::Creation)
        } else if string == "-" {
            Ok(LadderOperator::Annihilation)
        } else {
            Err(ParseLadderError)
        }
    }
}

impl LadderOperator {
    pub fn fermion_coeff(&self) -> Array1<Complex64> {
        match &self {
            LadderOperator::Creation => arr1(&[Complex64::new(0.5, 0.0), Complex64::new(0., -0.5)]),
            LadderOperator::Annihilation => {
                arr1(&[Complex64::new(0.5, 0.0), Complex64::new(0., 0.5)])
            }
        }
    }
}

#[cfg(test)]
mod ladder_tests {
    use crate::types::*;

    #[test]
    fn test_ladder_operators() {
        assert_eq!(
            LadderOperator::from_str("+").unwrap(),
            LadderOperator::Creation
        );
        assert_eq!(
            LadderOperator::from_str("-").unwrap(),
            LadderOperator::Annihilation
        );
        assert_eq!(LadderOperator::from_str("+-"), Err(ParseLadderError));
        assert_eq!(LadderOperator::from_str("-+"), Err(ParseLadderError));
    }
}

/*
Fermion
*/

// #[derive(Debug, PartialEq)]
// struct FermionHamiltonian<'coeff> {
//     terms: Vec<(Vec<LadderOperator>, Array
//
// View2<'coeff, f64>)>,
// }

#[derive(Debug, PartialEq, Clone, Copy)]
struct FermionOperator {
    op: LadderOperator,
    index: u32,
}

impl FermionOperator {
    fn new(op: LadderOperator, index: u32) -> Self {
        Self { op, index }
    }
}

struct SparseFermionTerm {
    signature: Vec<LadderOperator>,
    indices: Array2<u16>,
    coefficients: Array1<Complex64>,
}

#[derive(Debug, PartialEq, Clone)]
struct SparseFermionError;

impl SparseFermionTerm {
    pub fn new(
        signature: Vec<LadderOperator>,
        indices: Array2<u16>,
        coefficients: Array1<Complex64>,
    ) -> Result<Self, SparseFermionError> {
        if coefficients.len() != indices.len_of(Axis(0))
            || signature.len() != indices.len_of(Axis(1))
        {
            println!("{:#?}", coefficients.len());
            println!("{:#?}", signature.len());
            return Err(SparseFermionError);
        };

        Ok(Self {
            signature,
            indices,
            coefficients,
        })
    }
}

#[allow(dead_code)]
struct SparseFermionHamiltonian {
    terms: Vec<SparseFermionTerm>,
}

#[cfg(test)]
mod fermion_tests {
    use crate::types::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_operator_creation() {
        let c0 = FermionOperator::new(LadderOperator::Creation, 0);
        let a1 = FermionOperator::new(LadderOperator::Annihilation, 1);
        assert_eq!(
            c0,
            FermionOperator {
                op: LadderOperator::Creation,
                index: 0
            }
        );
        assert_eq!(
            a1,
            FermionOperator {
                op: LadderOperator::Annihilation,
                index: 1
            }
        );
    }

    #[test]
    fn test_signature_conversion() {
        let signature = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let im_coeffs: Array1<Complex64> = signature
            .iter()
            .map(|s| s.fermion_coeff())
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap();
        assert_eq!(
            im_coeffs,
            arr1(&[
                Complex64 { re: 0.25, im: 0.0 },
                Complex64 { re: 0.0, im: -0.25 },
                Complex64 { re: 0.0, im: 0.25 },
                Complex64 { re: 0.25, im: 0.0 }
            ])
        );
    }

    #[test]
    fn test_term_creation() {
        let signature = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let indices = arr2(&[[0, 1], [2, 3]]);
        let coefficients = arr1(&[Complex64::new(1.0, 0.), Complex64::new(-1., 0.)]);
        let _term = SparseFermionTerm::new(signature, indices, coefficients).unwrap();
    }
}

// /*
// Majorana
// */
#[derive(Debug, PartialEq, Clone)]
struct SparseMajoranaTerm {
    indices: Array2<u16>,
    coefficients: Array1<Complex64>,
}

#[derive(Debug, PartialEq, Clone)]
struct SparseMajoranaError;

impl SparseMajoranaTerm {
    pub fn new(
        indices: Array2<u16>,
        coefficients: Array1<Complex64>,
    ) -> Result<Self, SparseMajoranaError> {
        if coefficients.len() != indices.len_of(Axis(0)) {
            return Err(SparseMajoranaError);
        };
        Ok(Self {
            indices,
            coefficients,
        })
    }
}

impl From<SparseFermionTerm> for SparseMajoranaTerm {
    fn from(sft: SparseFermionTerm) -> Self {
        // Start off by creating a BTreeMap as we'll need to add a few fermionic terms
        // to each majorana term
        let term_length = sft.signature.len();
        //     .flatten()
        //     .collect::<Vec<u16>>();
        // let offset_array: Array2<u16> =
        //     Array2::from_shape_vec((2_i32.pow(term_length as u32), term_length), offset_vec);
        let mut majoranas: BTreeMap<Vec<u16>, Complex64> = BTreeMap::new();
        Zip::from(sft.indices.rows())
            .and(sft.coefficients.view())
            .for_each(|ind, coeff| {
                let signature_coeffs: Vec<Complex64> = sft
                    .signature
                    .iter()
                    .map(|s| s.fermion_coeff())
                    .reduce(|acc, s| vector_kron(&acc, &s))
                    .unwrap()
                    .to_vec();

                // println!("{:#?}", signature_coeffs);
                let offset = repeat_n(0u16..2u16, term_length).multi_cartesian_product();
                for (sc, offset) in zip(signature_coeffs, offset) {
                    println!("Majorana componentnts {:#?}", sc.clone());
                    println!("Majorana componentnts {:#?}", offset.clone());
                    let mut majorana_term = Array1::zeros(term_length);
                    majorana_term += &ind;
                    majorana_term *= 2;
                    majorana_term = majorana_term + Array1::from_vec(offset);
                    println!("Majorana componentnts {:#?}", majorana_term.clone());
                    *majoranas
                        .entry(majorana_term.to_vec())
                        .or_insert(Complex64 { re: 0.0, im: 0.0 }) += sc * coeff;
                }
            });

        // println!("Majoranas {:#?}", majoranas);
        let sparse_values: Array1<Complex64> = majoranas.values().cloned().collect();
        let mut sparse_indices: Array2<u16> = Array2::zeros((majoranas.keys().len(), term_length));
        // println!("{:#?}", sparse_values.clone());
        for (mut row, k) in zip(sparse_indices.rows_mut(), majoranas.keys()) {
            // let mut row_array: Array1<u16> = Array1::from_vec(k.clone());
            row.scaled_add(1, &Array1::from_vec(k.to_vec()));
        }
        println!("{:#?}", sparse_indices.clone());
        SparseMajoranaTerm::new(sparse_indices, sparse_values)
            .expect("Indices and coefficients should be same length.")
    }
}

#[allow(dead_code)]
struct SparseMajoranaHamiltonian {
    terms: Vec<SparseMajoranaTerm>,
}

#[cfg(test)]
mod majorana_tests {
    use crate::types::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_sparse_term() {
        let indices = arr2(&[[0, 1]]);
        let coefficients = arr1(&[Complex64::new(10.0, 0.)]);
        let signature = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        println!("{:#?}", indices.clone());
        println!("{:#?}", coefficients.clone());
        println!("{:#?}", signature.clone());

        let majorana_term = SparseMajoranaTerm::new(
            arr2(&[[0, 2], [0, 3], [1, 2], [1, 3]]),
            arr1(&[
                Complex64::new(2.5, 0.),
                Complex64::new(0., -2.5),
                Complex64::new(0.0, 2.5),
                Complex64::new(2.5, 0.),
            ]),
        )
        .unwrap();
        let fermion_term =
            SparseFermionTerm::new(signature.clone(), indices.clone(), coefficients.clone())
                .unwrap();
        assert_eq!(majorana_term, SparseMajoranaTerm::from(fermion_term));
    }
}
