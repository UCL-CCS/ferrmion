use ndarray::Axis;
/*
Shared Types.
*/
use crate::utils::vector_kron;
use numpy::ndarray::{arr1, arr2, Array1, Array2, ArrayView1};
use numpy::Complex64;
use std::collections::HashMap;
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
        if coefficients.len() != indices.len_of(Axis(1))
            || signature.len() != indices.len_of(Axis(0))
        {
            return Err(SparseFermionError);
        };

        Ok(Self {
            signature,
            indices,
            coefficients,
        })
    }
}

struct SparseFermionHamiltonian {
    terms: Vec<SparseFermionTerm>,
}

#[cfg(test)]
mod fermion_tests {
    use ndarray::arr1;

    use crate::types::*;

    fn test_fermion_operators() {
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

    fn test_sparse_term() {
        let signature = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let indices = arr2(&[[0, 1], [2, 3]]);
        let coefficients = arr1(&[Complex64::new(1.0, 0.), Complex64::new(-1., 0.)]);
        let term = SparseFermionTerm::new(signature, indices, coefficients).unwrap();
    }
}

// /*
// Majorana
// */
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
        if coefficients.len() != indices.len_of(Axis(1)) {
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
        // Start off by creating a hashmap as we'll need to add a few fermionic terms
        // to each majorana term
        let signature_coeffs = sft
            .signature
            .iter()
            .map(|s| s.fermion_coeff())
            .reduce(|acc, s| vector_kron(&acc, &s));
        let mut majoranas: HashMap<ArrayView1<u16>, Complex64> = HashMap::new();
        for fermion_index in sft.indices.rows() {
            majoranas.entry(fermion_index);
        }
        SparseMajoranaTerm::new(sft.indices, sft.coefficients)
            .expect("Indices and coefficients should be same length.")
    }
}

struct SparseMajoranaHamiltonian {
    terms: Vec<SparseMajoranaTerm>,
}

#[cfg(test)]
mod majorana_tests {
    use crate::types::*;
    fn test_sparse_term() {
        let indices = arr2(&[[0, 1], [2, 3]]);
        let coefficients = arr1(&[Complex64::new(1.0, 0.), Complex64::new(-1., 0.)]);
        let term = SparseMajoranaTerm::new(indices, coefficients).unwrap();
    }
}
