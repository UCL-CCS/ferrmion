use std::ops::Mul;
/*
Shared Types.
*/
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::Complex64;
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

#[derive(Debug, PartialEq)]
struct FermionHamiltonian<'coeff> {
    terms: Vec<(Vec<LadderOperator>, ArrayView2<'coeff, f64>)>,
}

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

impl Mul for FermionOperator {
    type Output = FermionProduct;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ops = Vec::<FermionOperator>::with_capacity(4);
        let mut coeff = Complex64::new(1., 0.);
        if self.index >= rhs.index {
            ops.push(self);
            ops.push(rhs);
        } else {
            ops.push(rhs);
            ops.push(self);
            coeff *= -1.;
        };
        FermionProduct::new(ops, Some(coeff))
    }
}

impl Mul<FermionProduct> for FermionOperator {
    type Output = FermionProduct;
    fn mul(self, rhs: FermionProduct) -> Self::Output {
        rhs.ops.push(self);
    }
}

#[derive(Debug, PartialEq)]
struct FermionProduct {
    ops: Vec<FermionOperator>,
    coeff: Complex64,
}

impl FermionProduct {
    pub fn new(ops: Vec<FermionOperator>, coeff: Option<Complex64>) -> Self {
        match coeff {
            Some(val) => Self {
                ops: ops,
                coeff: val,
            },
            None => Self {
                ops: ops,
                coeff: Complex64::new(1., 0.),
            },
        }
    }
}

struct SparseFermionHamiltonian<'coeff> {
    terms: Array2<FermionOperator>,
    coefficients: ArrayView1<'coeff, Complex64>,
}

impl<'coeff> SparseFermionHamiltonian<'coeff> {
    pub fn new(
        terms: Array2<FermionOperator>,
        coefficients: ArrayView1<'coeff, Complex64>,
    ) -> Self {
        Self {
            terms,
            coefficients,
        }
    }
}

#[cfg(test)]
mod fermion_tests {
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
        assert_eq!(c0 * a1, FermionProduct::new(vec![c0, a1], false));
        assert_eq!(a1 * c0, FermionProduct::new(vec![c0, a1], true));
    }
}

// /*
// Majorana
// */
// struct MajoranaOperator(u32);

// struct MajoranaHamiltonian<'coeff> {
//     terms: Vec<ArrayView2<'coeff, f64>>,
// }

// struct SparseMajoranaHamiltonian {
//     terms: Array2<MajoranaOperator>,
//     coefficients: Array1<Complex64>,
// }

// impl SparseMajoranaHamiltonian {
//     pub fn new(terms: Array2<MajoranaOperator>, coefficients: Array1<Complex64>) -> Self {
//         Self {
//             terms,
//             coefficients,
//         }
//     }
// }

// impl TryFrom<SparseFermionHamiltonian> for SparseMajoranaHamiltonian {
//     fn try_from(fermion_hamiltonian: SparseFermionHamiltonian) -> Self {}
// }
