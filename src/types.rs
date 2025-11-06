/*
Shared Types.
*/
use std::{error::Error, result::Result, str::FromStr};

#[allow(dead_code)]
#[derive(Debug, Default)]
pub enum Pauli {
    #[default]
    I,
    X,
    Y,
    Z,
}

#[derive(PartialEq, Eq, Debug)]
pub enum LadderOperator {
    Creation,
    Annihilation,
}

#[derive(Debug, PartialEq)]
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
mod tests {
    use super::*;

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

// struct FermionHamiltonian<'coeff> {
//     terms: Vec<(Vec<LadderOperator>, ArrayView2<'coeff, f64>)>,
// }

// struct FermionOperator(LadderOperator, u32);

// struct FermionProduct(Vec<FermionOperator>);

// struct SparseFermionHamiltonian<'coeff> {
//     terms: Array2<FermionOperator>,
//     coefficients: ArrayView1<'coeff, Complex64>,
// }

// impl<'coeff> SparseFermionHamiltonian<'coeff> {
//     pub fn new(
//         terms: Array2<FermionOperator>,
//         coefficients: ArrayView1<'coeff, Complex64>,
//     ) -> Self {
//         Self {
//             terms,
//             coefficients,
//         }
//     }
// }

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
