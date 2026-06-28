//! Ladder Operators, shared for Qubits and Fermions
use num_complex::c64;
use num_complex::Complex64;
use std::{result::Result, str::FromStr};

/// Operators for second quantisation.
///
/// These are primarily used in the signatures of fermionic operators, e.g. ['FermionProduct`].
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum LadderOperator {
    /// Particle creation operator.
    Creation,
    /// Particle annihilation operator.
    Annihilation,
}

/// Error for failure to parse ladder operators from strings.
///
/// Returned by [`LadderOperator::from_str`] when the input is not `"+"` or `"-"`.
#[derive(Debug, PartialEq, Clone)]
pub struct ParseLadderError;

impl FromStr for LadderOperator {
    type Err = ParseLadderError;
    /// Parse a string as a ladder operator.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::LadderOperator;
    /// use std::str::FromStr;
    ///
    /// assert_eq!(LadderOperator::from_str("+").unwrap(), LadderOperator::Creation);
    /// assert_eq!(LadderOperator::from_str("-").unwrap(), LadderOperator::Annihilation);
    /// assert!(LadderOperator::from_str("x").is_err());
    /// ```
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
    /// Returns the coefficients of a fermionic ladder operator in terms of Majorana operators.
    ///
    /// While ladder operators are general, the fermionic ladder operators can be expressed exactly as
    /// a combination of two majorana operators.
    ///
    /// This function is used when converting from fermionic operators with arbitrary signature, to a Majorana operator.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::LadderOperator;
    /// use num_complex::Complex64;
    ///
    /// let coeffs = LadderOperator::Creation.majorana_coefficients();
    /// assert_eq!(coeffs.len(), 2);
    /// assert_eq!(coeffs[0], Complex64::new(0.5, 0.0));
    /// assert_eq!(coeffs[1], Complex64::new(0.0, -0.5));
    /// ```
    /// Stack-allocated variant of [`LadderOperator::majorana_coefficients`].
    pub fn majorana_coefficients(&self) -> [Complex64; 2] {
        match self {
            LadderOperator::Creation => [c64(0.5, 0.0), c64(0., -0.5)],
            LadderOperator::Annihilation => [c64(0.5, 0.0), c64(0., 0.5)],
        }
    }
}
impl TryFrom<char> for LadderOperator {
    type Error = ParseLadderError;

    fn try_from(string: char) -> Result<Self, Self::Error> {
        if string == '+' {
            Ok(LadderOperator::Creation)
        } else if string == '-' {
            Ok(LadderOperator::Annihilation)
        } else {
            Err(ParseLadderError)
        }
    }
}

#[cfg(test)]
mod ladder_tests {
    use super::*;

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

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_ladder_try_from_char(c in proptest::sample::select(&['+', '-'])) {
            let op = LadderOperator::try_from(c);
            if c == '+' {
                prop_assert_eq!(op, Ok(LadderOperator::Creation));
            } else {
                prop_assert_eq!(op, Ok(LadderOperator::Annihilation));
            }
        }
    }
}
