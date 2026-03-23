//! Structs representing quantum states.

use itertools::CombinationsWithReplacement;
use ndarray::{s, Array1};
use num_complex::Complex64;
use std::{ops::Mul};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
enum BraKet {
    Bra,
    Ket,
}

/// Trait for a quantum state.
///
/// Note this is implicitly a Ket vector.
pub trait State {
    /// Normalize the state so that the coefficient has unit norm.
    fn normalize(&mut self);

    /// Return the dimension of the state space.
    #[allow(dead_code)]
    fn dimension(&self) -> usize;

    /// Return the adjoint (dagger) of the state.
    #[allow(dead_code)]
    fn adjoint(&mut self);

    fn reindex(&mut self, permutation: &[usize]);
}

#[derive(Error, Debug)]
pub enum StateError {
    #[error("Invalid bra/ket combination")]
    InvalidBraKet,
}

/// A quantum state in the computational (pauli Z) basis.
#[derive(Debug, Clone, PartialEq)]
pub struct ZBasisState {
    pub state: Array1<bool>,
    pub coefficient: Complex64,
    bra_ket: BraKet,
}

impl ZBasisState {
    /// Construct a new `ZBasisState` with the given state and coefficient.
    pub fn new(state: Array1<bool>, coefficient: Complex64) -> Self {
        let mut out = Self { state, coefficient, bra_ket: BraKet::Ket };
        out.normalize();
        out
    }

    /// Construct a new `ZBasisState` with all qubits set to zero and a unit coefficient.
    pub fn zeros(n_qubits: usize) -> Self {
        Self::new(Array1::from_elem(n_qubits, false), Complex64::new(1., 0.))
    }
}

impl State for ZBasisState {
    fn normalize(&mut self) {
        let norm = self.coefficient.norm();
        if norm != 0. {
            self.coefficient /= norm;
        }
    }
    fn dimension(&self) -> usize {
        self.state.len()
    }

    fn adjoint(&mut self) {
        self.state = self.state.slice(s![..;-1]).to_owned();
        self.coefficient = self.coefficient.conj();
        self.bra_ket = match self.bra_ket {
            BraKet::Bra => BraKet::Ket,
            BraKet::Ket => BraKet::Bra,
        };
    }

    fn reindex(&mut self, permutation: &[usize]) {
        let mut new_state = Array1::from_elem(self.state.len(), false);
        for (original, &new) in permutation.iter().enumerate() {
            new_state[new] = self.state[original];
        }
        self.state = new_state;
    }
}

impl Mul<Complex64> for ZBasisState {
    type Output = Self;

    fn mul(self, rhs: Complex64) -> Self::Output {
        ZBasisState::new(self.state, self.coefficient * rhs)
    }
}

impl Mul for ZBasisState {
    type Output = Result<Complex64, StateError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.bra_ket, rhs.bra_ket) {
            (BraKet::Bra, BraKet::Ket) => {
                if self.state == rhs.state {
                    Ok(self.coefficient * rhs.coefficient.conj())
                } else {
                    Ok(Complex64::ZERO)
                }
            }
            _ => Err(StateError::InvalidBraKet),
        }

    }
}

#[cfg(test)]
mod zbasis_tests {
    use super::*;

    #[test]
    fn test_zbasis_state() {
        let state = Array1::from_elem(3, false);
        let coefficient = Complex64::new(1., 0.);
        let zbasis_state = ZBasisState::new(state, coefficient);
        assert_eq!(zbasis_state.state, Array1::from_elem(3, false));
        assert_eq!(zbasis_state.coefficient, Complex64::new(1., 0.));
    }

    #[test]
    fn test_normalize() {
        let mut zbasis_state =
            ZBasisState::new(Array1::from_elem(3, false), Complex64::new(2., 0.));
        zbasis_state.normalize();
        assert_eq!(zbasis_state.coefficient, Complex64::new(1., 0.));
    }
}

pub struct FockState {
    pub state: Array1<bool>,
    pub coefficient: Complex64,
}

impl FockState {
    pub fn new(state: Array1<bool>, coefficient: Complex64) -> Self {
        Self { state, coefficient }
    }
}
impl State for FockState {
    fn normalize(&mut self) {
        let norm = self.coefficient.norm();
        if norm != 0. {
            self.coefficient /= norm;
        }
    }
    fn dimension(&self) -> usize {
        self.state.len()
    }

    fn adjoint(&mut self) {
        self.state = self.state.slice(s![..;-1]).to_owned();
        self.coefficient = self.coefficient.conj();
    }

    fn reindex(&mut self, permutation: &[usize]) {
        let mut new_state = Array1::from_elem(self.state.len(), false);
        for (original, &new) in permutation.iter().enumerate() {
            new_state[new] = self.state[original];
        }
        self.state = new_state;
    }
}

#[cfg(test)]
mod fock_tests {
    use super::*;

    #[test]
    fn test_fock_state() {
        let state = Array1::from_elem(3, false);
        let coefficient = Complex64::new(1., 0.);
        let fock_state = FockState::new(state, coefficient);
        assert_eq!(fock_state.state, Array1::from_elem(3, false));
        assert_eq!(fock_state.coefficient, Complex64::new(1., 0.));
    }
}
