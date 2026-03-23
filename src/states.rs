//! Structs representing quantum states.

use ndarray::{s, Array1};
use num_complex::Complex64;

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
}

/// A quantum state in the computational (pauli Z) basis.
#[derive(Debug, Clone, PartialEq)]
pub struct ZBasisState {
    pub state: Array1<bool>,
    pub coefficient: Complex64,
}

impl ZBasisState {
    /// Construct a new `ZBasisState` with the given state and coefficient.
    pub fn new(state: Array1<bool>, coefficient: Complex64) -> Self {
        let mut out = Self { state, coefficient };
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
    }
}

#[cfg(test)]
mod tests {
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
