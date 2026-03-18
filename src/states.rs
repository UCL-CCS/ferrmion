//! Structs representing quantum states

use ndarray::Array1;
use num_complex::Complex64;

pub trait Normalizable {
    fn normalize(&mut self);
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZBasisState {
    pub state: Array1<bool>,
    pub coefficient: Complex64,
}

impl ZBasisState {
    pub fn new(state: Array1<bool>, coefficient: Complex64) -> Self {
        let mut out = Self { state, coefficient };
        out.normalize();
        out
    }

    pub fn zeros(n_modes: usize) -> Self {
        Self::new(Array1::from_elem(n_modes, false), Complex64::new(1., 0.))
    }
}

impl Normalizable for ZBasisState {
    fn normalize(&mut self) {
        let norm = self.coefficient.norm();
        if norm != 0. {
            self.coefficient /= norm;
        }
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
