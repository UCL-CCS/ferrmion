use crate::operators::{CoefficientPauliWeight, PauliWeight};
use ahash::RandomState;
use num_complex::Complex64;
use std::collections::HashMap;

/// A qubit Hamiltonian represented as a sparse mapping from Pauli strings to complex coefficients.
///
/// Each key is a Pauli string (e.g. `"XYZII"`), and each value is the corresponding
/// complex coefficient. Uses a randomised hash state for performance.
///
/// # Examples
///
/// ```
/// use ferrmion_core::hamiltonians::QubitHamiltonian;
/// use num_complex::Complex64;
///
/// let mut ham = QubitHamiltonian::default();
/// ham.insert("XYZ".to_string(), Complex64::new(1.0, 0.0));
/// assert_eq!(ham.len(), 1);
/// ```
pub type QubitHamiltonian = HashMap<String, Complex64, RandomState>;

impl PauliWeight for QubitHamiltonian {
    fn pauli_weight(&self) -> usize {
        self.keys().fold(0, |acc, term: &String| {
            let n_identity = term.chars().filter(|c| c == &'I').count();
            acc + (term.len() - n_identity)
        })
    }
}
impl CoefficientPauliWeight for QubitHamiltonian {
    fn coeff_pauli_weight(&self) -> f64 {
        self.iter().fold(0., |acc, (term, coeff)| {
            let n_identity = term.chars().filter(|c| c == &'I').count();
            acc + (term.len() - n_identity) as f64 * coeff.norm()
        })
    }
}
