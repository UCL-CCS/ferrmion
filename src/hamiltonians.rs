use crate::operators::{CoefficientPauliWeight, PauliWeight};
use ahash::RandomState;
use numpy::Complex64;
use std::collections::HashMap;

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
