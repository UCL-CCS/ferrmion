/*
Functions relating to encoding optimisation.
*/

use crate::hamiltonians::*;
use num_complex::ComplexFloat;

#[allow(dead_code)]
pub fn pauli_coefficient_weight(hamiltonian: QubitHamiltonian) -> f64 {
    let weight = hamiltonian.iter().fold(0., |acc, (key, val)| {
        let n_identity = key.chars().filter(|c| c == &'I').count();
        acc + (key.len() - n_identity) as f64 * val.abs()
    });
    weight
}

// pub fn batch_template_weight<'template>(template: &'template QubitHamiltonianTemplate,
//     constant_energy: f64,
//     one_e_terms: ArrayView2<f64>,
//     two_e_terms: ArrayView4<f64>,
//     mode_op_map: HashMap<usize, usize>) {
//         pass
// }
