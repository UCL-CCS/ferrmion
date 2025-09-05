/*
Functions relating to encoding optimisation.
*/

use crate::hamiltonians::*;
use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor},
    solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use ndarray::{s, ArrayView1, ArrayViewMut, Axis, Zip};
use num_complex::ComplexFloat;
use numpy::ndarray::{Array1, ArrayView2, ArrayView4};
use permutation_iterator::Permutor;
use pyo3::type_object;
use rand::{distr::Uniform, prelude::*};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::{Arc, Mutex};

pub fn pauli_coefficient_weight(hamiltonian: QubitHamiltonian) -> f64 {
    let weight = hamiltonian.iter().fold(0., |acc, (key, val)| {
        let n_identity = key.chars().filter(|c| c == &'I').count();
        acc + (key.len() - n_identity) as f64 * val.abs()
    });
    weight
}

pub fn template_weight(
    template: &QubitHamiltonianTemplate,
    constant_energy: f64,
    one_e_coeffs: ArrayView2<f64>,
    two_e_coeffs: ArrayView4<f64>,
    n_permutations: usize,
) -> Array1<f64> {
    let n_modes = one_e_coeffs.len_of(Axis(0));
    let mut values: Array1<f64> = Array1::zeros(n_permutations);
    values.map_inplace(|v: &mut f64| {
        let permutor = Permutor::new(n_modes as u64);
        let permutation: Array1<usize> =
            Array1::from(permutor.map(|p| p as usize).collect::<Vec<usize>>());
        let hamiltonian = fill_template(
            template,
            constant_energy,
            one_e_coeffs,
            two_e_coeffs,
            permutation.view(),
        );
        *v = pauli_coefficient_weight(hamiltonian);
    });
    values
}

// pub fn batch_template_weight<'template>(template: &'template QubitHamiltonianTemplate,
//     constant_energy: f64,
//     one_e_coeffs: ArrayView2<f64>,
//     two_e_coeffs: ArrayView4<f64>,
//     mode_op_map: HashMap<usize, usize>) {
//         pass
// }

struct OptimalEnumeration<'template, 'coeff> {
    template: &'template QubitHamiltonianTemplate,
    constant_energy: f64,
    one_e_coeffs: ArrayView2<'coeff, f64>,
    two_e_coeffs: ArrayView4<'coeff, f64>,
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl CostFunction for OptimalEnumeration<'_, '_> {
    type Param = Array1<usize>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let filled_template = fill_template(
            self.template,
            self.constant_energy,
            self.one_e_coeffs,
            self.two_e_coeffs,
            param.view(),
        );
        Ok(pauli_coefficient_weight(filled_template))
    }
}

impl Anneal for OptimalEnumeration<'_, '_> {
    type Param = Array1<usize>;
    type Output = Array1<usize>;
    type Float = f64;

    fn anneal(&self, param: &Array1<usize>, temp: f64) -> Result<Array1<usize>, Error> {
        let mut next_perm = param.clone();
        let mut rng = self.rng.lock().unwrap();
        let distr = Uniform::try_from(0..param.len()).unwrap();

        for _ in 0..(temp.floor() as u64 + 1) {
            let pos: usize = rng.sample(distr);
            let left_stay_right: usize = rng.random_range(0..=1);
            let temp = next_perm[[pos]];
            next_perm[[pos]] = next_perm[[pos + 2 * left_stay_right - 1]];
            next_perm[[pos + 2 * left_stay_right - 1]] = temp;
        }
        Ok(next_perm)
    }
}
