/*
Functions relating to encoding optimisation.
*/

use crate::hamiltonians::*;
use argmin::{
    core::{CostFunction, Error, Executor},
    solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use ndarray::{ArrayView1, Axis};
use num_complex::ComplexFloat;
use numpy::ndarray::{Array1, ArrayView2, ArrayView4};
use permutation_iterator::Permutor;
use rand::{distr::Uniform, prelude::*};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::{Arc, Mutex};

/// Returns the mean Pauli-weight of Hamiltonian terms.
/// scaled by the coefficient of the term.
pub fn pauli_coefficient_weight(hamiltonian: QubitHamiltonian) -> f64 {
    let weight = hamiltonian.iter().fold(0., |acc, (key, val)| {
        let n_identity = key.chars().filter(|c| c == &'I').count();
        acc + (key.len() - n_identity) as f64 * val.abs()
    });
    weight / hamiltonian.len() as f64
}

#[allow(dead_code)]
/// Returns the mean Pauli-weight of Hamiltonian terms.
pub fn pauli_weight(hamiltonian: QubitHamiltonian) -> f64 {
    let weight = hamiltonian.keys().fold(0., |acc, key| {
        let n_identity = key.chars().filter(|c| c == &'I').count();
        acc + (key.len() - n_identity) as f64
    });
    weight / hamiltonian.len() as f64
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

struct OptimalEnumeration<'coeff> {
    template: QubitHamiltonianTemplate,
    one_e_coeffs: ArrayView2<'coeff, f64>,
    two_e_coeffs: ArrayView4<'coeff, f64>,
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl<'coeff> OptimalEnumeration<'coeff> {
    fn new(
        template: QubitHamiltonianTemplate,
        one_e_coeffs: ArrayView2<'coeff, f64>,
        two_e_coeffs: ArrayView4<'coeff, f64>,
    ) -> Self {
        OptimalEnumeration {
            template,
            one_e_coeffs,
            two_e_coeffs,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::seed_from_u64(1017))),
        }
    }
}

impl CostFunction for OptimalEnumeration<'_> {
    type Param = Array1<usize>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let filled_template = fill_template(
            &self.template,
            0.,
            self.one_e_coeffs,
            self.two_e_coeffs,
            param.view(),
        );
        Ok(pauli_coefficient_weight(filled_template))
    }
}

impl Anneal for OptimalEnumeration<'_> {
    type Param = Array1<usize>;
    type Output = Array1<usize>;
    type Float = f64;

    fn anneal(&self, param: &Array1<usize>, temp: f64) -> Result<Array1<usize>, Error> {
        let mut next_perm = param.clone();
        let n_modes = next_perm.len();
        let mut rng = self.rng.lock().unwrap();
        let distr = Uniform::try_from(0..n_modes).unwrap();
        let temp_int = temp.floor() as u64 + 1;

        for _ in 0..temp_int {
            let pos: usize = rng.sample(distr);
            let move_distance = rng.random_range(0..temp_int) as usize % n_modes;
            let pos2: usize = if rng.random_bool(0.5) {
                (pos + move_distance) % n_modes
            } else {
                (pos + n_modes - move_distance) % n_modes
            };
            let swap_val = next_perm[[pos]];
            next_perm[[pos]] = next_perm[[pos2]];
            next_perm[[pos2]] = swap_val;
        }
        Ok(next_perm)
    }
}

pub fn anneal_enumerations<'coeff>(
    template: QubitHamiltonianTemplate,
    one_e_coeffs: ArrayView2<'coeff, f64>,
    two_e_coeffs: ArrayView4<'coeff, f64>,
    temperature: f64,
    initial_guess: ArrayView1<usize>,
) -> Result<(f64, Array1<usize>), Error> {
    let operator = OptimalEnumeration::new(template, one_e_coeffs, two_e_coeffs);

    // Define initial parameter vector

    // Set up simulated annealing solver
    // An alternative random number generator (RNG) can be provided to `new_with_rng`:
    // SimulatedAnnealing::new_with_rng(temp, Xoshiro256PlusPlus::from_entropy())?
    let solver = SimulatedAnnealing::new(temperature)?
        // Optional: Define temperature function (defaults to `SATempFunc::TemperatureFast`)
        .with_temp_func(SATempFunc::Boltzmann)
        /////////////////////////
        // Stopping criteria   //
        /////////////////////////
        // Optional: stop if there was no new best solution after 1000 iterations
        .with_stall_best(1000);
    // Optional: stop if there was no accepted solution after 1000 iterations
    // .with_stall_accepted(1000);
    /////////////////////////
    // Reannealing         //
    /////////////////////////
    // Optional: Reanneal after 1000 iterations (resets temperature to initial temperature)
    // .with_reannealing_fixed(1000)
    // Optional: Reanneal after no accepted solution has been found for `iter` iterations
    // .with_reannealing_accepted(500)
    // Optional: Start reannealing after no new best solution has been found for 800 iterations
    // .with_reannealing_best(800);

    /////////////////////////
    // Run solver          //
    /////////////////////////
    let res = Executor::new(operator, solver)
        .configure(|state| {
            state
                .param(initial_guess.to_owned())
                // Optional: Set maximum number of iterations (defaults to `std::u64::MAX`)
                .max_iters(10_000)
                // Optional: Set target cost function value (defaults to `std::f64::NEG_INFINITY`)
                .target_cost(0.0)
        })
        // Optional: Attach a observer
        // .add_observer(SlogLogger::term(), ObserverMode::Never)
        .run()?;

    let final_state = res.state();
    let best_permutation = final_state
        .best_param
        .clone()
        .expect("No best param in final anneling state.");
    Ok((final_state.best_cost, best_permutation))
}
