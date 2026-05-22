//! Simulated annealing optimisation of Fermion-Qubit Encodings.

use crate::encode::majorana::Encode;
use crate::encode::majorana::MajoranaEncoding;
use crate::operators::{CoefficientPauliWeight, MajoranaSparse, PauliWeight};
use argmin::{
    core::{CostFunction, Error, Executor},
    solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use log::info;
use ndarray::Array1;
use ndarray::ArrayView1;
use rand::{distr::Uniform, prelude::*};
use rand_core_legacy::SeedableRng as LegacySeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro_legacy::Xoshiro256PlusPlus as LegacyXoshiro256PlusPlus;
use std::sync::{Arc, Mutex};

struct OptimalEnumeration {
    msparse: MajoranaSparse,
    encoding: MajoranaEncoding,
    coefficient_weighted: bool,
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl OptimalEnumeration {
    fn new(
        msparse: MajoranaSparse,
        encoding: MajoranaEncoding,
        coefficient_weighted: bool,
        seed: u64,
    ) -> Self {
        OptimalEnumeration {
            msparse,
            encoding,
            coefficient_weighted,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::seed_from_u64(seed))),
        }
    }
}

impl CostFunction for OptimalEnumeration {
    type Param = Array1<usize>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let enumerated_encoding = self.encoding.apply_mode_enumeration(param.to_vec());
        let qham = enumerated_encoding.encode(&self.msparse);
        let weight = match self.coefficient_weighted {
            true => qham.coeff_pauli_weight(),
            false => qham.pauli_weight() as f64,
        };
        Ok(weight)
    }
}

impl Anneal for OptimalEnumeration {
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

/// Optimise fermionic mode enumeration using simulated annealing.
///
/// Searches for a permutation of mode indices that minimises the Pauli weight
/// (or coefficient-weighted Pauli weight) of the encoded qubit Hamiltonian.
///
/// # Arguments
///
/// * `msparse` - The fermionic Hamiltonian in Majorana sparse form.
/// * `encoding` - The Majorana encoding to use.
/// * `temperature` - Initial temperature for the annealing schedule.
/// * `initial_guess` - Starting permutation of mode indices.
/// * `coefficient_weighted` - If `true`, minimise coefficient-weighted Pauli weight.
/// * `seed` - Seed for the `Xoshiro256PlusPlus` RNG driving permutation moves.
///
/// # Returns
///
/// A tuple of `(best_cost, best_permutation)`.
pub fn anneal_enumerations(
    msparse: MajoranaSparse,
    encoding: MajoranaEncoding,
    temperature: f64,
    initial_guess: ArrayView1<usize>,
    coefficient_weighted: bool,
    seed: u64,
) -> Result<(f64, Array1<usize>), Error> {
    assert_eq!(
        initial_guess.len(),
        encoding.operators.ipowers.len() / 2,
        "Initial enumeration length is not n_modes"
    );

    // Derive two decorrelated child seeds from the user-provided seed so the
    // permutation-move RNG (inside `OptimalEnumeration`) and the argmin solver
    // RNG (which drives acceptance decisions) consume independent streams
    // while remaining fully reproducible.
    let mut master = Xoshiro256PlusPlus::seed_from_u64(seed);
    let operator_seed: u64 = master.next_u64();
    let solver_seed: u64 = master.next_u64();

    let operator = OptimalEnumeration::new(msparse, encoding, coefficient_weighted, operator_seed);

    // Set up simulated annealing solver with our seeded RNG. The default
    // `SimulatedAnnealing::new` would seed from entropy, breaking
    // reproducibility. We use the rand_xoshiro 0.6 RNG type because argmin
    // 0.10's solver expects the older `rand_core 0.6::RngCore` trait.
    let solver = SimulatedAnnealing::new_with_rng(
        temperature,
        LegacyXoshiro256PlusPlus::seed_from_u64(solver_seed),
    )?
    .with_temp_func(SATempFunc::Boltzmann)
    .with_stall_best(250);

    let res = Executor::new(operator, solver)
        .configure(|state| state.param(initial_guess.to_owned()).max_iters(1_000))
        .run()?;

    let final_state = res.state();
    let best_permutation = final_state
        .best_param
        .clone()
        .expect("No best param in final anneling state.");
    Ok((final_state.best_cost, best_permutation))
}

struct CliffordHeuristic {
    msparse: MajoranaSparse,
    encoding: MajoranaEncoding,
    coefficient_weighted: bool,
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl CliffordHeuristic {
    fn new(
        msparse: MajoranaSparse,
        encoding: MajoranaEncoding,
        coefficient_weighted: bool,
        seed: u64,
    ) -> Self {
        CliffordHeuristic {
            msparse,
            encoding,
            coefficient_weighted,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::seed_from_u64(seed))),
        }
    }
}

impl CostFunction for CliffordHeuristic {
    // Lets start with CNOT_{ij}H_i as in
    // 10.48550/arXiv.2502.11933
    // Each row is a permutation of qubit indices.
    // To avoid clashes, read the row from left to right
    // taking pairs of indices as [(ij), (kl), (mn), ...]
    // First apply Hadamards on the left indices
    // Then apply CNOT with the left as control
    type Param = Vec<(usize, usize)>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let mut copy_encoding: MajoranaEncoding = self
            .encoding
            .apply_mode_enumeration((0..2 * self.encoding.n_modes).collect());
        let mut transpose = copy_encoding.operators.transpose();
        for (control, target) in param.iter() {
            transpose.haddamard(*control);
            transpose.cnot(*control, *target);
        }

        let qham = copy_encoding.encode(&self.msparse);
        let weight = match self.coefficient_weighted {
            true => qham.coeff_pauli_weight(),
            false => qham.pauli_weight() as f64,
        };
        Ok(weight)
    }
}

impl Anneal for CliffordHeuristic {
    type Param = Vec<(usize, usize)>;
    type Output = Vec<(usize, usize)>;
    type Float = f64;

    fn anneal(&self, param: &Vec<(usize, usize)>, temp: f64) -> Result<Vec<(usize, usize)>, Error> {
        let mut next_param = param.clone();
        let n_modes = next_param.len();
        let mut rng = self.rng.lock().unwrap();
        let distr = Uniform::try_from(0..n_modes).unwrap();
        let temp_int = temp.floor() as usize + 1;

        for _ in 0..temp_int {
            let control = rng.sample(distr);
            let target = rng.sample(distr);
            if control != target {
                next_param.push((control, target));
            }
        }
        Ok(next_param)
    }
}

pub fn clifford_heuristic_optimisation(
    msparse: MajoranaSparse,
    encoding: MajoranaEncoding,
    temperature: f64,
    coefficient_weighted: bool,
    seed: u64,
) -> Result<(f64, Vec<(usize, usize)>), Error> {
    info!("Beginning clifford heuristic encoding optimisation.");
    // Derive two decorrelated child seeds from the user-provided seed so the
    // permutation-move RNG (inside `OptimalEnumeration`) and the argmin solver
    // RNG (which drives acceptance decisions) consume independent streams
    // while remaining fully reproducible.
    let mut master = Xoshiro256PlusPlus::seed_from_u64(seed);
    let operator_seed: u64 = master.next_u64();
    let solver_seed: u64 = master.next_u64();

    let operator = CliffordHeuristic::new(msparse, encoding, coefficient_weighted, operator_seed);

    // Set up simulated annealing solver with our seeded RNG. The default
    // `SimulatedAnnealing::new` would seed from entropy, breaking
    // reproducibility. We use the rand_xoshiro 0.6 RNG type because argmin
    // 0.10's solver expects the older `rand_core 0.6::RngCore` trait.
    let solver = SimulatedAnnealing::new_with_rng(
        temperature,
        LegacyXoshiro256PlusPlus::seed_from_u64(solver_seed),
    )?
    .with_temp_func(SATempFunc::Boltzmann)
    .with_stall_best(250);

    let res = Executor::new(operator, solver)
        .configure(|state| state.param(Vec::<(usize, usize)>::new()).max_iters(1_000))
        .run()?;

    let final_state = res.state();
    let best_clifford_chain = final_state
        .best_param
        .clone()
        .expect("No best param in final anneling state.");

    info!("Best clifford operators: {:#?}", best_clifford_chain);

    Ok((final_state.best_cost, best_clifford_chain))
}
