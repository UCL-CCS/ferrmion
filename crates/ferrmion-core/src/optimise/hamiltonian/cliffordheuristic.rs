//! Clifford Heuristic optimisation of Hamiltonians.
//!
//! Based on http://arxiv.org/abs/2502.11933
//! This differs somewhat in that we don't use the same sampling
//! or temperature schedule.
//! Given that their original paper discussed computations taking multiple days
//! the version below is significantly faster to get to decent solutions.
//!
//! Changes to make it match:
//! - Anneal needs to be updated to only add one operator at a time
//! - Acceptance of new operators takes probability
//!   $e^{-\beta(t)[C(G^{\dagger}BG - B)]}$
//!   where B is the previous generation hamiltonian
//!   and G is the sampled clifford operator.
use crate::hamiltonians::SymplecticHamiltonian;
use crate::operators::{CliffordOperator, CoefficientPauliWeight, PauliWeight};
use argmin::{
    core::{CostFunction, Error, Executor},
    solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use log::info;
use rand::{distr::Uniform, prelude::*};
use rand_core_legacy::SeedableRng as LegacySeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro_legacy::Xoshiro256PlusPlus as LegacyXoshiro256PlusPlus;
use std::sync::Arc;
use std::sync::Mutex;

struct CliffordHeuristic {
    hamiltonian: SymplecticHamiltonian,
    coefficient_weighted: bool,
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl CliffordHeuristic {
    fn new(hamiltonian: SymplecticHamiltonian, coefficient_weighted: bool, seed: u64) -> Self {
        CliffordHeuristic {
            hamiltonian,
            coefficient_weighted,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::seed_from_u64(seed))),
        }
    }
}

impl CostFunction for CliffordHeuristic {
    type Param = Vec<CliffordOperator>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let mut ham = self.hamiltonian.clone();
        ham = apply_clifford_chain(ham, param.as_slice());

        let weight = match self.coefficient_weighted {
            true => ham.coeff_pauli_weight(),
            false => ham.pauli_weight() as f64,
        };
        Ok(weight)
    }
}

impl Anneal for CliffordHeuristic {
    type Param = Vec<CliffordOperator>;
    type Output = Vec<CliffordOperator>;
    type Float = f64;

    fn anneal(
        &self,
        param: &Vec<CliffordOperator>,
        temp: f64,
    ) -> Result<Vec<CliffordOperator>, Error> {
        let mut next_param = param.to_vec();
        let n_qubits = self.hamiltonian.n_qubits();
        let mut rng = self.rng.lock().unwrap();
        let distr = Uniform::try_from(0..n_qubits).unwrap();
        let op_flag_distr = Uniform::try_from(0..=3).unwrap();
        let temp_int = temp.floor() as usize + 1;

        for _ in 0..temp_int {
            let control = rng.sample(distr);
            let target = rng.sample(distr);
            if control == target {
                continue;
            }

            match rng.sample(op_flag_distr) {
                // H
                0 => {
                    next_param.push(CliffordOperator::H(control));
                    next_param.push(CliffordOperator::CNOT { control, target });
                }
                1 => {
                    next_param.push(CliffordOperator::S(control));
                    next_param.push(CliffordOperator::CNOT { control, target });
                }
                2 => {
                    next_param.push(CliffordOperator::CNOT { control, target });
                }
                // should be unreachable, but can just continue to
                // avoid a panic.
                _ => {
                    continue;
                }
            }
        }
        Ok(next_param)
    }
}

pub fn clifford_heuristic_optimisation(
    hamiltonian: SymplecticHamiltonian,
    temperature: f64,
    coefficient_weighted: bool,
    seed: u64,
) -> Result<(f64, Vec<CliffordOperator>), Error> {
    info!("Beginning clifford heuristic encoding optimisation.");
    // Derive two decorrelated child seeds from the user-provided seed so the
    // permutation-move RNG (inside `OptimalEnumeration`) and the argmin solver
    // RNG (which drives acceptance decisions) consume independent streams
    // while remaining fully reproducible.
    let mut master = Xoshiro256PlusPlus::seed_from_u64(seed);
    let operator_seed: u64 = master.next_u64();
    let solver_seed: u64 = master.next_u64();

    let operator = CliffordHeuristic::new(hamiltonian, coefficient_weighted, operator_seed);

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
        .configure(|state| state.param(Vec::<CliffordOperator>::new()).max_iters(1_000))
        .run()?;

    let final_state = res.state();
    let best_clifford_chain = final_state
        .best_param
        .clone()
        .expect("No best param in final anneling state.");

    info!("Best clifford operators: {:#?}", best_clifford_chain);

    Ok((final_state.best_cost, best_clifford_chain))
}

/// Apply a sequence of (control, target) Clifford gate pairs
///  to a [`SymplecticHamiltonian`].
///
/// Each pair applies H on `control` then CNOT with `control` as control,
/// matching the `CliffordHeuristic` cost function.
pub fn apply_clifford_chain(
    mut hamiltonian: SymplecticHamiltonian,
    chain: &[CliffordOperator],
) -> SymplecticHamiltonian {
    use CliffordOperator::{CNOT, H, S};
    let mut transpose = hamiltonian.operators.transpose();
    for op in chain {
        match op {
            H(idx) => transpose.haddamard(*idx),
            S(idx) => transpose.phasegate(*idx),
            CNOT { control, target } => transpose.cnot(*control, *target),
        }
    }
    hamiltonian
}
