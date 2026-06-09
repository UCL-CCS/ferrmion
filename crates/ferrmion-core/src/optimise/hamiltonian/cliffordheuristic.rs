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
use crate::operators::{
    Clifford, CoefficientPauliWeight, PauliWeight, SymplecticMatrixTranspose,
};
use crate::optimise::encoding::AnnealingParameters;
use argmin::{
    core::{CostFunction, Error, Executor},
    solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use log::info;
use rand::{
    distr::{weighted::WeightedIndex, Distribution, Uniform},
    prelude::*,
};
use rand_core_legacy::SeedableRng as LegacySeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro_legacy::Xoshiro256PlusPlus as LegacyXoshiro256PlusPlus;
use std::clone::Clone;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Mutex;
use thiserror::Error;

/// Errors produced by the Clifford-heuristic module.
#[derive(Debug, Error)]
pub enum CliffordHeuristicError {
    #[error("unknown clifford_subset '{0}'; expected one of all, c, ch, cs, chs, vp")]
    UnknownSubset(String),
}

/// Define a subset of clifford operators to sample from.
///
/// All: Sample any clifford gate H_i, S_i, CX_ij
/// CH: Sample H_i CX_ij
/// CS: Sample S_i CX_ij
/// CHS: Sample H_i CX_ij and S_i CX_ij
/// VP: Vacuum-preserving — uniform pick between a single S_i or a single
///     CX_ij. Both gates stabilise the |0…0⟩ vacuum, so chains drawn from
///     this distribution preserve any encoding's vacuum.
#[derive(Clone)]
pub enum CliffordSubset {
    All,
    C,
    CH,
    CS,
    CHS,
    VP,
}

impl CliffordSubset {
    fn sample(
        &self,
        rng: &mut Xoshiro256PlusPlus,
        control: usize,
        target: usize,
    ) -> Vec<Clifford> {
        match self {
            CliffordSubset::All => {
                let op = rng.random_range(0..3);
                match op {
                    0 => vec![Clifford::H(control)],
                    1 => vec![Clifford::S(control)],
                    2 => vec![Clifford::CNOT { control, target }],
                    _ => unreachable!(),
                }
            }
            CliffordSubset::C => vec![Clifford::CNOT { control, target }],
            CliffordSubset::CH => vec![
                Clifford::H(control),
                Clifford::CNOT { control, target },
            ],
            CliffordSubset::CS => vec![
                Clifford::S(control),
                Clifford::CNOT { control, target },
            ],
            CliffordSubset::CHS => {
                let op = rng.random_range(0..2);
                match op {
                    0 => vec![
                        Clifford::S(control),
                        Clifford::CNOT { control, target },
                    ],
                    1 => vec![
                        Clifford::H(control),
                        Clifford::CNOT { control, target },
                    ],
                    _ => unreachable!(),
                }
            }
            CliffordSubset::VP => {
                let op = rng.random_range(0..2);
                match op {
                    0 => vec![Clifford::S(control)],
                    _ => vec![Clifford::CNOT { control, target }],
                }
            }
        }
    }
}

impl FromStr for CliffordSubset {
    type Err = CliffordHeuristicError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "all" => Ok(CliffordSubset::All),
            "c" => Ok(CliffordSubset::C),
            "ch" => Ok(CliffordSubset::CH),
            "cs" => Ok(CliffordSubset::CS),
            "chs" => Ok(CliffordSubset::CHS),
            "vp" => Ok(CliffordSubset::VP),
            other => Err(CliffordHeuristicError::UnknownSubset(other.to_string())),
        }
    }
}

struct CliffordHeuristic<'ham> {
    hamiltonian: &'ham mut SymplecticHamiltonian,
    coefficient_weighted: bool,
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
    subsystem: Vec<usize>,
    clifford_subset: CliffordSubset,
}

impl<'ham> CliffordHeuristic<'ham> {
    fn new(
        hamiltonian: &'ham mut SymplecticHamiltonian,
        coefficient_weighted: bool,
        seed: u64,
        subsystem: Vec<usize>,
        clifford_subset: Option<CliffordSubset>,
    ) -> Self {
        CliffordHeuristic {
            hamiltonian,
            coefficient_weighted,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::seed_from_u64(seed))),
            subsystem,
            clifford_subset: clifford_subset.unwrap_or(CliffordSubset::CHS),
        }
    }
}

impl<'ham> CostFunction for CliffordHeuristic<'ham> {
    type Param = Vec<Clifford>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        // Cloning is fine for now but it would be better
        // to apply the chain and then uncompute it if the cost
        // is not improved.
        let mut ham = self.hamiltonian.clone();
        ham.operators.apply_clifford_chain(param.as_slice());

        let weight = match self.coefficient_weighted {
            true => ham.coeff_pauli_weight(),
            false => ham.pauli_weight() as f64,
        };
        Ok(weight)
    }
}

impl<'ham> Anneal for CliffordHeuristic<'ham> {
    type Param = Vec<Clifford>;
    type Output = Vec<Clifford>;
    type Float = f64;

    fn anneal(
        &self,
        param: &Vec<Clifford>,
        temp: f64,
    ) -> Result<Vec<Clifford>, Error> {
        let mut next_param = param.to_vec();
        let mut rng = self.rng.lock().unwrap();

        let distr = Uniform::try_from(0..self.subsystem.len()).unwrap();
        let temp_int = temp.floor() as usize + 1;

        for _ in 0..temp_int {
            let control = self.subsystem[rng.sample(distr)];
            let target = self.subsystem[rng.sample(distr)];
            if control == target {
                continue;
            }
            next_param.extend(self.clifford_subset.sample(&mut rng, control, target));
        }
        Ok(next_param)
    }
}

/// Result of a [`clifford_heuristic_optimisation`] run.
pub struct CliffordHeuristicResult {
    /// Final cost (Pauli weight or coefficient-weighted Pauli weight) of the best chain.
    pub cost: f64,
    /// The best Clifford operator chain found.
    pub chain: Vec<Clifford>,
}

/// Optimise a [`SymplecticHamiltonian`] using the clifford heuristic method.
///
/// If a subsystem is not provided, the full Hamiltonian is optimised.
///
/// Returns the optimised energy and the corresponding Clifford operator sequence.
pub fn clifford_heuristic_optimisation(
    hamiltonian: &mut SymplecticHamiltonian,
    temperature: f64,
    coefficient_weighted: bool,
    seed: u64,
    subsystem: Option<Vec<usize>>,
    clifford_subset: Option<CliffordSubset>,
) -> Result<CliffordHeuristicResult, Error> {
    info!("Beginning clifford heuristic encoding optimisation.");
    // Derive two decorrelated child seeds from the user-provided seed so the
    // permutation-move RNG (inside `OptimalEnumeration`) and the argmin solver
    // RNG (which drives acceptance decisions) consume independent streams
    // while remaining fully reproducible.
    let mut master = Xoshiro256PlusPlus::seed_from_u64(seed);
    let operator_seed: u64 = master.next_u64();
    let solver_seed: u64 = master.next_u64();

    let subsystem = subsystem.unwrap_or((0..hamiltonian.n_qubits()).collect());

    let operator = CliffordHeuristic::new(
        hamiltonian,
        coefficient_weighted,
        operator_seed,
        subsystem,
        clifford_subset,
    );

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
        .configure(|state| state.param(Vec::<Clifford>::new()).max_iters(1_000))
        .run()?;

    let final_state = res.state();
    let best_clifford_chain = final_state
        .best_param
        .clone()
        .expect("No best param in final anneling state.");

    info!("Best clifford operators: {:#?}", best_clifford_chain);

    Ok(CliffordHeuristicResult {
        cost: final_state.best_cost,
        chain: best_clifford_chain,
    })
}

/// Sampling heuristic to use for [`rsd`].
///
/// When providing a Custom sampler,
/// a function is needed which takes a [`SymplecticHamiltonian`]
/// and returns a probability for each qubit index to be sampled.
pub enum SubsystemSampler {
    /// Use all qubit indices.
    FullSystem,
    /// Sample uniformly.
    Uniform,
    /// Sample according to Hamming weight distribution.
    Hamming,
    /// Determine sampling probabilities from a custom function.
    Custom(fn(&SymplecticHamiltonian) -> Vec<f64>),
}

impl SubsystemSampler {
    fn get_subsystem(
        &self,
        hamiltonian: &mut SymplecticHamiltonian,
        dimension: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Vec<usize> {
        match self {
            SubsystemSampler::FullSystem => (0..hamiltonian.n_qubits()).collect(),
            SubsystemSampler::Uniform => WeightedIndex::new(vec![1; hamiltonian.n_qubits()])
                .expect("Should be able to make uniform distribution.")
                .sample_iter(rng)
                .take(dimension)
                .collect(),
            SubsystemSampler::Hamming => {
                let transpose: SymplecticMatrixTranspose = hamiltonian.operators.transpose();
                let weights = transpose.hamming_weights();
                WeightedIndex::new(weights)
                    .expect("Should be able to make hamming weight distribution.")
                    .sample_iter(rng)
                    .take(dimension)
                    .collect()
            }
            SubsystemSampler::Custom(f) => {
                let probs = f(hamiltonian);
                WeightedIndex::new(probs)
                    .expect("Should be able to make custom distribution.")
                    .sample_iter(rng)
                    .take(dimension)
                    .collect()
            }
        }
    }
}

pub fn randomised_subsystem_descent(
    mut hamiltonian: SymplecticHamiltonian,
    params: AnnealingParameters,
    coefficient_weighted: bool,
    sampler: SubsystemSampler,
    subsystem_dimension: usize,
    clifford_subset: Option<CliffordSubset>,
) -> SymplecticHamiltonian {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(params.seed);

    for _ in 0..params.max_iterations {
        let subsystem = sampler.get_subsystem(&mut hamiltonian, subsystem_dimension, &mut rng);
        let result = clifford_heuristic_optimisation(
            &mut hamiltonian,
            params.temperature,
            coefficient_weighted,
            rng.next_u64(),
            Some(subsystem),
            clifford_subset.clone(),
        )
        .expect("Should be able to optimise subsystem.");
        hamiltonian.operators.apply_clifford_chain(&result.chain);
    }
    hamiltonian
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::SymplecticMatrix;
    use ndarray::array;

    #[test]
    fn vp_subset_emits_only_s_and_cnot() {
        // Build a small SymplecticHamiltonian with two terms on 3 qubits.
        let x_block = array![[true, false, false], [false, true, false]];
        let z_block = array![[false, true, true], [true, false, true]];
        let mut sym_ham = SymplecticHamiltonian::new(
            SymplecticMatrix::new(x_block, z_block),
            ndarray::array![1.0, 0.5],
        );

        let result = clifford_heuristic_optimisation(
            &mut sym_ham,
            3.0,
            false,
            42,
            None,
            Some(CliffordSubset::VP),
        )
        .expect("vp optimisation should succeed");

        for op in &result.chain {
            assert!(
                matches!(op, Clifford::S(_) | Clifford::CNOT { .. }),
                "VP chain emitted non-stabilising gate: {op:?}"
            );
        }
    }
}
