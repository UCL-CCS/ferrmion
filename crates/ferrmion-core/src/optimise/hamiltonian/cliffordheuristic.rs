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
use crate::operators::{Clifford, DenseBlock, SymplecticMatrixTranspose};
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
use std::collections::HashMap;
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
    fn sample(&self, rng: &mut Xoshiro256PlusPlus, control: usize, target: usize) -> Vec<Clifford> {
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
            CliffordSubset::CH => vec![Clifford::H(control), Clifford::CNOT { control, target }],
            CliffordSubset::CS => vec![Clifford::S(control), Clifford::CNOT { control, target }],
            CliffordSubset::CHS => {
                let op = rng.random_range(0..2);
                match op {
                    0 => vec![Clifford::S(control), Clifford::CNOT { control, target }],
                    1 => vec![Clifford::H(control), Clifford::CNOT { control, target }],
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

/// Qubit-major transpose restricted to a subsystem's qubit columns.
///
/// [`CliffordHeuristic::anneal`] only ever samples `control`/`target` qubits
/// from the subsystem, so a candidate chain never touches any other qubit.
/// That means the rest of the Hamiltonian's Pauli weight is an additive
/// constant across every candidate in one optimisation run and can be
/// dropped from the cost entirely — this type holds only the subsystem's
/// columns, transposed once up front, so `cost()` can clone and mutate a
/// small buffer instead of re-transposing (and writing back) the whole
/// Hamiltonian on every evaluation.
#[derive(Clone)]
struct SubsystemTranspose {
    /// One term per subsystem qubit (in `local_index` order), each
    /// `words_for(n_rows)` words wide.
    x_transpose: DenseBlock,
    /// Same layout as `x_transpose`.
    z_transpose: DenseBlock,
}

impl SubsystemTranspose {
    /// Restrict `x_block`/`z_block` to `subsystem_qubits`' columns and
    /// transpose once into qubit-major form.
    fn new(x_block: &DenseBlock, z_block: &DenseBlock, subsystem_qubits: &[usize]) -> Self {
        Self {
            x_transpose: x_block.select_indices(subsystem_qubits).transpose(),
            z_transpose: z_block.select_indices(subsystem_qubits).transpose(),
        }
    }

    // Same bit-level identities as `SymplecticMatrixTranspose`
    // (`crate::operators::pauli`), minus i-power bookkeeping: `pauli_weight`/
    // `coeff_pauli_weight` never depend on phase, only on which qubits carry
    // a non-identity Pauli.
    fn haddamard(&mut self, qubit: usize) {
        let x_col = self.x_transpose.get_term(qubit).to_owned_block();
        self.x_transpose
            .set_term(qubit, self.z_transpose.get_term(qubit));
        self.z_transpose.set_term(qubit, x_col.as_ref());
    }

    fn phasegate(&mut self, qubit: usize) {
        let z_new = self
            .z_transpose
            .get_term(qubit)
            .xor(&self.x_transpose.get_term(qubit));
        self.z_transpose.set_term(qubit, z_new.as_ref());
    }

    fn cnot(&mut self, control: usize, target: usize) {
        let x_new = self
            .x_transpose
            .get_term(target)
            .xor(&self.x_transpose.get_term(control));
        self.x_transpose.set_term(target, x_new.as_ref());
        let z_new = self
            .z_transpose
            .get_term(control)
            .xor(&self.z_transpose.get_term(target));
        self.z_transpose.set_term(control, z_new.as_ref());
    }

    /// Total Pauli weight contributed by the subsystem's qubits alone.
    fn pauli_weight(&self) -> usize {
        (0..self.x_transpose.n_terms())
            .map(|q| {
                self.x_transpose
                    .get_term(q)
                    .or_count_ones(&self.z_transpose.get_term(q))
            })
            .sum()
    }

    /// Per-Hamiltonian-term Pauli weight contributed by the subsystem's
    /// qubits alone, indexed the same as `SymplecticHamiltonian::coefficients`.
    fn per_row_weight(&self) -> Vec<usize> {
        let mut weights = vec![0usize; self.x_transpose.n_indices()];
        for q in 0..self.x_transpose.n_terms() {
            let combined = self
                .x_transpose
                .get_term(q)
                .or(&self.z_transpose.get_term(q));
            for row in combined.iter_ones() {
                weights[row] += 1;
            }
        }
        weights
    }
}

struct CliffordHeuristic<'ham> {
    hamiltonian: &'ham mut SymplecticHamiltonian,
    coefficient_weighted: bool,
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
    subsystem: Vec<usize>,
    clifford_subset: CliffordSubset,
    /// Physical qubit index -> local column index into `base_transpose`.
    local_index: HashMap<usize, usize>,
    /// Subsystem-restricted qubit-major transpose, computed once and cloned
    /// per `cost()` evaluation. See [`SubsystemTranspose`].
    base_transpose: SubsystemTranspose,
}

impl<'ham> CliffordHeuristic<'ham> {
    fn new(
        hamiltonian: &'ham mut SymplecticHamiltonian,
        coefficient_weighted: bool,
        seed: u64,
        subsystem: Vec<usize>,
        clifford_subset: Option<CliffordSubset>,
    ) -> Self {
        // `subsystem` may repeat physical qubits (samplers like `Uniform`/
        // `Hamming` draw with replacement to bias `anneal`'s control/target
        // sampling), but the restricted transpose only needs each physical
        // qubit's column once.
        let mut subsystem_qubits = subsystem.clone();
        subsystem_qubits.sort_unstable();
        subsystem_qubits.dedup();

        let local_index: HashMap<usize, usize> = subsystem_qubits
            .iter()
            .enumerate()
            .map(|(local, &qubit)| (qubit, local))
            .collect();

        let base_transpose = SubsystemTranspose::new(
            &hamiltonian.operators.x_block,
            &hamiltonian.operators.z_block,
            &subsystem_qubits,
        );

        CliffordHeuristic {
            hamiltonian,
            coefficient_weighted,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::seed_from_u64(seed))),
            subsystem,
            clifford_subset: clifford_subset.unwrap_or(CliffordSubset::CHS),
            local_index,
            base_transpose,
        }
    }
}

impl<'ham> CostFunction for CliffordHeuristic<'ham> {
    type Param = Vec<Clifford>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        // Reuse the pre-transposed subsystem buffer: clone (cheap — it only
        // spans the subsystem's qubits) and replay the chain on it directly,
        // rather than cloning the whole Hamiltonian and transposing it (and
        // transposing back) on every evaluation.
        let mut state = self.base_transpose.clone();
        for op in param {
            match op {
                Clifford::H(idx) => state.haddamard(self.local_index[idx]),
                Clifford::S(idx) => state.phasegate(self.local_index[idx]),
                Clifford::CNOT { control, target } => {
                    state.cnot(self.local_index[control], self.local_index[target])
                }
            }
        }

        // This is the subsystem's contribution to the (coefficient-weighted)
        // Pauli weight, not the whole Hamiltonian's — the rest is an additive
        // constant across every candidate in this run (see
        // `SubsystemTranspose`), so minimising this is equivalent to
        // maximising the reduction in the subsystem's own weight.
        let weight = if self.coefficient_weighted {
            state
                .per_row_weight()
                .iter()
                .zip(&self.hamiltonian.coefficients)
                .map(|(&w, coeff)| coeff.abs() * w as f64)
                .sum()
        } else {
            state.pauli_weight() as f64
        };
        Ok(weight)
    }
}

impl<'ham> Anneal for CliffordHeuristic<'ham> {
    type Param = Vec<Clifford>;
    type Output = Vec<Clifford>;
    type Float = f64;

    fn anneal(&self, param: &Vec<Clifford>, temp: f64) -> Result<Vec<Clifford>, Error> {
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
    use crate::operators::{CoefficientPauliWeight, PauliWeight, SymplecticMatrix};
    use ndarray::array;

    /// The subsystem-restricted `cost()` must agree with the full Hamiltonian's
    /// weight *delta* exactly — not just up to a constant — since the constant
    /// (the Pauli weight contributed by qubits outside the subsystem) is what
    /// the subsystem-only formulation is designed to skip computing at all.
    #[test]
    fn subsystem_cost_matches_full_weight_delta() {
        // 6 qubits, a handful of terms with mixed X/Z/Y content so H/S/CNOT
        // all have something to do on the sampled subsystem qubits.
        let x_block = array![
            [true, false, false, true, false, true],
            [false, true, true, false, false, false],
            [true, true, false, false, true, false],
        ];
        let z_block = array![
            [false, true, false, true, true, false],
            [true, false, true, false, false, true],
            [false, false, true, true, false, true],
        ];
        let coefficients = array![1.0, -2.5, 0.5];
        let hamiltonian = SymplecticHamiltonian::new(
            SymplecticMatrix::from_arrays(x_block, z_block),
            coefficients,
        );

        // Includes a duplicate (qubit 1 twice), matching what `Uniform`/`Hamming`
        // samplers can actually produce.
        let subsystem = vec![1, 3, 4, 1];
        let chain = vec![
            Clifford::H(1),
            Clifford::CNOT {
                control: 1,
                target: 3,
            },
            Clifford::S(4),
            Clifford::CNOT {
                control: 4,
                target: 1,
            },
        ];

        for coefficient_weighted in [false, true] {
            let mut ham_for_operator = hamiltonian.clone();
            let operator = CliffordHeuristic::new(
                &mut ham_for_operator,
                coefficient_weighted,
                7,
                subsystem.clone(),
                None,
            );

            let subsystem_baseline = operator.cost(&Vec::new()).unwrap();
            let subsystem_after = operator.cost(&chain).unwrap();

            let full_baseline = if coefficient_weighted {
                hamiltonian.coeff_pauli_weight()
            } else {
                hamiltonian.pauli_weight() as f64
            };
            let mut full_after_ham = hamiltonian.clone();
            full_after_ham.operators.apply_clifford_chain(&chain);
            let full_after = if coefficient_weighted {
                full_after_ham.coeff_pauli_weight()
            } else {
                full_after_ham.pauli_weight() as f64
            };

            let subsystem_delta = subsystem_after - subsystem_baseline;
            let full_delta = full_after - full_baseline;
            assert!(
                (subsystem_delta - full_delta).abs() < 1e-9,
                "coefficient_weighted={coefficient_weighted}: subsystem delta {subsystem_delta} != full delta {full_delta}"
            );
        }
    }

    #[test]
    fn vp_subset_emits_only_s_and_cnot() {
        // Build a small SymplecticHamiltonian with two terms on 3 qubits.
        let x_block = array![[true, false, false], [false, true, false]];
        let z_block = array![[false, true, true], [true, false, true]];
        let mut sym_ham = SymplecticHamiltonian::new(
            SymplecticMatrix::from_arrays(x_block, z_block),
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
