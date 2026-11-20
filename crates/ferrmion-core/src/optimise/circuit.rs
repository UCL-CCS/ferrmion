//! TOPP-HATT as a compiler for fermionic circuits.
//!
//! A [`MajoranaCircuit`] is treated as a Hamiltonian in its own right, with
//! three differences: its terms are ordered, repeated terms are allowed, and
//! its coefficients are rotation angles modulo 2π. Compilation proceeds as:
//!
//! 1. Sum the absolute angle applied with each distinct operator, giving `V`
//!    ([`MajoranaCircuit::operator_weights`]).
//! 2. Split operators qDRIFT-style: starting from the heaviest, give an
//!    operator one more piece while the total number of pieces is below
//!    `split_factor · len(V)` ([`split_pieces`], greedy on the largest `V_j/k_j`).
//! 3. Run TOPP-HATT with each operator counting `k_j` times. Splitting into
//!    `k_j` uncombined copies is the same as giving one term the integer
//!    weight `k_j`, so the copies are never materialised; the
//!    [`ValueMerge::Additive`] backend makes sure terms that become identical
//!    during reduction add their counts instead of collapsing.
//! 4. Encode the circuit gate by gate into an ordered [`PauliCircuit`].

use crate::encode::majorana::{MajoranaEncoding, MajoranaEncodingError, TryEncode};
use crate::encode::ternarytree::{TernaryTree, TernaryTreeError};
use crate::operators::{MajoranaCircuit, PauliCircuit, MAX_MAJORANAS};
use thiserror::Error;
use tinyvec::ArrayVec;

use super::{
    split_pieces, topphatt, NodeOrderHeuristic, ToppHattError, WeightedBackendError,
    WeightedSparseBackend,
};

/// Default ratio of split pieces to distinct circuit operators.
pub const DEFAULT_SPLIT_FACTOR: f64 = 4.0;

/// Errors produced while compiling a circuit.
#[derive(Debug, Error)]
pub enum CircuitCompileError {
    #[error("split_factor must be finite and at least 1; got {0}.")]
    InvalidSplitFactor(f64),
    #[error(transparent)]
    Weighted(#[from] WeightedBackendError),
    #[error(transparent)]
    ToppHatt(#[from] ToppHattError),
    #[error(transparent)]
    Tree(#[from] TernaryTreeError),
    #[error(transparent)]
    Encoding(#[from] MajoranaEncodingError),
}

/// Number of split pieces for each distinct operator of a circuit.
///
/// Returns the distinct operators (first-occurrence order) and their piece
/// counts, which sum to `ceil(split_factor · n_distinct)`.
pub fn circuit_split_pieces(
    circuit: &MajoranaCircuit,
    split_factor: f64,
) -> Result<(Vec<ArrayVec<[u16; MAX_MAJORANAS]>>, Vec<u64>), CircuitCompileError> {
    if !split_factor.is_finite() || split_factor < 1.0 {
        return Err(CircuitCompileError::InvalidSplitFactor(split_factor));
    }
    let (terms, weights) = circuit.operator_weights();
    let budget = (split_factor * terms.len() as f64).ceil() as usize;
    let pieces = split_pieces(&weights, budget)?;
    Ok((terms, pieces.into_iter().map(|k| k as u64).collect()))
}

/// Build the additive weighted TOPP-HATT backend for a circuit.
pub fn circuit_backend(
    circuit: &MajoranaCircuit,
    split_factor: f64,
) -> Result<WeightedSparseBackend, CircuitCompileError> {
    let (terms, pieces) = circuit_split_pieces(circuit, split_factor)?;
    Ok(WeightedSparseBackend::from_multiplicities(terms, pieces))
}

/// Compile a [`MajoranaCircuit`] with TOPP-HATT.
///
/// Optimises the leaf assignment of `tree` (keeping its topology) for the
/// split, angle-weighted circuit operators, then encodes the circuit in order.
///
/// # Examples
///
/// ```
/// use ferrmion_core::encode::ternarytree::TernaryTree;
/// use ferrmion_core::operators::MajoranaCircuit;
/// use ferrmion_core::optimise::{compile_circuit, NodeOrderHeuristic};
/// use tinyvec::array_vec;
///
/// let circuit = MajoranaCircuit::new(
///     vec![array_vec!([u16; 7] => 0, 5), array_vec!([u16; 7] => 1, 2)],
///     vec![1.0, 0.1],
///     3,
/// )
/// .unwrap();
/// let (encoding, compiled) = compile_circuit(
///     &circuit,
///     TernaryTree::naive_jordan_wigner(3),
///     3,
///     4.0,
///     false,
///     NodeOrderHeuristic::MinWeight,
/// )
/// .unwrap();
/// assert_eq!(compiled.len(), 2);
/// assert_eq!(encoding.n_modes, 3);
/// ```
pub fn compile_circuit(
    circuit: &MajoranaCircuit,
    tree: TernaryTree,
    n_qubits: usize,
    split_factor: f64,
    parallelize: bool,
    heuristic: NodeOrderHeuristic,
) -> Result<(MajoranaEncoding, PauliCircuit), CircuitCompileError> {
    let backend = circuit_backend(circuit, split_factor)?;
    let tree = topphatt(backend, tree, parallelize, heuristic)?;
    let encoding = tree.build_encoding(n_qubits)?;
    let compiled = encoding.try_encode(circuit)?;
    Ok((encoding, compiled))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{hermitian_phase_power, MajoranaSparse};
    use crate::optimise::Backend;
    use crate::optimise::ValueMerge;
    use ndarray::{linalg::kron, Array2};
    use num_complex::Complex64;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::collections::BTreeSet;
    use std::sync::atomic::AtomicUsize;
    use tinyvec::array_vec;

    type Term = ArrayVec<[u16; 7]>;

    fn term(indices: &[u16]) -> Term {
        let mut av = ArrayVec::new();
        av.extend_from_slice(indices);
        av
    }

    fn random_circuit(n_modes: usize, n_gates: usize, seed: u64) -> MajoranaCircuit {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut terms = Vec::with_capacity(n_gates);
        let mut angles = Vec::with_capacity(n_gates);
        for _ in 0..n_gates {
            let len = [1usize, 2, 3, 4][rng.random_range(0..4)];
            let mut chosen: BTreeSet<u16> = BTreeSet::new();
            while chosen.len() < len {
                chosen.insert(rng.random_range(0..2 * n_modes) as u16);
            }
            // Shuffle so construction has to track the reordering sign.
            let mut t: Vec<u16> = chosen.into_iter().collect();
            if t.len() > 1 && rng.random_bool(0.5) {
                t.swap(0, 1);
            }
            terms.push(term(&t));
            angles.push(rng.random_range(-7.0..7.0));
        }
        MajoranaCircuit::new(terms, angles, n_modes).unwrap()
    }

    fn pauli_matrix(c: char) -> Array2<Complex64> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let data = match c {
            'I' => [o, z, z, o],
            'X' => [z, o, o, z],
            'Y' => [z, -i, i, z],
            'Z' => [o, z, z, -o],
            _ => unreachable!(),
        };
        Array2::from_shape_vec((2, 2), data.to_vec()).unwrap()
    }

    fn pauli_string_matrix(pauli: &str) -> Array2<Complex64> {
        pauli
            .chars()
            .fold(Array2::eye(1), |acc, c| kron(&acc, &pauli_matrix(c)))
    }

    /// `exp(-iθ A)` for an involution `A` (`A² = I`).
    fn involution_rotation(a: &Array2<Complex64>, theta: f64) -> Array2<Complex64> {
        let dim = a.nrows();
        Array2::<Complex64>::eye(dim) * Complex64::new(theta.cos(), 0.0)
            - a * Complex64::new(0.0, theta.sin())
    }

    /// The circuit unitary built directly from the encoding's Majorana matrices.
    fn majorana_unitary(
        encoding: &MajoranaEncoding,
        circuit: &MajoranaCircuit,
    ) -> Array2<Complex64> {
        let dim = 1 << encoding.n_qubits;
        let majorana = |a: u16| {
            let (pauli, ipower) = encoding.operators.view_row(a as usize).to_pauli_string();
            pauli_string_matrix(&pauli) * crate::utils::icount_to_sign(ipower as usize)
        };
        let mut u = Array2::<Complex64>::eye(dim);
        for (t, &theta) in circuit.terms.iter().zip(&circuit.angles) {
            let mut gamma = Array2::<Complex64>::eye(dim);
            for &a in t.iter() {
                gamma = gamma.dot(&majorana(a));
            }
            gamma *= crate::utils::icount_to_sign(hermitian_phase_power(t.len()) as usize);
            u = involution_rotation(&gamma, theta).dot(&u);
        }
        u
    }

    fn pauli_unitary(circuit: &PauliCircuit) -> Array2<Complex64> {
        let dim = 1 << circuit.n_qubits;
        let mut u = Array2::<Complex64>::eye(dim);
        for (p, &theta) in circuit.paulis.iter().zip(&circuit.angles) {
            u = involution_rotation(&pauli_string_matrix(p), theta).dot(&u);
        }
        u
    }

    fn trees(n_modes: usize) -> Vec<TernaryTree> {
        vec![
            TernaryTree::naive_jordan_wigner(n_modes),
            TernaryTree::naive_parity(n_modes),
            TernaryTree::naive_bravyi_kitaev(n_modes),
            TernaryTree::naive_jkmn(n_modes),
        ]
    }

    fn weight(backend: &impl Backend, comb: [u16; 3]) -> usize {
        backend.evaluate_combination(comb, &AtomicUsize::new(usize::MAX))
    }

    #[test]
    fn additive_merge_never_cancels() {
        // γ1γ0 and γ0γ1 differ by a sign: a signed backend cancels them, an
        // additive one counts both gates.
        let terms = vec![term(&[1, 0]), term(&[0, 1])];
        let signed = WeightedSparseBackend::from_values(terms.clone(), vec![2, 2]);
        assert!(signed.values().is_empty());
        let additive = WeightedSparseBackend::from_multiplicities(terms, vec![2, 2]);
        assert_eq!(additive.merge_rule(), ValueMerge::Additive);
        assert_eq!(additive.indices(), &[term(&[0, 1])]);
        assert_eq!(additive.values(), &[4]);

        // Terms that become identical on reduction add their counts.
        let mut additive = WeightedSparseBackend::from_multiplicities(
            vec![term(&[0, 4]), term(&[1, 4]), term(&[2, 4])],
            vec![5, 2, 1],
        );
        additive.reduce(0, [0, 1, 2], 9);
        assert_eq!(additive.indices(), &[term(&[4, 9])]);
        assert_eq!(additive.values(), &[8]);
    }

    #[test]
    fn split_factor_one_with_equal_angles_matches_unweighted() {
        let terms = vec![term(&[0, 1]), term(&[0, 1, 2]), term(&[0, 3, 4, 5])];
        let circuit = MajoranaCircuit::new(terms.clone(), vec![0.3; 3], 3).unwrap();
        let backend = circuit_backend(&circuit, 1.0).unwrap();
        assert_eq!(backend.values(), &[1, 1, 1]);
        let plain = MajoranaSparse::new(terms, vec![Complex64::ONE; 3], 0.0).unwrap();
        for comb in [[0u16, 1, 2], [0, 1, 3], [2, 3, 4], [1, 4, 5]] {
            assert_eq!(weight(&plain, comb), weight(&backend, comb), "{comb:?}");
        }
    }

    #[test]
    fn repeats_and_large_angles_get_more_pieces() {
        let circuit = MajoranaCircuit::new(
            vec![term(&[0, 1]), term(&[2, 3]), term(&[0, 1]), term(&[4, 5])],
            vec![0.5, 0.1, 0.5, 2.0],
            3,
        )
        .unwrap();
        let (terms, pieces) = circuit_split_pieces(&circuit, 4.0).unwrap();
        assert_eq!(terms, vec![term(&[0, 1]), term(&[2, 3]), term(&[4, 5])]);
        assert_eq!(pieces.iter().sum::<u64>(), 12);
        assert!(pieces[2] > pieces[0] && pieces[0] > pieces[1]);
        assert!(matches!(
            circuit_split_pieces(&circuit, 0.5),
            Err(CircuitCompileError::InvalidSplitFactor(_))
        ));
    }

    #[test]
    fn jordan_wigner_encodes_known_rotations() {
        let encoding = TernaryTree::naive_jordan_wigner(2)
            .build_encoding(2)
            .unwrap();
        let circuit = MajoranaCircuit::new(
            vec![term(&[0, 1]), term(&[1, 0]), term(&[1, 2])],
            vec![0.25, 0.25, 0.5],
            2,
        )
        .unwrap();
        let pc = encoding.try_encode(&circuit).unwrap();
        // iγ0γ1 is ±Z on qubit 0; reversing the indices flips the angle.
        let z0 = pc.paulis[0]
            .chars()
            .filter(|&c| c != 'I')
            .collect::<String>();
        assert_eq!(z0, "Z");
        assert_eq!(pc.paulis[0], pc.paulis[1]);
        assert_eq!(pc.angles[0], -pc.angles[1]);
        assert_eq!(pc.angles[0].abs(), 0.25);
        // iγ1γ2 hops between neighbouring modes: weight two.
        assert_eq!(pc.paulis[2].chars().filter(|&c| c != 'I').count(), 2);
    }

    #[test]
    fn encoded_circuit_matches_majorana_unitary() {
        for n_modes in [2usize, 3] {
            for seed in 0..4u64 {
                let circuit = random_circuit(n_modes, 10, seed);
                for tree in trees(n_modes) {
                    let encoding = tree.build_encoding(n_modes).unwrap();
                    let pc = encoding.try_encode(&circuit).unwrap();
                    assert_eq!(pc.len(), circuit.n_terms());
                    let expected = majorana_unitary(&encoding, &circuit);
                    let actual = pauli_unitary(&pc);
                    let diff = (&expected - &actual)
                        .iter()
                        .map(|z| z.norm())
                        .fold(0.0, f64::max);
                    assert!(diff < 1e-10, "n={n_modes} seed={seed} diff={diff}");
                }
            }
        }
    }

    #[test]
    fn compile_circuit_gives_valid_encodings_and_keeps_order() {
        let n_modes = 5;
        for seed in 0..4u64 {
            let circuit = random_circuit(n_modes, 30, seed);
            for tree in trees(n_modes) {
                for split_factor in [1.0, 2.0, 4.0] {
                    let (encoding, compiled) = compile_circuit(
                        &circuit,
                        tree.clone(),
                        n_modes,
                        split_factor,
                        false,
                        NodeOrderHeuristic::MinWeight,
                    )
                    .unwrap();
                    assert_eq!(encoding.operators.ipowers.len(), 2 * n_modes);
                    assert_eq!(compiled.len(), circuit.n_terms());
                    // The compiled circuit is the same as encoding afresh.
                    assert_eq!(compiled, encoding.try_encode(&circuit).unwrap());
                }
            }
        }
    }

    #[test]
    fn compile_circuit_weights_heavy_operators() {
        // Two operators compete for the cheap qubit; the heavily rotated one
        // should end up no heavier than when angles are ignored.
        let n_modes = 4;
        let circuit = MajoranaCircuit::new(
            vec![
                term(&[0, 7]),
                term(&[0, 7]),
                term(&[0, 7]),
                term(&[1, 2]),
                term(&[3, 4]),
                term(&[5, 6]),
            ],
            vec![1.5, 1.5, 1.5, 0.01, 0.01, 0.01],
            n_modes,
        )
        .unwrap();
        let tree = TernaryTree::naive_jordan_wigner(n_modes);
        let plain = topphatt(
            circuit.to_majorana_sparse(),
            tree.clone(),
            false,
            NodeOrderHeuristic::MinWeight,
        )
        .unwrap()
        .build_encoding(n_modes)
        .unwrap()
        .try_encode(&circuit)
        .unwrap();
        let (_, weighted) = compile_circuit(
            &circuit,
            tree,
            n_modes,
            4.0,
            false,
            NodeOrderHeuristic::MinWeight,
        )
        .unwrap();
        assert!(
            weighted.angle_weighted_pauli_weight() <= plain.angle_weighted_pauli_weight() + 1e-12,
            "weighted {} > plain {}",
            weighted.angle_weighted_pauli_weight(),
            plain.angle_weighted_pauli_weight()
        );
    }

    #[test]
    fn encoding_rejects_too_many_modes() {
        let encoding = TernaryTree::naive_jordan_wigner(2)
            .build_encoding(2)
            .unwrap();
        let circuit =
            MajoranaCircuit::new(vec![array_vec!([u16; 7] => 0, 5)], vec![0.1], 3).unwrap();
        assert!(matches!(
            encoding.try_encode(&circuit),
            Err(MajoranaEncodingError::CircuitModeMismatch { .. })
        ));
    }
}
