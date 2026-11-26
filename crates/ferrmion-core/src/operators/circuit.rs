//! Ordered circuits of Majorana and Pauli rotation gates.
//!
//! Unlike [`MajoranaSparse`], which describes a Hamiltonian, a
//! [`MajoranaCircuit`] is an *ordered sequence* of rotation gates: term order
//! and multiplicity are preserved exactly, because every occurrence is a real
//! gate. Encoding a [`MajoranaCircuit`] with a
//! [`crate::encode::majorana::MajoranaEncoding`] gives a [`PauliCircuit`] with
//! the same order and length.
//!
//! # Gate convention
//!
//! Gate `j` of a [`MajoranaCircuit`] is
//!
//! $$ G_j = \exp(-i\theta_j \Gamma_{S_j}), \qquad
//!    \Gamma_S = i^{m(k)} \gamma_{s_1}\cdots\gamma_{s_k}, \qquad
//!    m(k) = \tfrac{k(k-1)}{2} \bmod 2, $$
//!
//! where `S = (s_1 < … < s_k)` are distinct Majorana indices. The phase
//! `i^{m(k)}` makes `Γ_S` Hermitian (and `Γ_S² = I`), so every gate is unitary
//! for real `θ`. A [`PauliCircuit`] gate is `exp(-iθ P)` for a Pauli string `P`.
//! Angles are stored wrapped into `(-π, π]`.

use super::{cancel_pairs, cannonicalize, MajoranaSparse, MAX_MAJORANAS};
use crate::spaces::{Fermion, Qubit};
use crate::utils::COEFFICIENT_TOLERANCE;
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use thiserror::Error;
use tinyvec::ArrayVec;

/// Relative tolerance on the imaginary part of a Hermitian rotation angle when
/// converting from a [`MajoranaSparse`].
const HERMITIAN_TOLERANCE: f64 = 1e-9;

/// Errors produced when constructing a [`MajoranaCircuit`].
#[derive(Debug, Error, PartialEq)]
pub enum MajoranaCircuitError {
    #[error("terms ({terms}) and angles ({angles}) length mismatch")]
    LengthMismatch { terms: usize, angles: usize },
    #[error("term {term} index {index} out of bounds for {n_modes} modes")]
    IndexOutOfBounds {
        term: usize,
        index: u16,
        n_modes: usize,
    },
    #[error("term {0} is empty")]
    EmptyTerm(usize),
    #[error("term {term} has {len} indices; the maximum is {MAX_MAJORANAS}")]
    TooManyIndices { term: usize, len: usize },
    #[error("term {0} has a non-finite angle")]
    NonFiniteAngle(usize),
    #[error("term {term} with coefficient {coefficient} is not Hermitian, so is not a rotation")]
    NonHermitianTerm { term: usize, coefficient: Complex64 },
    #[error("expected a permutation of 0..{expected}")]
    InvalidPermutation { expected: usize },
}

/// Power of `i` that makes a product of `k` distinct Majoranas Hermitian.
///
/// $(\gamma_{s_1}\cdots\gamma_{s_k})^\dagger = (-1)^{k(k-1)/2}\gamma_{s_1}\cdots\gamma_{s_k}$,
/// so the product is Hermitian when `k(k-1)/2` is even and anti-Hermitian
/// otherwise; multiplying by `i^{m(k)}` fixes the latter.
///
/// # Examples
///
/// ```
/// use ferrmion_core::operators::hermitian_phase_power;
///
/// assert_eq!(hermitian_phase_power(1), 0); // γ_a is Hermitian
/// assert_eq!(hermitian_phase_power(2), 1); // iγ_aγ_b is Hermitian
/// assert_eq!(hermitian_phase_power(3), 1);
/// assert_eq!(hermitian_phase_power(4), 0);
/// ```
pub fn hermitian_phase_power(k: usize) -> u8 {
    ((k * k.saturating_sub(1) / 2) % 2) as u8
}

/// Wrap a rotation angle into `(-π, π]`.
///
/// # Examples
///
/// ```
/// use ferrmion_core::operators::wrap_angle;
/// use std::f64::consts::PI;
///
/// assert!((wrap_angle(2.0 * PI - 0.1) + 0.1).abs() < 1e-12);
/// assert_eq!(wrap_angle(PI), PI);
/// assert_eq!(wrap_angle(-PI), PI);
/// ```
pub fn wrap_angle(theta: f64) -> f64 {
    // Leave in-range angles untouched so wrapping never costs precision.
    if theta > -PI && theta <= PI {
        return theta;
    }
    let r = theta.rem_euclid(2.0 * PI);
    if r > PI {
        r - 2.0 * PI
    } else {
        r
    }
}

/// An ordered circuit of Majorana rotation gates `exp(-iθ Γ_S)`.
///
/// Order and multiplicity are preserved: nothing is merged during
/// construction. Each term is canonicalised — indices sorted (negating the
/// angle once per swap of unequal indices), repeated indices cancelled
/// pairwise (`γ² = I`, no sign) — and angles are wrapped into `(-π, π]`. The
/// Hermitian phase `i^{m(k)}` is that of the *canonical* term, so a gate given
/// as `[0, 0, 1]` is the gate on `[1]`. Gates
/// that reduce to the identity (a global phase) or have a zero angle are
/// dropped, so [`Self::n_terms`] may be smaller than the input length.
///
/// See the [module documentation](self) for the gate convention.
#[derive(Debug, Clone, PartialEq)]
pub struct MajoranaCircuit {
    /// Canonical (sorted, distinct) Majorana indices of each gate.
    pub terms: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
    /// Rotation angle of each gate, wrapped into `(-π, π]`.
    pub angles: Vec<f64>,
    /// Number of fermionic modes; every index is `< 2 * n_modes`.
    pub n_modes: usize,
}

impl Fermion for MajoranaCircuit {}

impl Fermion for &MajoranaCircuit {}

impl MajoranaCircuit {
    /// Construct an ordered circuit from parallel term and angle lists.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::MajoranaCircuit;
    /// use tinyvec::array_vec;
    ///
    /// let circuit = MajoranaCircuit::new(
    ///     vec![array_vec!([u16; 7] => 1, 0), array_vec!([u16; 7] => 0, 1)],
    ///     vec![0.25, 0.5],
    ///     1,
    /// )
    /// .unwrap();
    /// // γ1γ0 = -γ0γ1, so the first angle is negated; the repeat is kept.
    /// assert_eq!(circuit.angles, vec![-0.25, 0.5]);
    /// assert_eq!(circuit.n_terms(), 2);
    /// ```
    pub fn new(
        terms: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
        angles: Vec<f64>,
        n_modes: usize,
    ) -> Result<Self, MajoranaCircuitError> {
        if terms.len() != angles.len() {
            return Err(MajoranaCircuitError::LengthMismatch {
                terms: terms.len(),
                angles: angles.len(),
            });
        }
        let bound = 2 * n_modes;
        let mut canonical_terms = Vec::with_capacity(terms.len());
        let mut canonical_angles = Vec::with_capacity(angles.len());
        for (t, (mut term, angle)) in terms.into_iter().zip(angles).enumerate() {
            if term.is_empty() {
                return Err(MajoranaCircuitError::EmptyTerm(t));
            }
            if !angle.is_finite() {
                return Err(MajoranaCircuitError::NonFiniteAngle(t));
            }
            let max = *term.iter().max().expect("term verified non-empty");
            if max as usize >= bound {
                return Err(MajoranaCircuitError::IndexOutOfBounds {
                    term: t,
                    index: max,
                    n_modes,
                });
            }
            let sign = cannonicalize(term.as_mut_slice());
            let kept = cancel_pairs(term.as_mut_slice());
            if kept == 0 {
                continue;
            }
            term.truncate(kept);
            let theta = wrap_angle(angle * sign);
            if theta.abs() < COEFFICIENT_TOLERANCE {
                continue;
            }
            canonical_terms.push(term);
            canonical_angles.push(theta);
        }
        Ok(Self {
            terms: canonical_terms,
            angles: canonical_angles,
            n_modes,
        })
    }

    /// Construct from plain index lists, checking each fits in a term.
    pub fn from_index_lists(
        terms: Vec<Vec<u16>>,
        angles: Vec<f64>,
        n_modes: usize,
    ) -> Result<Self, MajoranaCircuitError> {
        let packed = terms
            .into_iter()
            .enumerate()
            .map(|(t, indices)| {
                if indices.len() > MAX_MAJORANAS {
                    return Err(MajoranaCircuitError::TooManyIndices {
                        term: t,
                        len: indices.len(),
                    });
                }
                let mut av: ArrayVec<[u16; MAX_MAJORANAS]> = ArrayVec::new();
                av.extend_from_slice(&indices);
                Ok(av)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(packed, angles, n_modes)
    }

    /// Build the circuit `∏_j exp(-i c_j γ_{S_j})` from a Hamiltonian's terms,
    /// in their stored order (one first-order Trotter step of unit time).
    ///
    /// Each term `c·γ_S` must be Hermitian: `θ = c · i^{-m(k)}` must be real
    /// (relative tolerance `1e-9`). The Hamiltonian's constant is a global
    /// phase and is dropped.
    pub fn from_majorana_sparse(
        sparse: &MajoranaSparse,
        n_modes: usize,
    ) -> Result<Self, MajoranaCircuitError> {
        let mut terms = Vec::with_capacity(sparse.indices.len());
        let mut angles = Vec::with_capacity(sparse.indices.len());
        for (t, (term, &coefficient)) in sparse.indices.iter().zip(&sparse.coefficients).enumerate()
        {
            let mut term = *term;
            let sign = cannonicalize(term.as_mut_slice());
            let kept = cancel_pairs(term.as_mut_slice());
            if kept == 0 {
                // A pure constant: a global phase.
                continue;
            }
            term.truncate(kept);
            // c γ_T = c · sign · γ_S = c · sign · i^{-m} Γ_S.
            let theta = match hermitian_phase_power(kept) {
                0 => coefficient * sign,
                _ => coefficient * sign * Complex64::new(0.0, -1.0),
            };
            if theta.im.abs() > HERMITIAN_TOLERANCE * theta.norm().max(1.0) {
                return Err(MajoranaCircuitError::NonHermitianTerm {
                    term: t,
                    coefficient,
                });
            }
            terms.push(term);
            angles.push(theta.re);
        }
        Self::new(terms, angles, n_modes)
    }

    /// The circuit's generators as a [`MajoranaSparse`], term by term in order
    /// and without merging: gate `exp(-iθ Γ_S)` becomes `θ · i^{m(k)} · γ_S`.
    pub fn to_majorana_sparse(&self) -> MajoranaSparse {
        let coefficients = self
            .terms
            .iter()
            .zip(&self.angles)
            .map(|(term, &theta)| match hermitian_phase_power(term.len()) {
                0 => Complex64::new(theta, 0.0),
                _ => Complex64::new(0.0, theta),
            })
            .collect();
        MajoranaSparse {
            indices: self.terms.clone(),
            coefficients,
            constant: 0.0,
        }
    }

    /// Number of gates.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Whether the circuit has no gates.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Merge *consecutive* gates on the same term by summing their angles.
    ///
    /// Non-adjacent repeats are not merged (in general they do not commute
    /// with the gates in between). Gates whose merged angle wraps to ~0 are
    /// dropped, and the gates either side of them then become adjacent:
    /// `A, B(θ), B(-θ), A` merges to a single `A`.
    pub fn merge_adjacent(&self) -> MajoranaCircuit {
        let mut terms: Vec<ArrayVec<[u16; MAX_MAJORANAS]>> = Vec::with_capacity(self.terms.len());
        let mut angles: Vec<f64> = Vec::with_capacity(self.angles.len());
        for (term, &angle) in self.terms.iter().zip(&self.angles) {
            if terms.last() == Some(term) {
                let last = angles.last_mut().expect("terms and angles stay parallel");
                *last = wrap_angle(*last + angle);
                if last.abs() < COEFFICIENT_TOLERANCE {
                    terms.pop();
                    angles.pop();
                }
            } else {
                terms.push(*term);
                angles.push(angle);
            }
        }
        MajoranaCircuit {
            terms,
            angles,
            n_modes: self.n_modes,
        }
    }

    /// Total absolute rotation applied with each distinct operator.
    ///
    /// Returns the distinct terms in order of first occurrence, with
    /// `V_j = Σ |θ|` over every gate on term `j`. Repeated gates and large
    /// angles both increase `V_j`.
    pub fn operator_weights(&self) -> (Vec<ArrayVec<[u16; MAX_MAJORANAS]>>, Vec<f64>) {
        let mut position: HashMap<ArrayVec<[u16; MAX_MAJORANAS]>, usize> = HashMap::new();
        let mut terms = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        for (term, &theta) in self.terms.iter().zip(&self.angles) {
            match position.get(term) {
                Some(&j) => weights[j] += theta.abs(),
                None => {
                    position.insert(*term, terms.len());
                    terms.push(*term);
                    weights.push(theta.abs());
                }
            }
        }
        (terms, weights)
    }

    /// [`Self::operator_weights`] normalised to sum to one.
    pub fn relative_operator_weights(&self) -> (Vec<ArrayVec<[u16; MAX_MAJORANAS]>>, Vec<f64>) {
        let (terms, mut weights) = self.operator_weights();
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            weights.iter_mut().for_each(|w| *w /= total);
        }
        (terms, weights)
    }

    /// Relabel the Majorana operators: every `γ_a` becomes `γ_{perm[a]}`.
    ///
    /// Gate order and multiplicity are unchanged. Relabelled terms are
    /// re-canonicalised, so a gate's angle is negated when its new indices
    /// sort with an odd permutation.
    ///
    /// With `perm` from
    /// [`crate::encode::majorana::MajoranaEncoding::majorana_permutation_to`],
    /// encoding `self` under one encoding equals encoding the permuted circuit
    /// under the other, which puts circuits optimised for different encodings
    /// into a common labelling for comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::MajoranaCircuit;
    /// use tinyvec::array_vec;
    ///
    /// let circuit =
    ///     MajoranaCircuit::new(vec![array_vec!([u16; 7] => 0, 1)], vec![0.5], 2).unwrap();
    /// // Swap γ0 and γ1: γ1γ0 = -γ0γ1.
    /// let swapped = circuit.permuted(&[1, 0, 2, 3]).unwrap();
    /// assert_eq!(swapped.angles, vec![-0.5]);
    /// // Move the gate onto γ2γ3.
    /// let moved = circuit.permuted(&[2, 3, 0, 1]).unwrap();
    /// assert_eq!(moved.terms[0].as_slice(), &[2, 3]);
    /// ```
    pub fn permuted(&self, perm: &[usize]) -> Result<MajoranaCircuit, MajoranaCircuitError> {
        let n_majoranas = 2 * self.n_modes;
        let mut seen = vec![false; n_majoranas];
        if perm.len() != n_majoranas {
            return Err(MajoranaCircuitError::InvalidPermutation {
                expected: n_majoranas,
            });
        }
        for &b in perm {
            if b >= n_majoranas || std::mem::replace(&mut seen[b], true) {
                return Err(MajoranaCircuitError::InvalidPermutation {
                    expected: n_majoranas,
                });
            }
        }
        let terms = self
            .terms
            .iter()
            .map(|term| term.iter().map(|&a| perm[a as usize] as u16).collect())
            .collect();
        MajoranaCircuit::new(terms, self.angles.clone(), self.n_modes)
    }

    /// Reorder the gates by where their Majoranas sit in an encoding tree.
    ///
    /// `rank[a]` is the position of Majorana `a` in the tree, e.g. from
    /// [`crate::encode::majorana::MajoranaEncoding::majorana_tree_order`]. Gates
    /// are stably sorted by their Majoranas' ranks (each gate's ranks sorted,
    /// then compared lexicographically), so gates acting near each other in
    /// the tree become neighbours and their CNOT ladders can cancel.
    ///
    /// **This changes the circuit's unitary** unless the gates commute. It is
    /// meant for circuits whose order is free, such as a single Trotter step,
    /// where any order is an equally valid product formula.
    ///
    /// # Panics
    ///
    /// If `rank` has fewer than `2 * n_modes` entries.
    pub fn sorted_by_rank(&self, rank: &[usize]) -> MajoranaCircuit {
        assert!(
            rank.len() >= 2 * self.n_modes,
            "rank needs an entry for each of the {} Majoranas",
            2 * self.n_modes
        );
        let keys: Vec<ArrayVec<[usize; MAX_MAJORANAS]>> = self
            .terms
            .iter()
            .map(|term| {
                let mut key: ArrayVec<[usize; MAX_MAJORANAS]> =
                    term.iter().map(|&a| rank[a as usize]).collect();
                key.sort_unstable();
                key
            })
            .collect();
        let mut order: Vec<usize> = (0..self.terms.len()).collect();
        order.sort_by(|&i, &j| keys[i].cmp(&keys[j]));
        MajoranaCircuit {
            terms: order.iter().map(|&i| self.terms[i]).collect(),
            angles: order.iter().map(|&i| self.angles[i]).collect(),
            n_modes: self.n_modes,
        }
    }
}

/// An ordered circuit of Pauli rotation gates `exp(-iθ P)`.
///
/// Pauli strings use ferrmion's qubit-0-first convention: character `q` acts
/// on qubit `q`.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliCircuit {
    /// Number of qubits.
    pub n_qubits: usize,
    /// Pauli string of each gate.
    pub paulis: Vec<String>,
    /// Rotation angle of each gate, wrapped into `(-π, π]`.
    pub angles: Vec<f64>,
}

impl Qubit for PauliCircuit {}

impl PauliCircuit {
    /// Number of gates.
    pub fn len(&self) -> usize {
        self.paulis.len()
    }

    /// Whether the circuit has no gates.
    pub fn is_empty(&self) -> bool {
        self.paulis.is_empty()
    }

    /// Summed Pauli weight (number of non-identity letters) over all gates.
    pub fn pauli_weight(&self) -> usize {
        self.paulis.iter().map(|p| pauli_string_weight(p)).sum()
    }

    /// Summed `|θ| · weight(P)` over all gates.
    pub fn angle_weighted_pauli_weight(&self) -> f64 {
        self.paulis
            .iter()
            .zip(&self.angles)
            .map(|(p, theta)| theta.abs() * pauli_string_weight(p) as f64)
            .sum()
    }
}

/// Number of non-identity letters in a Pauli string.
fn pauli_string_weight(pauli: &str) -> usize {
    pauli.chars().filter(|&c| c != 'I').count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::c64;
    use tinyvec::array_vec;

    fn term(indices: &[u16]) -> ArrayVec<[u16; MAX_MAJORANAS]> {
        let mut av = ArrayVec::new();
        av.extend_from_slice(indices);
        av
    }

    #[test]
    fn constructor_validation() {
        assert_eq!(
            MajoranaCircuit::new(vec![term(&[0, 1])], vec![], 1),
            Err(MajoranaCircuitError::LengthMismatch {
                terms: 1,
                angles: 0
            })
        );
        assert_eq!(
            MajoranaCircuit::new(vec![term(&[0, 4])], vec![0.1], 2),
            Err(MajoranaCircuitError::IndexOutOfBounds {
                term: 0,
                index: 4,
                n_modes: 2
            })
        );
        assert_eq!(
            MajoranaCircuit::new(vec![term(&[])], vec![0.1], 2),
            Err(MajoranaCircuitError::EmptyTerm(0))
        );
        assert_eq!(
            MajoranaCircuit::new(vec![term(&[0, 1])], vec![f64::NAN], 2),
            Err(MajoranaCircuitError::NonFiniteAngle(0))
        );
        assert_eq!(
            MajoranaCircuit::from_index_lists(vec![vec![0, 1, 2, 3, 4, 5, 6, 7]], vec![0.1], 4),
            Err(MajoranaCircuitError::TooManyIndices { term: 0, len: 8 })
        );
    }

    #[test]
    fn construction_applies_the_reordering_sign() {
        let circuit = MajoranaCircuit::new(
            vec![term(&[1, 0]), term(&[3, 2, 1, 0]), term(&[0, 1])],
            vec![0.1, 0.2, 0.3],
            2,
        )
        .unwrap();
        assert_eq!(circuit.terms[0].as_slice(), &[0, 1]);
        assert_eq!(circuit.terms[1].as_slice(), &[0, 1, 2, 3]);
        assert_eq!(circuit.angles, vec![-0.1, 0.2, 0.3]);
    }

    #[test]
    fn construction_cancels_gamma_squared_pairs() {
        let circuit = MajoranaCircuit::new(
            vec![
                term(&[0, 1, 2, 2]),
                term(&[1, 1, 2, 2]),
                term(&[2, 0, 2]),
                term(&[0, 3]),
            ],
            vec![0.1, 0.2, 0.3, 0.4],
            2,
        )
        .unwrap();
        // [1,1,2,2] is the identity (a global phase) and is dropped.
        assert_eq!(circuit.n_terms(), 3);
        assert_eq!(circuit.terms[0].as_slice(), &[0, 1]);
        // γ2γ0γ2 = -γ0γ2γ2 = -γ0.
        assert_eq!(circuit.terms[1].as_slice(), &[0]);
        assert_eq!(circuit.terms[2].as_slice(), &[0, 3]);
        assert_eq!(circuit.angles, vec![0.1, -0.3, 0.4]);
    }

    #[test]
    fn angles_wrap_and_zero_angles_drop() {
        let circuit = MajoranaCircuit::new(
            vec![term(&[0, 1]), term(&[2, 3]), term(&[0, 3])],
            vec![2.0 * PI - 0.25, 2.0 * PI, -PI],
            2,
        )
        .unwrap();
        assert_eq!(circuit.n_terms(), 2);
        assert!((circuit.angles[0] + 0.25).abs() < 1e-12);
        assert_eq!(circuit.angles[1], PI);
    }

    #[test]
    fn repeated_terms_preserved_until_merge() {
        let circuit = MajoranaCircuit::new(
            vec![term(&[0, 1]), term(&[0, 1]), term(&[2, 3]), term(&[0, 1])],
            vec![0.5, 0.25, 1.0, 0.125],
            2,
        )
        .unwrap();
        assert_eq!(circuit.n_terms(), 4);
        let merged = circuit.merge_adjacent();
        assert_eq!(merged.n_terms(), 3);
        assert_eq!(merged.angles, vec![0.75, 1.0, 0.125]);
    }

    #[test]
    fn merge_adjacent_drops_cancelled_terms() {
        let circuit = MajoranaCircuit::new(
            vec![term(&[0, 1]), term(&[0, 1]), term(&[2, 3])],
            vec![0.5, -0.5, 1.0],
            2,
        )
        .unwrap();
        let merged = circuit.merge_adjacent();
        assert_eq!(merged.n_terms(), 1);
        assert_eq!(merged.terms[0].as_slice(), &[2, 3]);
    }

    #[test]
    fn merge_adjacent_cascades_through_cancellations() {
        // B(θ)B(-θ) = I, so the two A gates either side become adjacent.
        let circuit = MajoranaCircuit::new(
            vec![term(&[0, 1]), term(&[2, 3]), term(&[3, 2]), term(&[1, 0])],
            vec![0.25, 0.5, 0.5, -0.5],
            2,
        )
        .unwrap();
        let merged = circuit.merge_adjacent();
        assert_eq!(merged.n_terms(), 1);
        assert_eq!(merged.terms[0].as_slice(), &[0, 1]);
        assert!((merged.angles[0] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn operator_weights_sum_absolute_angles() {
        let circuit = MajoranaCircuit::new(
            vec![term(&[2, 3]), term(&[0, 1]), term(&[2, 3]), term(&[3, 2])],
            vec![0.5, 0.25, -1.0, 0.5],
            2,
        )
        .unwrap();
        let (terms, weights) = circuit.operator_weights();
        assert_eq!(terms, vec![term(&[2, 3]), term(&[0, 1])]);
        assert_eq!(weights, vec![2.0, 0.25]);

        // r repeats at θ weigh the same as one gate at rθ.
        let repeated = MajoranaCircuit::new(vec![term(&[0, 1]); 3], vec![0.2; 3], 1).unwrap();
        let single = MajoranaCircuit::new(vec![term(&[0, 1])], vec![0.6], 1).unwrap();
        let (_, wr) = repeated.operator_weights();
        let (_, ws) = single.operator_weights();
        assert!((wr[0] - ws[0]).abs() < 1e-12);
        assert_eq!(repeated.n_terms(), 3);

        let (_, relative) = circuit.relative_operator_weights();
        assert!((relative.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn majorana_sparse_round_trip_uses_hermitian_phase() {
        // i·γ0γ1 is Hermitian with θ = 1 for the convention Γ = iγ0γ1.
        let ms = MajoranaSparse::new(
            vec![
                array_vec!([u16; 7] => 0, 1),
                array_vec!([u16; 7] => 2),
                array_vec!([u16; 7] => 0, 1, 2, 3),
            ],
            vec![c64(0.0, 0.5), c64(0.25, 0.0), c64(-0.125, 0.0)],
            3.0,
        )
        .unwrap();
        let circuit = MajoranaCircuit::from_majorana_sparse(&ms, 2).unwrap();
        assert_eq!(circuit.angles, vec![0.5, 0.25, -0.125]);
        let back = circuit.to_majorana_sparse();
        assert_eq!(back.indices, ms.indices);
        assert_eq!(back.coefficients, ms.coefficients);

        // A real coefficient on γ0γ1 is anti-Hermitian: not a rotation.
        let bad = MajoranaSparse::new(vec![array_vec!([u16; 7] => 0, 1)], vec![c64(1.0, 0.0)], 0.0)
            .unwrap();
        assert!(matches!(
            MajoranaCircuit::from_majorana_sparse(&bad, 1),
            Err(MajoranaCircuitError::NonHermitianTerm { term: 0, .. })
        ));
    }

    #[test]
    fn sorted_by_rank_reorders_stably() {
        let circuit = MajoranaCircuit::new(
            vec![term(&[2, 3]), term(&[0, 3]), term(&[0, 1]), term(&[2, 3])],
            vec![0.1, 0.2, 0.3, 0.4],
            2,
        )
        .unwrap();
        // Identity ranks: lexicographic order of the (sorted) index tuples.
        let sorted = circuit.sorted_by_rank(&[0, 1, 2, 3]);
        assert_eq!(
            sorted.terms,
            vec![term(&[0, 1]), term(&[0, 3]), term(&[2, 3]), term(&[2, 3])]
        );
        assert_eq!(sorted.angles, vec![0.3, 0.2, 0.1, 0.4]);
        // Reversed ranks put Majoranas 2, 3 first.
        let reversed = circuit.sorted_by_rank(&[3, 2, 1, 0]);
        assert_eq!(reversed.terms[0], term(&[2, 3]));
        assert_eq!(reversed.angles[..2], [0.1, 0.4]);
        assert_eq!(reversed.terms[3], term(&[0, 1]));
    }

    #[test]
    fn permuted_validates_and_inverts() {
        let circuit = MajoranaCircuit::new(
            vec![term(&[0, 3]), term(&[1, 2]), term(&[0, 1, 2, 3])],
            vec![0.1, 0.2, 0.3],
            2,
        )
        .unwrap();
        for bad in [vec![0, 1, 2], vec![0, 1, 1, 3], vec![0, 1, 2, 4]] {
            assert_eq!(
                circuit.permuted(&bad),
                Err(MajoranaCircuitError::InvalidPermutation { expected: 4 })
            );
        }
        let perm = [2, 0, 3, 1];
        let mut inverse = [0; 4];
        for (a, &b) in perm.iter().enumerate() {
            inverse[b] = a;
        }
        let there = circuit.permuted(&perm).unwrap();
        assert_ne!(there, circuit);
        assert_eq!(there.permuted(&inverse).unwrap(), circuit);
        assert_eq!(circuit.permuted(&[0, 1, 2, 3]).unwrap(), circuit);
    }

    #[test]
    fn pauli_circuit_weights() {
        let pc = PauliCircuit {
            n_qubits: 3,
            paulis: vec!["XIZ".into(), "IIY".into()],
            angles: vec![0.5, -2.0],
        };
        assert_eq!(pc.len(), 2);
        assert_eq!(pc.pauli_weight(), 3);
        assert!((pc.angle_weighted_pauli_weight() - 3.0).abs() < 1e-12);
    }
}
