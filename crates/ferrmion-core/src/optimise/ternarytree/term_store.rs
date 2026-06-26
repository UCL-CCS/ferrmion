//! Pluggable Majorana-term storage for TOPP-HATT.
//!
//! TOPP-HATT's inner loop only needs three things from the Hamiltonian: the
//! number of terms, the Pauli weight of a candidate leaf-triple, and an
//! in-place reduction step. [`MajoranaTermStore`] abstracts those operations so
//! the orchestration in [`super::topphatt`] can run unchanged over different
//! term representations.
//!
//! Two backends are provided:
//! - [`ArrayVecTermStore`]: the production representation — a `Vec` of
//!   `ArrayVec<[u16; MAJORANA_MAX]>` index lists. Delegates to the existing
//!   [`qubit_term_weight`] / [`reduce_hamiltonian`] helpers, so behaviour is
//!   identical to the original algorithm.
//! - [`BitTermStore`]: a prototype where each term is a single `u64` bitmask
//!   (bit `i` set ⇔ Majorana `i` is present with odd parity). The per-term
//!   weight loop collapses to a couple of bit ops.
//!
//! # Node representatives
//!
//! When a node is formed, its three child indices are folded into a single
//! "representative" index that future iterations use in place of the node. The
//! ArrayVec backend keeps the original convention — a fresh token in the upper
//! index range (`min_parent + n_leaves`, i.e. `node_offset + 2*n_nodes + 1`).
//! The bit backend instead reuses the **maximum real index of the selection**,
//! which keeps every bit index `≤ 2*n_nodes` so an `n`-mode problem fits in a
//! `u64` for `n ≤ 31`. [`MajoranaTermStore::reduce`] returns the representative
//! it chose; the caller threads it back into the restriction system.
//!
//! # Parity vs. multiplicity (semantic note)
//!
//! [`reduce_hamiltonian`] pads each term with repeated copies of the parent
//! token and deduplicates on the resulting *multiset*. The weight function only
//! depends on the *parity* of each index, so the bit backend deduplicates on the
//! parity-set instead. These agree except in the rare case where two distinct
//! ArrayVec multisets share a parity-set (e.g. `[7,8]` and `[7,8,p,p]`); there
//! the bit backend collapses them to one term while the ArrayVec backend keeps
//! both. The bit behaviour is arguably the more physical one (equal operators
//! are merged), but it means the two backends can occasionally pick different
//! optimal selections. Both still yield valid encodings.

use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use std::sync::atomic::{AtomicUsize, Ordering};
use tinyvec::ArrayVec;

use super::hatt::{qubit_term_weight, reduce_hamiltonian, MAJORANA_MAX};

/// The result of a single TOPP-HATT assignment step.
///
/// Stores the minimum Pauli weight found, the parent node chosen,
/// and the three leaf indices assigned to that node's edges.
#[derive(Debug, Clone, Copy)]
pub struct ToppHattSelection {
    pub(crate) min_weight: usize,
    pub(crate) min_parent: usize,
    pub(crate) leaf_indices: [u16; 3],
}

impl ToppHattSelection {
    /// Sentinel "no selection" value. Used both to initialise the search and as
    /// the identity element of [`combine`]: every real candidate compares better.
    pub(crate) const WORST: Self = Self {
        min_weight: usize::MAX,
        min_parent: usize::MAX,
        leaf_indices: [u16::MAX; 3],
    };
}

/// Pack three leaf indices into a single `u64` for deterministic tie-breaking.
#[inline(always)]
pub(crate) fn packed_leaf_indices(leaf_indices: [u16; 3]) -> u64 {
    (leaf_indices[0] as u64) << 16 | (leaf_indices[1] as u64) << 32 | (leaf_indices[2] as u64) << 48
}

/// Combine two candidate selections, keeping the lower Pauli weight and breaking
/// ties deterministically by the packed leaf indices (see
/// [`packed_leaf_indices`]).
///
/// This combine is associative and order-independent, so the reduction yields the
/// same selection whether combinations are reduced in parallel or folded
/// sequentially.
#[inline(always)]
pub(crate) fn combine(
    current: ToppHattSelection,
    candidate: ToppHattSelection,
) -> ToppHattSelection {
    let take_candidate = candidate.min_weight < current.min_weight
        || (candidate.min_weight == current.min_weight
            && current.min_weight != usize::MAX
            && packed_leaf_indices(candidate.leaf_indices)
                > packed_leaf_indices(current.leaf_indices));
    if take_candidate {
        candidate
    } else {
        current
    }
}

/// Normalise a raw combination (length 2 or 3) into three leaf indices.
///
/// A length-2 combination encodes an even-Majorana / Z pair; the odd partner is
/// inferred. Returns `None` for invalid combinations (a repeated Z leaf).
#[inline(always)]
fn normalise_comb(comb: &[u16]) -> Option<[u16; 3]> {
    let comb: [u16; 3] = if comb.len() == 3 {
        [comb[0], comb[1], comb[2]]
    } else {
        let pair = if comb[0].is_multiple_of(2) {
            comb[0] + 1
        } else {
            comb[0] - 1
        };
        [comb[0], pair, comb[1]]
    };

    if comb[0] == comb[2] || comb[1] == comb[2] {
        return None;
    }
    Some(comb)
}

/// Storage and hot-path operations for a Majorana Hamiltonian during TOPP-HATT.
///
/// Implementors own the term collection and expose only what the assignment loop
/// needs, so the orchestration can be written once and run over any backend.
pub trait MajoranaTermStore {
    /// Number of terms currently held.
    fn len(&self) -> usize;

    /// Whether the Hamiltonian has no terms left.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Score a single leaf-index combination, returning a candidate selection.
    ///
    /// `bound` holds the lowest weight found so far across every thread and
    /// active node: it is read to drive the branch-and-bound early-exit and
    /// lowered (lock-free) whenever a smaller weight is computed. Because pruning
    /// only abandons combinations whose partial weight already exceeds `bound`,
    /// no combination that ties or beats the running minimum is discarded, so the
    /// selection stays deterministic.
    fn evaluate_combination(
        &self,
        comb: &[u16],
        active: usize,
        bound: &AtomicUsize,
    ) -> ToppHattSelection;

    /// Simplify the Hamiltonian after a node's children have been chosen.
    ///
    /// Substitutes the three `selection` indices with a single representative
    /// index and returns that representative, which the caller records so the
    /// restriction system can refer to the new node.
    fn reduce(&mut self, min_parent: usize, selection: [u16; 3], n_leaves: usize) -> u16;
}

/// Production backend: a `Vec` of stack-allocated Majorana-index lists.
///
/// Delegates to the original [`qubit_term_weight`] and [`reduce_hamiltonian`]
/// helpers, so it reproduces the existing algorithm exactly.
pub struct ArrayVecTermStore {
    pub(crate) terms: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
}

impl ArrayVecTermStore {
    /// Wrap an existing list of Majorana terms.
    pub fn new(terms: Vec<ArrayVec<[u16; MAJORANA_MAX]>>) -> Self {
        Self { terms }
    }
}

impl MajoranaTermStore for ArrayVecTermStore {
    fn len(&self) -> usize {
        self.terms.len()
    }

    fn evaluate_combination(
        &self,
        comb: &[u16],
        active: usize,
        bound: &AtomicUsize,
    ) -> ToppHattSelection {
        let comb = match normalise_comb(comb) {
            Some(comb) => comb,
            None => return ToppHattSelection::WORST,
        };

        let mut sorted_comb: [u16; 3] = comb;
        sorted_comb.sort_unstable();
        let comb_min = sorted_comb[0];
        let comb_max = sorted_comb[2];

        let min_weight = bound.load(Ordering::Relaxed);

        // We expect that the hamiltonian terms are sorted!
        let weight = self
            .terms
            .iter()
            .filter(|inds| {
                // Safety
                //
                // We know that `inds` is sorted, and non-empty as
                // `MajoranaSparse` is prepared with sorted indices, and
                // `reduce_hamiltonian` preserves sorted order while removing
                // duplicate indices.
                let inds_min = unsafe { inds.first().unwrap_unchecked() };
                let inds_max = unsafe { inds.last().unwrap_unchecked() };
                (comb_min <= *inds_max) & (comb_max >= *inds_min)
            })
            .fold_while(0, |acc, inds| {
                if acc > min_weight {
                    Done(acc)
                } else {
                    Continue(acc + qubit_term_weight(inds, &comb))
                }
            })
            .into_inner();

        bound.fetch_min(weight, Ordering::Relaxed);

        ToppHattSelection {
            min_weight: weight,
            min_parent: active,
            leaf_indices: comb,
        }
    }

    fn reduce(&mut self, min_parent: usize, selection: [u16; 3], n_leaves: usize) -> u16 {
        let parent = (min_parent + n_leaves) as u16;
        let terms = std::mem::take(&mut self.terms);
        self.terms = reduce_hamiltonian(terms, parent, selection);
        parent
    }
}

/// Prototype backend: one `u64` bitmask per term.
///
/// Bit `i` set ⇔ Majorana `i` acts with odd parity in the term. Supports up to
/// `n_modes = 31` (highest index is the all-Z leaf at `2*n_nodes ≤ 62`).
pub struct BitTermStore {
    pub(crate) terms: Vec<u64>,
}

/// Maximum number of fermionic modes the `u64` bit backend can represent.
///
/// The highest index touched is the all-Z terminator leaf at `2*n_nodes`; with
/// `n_nodes == n_modes` this must fit in a `u64`, so `2*n_modes ≤ 63`.
pub const BIT_STORE_MAX_MODES: usize = 31;

impl BitTermStore {
    /// Build a bit-packed store from Majorana-index terms.
    ///
    /// Each term is folded into a mask by XOR, so any even-multiplicity input
    /// collapses to the correct parity. Returns `None` if any index does not fit
    /// in a `u64` (i.e. the problem exceeds [`BIT_STORE_MAX_MODES`]).
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAJORANA_MAX]>]) -> Option<Self> {
        let mut packed = Vec::with_capacity(terms.len());
        for term in terms {
            let mut mask = 0u64;
            for &idx in term.iter() {
                if idx >= 64 {
                    return None;
                }
                mask ^= 1u64 << idx;
            }
            packed.push(mask);
        }
        packed.sort_unstable();
        packed.dedup();
        Some(Self { terms: packed })
    }
}

impl MajoranaTermStore for BitTermStore {
    fn len(&self) -> usize {
        self.terms.len()
    }

    fn evaluate_combination(
        &self,
        comb: &[u16],
        active: usize,
        bound: &AtomicUsize,
    ) -> ToppHattSelection {
        let comb = match normalise_comb(comb) {
            Some(comb) => comb,
            None => return ToppHattSelection::WORST,
        };

        // Mask of the three child indices. The number of these bits set in a term
        // is exactly the count of odd-parity children; the qubit weight is 0 iff
        // that count is 0 or 3 (PP = I and XYZ = -iI), else 1.
        let child_mask = (1u64 << comb[0]) | (1u64 << comb[1]) | (1u64 << comb[2]);

        let min_weight = bound.load(Ordering::Relaxed);

        let weight = self
            .terms
            .iter()
            .fold_while(0, |acc, &term| {
                if acc > min_weight {
                    Done(acc)
                } else {
                    let odd = (term & child_mask).count_ones() as usize;
                    Continue(acc + !odd.is_multiple_of(3) as usize)
                }
            })
            .into_inner();

        bound.fetch_min(weight, Ordering::Relaxed);

        ToppHattSelection {
            min_weight: weight,
            min_parent: active,
            leaf_indices: comb,
        }
    }

    fn reduce(&mut self, _min_parent: usize, selection: [u16; 3], n_leaves: usize) -> u16 {
        // The all-Z terminator leaf occupies bit `2*n_nodes == n_leaves - 1` and
        // never appears in real terms; keep it reserved by choosing the
        // representative from the real selection members only.
        let all_z = (n_leaves - 1) as u16;
        let repr = selection
            .iter()
            .copied()
            .filter(|&idx| idx < all_z)
            .max()
            .expect("selection always has a real (non-all-Z) member on the X/Y edges");

        let sel_mask = (1u64 << selection[0]) | (1u64 << selection[1]) | (1u64 << selection[2]);
        let repr_bit = 1u64 << repr;

        let mut reduced: Vec<u64> = self
            .terms
            .iter()
            .filter_map(|&term| {
                let remainder = term & !sel_mask;
                // Drop terms that were entirely selection indices: their weight
                // is already accounted for and they carry nothing upward. This
                // mirrors `reduce_hamiltonian`'s "all parent token" filter.
                if remainder == 0 {
                    return None;
                }
                // Re-introduce the representative with the parity of the removed
                // indices, matching the parent-token padding in the index-list
                // reduction.
                let parent_odd = (term & sel_mask).count_ones() & 1 == 1;
                if parent_odd {
                    Some(remainder | repr_bit)
                } else {
                    Some(remainder)
                }
            })
            .collect();

        reduced.sort_unstable();
        reduced.dedup();
        self.terms = reduced;
        repr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tinyvec::array_vec;

    fn weight_via_store<S: MajoranaTermStore>(store: &S, comb: &[u16]) -> usize {
        let bound = AtomicUsize::new(usize::MAX);
        store.evaluate_combination(comb, 0, &bound).min_weight
    }

    #[test]
    fn bit_weight_matches_index_list() {
        // Mirrors the qubit_term_weight cases in topphatt's tests, summed over a
        // small Hamiltonian, for both backends.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1),
            array_vec!([u16; 7] => 0u16, 1, 2),
            array_vec!([u16; 7] => 0u16, 3, 4, 5),
        ];
        let av = ArrayVecTermStore::new(terms.clone());
        let bits = BitTermStore::from_arrayvecs(&terms).unwrap();

        for comb in [[0u16, 1, 2], [0, 1, 3], [2, 3, 4], [1, 4, 5]] {
            assert_eq!(
                weight_via_store(&av, &comb),
                weight_via_store(&bits, &comb),
                "weights differ for comb {comb:?}"
            );
        }
    }

    #[test]
    fn bit_reduce_parity_matches() {
        // The reduce case from topphatt.rs::test_reduce_hamiltonian_substitutes_inplace:
        // selection [2,3,55] over [0,1,2,3] and [0,2,3,4].
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1, 2, 3),
            array_vec!([u16; 7] => 0u16, 2, 3, 4),
        ];
        // n_leaves chosen so all-Z bit (n_leaves-1 = 56) is above index 55.
        let n_leaves = 57;

        let mut bits = BitTermStore::from_arrayvecs(&terms).unwrap();
        let repr = bits.reduce(0, [2, 3, 55], n_leaves);
        // Representative is the max real selection member.
        assert_eq!(repr, 55);

        // [0,1,2,3]: removes 2,3 (even) -> {0,1}; [0,2,3,4]: removes 2,3 (even) -> {0,4}.
        let expected: Vec<u64> = {
            let mut v = vec![(1u64 << 0) | (1 << 1), (1u64 << 0) | (1 << 4)];
            v.sort_unstable();
            v
        };
        assert_eq!(bits.terms, expected);
    }

    #[test]
    fn bit_reduce_odd_parity_sets_representative() {
        // A term with an odd number of selection members must carry the
        // representative bit upward.
        let terms = vec![array_vec!([u16; 7] => 0u16, 2)];
        let n_leaves = 57;
        let mut bits = BitTermStore::from_arrayvecs(&terms).unwrap();
        let repr = bits.reduce(0, [2, 3, 55], n_leaves);
        assert_eq!(repr, 55);
        // {0,2} - {2} = {0}, odd count (1) so add repr 55 -> {0, 55}.
        assert_eq!(bits.terms, vec![(1u64 << 0) | (1u64 << 55)]);
    }
}
