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
//! - [`BitTermStore`]: a prototype where each term is a single fixed-width
//!   bitmask (bit `i` set ⇔ Majorana `i` is present with odd parity). The
//!   per-term weight loop collapses to a couple of bit ops. It is generic over a
//!   [`BitWord`] (`u64`, ≤ 31 modes; `u128`, ≤ 63 modes; `U256`, ≤ 127 modes),
//!   with type aliases [`BitTermStore64`], [`BitTermStore128`] and
//!   [`BitTermStore256`].
//! - [`BitSlicedTermStore`]: the *transpose* of `BitTermStore` — one `u64`
//!   bit-vector per Majorana index, with bits indexing terms. Scoring a selection
//!   reads only the three relevant vectors. No mode ceiling.
//!
//! # Node representatives
//!
//! When a node is formed, its three child indices are folded into a single
//! "representative" index that future iterations use in place of the node. The
//! ArrayVec backend keeps the original convention — a fresh token in the upper
//! index range (`min_parent + n_leaves`, i.e. `node_offset + 2*n_nodes + 1`).
//! The bit backend instead reuses the **maximum real index of the selection**,
//! which keeps every bit index `≤ 2*n_nodes` so an `n`-mode problem fits in the
//! chosen word width (`n ≤ 31` for `u64`, `n ≤ 63` for `u128`, `n ≤ 127` for
//! `U256`). [`MajoranaTermStore::reduce`] returns the representative it chose;
//! the caller threads it back into the restriction system.
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

/// A fixed-width unsigned integer usable as a term bitmask.
///
/// Implemented for the native `u64` and `u128`, plus a 256-bit `bnum` word
/// ([`U256`]) for problems beyond 63 modes. The few primitives below are all the
/// generic [`BitTermStore`] needs; for the native words each maps to a single
/// machine instruction, while `u128`/`U256` are emulated from 64-bit limbs.
pub trait BitWord: Copy + Ord + Send + Sync {
    /// Width of the word in bits (e.g. 64 or 128).
    const BITS: u32;
    /// The all-zero word.
    const ZERO: Self;
    /// `1 << index`. `index` must be `< BITS`.
    fn single_bit(index: u16) -> Self;
    fn and(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
    fn not(self) -> Self;
    fn xor_assign(&mut self, rhs: Self);
    fn count_ones(self) -> u32;
}

macro_rules! impl_bit_word {
    ($t:ty) => {
        impl BitWord for $t {
            const BITS: u32 = <$t>::BITS;
            const ZERO: Self = 0;
            #[inline(always)]
            fn single_bit(index: u16) -> Self {
                1 << index
            }
            #[inline(always)]
            fn and(self, rhs: Self) -> Self {
                self & rhs
            }
            #[inline(always)]
            fn or(self, rhs: Self) -> Self {
                self | rhs
            }
            #[inline(always)]
            fn not(self) -> Self {
                !self
            }
            #[inline(always)]
            fn xor_assign(&mut self, rhs: Self) {
                *self ^= rhs;
            }
            #[inline(always)]
            fn count_ones(self) -> u32 {
                <$t>::count_ones(self)
            }
        }
    };
}

impl_bit_word!(u64);
impl_bit_word!(u128);

/// 256-bit unsigned word (four 64-bit limbs) for the widest bit backend.
pub use bnum::types::U256;

impl BitWord for U256 {
    const BITS: u32 = U256::BITS;
    const ZERO: Self = U256::ZERO;
    #[inline(always)]
    fn single_bit(index: u16) -> Self {
        U256::ONE << u32::from(index)
    }
    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        self & rhs
    }
    #[inline(always)]
    fn or(self, rhs: Self) -> Self {
        self | rhs
    }
    #[inline(always)]
    fn not(self) -> Self {
        !self
    }
    #[inline(always)]
    fn xor_assign(&mut self, rhs: Self) {
        *self ^= rhs;
    }
    #[inline(always)]
    fn count_ones(self) -> u32 {
        U256::count_ones(self)
    }
}

/// Prototype backend: one fixed-width bitmask per term.
///
/// Bit `i` set ⇔ Majorana `i` acts with odd parity in the term. The word type
/// `W` sets the mode ceiling: the highest index touched is the all-Z leaf at
/// `2*n_nodes`, which must fit in `W`, so `n_modes ≤ (W::BITS - 1) / 2`
/// (31 for `u64`, 63 for `u128`, 127 for `U256`). See [`BitTermStore::MAX_MODES`].
pub struct BitTermStore<W: BitWord = u64> {
    pub(crate) terms: Vec<W>,
}

/// `u64`-backed bit store (≤ 31 modes).
pub type BitTermStore64 = BitTermStore<u64>;
/// `u128`-backed bit store (≤ 63 modes).
pub type BitTermStore128 = BitTermStore<u128>;
/// `U256`-backed bit store (≤ 127 modes).
pub type BitTermStore256 = BitTermStore<U256>;

impl<W: BitWord> BitTermStore<W> {
    /// Maximum number of fermionic modes this word width can represent.
    ///
    /// The highest index touched is the all-Z terminator leaf at `2*n_nodes`;
    /// with `n_nodes == n_modes` this must fit in `W`, so `2*n_modes ≤ BITS-1`.
    pub const MAX_MODES: usize = ((W::BITS - 1) / 2) as usize;

    /// Build a bit-packed store from Majorana-index terms.
    ///
    /// Each term is folded into a mask by XOR, so any even-multiplicity input
    /// collapses to the correct parity. Returns `None` if any index does not fit
    /// in `W` (i.e. the problem exceeds [`BitTermStore::MAX_MODES`]).
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAJORANA_MAX]>]) -> Option<Self> {
        let mut packed = Vec::with_capacity(terms.len());
        for term in terms {
            let mut mask = W::ZERO;
            for &idx in term.iter() {
                if u32::from(idx) >= W::BITS {
                    return None;
                }
                mask.xor_assign(W::single_bit(idx));
            }
            packed.push(mask);
        }
        packed.sort_unstable();
        packed.dedup();
        Some(Self { terms: packed })
    }
}

impl<W: BitWord> MajoranaTermStore for BitTermStore<W> {
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
        let child_mask = W::single_bit(comb[0])
            .or(W::single_bit(comb[1]))
            .or(W::single_bit(comb[2]));

        let min_weight = bound.load(Ordering::Relaxed);

        let weight = self
            .terms
            .iter()
            .fold_while(0, |acc, &term| {
                if acc > min_weight {
                    Done(acc)
                } else {
                    let odd = term.and(child_mask).count_ones() as usize;
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

        let sel_mask = W::single_bit(selection[0])
            .or(W::single_bit(selection[1]))
            .or(W::single_bit(selection[2]));
        let repr_bit = W::single_bit(repr);

        let mut reduced: Vec<W> = self
            .terms
            .iter()
            .filter_map(|&term| {
                let remainder = term.and(sel_mask.not());
                // Drop terms that were entirely selection indices: their weight
                // is already accounted for and they carry nothing upward. This
                // mirrors `reduce_hamiltonian`'s "all parent token" filter.
                if remainder == W::ZERO {
                    return None;
                }
                // Re-introduce the representative with the parity of the removed
                // indices, matching the parent-token padding in the index-list
                // reduction.
                let parent_odd = term.and(sel_mask).count_ones() & 1 == 1;
                if parent_odd {
                    Some(remainder.or(repr_bit))
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

/// Transposed ("bit-sliced") prototype backend.
///
/// Where [`BitTermStore`] is row-major (one word per term, bits = Majorana
/// indices), this is column-major: one `u64` bit-vector per **Majorana index**,
/// whose bits correspond to **terms** (`columns[i]` bit `t` set ⇔ term `t`
/// contains index `i`). Scoring a candidate selection reads only the three
/// vectors for that selection and computes the whole Pauli weight with
/// word-parallel bit ops over `⌈T/64⌉` words, instead of touching every term.
///
/// Because terms are fixed bit positions shared across all columns, this backend
/// performs **no term drop/dedup** during reduction: emptied terms become
/// all-zero columns (and contribute 0 weight automatically), but duplicates are
/// counted with multiplicity. It matches the row-major store's per-selection
/// weight, yet — like the other bit backends — can pick a different (still valid)
/// encoding than the index-list backend. It has no mode ceiling: columns are
/// indexed by Majorana index, and the `u64` words slice terms.
pub struct BitSlicedTermStore {
    n_terms: usize,
    n_words: usize,
    /// One bit-vector per Majorana index (length `2*n_modes + 1`).
    columns: Vec<Vec<u64>>,
}

impl BitSlicedTermStore {
    /// Build a bit-sliced store from Majorana-index terms.
    ///
    /// `n_modes` sizes the column table (`2*n_modes + 1` indices: real Majoranas,
    /// the all-Z leaf, and node representatives — all `≤ 2*n_modes`).
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAJORANA_MAX]>], n_modes: usize) -> Self {
        let n_terms = terms.len();
        let n_words = n_terms.div_ceil(64);
        let n_cols = 2 * n_modes + 1;
        let mut columns = vec![vec![0u64; n_words]; n_cols];
        for (t, term) in terms.iter().enumerate() {
            let word = t / 64;
            let bit = 1u64 << (t % 64);
            for &idx in term.iter() {
                columns[idx as usize][word] |= bit;
            }
        }
        Self {
            n_terms,
            n_words,
            columns,
        }
    }
}

impl MajoranaTermStore for BitSlicedTermStore {
    fn len(&self) -> usize {
        self.n_terms
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

        let a = &self.columns[comb[0] as usize];
        let b = &self.columns[comb[1] as usize];
        let c = &self.columns[comb[2] as usize];

        let min_weight = bound.load(Ordering::Relaxed);

        // Per term the Pauli weight is 1 iff exactly one or two of the three
        // children are present: `(a|b|c) & !(a&b&c)` (count 0 or 3 ⇒ identity).
        // Accumulate word by word, keeping the branch-and-bound early-exit.
        let mut weight = 0usize;
        for w in 0..self.n_words {
            let any = a[w] | b[w] | c[w];
            let all = a[w] & b[w] & c[w];
            weight += (any & !all).count_ones() as usize;
            if weight > min_weight {
                break;
            }
        }

        bound.fetch_min(weight, Ordering::Relaxed);

        ToppHattSelection {
            min_weight: weight,
            min_parent: active,
            leaf_indices: comb,
        }
    }

    fn reduce(&mut self, _min_parent: usize, selection: [u16; 3], n_leaves: usize) -> u16 {
        // The all-Z terminator leaf is at bit `n_leaves - 1`; keep it reserved by
        // choosing the representative from the real selection members only.
        let all_z = (n_leaves - 1) as u16;
        let repr = selection
            .iter()
            .copied()
            .filter(|&idx| idx < all_z)
            .max()
            .expect("selection always has a real (non-all-Z) member on the X/Y edges");

        let (c0, c1, c2) = (
            selection[0] as usize,
            selection[1] as usize,
            selection[2] as usize,
        );

        // Per term, the representative carries the parity of the removed indices
        // (matching the parent-token padding in the index-list reduction). Read
        // the three columns into a parity buffer, clear them, then write the
        // parity into the representative's column. No terms are dropped.
        let parity: Vec<u64> = (0..self.n_words)
            .map(|w| self.columns[c0][w] ^ self.columns[c1][w] ^ self.columns[c2][w])
            .collect();
        for &col in &[c0, c1, c2] {
            self.columns[col].iter_mut().for_each(|w| *w = 0);
        }
        self.columns[repr as usize].copy_from_slice(&parity);

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
        let bits64 = BitTermStore64::from_arrayvecs(&terms).unwrap();
        let bits128 = BitTermStore128::from_arrayvecs(&terms).unwrap();
        let bits256 = BitTermStore256::from_arrayvecs(&terms).unwrap();
        let sliced = BitSlicedTermStore::from_arrayvecs(&terms, 4);

        for comb in [[0u16, 1, 2], [0, 1, 3], [2, 3, 4], [1, 4, 5]] {
            let expected = weight_via_store(&av, &comb);
            assert_eq!(
                expected,
                weight_via_store(&bits64, &comb),
                "u64 weights differ for comb {comb:?}"
            );
            assert_eq!(
                expected,
                weight_via_store(&bits128, &comb),
                "u128 weights differ for comb {comb:?}"
            );
            assert_eq!(
                expected,
                weight_via_store(&bits256, &comb),
                "u256 weights differ for comb {comb:?}"
            );
            assert_eq!(
                expected,
                weight_via_store(&sliced, &comb),
                "bit-sliced weights differ for comb {comb:?}"
            );
        }
    }

    #[test]
    fn bit_sliced_weight_no_mode_ceiling() {
        // Indices well past any native-word ceiling: the transposed store has no
        // limit because words slice terms, not indices.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 200),
            array_vec!([u16; 7] => 100u16, 200, 250),
        ];
        let av = ArrayVecTermStore::new(terms.clone());
        let sliced = BitSlicedTermStore::from_arrayvecs(&terms, 130);
        for comb in [[0u16, 100, 200], [0, 200, 250], [100, 200, 250]] {
            assert_eq!(
                weight_via_store(&av, &comb),
                weight_via_store(&sliced, &comb),
                "bit-sliced weight differs for comb {comb:?}"
            );
        }
    }

    #[test]
    fn bit_sliced_reduce_parity_matches() {
        // Same reduce case as `bit_reduce_parity_matches`, checked via the
        // post-reduction weights (the transposed store has no flat term list).
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1, 2, 3),
            array_vec!([u16; 7] => 0u16, 2, 3, 4),
        ];
        let n_leaves = 57;
        let mut sliced = BitSlicedTermStore::from_arrayvecs(&terms, 28);
        let repr = sliced.reduce(0, [2, 3, 55], n_leaves);
        assert_eq!(repr, 55);

        // After reduction terms are {0,1} and {0,4}. Spot-check a few combs
        // against an index-list store holding those reduced terms.
        let reduced = vec![
            array_vec!([u16; 7] => 0u16, 1),
            array_vec!([u16; 7] => 0u16, 4),
        ];
        let expected = ArrayVecTermStore::new(reduced);
        for comb in [[0u16, 1, 4], [1, 4, 5], [0, 1, 2]] {
            assert_eq!(
                weight_via_store(&expected, &comb),
                weight_via_store(&sliced, &comb),
                "reduced bit-sliced weight differs for comb {comb:?}"
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

        let mut bits = BitTermStore64::from_arrayvecs(&terms).unwrap();
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
    fn bit_reduce_u128_high_index() {
        // A representative beyond the u64 range (index 80) is only expressible
        // with the u128 backend; exercise it end to end.
        let terms = vec![array_vec!([u16; 7] => 0u16, 80)];
        let n_leaves = 122; // all-Z bit at 121, above index 80.
        let mut bits = BitTermStore128::from_arrayvecs(&terms).unwrap();
        let repr = bits.reduce(0, [80, 81, 90], n_leaves);
        assert_eq!(repr, 90);
        // {0,80} - {80} = {0}, odd count (1) so add repr 90 -> {0, 90}.
        assert_eq!(bits.terms, vec![(1u128 << 0) | (1u128 << 90)]);
    }

    #[test]
    fn bit_store_mode_ceilings() {
        assert_eq!(BitTermStore64::MAX_MODES, 31);
        assert_eq!(BitTermStore128::MAX_MODES, 63);
        assert_eq!(BitTermStore256::MAX_MODES, 127);
        // An index that overflows u64 (64) but fits the wider words is rejected
        // only by u64.
        let terms = vec![array_vec!([u16; 7] => 0u16, 64)];
        assert!(BitTermStore64::from_arrayvecs(&terms).is_none());
        assert!(BitTermStore128::from_arrayvecs(&terms).is_some());
        assert!(BitTermStore256::from_arrayvecs(&terms).is_some());
        // An index past u128 (130) is only representable in U256.
        let wide = vec![array_vec!([u16; 7] => 0u16, 130)];
        assert!(BitTermStore128::from_arrayvecs(&wide).is_none());
        assert!(BitTermStore256::from_arrayvecs(&wide).is_some());
    }

    #[test]
    fn bit_reduce_odd_parity_sets_representative() {
        // A term with an odd number of selection members must carry the
        // representative bit upward.
        let terms = vec![array_vec!([u16; 7] => 0u16, 2)];
        let n_leaves = 57;
        let mut bits = BitTermStore64::from_arrayvecs(&terms).unwrap();
        let repr = bits.reduce(0, [2, 3, 55], n_leaves);
        assert_eq!(repr, 55);
        // {0,2} - {2} = {0}, odd count (1) so add repr 55 -> {0, 55}.
        assert_eq!(bits.terms, vec![(1u64 << 0) | (1u64 << 55)]);
    }
}
