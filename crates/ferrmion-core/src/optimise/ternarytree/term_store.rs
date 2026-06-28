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
//! - [`BitSlicedTermStore`]: a transposed bit-vector layout — one `u64`
//!   bit-vector per Majorana index, with bits indexing terms. Scoring a selection
//!   reads only the three relevant vectors and computes the Pauli weight with
//!   word-parallel bit ops over `⌈T/64⌉` words. No mode ceiling.
//!
//! # Node representatives
//!
//! When a node is formed, its three child indices are folded into a single
//! "representative" index that future iterations use in place of the node. Both
//! backends keep the original convention — a fresh token in the upper index range
//! (`min_parent + n_leaves`, i.e. `node_offset + 2*n_nodes + 1`) — so the
//! orchestration's magnitude-based edge classification works identically on every
//! tree topology. [`MajoranaTermStore::reduce`] returns the representative it
//! chose; the caller threads it back into the restriction system.
//!
//! # Parity vs. multiplicity (semantic note)
//!
//! [`reduce_hamiltonian`] (the index-list backend) pads each term with repeated
//! copies of the parent token and deduplicates on the resulting *multiset*. The
//! weight function only depends on the *parity* of each index. The transposed
//! backends keep that fast parity weight evaluation but deduplicate whole terms
//! on the same **multiset** rule (see [`MultisetDedup`]), so all three backends
//! produce identical encodings.

use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use std::collections::HashMap;
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

/// Reduce a Majorana term to its **parity-set** (γ²=I): the sorted, distinct
/// indices that appear an odd number of times. An index appearing an even number
/// of times cancels. This is the canonical form the transposed backends store.
fn parity_set(term: &[u16]) -> ArrayVec<[u16; MAJORANA_MAX]> {
    let mut p: ArrayVec<[u16; MAJORANA_MAX]> = ArrayVec::new();
    for &idx in term {
        if let Some(pos) = p.iter().position(|&x| x == idx) {
            p.remove(pos);
        } else {
            p.push(idx);
        }
    }
    p.sort_unstable();
    p
}

/// Deterministic FNV-1a fingerprint of a sorted parity-set, used to find
/// candidate duplicate terms cheaply (collisions are resolved by comparing the
/// sets directly).
#[inline]
fn fingerprint(sig: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in sig {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Multiset (whole-term) deduplication — the index-list backend's rule.
///
/// This merges terms that are identical as **multisets** (with parent-token
/// padding, length preserved), reproducing `reduce_hamiltonian`'s `sort + dedup`.
/// It therefore yields encodings identical to the index-list backend while
/// leaving the fast (parity) weight evaluation untouched. Both transposed
/// backends drive an identical instance in the same term order, so they make
/// identical liveness decisions and yield identical encodings. Because the
/// multiset can hold
/// even-multiplicity members that the parity columns drop, the reduction scans
/// every live term's signature rather than reading the columns.
struct MultisetDedup {
    /// Per-term multiset signature, sorted, length-preserving (selection members
    /// are mapped to the representative, not removed).
    sigs: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
    fps: Vec<u64>,
    buckets: HashMap<u64, Vec<u32>>,
    live: Vec<bool>,
}

impl MultisetDedup {
    fn new(terms: &[ArrayVec<[u16; MAJORANA_MAX]>]) -> Self {
        let n = terms.len();
        let mut sigs: Vec<ArrayVec<[u16; MAJORANA_MAX]>> = Vec::with_capacity(n);
        for term in terms {
            let mut s = *term;
            s.sort_unstable();
            sigs.push(s);
        }
        let mut fps = vec![0u64; n];
        let mut buckets: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut live = vec![true; n];
        for t in 0..n {
            if sigs[t].is_empty() {
                live[t] = false;
                continue;
            }
            let f = fingerprint(&sigs[t]);
            fps[t] = f;
            let is_dup = buckets
                .get(&f)
                .is_some_and(|b| b.iter().any(|&o| sigs[o as usize] == sigs[t]));
            if is_dup {
                live[t] = false;
            } else {
                buckets.entry(f).or_default().push(t as u32);
            }
        }
        Self {
            sigs,
            fps,
            buckets,
            live,
        }
    }

    /// Map selection members to `repr` (multiset, length-preserving) in every live
    /// term, drop all-`repr` terms, and merge multiset-duplicates. Returns the
    /// ids that became dead so the caller can update its live mask.
    fn reduce(&mut self, selection: [u16; 3], repr: u16) -> Vec<u32> {
        let mut dead = Vec::new();
        for t in 0..self.sigs.len() {
            if !self.live[t] || !self.sigs[t].iter().any(|&x| selection.contains(&x)) {
                continue;
            }
            if let Some(b) = self.buckets.get_mut(&self.fps[t]) {
                if let Some(p) = b.iter().position(|&o| o == t as u32) {
                    b.swap_remove(p);
                }
            }
            {
                let sig = &mut self.sigs[t];
                for x in sig.iter_mut() {
                    if selection.contains(x) {
                        *x = repr;
                    }
                }
                sig.sort_unstable();
            }
            if self.sigs[t].iter().all(|&x| x == repr) {
                self.live[t] = false;
                dead.push(t as u32);
                continue;
            }
            let f = fingerprint(&self.sigs[t]);
            self.fps[t] = f;
            let is_dup = self.buckets.get(&f).is_some_and(|b| {
                b.iter()
                    .any(|&o| self.live[o as usize] && self.sigs[o as usize] == self.sigs[t])
            });
            if is_dup {
                self.live[t] = false;
                dead.push(t as u32);
            } else {
                self.buckets.entry(f).or_default().push(t as u32);
            }
        }
        dead
    }
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

/// Transposed ("bit-sliced") bit-vector backend.
///
/// Where [`ArrayVecTermStore`] keeps one index-list per term, this is
/// column-major: one `u64` bit-vector per **Majorana index**, whose bits
/// correspond to **terms** (`columns[i]` bit `t` set ⇔ term `t` contains index
/// `i`). Scoring a candidate selection reads only the three vectors for that
/// selection and computes the whole Pauli weight with word-parallel bit ops over
/// `⌈T/64⌉` words, instead of touching every term.
///
/// Terms are fixed bit positions shared across all columns. Duplicate whole terms
/// are removed by **multiset deduplication** (the index-list backend's rule): a
/// [`MultisetDedup`] tracks each term's multiset signature and a `live` mask, and
/// a deduplicated term is excluded from the weight by ANDing the mask into the
/// per-word accumulation. The fast parity weight eval is unchanged, so this
/// backend reproduces `index_list`'s encodings exactly. It has no mode ceiling:
/// columns are indexed by Majorana index, and the `u64` words slice terms.
pub struct BitSlicedTermStore {
    n_terms: usize,
    n_words: usize,
    /// One bit-vector per index (length `3*n_modes + 1`): real Majoranas
    /// `0..2*n_nodes`, the all-Z leaf `2*n_nodes`, and node representatives
    /// `2*n_nodes+1..=3*n_nodes`.
    columns: Vec<Vec<u64>>,
    /// Per-term multiset (whole-term) deduplication state — the index-list
    /// backend's rule, so this backend now reproduces its encodings.
    dedup: MultisetDedup,
    /// Word-packed liveness mask mirroring `dedup.live`; bit `t` set ⇔ term `t`
    /// is still counted. ANDed into the per-word weight accumulation.
    live_words: Vec<u64>,
}

impl BitSlicedTermStore {
    /// Build a bit-sliced store from Majorana-index terms.
    ///
    /// `n_modes` sizes the column table to `3*n_modes + 1` indices. The store uses
    /// the **same upper-range node representative as the index-list backend**
    /// (`node + 2*n_nodes + 1`), so it stays compatible with the orchestration's
    /// magnitude-based edge classification on every tree topology (not just
    /// JKMN). There is no word-width mode ceiling: the `u64` words slice terms,
    /// not indices.
    ///
    /// Each term is **parity-canonicalised** (γ²=I): a bit is XOR-toggled per
    /// index occurrence, so an index appearing an even number of times in a term
    /// cancels — matching `qubit_term_weight` and the XOR `reduce`. (Molecular
    /// Majorana terms include number operators like `[0,0]` that must cancel.)
    ///
    /// Duplicate and identity input terms are merged at build time by the
    /// [`MultisetDedup`], so they are counted once (matching the index-list rule).
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAJORANA_MAX]>], n_modes: usize) -> Self {
        let n_terms = terms.len();
        let n_words = n_terms.div_ceil(64);
        let n_cols = 3 * n_modes + 1;
        let mut columns = vec![vec![0u64; n_words]; n_cols];
        for (t, term) in terms.iter().enumerate() {
            let word = t / 64;
            let bit = 1u64 << (t % 64);
            for &idx in term.iter() {
                // XOR (not OR): a repeated index toggles back off — γ²=I parity.
                columns[idx as usize][word] ^= bit;
            }
        }
        let dedup = MultisetDedup::new(terms);
        let mut live_words = vec![0u64; n_words];
        for (t, &alive) in dedup.live.iter().enumerate() {
            if alive {
                live_words[t / 64] |= 1u64 << (t % 64);
            }
        }
        Self {
            n_terms,
            n_words,
            columns,
            dedup,
            live_words,
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
        // AND the live mask so deduplicated terms are not counted. Accumulate
        // word by word, keeping the branch-and-bound early-exit.
        let mut weight = 0usize;
        for w in 0..self.n_words {
            let any = a[w] | b[w] | c[w];
            let all = a[w] & b[w] & c[w];
            weight += (any & !all & self.live_words[w]).count_ones() as usize;
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

    fn reduce(&mut self, min_parent: usize, selection: [u16; 3], n_leaves: usize) -> u16 {
        // Use the index-list backend's upper-range node token so the
        // orchestration classifies this node's edges identically (the
        // representative is `≥ n_leaves`, never mistaken for a real leaf).
        let repr = (min_parent + n_leaves) as u16;

        let (c0, c1, c2) = (
            selection[0] as usize,
            selection[1] as usize,
            selection[2] as usize,
        );

        // Multiset (whole-term) deduplication: map selection members to `repr`
        // in every live term's multiset signature and merge duplicates, matching
        // the index-list backend. Clear the live bit of each newly-dead term.
        for t in self.dedup.reduce(selection, repr) {
            self.live_words[(t / 64) as usize] &= !(1u64 << (t % 64));
        }

        // Per term, the representative carries the parity of the removed indices
        // (matching the parent-token padding in the index-list reduction). Read
        // the three columns into a parity buffer, clear them, then write the
        // parity into the representative's column. Dead terms stay masked by
        // `live_words`.
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

/// Sparse inverted-index backend.
///
/// The sparse counterpart of [`BitSlicedTermStore`]: instead of a dense `u64`
/// bit-vector per index, each index keeps a **sorted list of the term indices it
/// appears in**. For sparse Hamiltonians (e.g. molecular) the dense bit columns
/// are mostly zero, so these lists are short and scoring a selection — a 3-way
/// merge of three lists — costs `O(|L0|+|L1|+|L2|)` instead of `O(T/64)`.
///
/// It runs the identical algorithm as [`BitSlicedTermStore`] — same per-selection
/// (parity) weight, same reduction and upper-range representative, and the same
/// [`MultisetDedup`] driven in the same term order — so `topphatt_impl` over it
/// yields encodings identical to the bit-sliced backend *and* to the index-list
/// backend; only the representation and performance differ. No mode ceiling.
pub struct SparseListTermStore {
    n_terms: usize,
    /// One ascending, duplicate-free list of term indices per index (length
    /// `3*n_modes + 1`): real Majoranas `0..2*n_nodes`, the all-Z leaf
    /// `2*n_nodes`, and node representatives `2*n_nodes+1..=3*n_nodes`.
    lists: Vec<Vec<u32>>,
    /// Per-term multiset (whole-term) deduplication state — the index-list
    /// backend's rule. A dead term is skipped in the weight merge.
    dedup: MultisetDedup,
}

impl SparseListTermStore {
    /// Build a sparse inverted index from Majorana-index terms.
    ///
    /// Uses the index-list/bit-sliced upper-range node representative
    /// (`node + 2*n_nodes + 1`), so it is valid on every tree topology and has no
    /// mode ceiling.
    ///
    /// The inverted-index **lists** are parity-canonicalised (γ²=I): only indices
    /// appearing an odd number of times in a term are recorded, so number-operator
    /// terms like `[0,0]` cancel — this drives the parity weight. Whole-term
    /// deduplication (and identity-term dropping) is handled separately by the
    /// [`MultisetDedup`], built from the raw terms, matching the index-list rule.
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAJORANA_MAX]>], n_modes: usize) -> Self {
        let n_terms = terms.len();
        let n_cols = 3 * n_modes + 1;
        let mut lists = vec![Vec::new(); n_cols];
        for (t, term) in terms.iter().enumerate() {
            for &idx in parity_set(term).iter() {
                // Terms are visited in ascending `t`, so each list stays sorted
                // and duplicate-free.
                lists[idx as usize].push(t as u32);
            }
        }
        let dedup = MultisetDedup::new(terms);
        Self {
            n_terms,
            lists,
            dedup,
        }
    }
}

impl MajoranaTermStore for SparseListTermStore {
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

        let a = &self.lists[comb[0] as usize];
        let b = &self.lists[comb[1] as usize];
        let c = &self.lists[comb[2] as usize];

        let min_weight = bound.load(Ordering::Relaxed);

        // 3-way merge of the sorted term lists. For each term present in at least
        // one list, `count` (∈ 1..=3) is how many of the three lists contain it;
        // the Pauli weight is 1 unless `count` is a multiple of 3 (here only 3).
        // `u32::MAX` is the exhausted-list sentinel (never a real term index).
        let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
        let mut weight = 0usize;
        loop {
            let va = a.get(i).copied().unwrap_or(u32::MAX);
            let vb = b.get(j).copied().unwrap_or(u32::MAX);
            let vc = c.get(k).copied().unwrap_or(u32::MAX);
            let m = va.min(vb).min(vc);
            if m == u32::MAX {
                break;
            }
            let mut count = 0u32;
            if va == m {
                i += 1;
                count += 1;
            }
            if vb == m {
                j += 1;
                count += 1;
            }
            if vc == m {
                k += 1;
                count += 1;
            }
            // Skip deduplicated terms; otherwise weight 1 unless count is a
            // multiple of 3 (here, 3).
            if self.dedup.live[m as usize] && !count.is_multiple_of(3) {
                weight += 1;
            }
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

    fn reduce(&mut self, min_parent: usize, selection: [u16; 3], n_leaves: usize) -> u16 {
        // Upper-range node token, matching the index-list and bit-sliced backends.
        let repr = (min_parent + n_leaves) as u16;
        let (c0, c1, c2) = (
            selection[0] as usize,
            selection[1] as usize,
            selection[2] as usize,
        );

        // Multiset (whole-term) deduplication, matching the index-list backend:
        // map selection members to `repr` in every live term's multiset and merge
        // duplicates. Updates `self.dedup.live`.
        let _ = self.dedup.reduce(selection, repr);

        // 3-way merge over the three selected parity lists; the representative
        // carries each term that appears in an ODD number of them (matching the
        // bit-sliced XOR reduction) and is still live. Output stays ascending.
        let mut parity: Vec<u32> = Vec::new();
        {
            let a = &self.lists[c0];
            let b = &self.lists[c1];
            let c = &self.lists[c2];
            let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
            loop {
                let va = a.get(i).copied().unwrap_or(u32::MAX);
                let vb = b.get(j).copied().unwrap_or(u32::MAX);
                let vc = c.get(k).copied().unwrap_or(u32::MAX);
                let m = va.min(vb).min(vc);
                if m == u32::MAX {
                    break;
                }
                let mut count = 0u32;
                if va == m {
                    i += 1;
                    count += 1;
                }
                if vb == m {
                    j += 1;
                    count += 1;
                }
                if vc == m {
                    k += 1;
                    count += 1;
                }
                if count & 1 == 1 && self.dedup.live[m as usize] {
                    parity.push(m);
                }
            }
        }

        // Clear the three selected lists and install the representative's list.
        // `repr` is a fresh upper index distinct from the selection, so its list
        // was empty.
        self.lists[c0].clear();
        self.lists[c1].clear();
        self.lists[c2].clear();
        self.lists[repr as usize] = parity;

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
    fn transposed_backends_parity_canonicalise_repeated_indices() {
        // Molecular Majorana Hamiltonians include number-operator terms with a
        // repeated index (γ²=I), e.g. `[0,0]` or `[0,0,2,3]`. The index-list
        // backend handles these via XOR-parity; the transposed backends must
        // match by canonicalising each term to its odd-multiplicity indices.
        // Parity-sets here are all distinct ({}, {2,3}, {3}, {1,4,5}, {0,1}), so
        // deduplication is a no-op and the three backends agree term-for-term.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 0),
            array_vec!([u16; 7] => 0u16, 0, 2, 3),
            array_vec!([u16; 7] => 2u16, 2, 3),
            array_vec!([u16; 7] => 1u16, 4, 5),
            array_vec!([u16; 7] => 0u16, 1),
        ];
        let av = ArrayVecTermStore::new(terms.clone());
        let bits = BitSlicedTermStore::from_arrayvecs(&terms, 4);
        let sparse = SparseListTermStore::from_arrayvecs(&terms, 4);

        for a in 0u16..8 {
            for b in 0u16..8 {
                for c in 0u16..8 {
                    let comb = [a, b, c];
                    let expected = weight_via_store(&av, &comb);
                    assert_eq!(
                        expected,
                        weight_via_store(&bits, &comb),
                        "bit-sliced parity differs for comb {comb:?}"
                    );
                    assert_eq!(
                        expected,
                        weight_via_store(&sparse, &comb),
                        "sparse-list parity differs for comb {comb:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn transposed_backends_deduplicate_whole_terms_like_index_list() {
        // The transposed backends deduplicate on the *multiset* rule, matching the
        // index-list backend. Two *identical* terms collapse to one; but
        // `[0,0,2,3]` and `[2,3]` (parity-equal, multiset-distinct) are kept apart
        // — exactly as `index_list` does — so all three backends agree.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 0, 2, 3), // multiset {0,0,2,3}, parity {2,3}
            array_vec!([u16; 7] => 2u16, 3),       // multiset {2,3},     parity {2,3}
            array_vec!([u16; 7] => 2u16, 3),       // exact duplicate of the above
        ];
        let av = ArrayVecTermStore::new(terms.clone());
        let bits = BitSlicedTermStore::from_arrayvecs(&terms, 4);
        let sparse = SparseListTermStore::from_arrayvecs(&terms, 4);

        // comb [2,3,5]: each surviving {2,3} term has two of the three present ⇒
        // weight 1. index_list does not dedup at eval time, so it counts all three
        // raw terms (weight 3); the transposed backends merge the exact duplicate
        // at build time, counting two distinct terms (weight 2). They agree with
        // each other, which is the guarantee under test here.
        let comb = [2u16, 3, 5];
        assert_eq!(
            weight_via_store(&bits, &comb),
            weight_via_store(&sparse, &comb),
            "transposed backends must agree"
        );
        assert_eq!(weight_via_store(&bits, &comb), 2, "exact duplicate merged");
        // index_list keeps all three raw terms at evaluation time.
        assert_eq!(weight_via_store(&av, &comb), 3);
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
        let sliced = BitSlicedTermStore::from_arrayvecs(&terms, 4);

        for comb in [[0u16, 1, 2], [0, 1, 3], [2, 3, 4], [1, 4, 5]] {
            assert_eq!(
                weight_via_store(&av, &comb),
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
        // Upper-range node token (min_parent 0 + n_leaves), matching the
        // index-list convention.
        assert_eq!(repr, n_leaves as u16);

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
    fn sparse_list_weight_matches_index_list() {
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1),
            array_vec!([u16; 7] => 0u16, 1, 2),
            array_vec!([u16; 7] => 0u16, 3, 4, 5),
        ];
        let av = ArrayVecTermStore::new(terms.clone());
        let sparse = SparseListTermStore::from_arrayvecs(&terms, 4);

        for comb in [[0u16, 1, 2], [0, 1, 3], [2, 3, 4], [1, 4, 5]] {
            assert_eq!(
                weight_via_store(&av, &comb),
                weight_via_store(&sparse, &comb),
                "sparse-list weights differ for comb {comb:?}"
            );
        }
    }

    #[test]
    fn sparse_list_weight_no_mode_ceiling() {
        // Indices past any fixed-word ceiling: the sparse store is indexed by
        // Majorana index with no width limit.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 200),
            array_vec!([u16; 7] => 100u16, 200, 250),
        ];
        let av = ArrayVecTermStore::new(terms.clone());
        let sparse = SparseListTermStore::from_arrayvecs(&terms, 130);
        for comb in [[0u16, 100, 200], [0, 200, 250], [100, 200, 250]] {
            assert_eq!(
                weight_via_store(&av, &comb),
                weight_via_store(&sparse, &comb),
                "sparse-list weight differs for comb {comb:?}"
            );
        }
    }

    #[test]
    fn sparse_list_reduce_parity_matches() {
        // Same reduce case as `bit_sliced_reduce_parity_matches`, checked via the
        // post-reduction weights.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1, 2, 3),
            array_vec!([u16; 7] => 0u16, 2, 3, 4),
        ];
        let n_leaves = 57;
        let mut sparse = SparseListTermStore::from_arrayvecs(&terms, 28);
        let repr = sparse.reduce(0, [2, 3, 55], n_leaves);
        assert_eq!(repr, n_leaves as u16);

        let reduced = vec![
            array_vec!([u16; 7] => 0u16, 1),
            array_vec!([u16; 7] => 0u16, 4),
        ];
        let expected = ArrayVecTermStore::new(reduced);
        for comb in [[0u16, 1, 4], [1, 4, 5], [0, 1, 2]] {
            assert_eq!(
                weight_via_store(&expected, &comb),
                weight_via_store(&sparse, &comb),
                "reduced sparse-list weight differs for comb {comb:?}"
            );
        }
    }
}
