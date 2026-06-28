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
//! backends instead deduplicate on each term's **parity-set** (two terms that are
//! the same Pauli operator are counted once; see [`ParitySetDedup`]) — the
//! physically meaningful notion. Both remove duplicate terms, but the rules
//! differ (multiset vs. parity-set), so the transposed backends can still pick a
//! different — but equally valid — encoding than the index-list backend.

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

/// Parity-set deduplication shared by the transposed backends.
///
/// Two Majorana terms with the same parity-set are the same Pauli operator, so
/// they should be counted once. This tracks, per term, its current parity-set
/// signature plus a `live` flag; exact duplicates (and empty/identity terms) are
/// marked dead and excluded from the per-selection weight. Detection is
/// **incremental**: a reduction only re-examines the terms it touches.
///
/// Both transposed backends drive an identical instance in the same term order,
/// so they make identical liveness decisions and yield identical encodings.
struct ParitySetDedup {
    /// Per-term parity-set, sorted ascending. Kept in sync with the backend's
    /// columns/lists.
    sigs: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
    /// Per-term fingerprint of `sigs`.
    fps: Vec<u64>,
    /// Fingerprint → live term ids sharing it (for collision lookup).
    buckets: HashMap<u64, Vec<u32>>,
    /// Per-term liveness; a dead term is no longer counted.
    live: Vec<bool>,
}

impl ParitySetDedup {
    /// Build from each term's parity-set, dropping empty (identity) terms and
    /// merging exact-duplicate inputs. Terms are processed in ascending id, so
    /// the smallest id of each duplicate group survives — deterministically.
    fn new(sigs: Vec<ArrayVec<[u16; MAJORANA_MAX]>>) -> Self {
        let n = sigs.len();
        let mut fps = vec![0u64; n];
        let mut buckets: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut live = vec![true; n];
        for t in 0..n {
            if sigs[t].is_empty() {
                live[t] = false; // identity term: never contributes weight
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

    /// Apply a reduction to one affected term: replace any selection members in
    /// its parity-set with `repr` (parity — `repr` is added iff an odd number of
    /// members were present), then re-detect empties and duplicates. Returns the
    /// term's liveness afterwards.
    fn update_term(&mut self, t: u32, selection: [u16; 3], repr: u16) -> bool {
        let ti = t as usize;
        if !self.live[ti] {
            return false;
        }
        // Detach t from its current fingerprint bucket.
        if let Some(bucket) = self.buckets.get_mut(&self.fps[ti]) {
            if let Some(pos) = bucket.iter().position(|&o| o == t) {
                bucket.swap_remove(pos);
            }
        }
        // Parity update of the signature (mirrors the column/list XOR).
        {
            let sig = &mut self.sigs[ti];
            let mut cnt = 0u32;
            for &s in &selection {
                if let Some(pos) = sig.iter().position(|&x| x == s) {
                    sig.remove(pos);
                    cnt += 1;
                }
            }
            if cnt & 1 == 1 {
                let pos = sig.iter().position(|&x| x > repr).unwrap_or(sig.len());
                sig.insert(pos, repr);
            }
        }
        if self.sigs[ti].is_empty() {
            self.live[ti] = false;
            return false;
        }
        let f = fingerprint(&self.sigs[ti]);
        self.fps[ti] = f;
        let is_dup = self.buckets.get(&f).is_some_and(|b| {
            b.iter()
                .any(|&o| self.live[o as usize] && self.sigs[o as usize] == self.sigs[ti])
        });
        if is_dup {
            self.live[ti] = false;
            return false;
        }
        self.buckets.entry(f).or_default().push(t);
        true
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
/// Terms are fixed bit positions shared across all columns. Duplicate terms (two
/// terms that became the same Pauli operator) are removed by **parity-set
/// deduplication**: a [`ParitySetDedup`] tracks each term's parity-set and a
/// `live` mask, and a deduplicated term is excluded from the weight by ANDing the
/// mask into the per-word accumulation. Because this merges terms on their
/// *parity*-set (the physically meaningful notion) rather than the index-list
/// backend's coarser multiset rule, it can still pick a different (but valid)
/// encoding than `index_list`. It has no mode ceiling: columns are indexed by
/// Majorana index, and the `u64` words slice terms.
pub struct BitSlicedTermStore {
    n_terms: usize,
    n_words: usize,
    /// One bit-vector per index (length `3*n_modes + 1`): real Majoranas
    /// `0..2*n_nodes`, the all-Z leaf `2*n_nodes`, and node representatives
    /// `2*n_nodes+1..=3*n_nodes`.
    columns: Vec<Vec<u64>>,
    /// Per-term parity-set deduplication state (shared logic with the sparse
    /// backend, so both make identical liveness decisions).
    dedup: ParitySetDedup,
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
    /// Duplicate and identity (empty parity-set) input terms are also merged at
    /// build time by the [`ParitySetDedup`], so they are counted once.
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAJORANA_MAX]>], n_modes: usize) -> Self {
        let n_terms = terms.len();
        let n_words = n_terms.div_ceil(64);
        let n_cols = 3 * n_modes + 1;
        let mut columns = vec![vec![0u64; n_words]; n_cols];
        let mut sigs: Vec<ArrayVec<[u16; MAJORANA_MAX]>> = Vec::with_capacity(n_terms);
        for (t, term) in terms.iter().enumerate() {
            let word = t / 64;
            let bit = 1u64 << (t % 64);
            for &idx in term.iter() {
                // XOR (not OR): a repeated index toggles back off — γ²=I parity.
                columns[idx as usize][word] ^= bit;
            }
            sigs.push(parity_set(term));
        }
        let dedup = ParitySetDedup::new(sigs);
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

        // Terms touched by this reduction are exactly those with one of the three
        // selected indices in their parity-set — the set bits of the union of the
        // three columns (read before the columns are cleared). Re-examine each for
        // emptied/duplicate parity-sets, updating the live mask. Ascending term-id
        // order matches the sparse backend's merge order, so both dedup alike.
        for w in 0..self.n_words {
            let mut u = self.columns[c0][w] | self.columns[c1][w] | self.columns[c2][w];
            while u != 0 {
                let t = (w as u32) * 64 + u.trailing_zeros();
                if !self.dedup.update_term(t, selection, repr) {
                    self.live_words[w] &= !(1u64 << (t % 64));
                }
                u &= u - 1;
            }
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
/// It runs the identical parity algorithm as [`BitSlicedTermStore`] — same
/// per-selection weight, same reduction and upper-range representative, and the
/// same [`ParitySetDedup`] driven in the same term order — so `topphatt_impl`
/// over it yields encodings identical to the bit-sliced backend; only the
/// representation and performance differ. No mode ceiling.
pub struct SparseListTermStore {
    n_terms: usize,
    /// One ascending, duplicate-free list of term indices per index (length
    /// `3*n_modes + 1`): real Majoranas `0..2*n_nodes`, the all-Z leaf
    /// `2*n_nodes`, and node representatives `2*n_nodes+1..=3*n_nodes`.
    lists: Vec<Vec<u32>>,
    /// Per-term parity-set deduplication state. A dead term is skipped in the
    /// weight merge; shared logic with the bit-sliced backend.
    dedup: ParitySetDedup,
}

impl SparseListTermStore {
    /// Build a sparse inverted index from Majorana-index terms.
    ///
    /// Uses the index-list/bit-sliced upper-range node representative
    /// (`node + 2*n_nodes + 1`), so it is valid on every tree topology and has no
    /// mode ceiling.
    ///
    /// Each term is **parity-canonicalised** (γ²=I): only indices appearing an
    /// odd number of times in the term are recorded, so number-operator terms
    /// like `[0,0]` cancel — matching `qubit_term_weight` and the bit-sliced
    /// backend. Duplicate and identity input terms are merged at build time by
    /// the [`ParitySetDedup`].
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAJORANA_MAX]>], n_modes: usize) -> Self {
        let n_terms = terms.len();
        let n_cols = 3 * n_modes + 1;
        let mut lists = vec![Vec::new(); n_cols];
        let mut sigs: Vec<ArrayVec<[u16; MAJORANA_MAX]>> = Vec::with_capacity(n_terms);
        for (t, term) in terms.iter().enumerate() {
            let parity = parity_set(term);
            for &idx in parity.iter() {
                // Terms are visited in ascending `t`, so each list stays sorted
                // and duplicate-free.
                lists[idx as usize].push(t as u32);
            }
            sigs.push(parity);
        }
        let dedup = ParitySetDedup::new(sigs);
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

        // 3-way merge over the three selected lists, recording each touched term
        // (ascending) and whether it appears in an ODD number of them (so the
        // representative carries it, matching the bit-sliced XOR reduction).
        let mut merged: Vec<(u32, bool)> = Vec::new();
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
                merged.push((m, count & 1 == 1));
            }
        }

        // Re-examine each touched term for emptied/duplicate parity-sets, then
        // install only the live representatives into `repr`'s list (ascending).
        let mut parity: Vec<u32> = Vec::with_capacity(merged.len());
        for (m, odd) in merged {
            let live_after = self.dedup.update_term(m, selection, repr);
            if live_after && odd {
                parity.push(m);
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
    fn transposed_backends_deduplicate_parity_equal_terms() {
        // `[0,0,2,3]` and `[2,3]` are the same Pauli operator (parity-set {2,3}).
        // The transposed backends merge them, so {2,3} is counted once; the
        // index-list backend does not deduplicate at evaluation time, so it
        // counts both. The two transposed backends must still agree with each
        // other.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 0, 2, 3),
            array_vec!([u16; 7] => 2u16, 3),
        ];
        let av = ArrayVecTermStore::new(terms.clone());
        let bits = BitSlicedTermStore::from_arrayvecs(&terms, 4);
        let sparse = SparseListTermStore::from_arrayvecs(&terms, 4);

        // comb [2,3,5]: term {2,3} has two of the three present ⇒ weight 1 each.
        let comb = [2u16, 3, 5];
        assert_eq!(weight_via_store(&av, &comb), 2, "index-list counts both");
        assert_eq!(weight_via_store(&bits, &comb), 1, "bit-sliced dedups");
        assert_eq!(weight_via_store(&sparse, &comb), 1, "sparse-list dedups");
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
