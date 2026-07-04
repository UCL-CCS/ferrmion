use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use log::debug;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::iter::zip;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use thiserror::Error;
use tinyvec::ArrayVec;

use crate::encode::ternarytree::{Child, Edge, TernaryTree, YParity};
use crate::operators::{MajoranaDenseTranspose, MajoranaSparse, MajoranaSparseTranspose};

use super::hatt::{qubit_term_weight, reduce_hamiltonian, MAX_MAJORANAS};

trait Sealed {}

/// Storage and hot-path operations for a Majorana Hamiltonian during TOPP-HATT.
///
/// Implementors own the term collection and expose only what the assignment loop
/// needs, so the orchestration can be written once and run over any backend.
#[allow(private_bounds)]
pub trait Backend: Sealed {
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
    fn evaluate_combination(&self, comb: [u16; 3], bound: &AtomicUsize) -> usize;

    /// Simplify the Hamiltonian after a node's children have been chosen.
    ///
    /// Substitutes the three `selection` indices with a single representative
    /// index and returns that representative, which the caller records so the
    /// restriction system can refer to the new node.
    fn reduce(&mut self, min_parent: usize, selection: [u16; 3], n_leaves: usize) -> u16;
}

impl Sealed for MajoranaSparse {}

impl Backend for MajoranaSparse {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn evaluate_combination(&self, comb: [u16; 3], bound: &AtomicUsize) -> usize {
        let mut sorted_comb: [u16; 3] = comb;
        sorted_comb.sort_unstable();
        let comb_min = sorted_comb[0];
        let comb_max = sorted_comb[2];

        let min_weight = bound.load(Ordering::Relaxed);

        // We expect that the hamiltonian terms are sorted!
        let weight = self
            .indices
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

        weight
    }

    fn reduce(&mut self, min_parent: usize, selection: [u16; 3], n_leaves: usize) -> u16 {
        let parent = (min_parent + n_leaves) as u16;
        let terms = std::mem::take(&mut self.indices);
        self.indices = reduce_hamiltonian(terms, parent, selection);
        parent
    }
}

/// TOPP-HATT backend wrapping a [`MajoranaDenseTranspose`] with whole-term
/// deduplication.
///
/// The transpose is a pure operator representation; this wrapper adds the
/// TOPP-HATT-specific [`MultisetDedup`] (the index-list backend's whole-term
/// rule) plus a packed `live_words` mask. A deduplicated term is excluded from
/// the weight by ANDing the mask into the per-word accumulation, so the fast
/// parity weight evaluation is unchanged and the backend reproduces the
/// index-list backend's encodings exactly.
pub struct DenseTransposeBackend {
    transpose: MajoranaDenseTranspose,
    dedup: MultisetDedup,
    /// Word-packed liveness mask mirroring `dedup.live`; bit `t` set ⇔ term `t`
    /// is still counted.
    live_words: Vec<u64>,
}

impl DenseTransposeBackend {
    /// Build a dense-transpose backend from Majorana-index terms. Builds the pure
    /// transpose, the multiset dedup, and the derived liveness mask. Duplicate
    /// and identity input terms are merged at build time by the [`MultisetDedup`].
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAX_MAJORANAS]>], n_modes: usize) -> Self {
        let transpose = MajoranaDenseTranspose::from_arrayvecs(terms, n_modes);
        let dedup = MultisetDedup::new(terms);
        let mut live_words = vec![0u64; transpose.n_words];
        for (t, &alive) in dedup.live.iter().enumerate() {
            if alive {
                live_words[t / 64] |= 1u64 << (t % 64);
            }
        }
        Self {
            transpose,
            dedup,
            live_words,
        }
    }
}

impl Sealed for DenseTransposeBackend {}

impl Backend for DenseTransposeBackend {
    fn len(&self) -> usize {
        self.transpose.n_terms
    }

    fn evaluate_combination(&self, comb: [u16; 3], bound: &AtomicUsize) -> usize {
        let a = &self.transpose.columns[comb[0] as usize];
        let b = &self.transpose.columns[comb[1] as usize];
        let c = &self.transpose.columns[comb[2] as usize];

        let min_weight = bound.load(Ordering::Relaxed);

        // Per term the Pauli weight is 1 iff exactly one or two of the three
        // children are present: `(a|b|c) & !(a&b&c)` (count 0 or 3 ⇒ identity).
        // AND the live mask so deduplicated terms are not counted. Accumulate
        // word by word, keeping the branch-and-bound early-exit.
        let mut weight = 0usize;
        for w in 0..self.transpose.n_words {
            let any = a[w] | b[w] | c[w];
            let all = a[w] & b[w] & c[w];
            weight += (any & !all & self.live_words[w]).count_ones() as usize;
            if weight > min_weight {
                break;
            }
        }

        bound.fetch_min(weight, Ordering::Relaxed);

        weight
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
        let columns = &mut self.transpose.columns;
        let parity: Vec<u64> = (0..self.transpose.n_words)
            .map(|w| columns[c0][w] ^ columns[c1][w] ^ columns[c2][w])
            .collect();
        for &col in &[c0, c1, c2] {
            columns[col].iter_mut().for_each(|w| *w = 0);
        }
        columns[repr as usize].copy_from_slice(&parity);

        repr
    }
}

/// TOPP-HATT backend wrapping a [`MajoranaSparseTranspose`] with whole-term
/// deduplication.
///
/// The sparse counterpart of [`DenseTransposeBackend`]: the transpose is a pure
/// operator representation and this wrapper adds the same [`MultisetDedup`]
/// driven in the same term order, so it yields encodings identical to the
/// dense-transpose backend *and* to the index-list backend; only the
/// representation and performance differ. Deduplicated terms are skipped in the
/// 3-way merge.
pub struct SparseTransposeBackend {
    transpose: MajoranaSparseTranspose,
    dedup: MultisetDedup,
}

impl SparseTransposeBackend {
    /// Build a sparse-transpose backend from Majorana-index terms. Builds the pure
    /// transpose and the multiset dedup (built from the raw terms, matching the
    /// index-list rule).
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAX_MAJORANAS]>], n_modes: usize) -> Self {
        Self {
            transpose: MajoranaSparseTranspose::from_arrayvecs(terms, n_modes),
            dedup: MultisetDedup::new(terms),
        }
    }
}

impl Sealed for SparseTransposeBackend {}

impl Backend for SparseTransposeBackend {
    fn len(&self) -> usize {
        self.transpose.n_terms
    }

    fn evaluate_combination(&self, comb: [u16; 3], bound: &AtomicUsize) -> usize {
        let a = &self.transpose.lists[comb[0] as usize];
        let b = &self.transpose.lists[comb[1] as usize];
        let c = &self.transpose.lists[comb[2] as usize];

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

        weight
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
            let a = &self.transpose.lists[c0];
            let b = &self.transpose.lists[c1];
            let c = &self.transpose.lists[c2];
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
        self.transpose.lists[c0].clear();
        self.transpose.lists[c1].clear();
        self.transpose.lists[c2].clear();
        self.transpose.lists[repr as usize] = parity;

        repr
    }
}

/// The result of a single TOPP-HATT assignment step.
///
/// Stores the minimum Pauli weight found, the parent node chosen,
/// and the three leaf indices assigned to that node's edges.
#[derive(Debug, Clone, Copy)]
struct Selection {
    pub(crate) min_weight: usize,
    pub(crate) min_parent: usize,
    pub(crate) leaf_indices: [u16; 3],
}

impl Selection {
    /// Sentinel "no selection" value. Used both to initialise the search and as
    /// the identity element of [`ToppHattSelection::combine`]: every real
    /// candidate compares better.
    pub(crate) const WORST: Self = Self {
        min_weight: usize::MAX,
        min_parent: usize::MAX,
        leaf_indices: [u16::MAX; 3],
    };

    /// Pack this selection's three leaf indices into a single `u64` for
    /// deterministic tie-breaking.
    #[inline(always)]
    pub(crate) fn packed_leaf_indices(&self) -> u64 {
        (self.leaf_indices[0] as u64) << 16
            | (self.leaf_indices[1] as u64) << 32
            | (self.leaf_indices[2] as u64) << 48
    }

    /// Combine two candidate selections, keeping the lower Pauli weight and
    /// breaking ties deterministically by the packed leaf indices (see
    /// [`ToppHattSelection::packed_leaf_indices`]).
    ///
    /// This combine is associative and order-independent, so the reduction yields
    /// the same selection whether combinations are reduced in parallel or folded
    /// sequentially.
    #[inline(always)]
    pub(crate) fn combine(self, candidate: Self) -> Self {
        let take_candidate = candidate.min_weight < self.min_weight
            || (candidate.min_weight == self.min_weight
                && self.min_weight != usize::MAX
                && candidate.packed_leaf_indices() > self.packed_leaf_indices());
        if take_candidate {
            candidate
        } else {
            self
        }
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
    sigs: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
    fingerprints: Vec<u64>,
    buckets: HashMap<u64, Vec<u32>>,
    live: Vec<bool>,
}

impl MultisetDedup {
    /// Deterministic FNV-1a fingerprint of a sorted parity-set, used to find
    /// candidate duplicate terms cheaply (collisions are resolved by comparing
    /// the sets directly).
    #[inline]
    fn fingerprint(sig: &[u16]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &x in sig {
            h ^= x as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn new(terms: &[ArrayVec<[u16; MAX_MAJORANAS]>]) -> Self {
        let n = terms.len();
        let mut sigs: Vec<ArrayVec<[u16; MAX_MAJORANAS]>> = Vec::with_capacity(n);
        for term in terms {
            let mut s = *term;
            s.sort_unstable();
            sigs.push(s);
        }
        let mut fingerprints = vec![0u64; n];
        let mut buckets: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut live = vec![true; n];
        for t in 0..n {
            if sigs[t].is_empty() {
                live[t] = false;
                continue;
            }
            let f = Self::fingerprint(&sigs[t]);
            fingerprints[t] = f;
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
            fingerprints,
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
            if let Some(b) = self.buckets.get_mut(&self.fingerprints[t]) {
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
            let f = Self::fingerprint(&self.sigs[t]);
            self.fingerprints[t] = f;
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

/// Strategy for selecting which active node to expand at each TOPP-HATT step.
///
/// `MinWeight` reproduces the original algorithm: every active node is
/// evaluated and the one yielding the lowest Pauli weight is kept. The
/// remaining variants pre-filter `active_nodes` to a single candidate, so the
/// inner weight search only ranges over leaf-index combinations of that one
/// node.
#[derive(Debug, Clone, Copy)]
pub enum NodeOrderHeuristic {
    /// Try every active node, keep the (node, leaves) with the lowest weight.
    MinWeight,
    /// Pick the lowest-indexed active node, then minimise weight over its leaves.
    XFirst,
    /// Pick the highest-indexed active node, then minimise weight over its leaves.
    ZFirst,
    /// Pick a uniformly random active node using a seeded RNG.
    Random { seed: u64 },
}

impl NodeOrderHeuristic {
    /// Build a heuristic from a name
    /// (`"min_weight" | "x_first" | "z_first" | "random"`) and an optional
    /// seed. The seed is only used for `random`; for other variants it is
    /// ignored. When `random` is requested without a seed, the RNG is seeded
    /// with `0` for reproducibility.
    pub fn parse(name: &str, seed: Option<u64>) -> Result<Self, String> {
        match name {
            "min_weight" => Ok(NodeOrderHeuristic::MinWeight),
            "x_first" => Ok(NodeOrderHeuristic::XFirst),
            "z_first" => Ok(NodeOrderHeuristic::ZFirst),
            "random" => Ok(NodeOrderHeuristic::Random {
                seed: seed.unwrap_or(0),
            }),
            other => Err(format!(
                "unknown TOPP-HATT heuristic: {other:?} (expected one of \
                 \"min_weight\", \"x_first\", \"z_first\", \"random\")"
            )),
        }
    }

    /// Reduce `active_nodes` in place according to this heuristic.
    ///
    /// `MinWeight` leaves `active_nodes` untouched (every candidate is later
    /// evaluated). The other variants trim it to a single chosen index, so the
    /// inner search only ranges over leaf-index combinations of one node.
    ///
    /// `rng` must be `Some` whenever `self` is `Random`. It is constructed
    /// once outside the assignment loop so a single seeded stream is consumed
    /// across all iterations.
    pub fn apply(&self, active_nodes: &mut Vec<usize>, rng: Option<&mut Xoshiro256PlusPlus>) {
        match self {
            NodeOrderHeuristic::MinWeight => {}
            NodeOrderHeuristic::XFirst => {
                if let Some(&n) = active_nodes.iter().min() {
                    active_nodes.clear();
                    active_nodes.push(n);
                }
            }
            NodeOrderHeuristic::ZFirst => {
                if let Some(&n) = active_nodes.iter().max() {
                    active_nodes.clear();
                    active_nodes.push(n);
                }
            }
            NodeOrderHeuristic::Random { .. } => {
                let rng = rng.expect("RNG must be provided for Random heuristic.");
                if let Some(&n) = active_nodes.iter().choose(rng) {
                    active_nodes.clear();
                    active_nodes.push(n);
                }
            }
        }
    }
}

/// Error types possible during TOPP-HATT
#[derive(Debug, Error)]
pub enum ToppHattError {
    #[error("Found invalid restriction: {0:?}.")]
    InvalidRestriction(NodeRestriction),
    #[error("No selection made for loop index {0}.")]
    NoSelectionMade(usize),
    #[error("No min parent for loop index {0}.")]
    NoMinParentFound(usize),
}

/// Restrictons on which Majorana operator can be assigned
///
/// Each edge of each node connects to one of:
/// - a node
/// - a leaf, with  or without an assiged Majorana.
/// - nothing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeRestriction {
    /// The edge can have any assignment.
    Any,
    /// The edge must have an odd-indexed Majorana.
    OddLeaf,
    /// The edge must have an even-indexed Majorana.
    EvenLeaf,
    /// The edge must have a specific child node.
    ChildNode(u8),
    /// The edge must have a specific Majorana.
    Majorana(u16),
    /// The edge must have no assignment.
    Empty,
}

impl NodeRestriction {
    /// Find the available subset of Majorana indices which a restricion allows.
    ///
    /// As the procedure progresses, the set of unassigned indices will become
    /// more restrictive. `representatives[c]` gives the index that currently
    /// stands in for child node `c`; this is the term-store's chosen
    /// representative — the upper-range token `c + 2*n_nodes + 1` for both the
    /// index-list and bit-sliced backends.
    fn get_index_subset(
        &self,
        unassigned: &BTreeSet<usize>,
        n_nodes: usize,
        representatives: &[u16],
    ) -> Vec<u16> {
        match self {
            // Incomplete selections.
            NodeRestriction::EvenLeaf => unassigned.iter().map(|v| (2 * v) as u16).collect(),
            NodeRestriction::OddLeaf => unassigned.iter().map(|v| ((2 * v) + 1) as u16).collect(),
            NodeRestriction::Any => {
                let mut allowed: Vec<u16> = unassigned
                    .iter()
                    .map(|v| (2 * v) as u16)
                    .collect::<Vec<u16>>();
                allowed.extend(unassigned.iter().map(|v| (2 * v + 1) as u16));
                allowed
            }
            // Completed selections.
            NodeRestriction::ChildNode(child_index) => {
                vec![representatives[*child_index as usize]]
            }
            NodeRestriction::Empty => vec![(2 * n_nodes) as u16],
            NodeRestriction::Majorana(index) => vec![*index],
        }
    }
}

/// Newtype for the location of a leaf.
///
/// The first field is the node index of its parent node.
/// The second field is the edge on that parent node.
#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
struct LeafLocation(usize, Edge);

/// A pair of leaves.
///
/// Each pair defines the Majorana operators which make up one fermionic operator.
///
#[derive(Debug, PartialEq)]
struct LeafPair {
    x: LeafLocation,
    y: LeafLocation,
}

/// A set of restrictons on which Majorana operators can be assigned to which leaves.
///
/// This is defined for a specific input tree, and guarantees that TOPP-HATT will
/// - generate a valid encoding.
/// - retain the original tree structure.
/// - retain qubit indices on nodes.
/// - produce real-valued.
#[derive(Debug, PartialEq)]
struct TreeRestrictions {
    x: Vec<NodeRestriction>,
    y: Vec<NodeRestriction>,
    z: Vec<NodeRestriction>,
    pairs: HashMap<LeafLocation, LeafLocation>,
}

impl TreeRestrictions {
    /// Create a set of [`TreeRestrictons`] for  a [`TernaryTree`].
    fn new(tree: &TernaryTree) -> Self {
        let x: Vec<NodeRestriction> = vec![NodeRestriction::Any; tree.n_nodes];
        let y: Vec<NodeRestriction> = vec![NodeRestriction::Any; tree.n_nodes];
        let z: Vec<NodeRestriction> = vec![NodeRestriction::Any; tree.n_nodes];
        let pairs: HashMap<LeafLocation, LeafLocation> = HashMap::new();

        let mut output = Self { x, y, z, pairs };

        output.apply_all_z(tree);
        output.apply_retain_children(tree);
        output.apply_leaf_parity(tree);
        output.apply_leaf_pairs(tree);

        output
    }

    /// Add the All-Z leaf restriction.
    ///
    /// For a valid encoding, we need both linear and algebraic independence of
    /// operators. Ternary trees have 2*n_modes+1 leaves, from which we
    /// create a set of 2*n_modes Majorana operators, ensuring both properties.
    /// By convention, the leaf which is reached by the all-Z path is omitted.
    fn apply_all_z(&mut self, tree: &TernaryTree) {
        let all_z_index = tree
            .z_child_of
            .iter()
            .position(|&v| v.is_none())
            .expect("Input tree should not have all-z leaf assigned.");
        self.z[all_z_index] = NodeRestriction::Empty;
    }

    /// Add restrictions to keep parent-child relationships.
    ///
    /// For TOPP-HATT, we wish to keep the structure of the tree constant,
    /// while retaining specific qubit labels for specific nodes.
    /// This allows us to map a tree to the qubit-connectivity of a QPU.
    fn apply_retain_children(&mut self, tree: &TernaryTree) {
        for (restriction, children) in zip(
            [&mut self.x, &mut self.y, &mut self.z],
            [&tree.x_child_of, &tree.y_child_of, &tree.z_child_of],
        ) {
            for (r, c) in zip(restriction, children) {
                if let Some(Child::Node(child_index)) = c {
                    *r = NodeRestriction::ChildNode(*child_index)
                }
            }
        }
    }

    /// Add restrictions to ensure reals-valued terms.
    ///
    /// Each Majorana operator is generated by following the path from
    /// a leaf to the root node.
    /// To make sure we produce Qubit Hamiltonians which have real-valued terms
    /// we need to order pairs of leaves so that the fermionic operators
    /// they define have real values.
    fn apply_leaf_parity(&mut self, tree: &TernaryTree) {
        for (restriction, children) in zip(
            [&mut self.x, &mut self.y, &mut self.z],
            [&tree.x_child_of, &tree.y_child_of, &tree.z_child_of],
        ) {
            for (r, c) in zip(restriction, children) {
                // Late init is helpful here as we
                // want to be able to continue.
                match c {
                    Some(Child::XLeaf(_)) => {
                        *r = NodeRestriction::EvenLeaf;
                    }
                    Some(Child::YLeaf(_)) => {
                        *r = NodeRestriction::OddLeaf;
                    }
                    _ => {
                        continue;
                    }
                }
            }
        }
    }

    /// Add restrictions to enforce vacuum state preservation.
    ///
    /// Each fermionic operator is defined in terms of a pair of
    /// Majorana operators. Within a valid encoding, any set of pairs
    /// of Majoranas would work. However, we can enforce vacuum state preservation
    /// by taking a pair of operators which take each of the X and Y
    /// edge out of some node, and then continue on the Z-edges until
    /// they reach a leaf.
    fn apply_leaf_pairs(&mut self, tree: &TernaryTree) {
        let mut leaf_pairs: Vec<LeafPair> = (0..tree.n_nodes)
            .map(|v| LeafPair {
                x: LeafLocation(v, Edge::X),
                y: LeafLocation(v, Edge::Y),
            })
            .collect();

        for (edge, child_of) in zip(
            [Edge::X, Edge::Y, Edge::Z],
            [&tree.x_child_of, &tree.y_child_of, &tree.z_child_of],
        ) {
            child_of
                .iter()
                .enumerate()
                .for_each(|(parent_index, &child)| {
                    let leaf_index: usize;
                    let y_parity: YParity;
                    match child {
                        Some(Child::XLeaf(ind)) => {
                            leaf_index = ind as usize;
                            y_parity = tree.y_parity_of[leaf_index];
                        }
                        Some(Child::YLeaf(ind)) => {
                            leaf_index = ind as usize;
                            y_parity = !tree.y_parity_of[leaf_index];
                        }
                        _ => {
                            // If  the child is a Node, we continue.
                            return;
                        }
                    }
                    match y_parity {
                        YParity::Even => {
                            let pair = &mut leaf_pairs[leaf_index];
                            pair.x = LeafLocation(parent_index, edge)
                        }
                        YParity::Odd => {
                            let pair = &mut leaf_pairs[leaf_index];
                            pair.y = LeafLocation(parent_index, edge)
                        }
                    }
                });
        }
        leaf_pairs.iter().for_each(|pair| {
            self.pairs.insert(pair.x, pair.y);
            self.pairs.insert(pair.y, pair.x);
        });
    }
}

impl TreeRestrictions {
    /// Assign Majorana indices to the leaves of a tree.
    fn update_tree(self, tree: &mut TernaryTree) -> Result<(), ToppHattError> {
        let n_nodes = &self.x.len();
        debug!("Updatign tree {self:?}");
        debug_assert_eq!(
            &self.y.len(),
            n_nodes,
            "XYZ restrictions should be same length."
        );
        debug_assert_eq!(
            &self.z.len(),
            n_nodes,
            "XYZ restrictions should be same length."
        );
        for (res, child_of) in zip(
            [&self.x, &self.y, &self.z],
            [
                &mut tree.x_child_of,
                &mut tree.y_child_of,
                &mut tree.z_child_of,
            ],
        ) {
            for (r, c) in zip(res, child_of) {
                match r {
                    NodeRestriction::Majorana(index) => {
                        if index % 2 == 0 {
                            *c = Some(Child::XLeaf((index / 2) as u8));
                        } else {
                            *c = Some(Child::YLeaf(((index - 1) / 2) as u8));
                        };
                        debug_assert!(
                            *index < (2 * tree.n_nodes) as u16,
                            "Index too high: {index}"
                        );
                    }
                    NodeRestriction::ChildNode(_) => {
                        debug_assert!(matches!(c, Some(Child::Node(_))));
                    }
                    NodeRestriction::Empty => {
                        debug_assert!(c.is_none())
                    }
                    _ => return Err(ToppHattError::InvalidRestriction(*r)),
                }
            }
        }
        Ok(())
    }
}

/// A flat map of parent-child dependencies between nodes.
#[derive(Debug, PartialEq)]
struct NodeDependencies {
    /// The distance of each node from the root node.
    root_distances: BTreeMap<usize, usize>,
    /// Child nodes of each node which are still to be assigned Majoranas.
    children_without_leaves: BTreeMap<usize, ArrayVec<[usize; 3]>>,
}

impl NodeDependencies {
    /// Create a new set of [`NodeDependencies`].
    fn new(tree: &TernaryTree) -> Self {
        // find the root node by traversing up
        // it will usually be the 0th position so start there
        let mut parent_index: usize = 0;
        while let Some(parent) = tree.parent_of[parent_index] {
            parent_index = parent.node_index();
        }
        debug!("Parent index: {parent_index:?}");
        let mut root_distances: BTreeMap<usize, usize> = BTreeMap::new();
        debug!("{:?}", tree.n_nodes);
        let mut children_without_leaves: BTreeMap<usize, ArrayVec<[usize; 3]>> = BTreeMap::new();

        let mut nodes_to_check: VecDeque<usize> = VecDeque::new();
        nodes_to_check.push_front(parent_index);

        while !nodes_to_check.is_empty() {
            debug!("TO check {:?}", nodes_to_check);
            debug!("RD {:?}", root_distances);
            debug!("UC {:?}", children_without_leaves);
            if let Some(node) = nodes_to_check.pop_front() {
                debug_assert!(!children_without_leaves.contains_key(&node));
                children_without_leaves.insert(node, ArrayVec::new());
                match tree.parent_of[node] {
                    Some(parent) => {
                        root_distances.insert(
                            node,
                            root_distances
                                .get(&parent.node_index())
                                .expect("Parent root distance should be set before getting child.")
                                + 1,
                        );
                    }
                    None => {
                        root_distances.insert(node, 0);
                    }
                }
                for child_of in [&tree.x_child_of, &tree.y_child_of, &tree.z_child_of] {
                    if let Some(Child::Node(child_index)) = child_of[node] {
                        children_without_leaves
                            .entry(node)
                            .and_modify(|v| v.push(child_index as usize));
                        nodes_to_check.push_back(child_index as usize);
                    }
                }
            }
        }
        debug!("{root_distances:?}");
        debug!("{children_without_leaves:?}");
        Self {
            root_distances,
            children_without_leaves,
        }
    }

    /// Remove a node from the set of  [`NodeDependencies`].
    ///
    /// After all the edges of a node are assigned,
    /// it is dropped from the set.
    fn drop_node(&mut self, index: usize) {
        debug!("Dropping Node {:?}", index);
        if !self.root_distances.contains_key(&index) {
            return;
        }
        self.root_distances.remove(&index);
        self.children_without_leaves.remove(&index);
        debug!("{:?}", self.children_without_leaves);
        for v in self.children_without_leaves.values_mut() {
            v.retain(|&i| i != index);
        }
        debug!("{:?}", self.children_without_leaves);
        debug!("Dopped node {:?}", index);
    }
}

/// Toplogy-Preserving Hamiltonian-Adaptive Ternary Tree
///
/// Optimises a given [`TernaryTree`] to minimise the Pauli-weight
/// of the qubit hamiltonian obtained by encoding the input [`MajoranaSparse`] hamiltonian.
///
/// This is a thin wrapper that runs the algorithm over the production
/// [`MajoranaSparse`] backend. See [`topphatt_impl`] to run it over an
/// alternative [`MajoranaTermStore`] (e.g. the bit-packed prototype).
pub fn topphatt<S: Backend + Sync>(
    mut backend: S,
    mut tree: TernaryTree,
    parallelize: bool,
    heuristic: NodeOrderHeuristic,
) -> Result<TernaryTree, ToppHattError> {
    let mut restrictions = TreeRestrictions::new(&tree);
    let mut node_dependencies = NodeDependencies::new(&tree);

    // Rough threshold at which parallelism is worth the overhead. When enabled the
    // weight search runs on rayon's global thread pool.
    let mut use_parallel = parallelize && backend.len() > 1000;

    // Created once outside the assignment loop so a single RNG stream is
    // consumed across all iterations, rather than reseeded each step.
    let mut rng = match heuristic {
        NodeOrderHeuristic::Random { seed } => Some(Xoshiro256PlusPlus::seed_from_u64(seed)),
        _ => None,
    };

    // Reversing the direction tends to give better results for molecules
    let mut unassigned_modes: BTreeSet<usize> = BTreeSet::from_iter(0..tree.n_nodes);

    // Index that currently represents each (eventually formed) node. Initialised
    // to the index-list backend's upper-range token `node + 2*n_nodes + 1`;
    // `store.reduce` overwrites each entry with the backend's own representative
    // as nodes are formed, and entries are only ever read after the node they
    // describe has been reduced (children are reduced before their parents become
    // active).
    let n_leaves_total = 2 * tree.n_nodes + 1;
    let mut representatives: Vec<u16> = (0..tree.n_nodes)
        .map(|node| (node + n_leaves_total) as u16)
        .collect();

    let mut total_weight = 0;
    debug!("Number of hamiltonian terms {:?}", backend.len());
    'assign: for loop_index in 0..tree.n_nodes {
        debug!("loop {:}", loop_index);
        debug!("Restrictions {:?}", restrictions);
        debug!("Dependencies {:?}", node_dependencies);
        debug!("Unassigned Modes {:?}", unassigned_modes);
        let n_leaves = 2 * tree.n_nodes + 1;

        // Best (lowest-weight) candidate found across all active nodes this step.
        let mut best = Selection::WORST;
        // Lowest weight found so far, shared across threads and active nodes to
        // drive the branch-and-bound early-exit in `evaluate_combination`.
        let bound = AtomicUsize::new(usize::MAX);

        debug!("root distances {:?}", node_dependencies.root_distances);
        let max_root_distance: &usize = node_dependencies
            .root_distances
            .values()
            .max()
            .expect("Root distances should have a maximum length.");
        debug!("Max root distance {:?}", max_root_distance);

        let mut active_nodes: Vec<usize> = node_dependencies
            .root_distances
            .iter()
            .zip(node_dependencies.children_without_leaves.values())
            .filter(|&((_, rd), &uc)| (rd == max_root_distance) & (uc == ArrayVec::new()))
            .map(|((&ind, _), _)| ind)
            .collect();

        heuristic.apply(&mut active_nodes, rng.as_mut());

        // This is an optimisation for the case when there are multiple terminal
        // nodes at the same length.
        // Since they can only have one of each of EvenLeaf and Oddleaf on the x and y branches,
        // while the z branch can be either EvenLeaf or OddLeaf.
        if active_nodes.len() > 1 {
            let mut unique_choices: HashSet<(
                &NodeRestriction,
                &NodeRestriction,
                &NodeRestriction,
            )> = HashSet::with_capacity(active_nodes.len());

            active_nodes = active_nodes
                .into_iter()
                .filter(|&active| {
                    let xyz = unique_choices.insert((
                        &restrictions.x[active],
                        &restrictions.y[active],
                        &restrictions.z[active],
                    ));
                    let yxz = unique_choices.insert((
                        &restrictions.y[active],
                        &restrictions.x[active],
                        &restrictions.z[active],
                    ));
                    xyz && yxz
                })
                .collect::<Vec<usize>>();
        }

        debug!("Active Nodes {:?}", active_nodes);
        for active in active_nodes {
            let mut allowed_x = restrictions.x[active].get_index_subset(
                &unassigned_modes,
                tree.n_nodes,
                &representatives,
            );
            // Optimisation:
            // Reversing x, y but leaving z increadsing order reduces the runtime for
            // for hamiltonians in tests.
            allowed_x.reverse();
            let mut allowed_y = restrictions.y[active].get_index_subset(
                &unassigned_modes,
                tree.n_nodes,
                &representatives,
            );
            allowed_y.reverse();
            let allowed_z = restrictions.z[active].get_index_subset(
                &unassigned_modes,
                tree.n_nodes,
                &representatives,
            );

            debug!("Allowed X {:?}", allowed_x);
            debug!("Allowed Y {:?}", allowed_y);
            debug!("Allowed Z {:?}", allowed_z);

            let product = match (restrictions.x[active], restrictions.y[active]) {
                (
                    NodeRestriction::EvenLeaf | NodeRestriction::OddLeaf,
                    NodeRestriction::EvenLeaf | NodeRestriction::OddLeaf,
                ) => [allowed_x, allowed_z].into_iter().multi_cartesian_product(),
                _ => [allowed_x, allowed_y, allowed_z]
                    .into_iter()
                    .multi_cartesian_product(),
            };

            debug!("Product {:?}", product);

            // Find the combination of possible assignments which has the minimum
            // Pauli weight. Materialise the cartesian product so rayon can split it
            // for work-stealing, which load-balances the uneven per-combination cost
            // left by the branch-and-bound early-exit.
            let combos: Vec<Vec<u16>> = product.collect();

            // Each combination is scored independently. The shared `bound` preserves
            // the early-exit across threads, and an associative, deterministic
            // reduction selects the winner regardless of evaluation order.
            //
            // For most trees, using `<` gives the best results (counter example:
            // JKMN(14) benefits from `<=`). This interacts with the ordering of
            // active nodes, which is X-most to Z-most; `combine` keeps the earliest
            // candidate on an exact tie to preserve that behaviour.
            let node_best = if use_parallel {
                combos
                    .par_iter()
                    .filter_map(|comb| normalise_comb(comb))
                    .map(|comb| {
                        let weight = backend.evaluate_combination(comb, &bound);
                        Selection {
                            min_weight: weight,
                            min_parent: active,
                            leaf_indices: comb,
                        }
                    })
                    .reduce(|| Selection::WORST, Selection::combine)
            } else {
                combos
                    .iter()
                    .filter_map(|comb| normalise_comb(comb))
                    .map(|comb| {
                        let weight = backend.evaluate_combination(comb, &bound);
                        Selection {
                            min_weight: weight,
                            min_parent: active,
                            leaf_indices: comb,
                        }
                    })
                    .fold(Selection::WORST, Selection::combine)
            };
            best = best.combine(node_best);
        }
        // debug!("Selection {:?}", &selection);
        let selection = best;
        match selection.leaf_indices {
            [u16::MAX, u16::MAX, u16::MAX] => {
                return Err(ToppHattError::NoSelectionMade(loop_index))
            }
            _ => {
                debug!("Removing selection from unassigned");
                selection
                    .leaf_indices
                    .into_iter()
                    .filter(|&v| n_leaves > v as usize)
                    .map(|v| if v % 2 == 0 { v / 2 } else { (v - 1) / 2 })
                    .for_each(|v| {
                        unassigned_modes.remove(&(v as usize));
                    });
            }
        }
        debug!("Unassigned {:?}", unassigned_modes);
        total_weight += selection.min_weight;
        debug!("Total weight {:?}", total_weight);

        match selection.min_parent {
            usize::MAX => return Err(ToppHattError::NoMinParentFound(loop_index)),
            _ => node_dependencies.drop_node(selection.min_parent),
        }

        debug!("Dropped dependencies");
        for (&sel, res) in zip(
            &selection.leaf_indices,
            [
                &mut restrictions.x,
                &mut restrictions.y,
                &mut restrictions.z,
            ],
        ) {
            if (sel as usize) < n_leaves - 1 {
                res[selection.min_parent] = NodeRestriction::Majorana(sel);
            } else if (sel as usize) == n_leaves {
                res[selection.min_parent] = NodeRestriction::Empty;
            }
        }

        debug!("Selection {:?}", selection);
        // Need to subtract one so that the all-z leaf
        // which is set at index 2*n_nodes doesn't look for a pair.
        // Be careful about zero indexing here too.
        if (selection.leaf_indices[2] as usize) < n_leaves - 1 {
            let pair_index: u16 = if selection.leaf_indices[2].is_multiple_of(2) {
                selection.leaf_indices[2] + 1
            } else {
                selection.leaf_indices[2] - 1
            };
            debug!("pair index {:?}", pair_index);
            let partner_location: LeafLocation = {
                *restrictions
                    .pairs
                    .get(&LeafLocation(selection.min_parent, Edge::Z))
                    .expect("All leaves should have pairs.")
            };
            debug!("partner location {:?}", partner_location);

            match partner_location.1 {
                Edge::X => {
                    restrictions.x[partner_location.0] = NodeRestriction::Majorana(pair_index)
                }
                Edge::Y => {
                    restrictions.y[partner_location.0] = NodeRestriction::Majorana(pair_index)
                }
                Edge::Z => {
                    restrictions.z[partner_location.0] = NodeRestriction::Majorana(pair_index)
                }
            }
        }

        // Check for nods which are now complete thanks to assigning leaf pairs.
        let complete_nodes: Vec<usize> = (0..tree.n_nodes)
            .filter(|&ind| {
                matches!(
                    restrictions.x[ind],
                    NodeRestriction::Majorana(_) | NodeRestriction::ChildNode(_)
                ) & matches!(
                    restrictions.y[ind],
                    NodeRestriction::Majorana(_) | NodeRestriction::ChildNode(_)
                ) & matches!(
                    restrictions.z[ind],
                    NodeRestriction::Majorana(_)
                        | NodeRestriction::ChildNode(_)
                        | NodeRestriction::Empty
                )
            })
            .collect();
        debug!("Complete nodes {:?}", complete_nodes);
        complete_nodes
            .iter()
            .for_each(|&ind| node_dependencies.drop_node(ind));

        let representative = backend.reduce(selection.min_parent, selection.leaf_indices, n_leaves);
        representatives[selection.min_parent] = representative;
        debug!(
            "Node {} represented by index {representative}.",
            selection.min_parent
        );
        if backend.len() < 1000 {
            use_parallel = false;
        }
        debug!("Reduced Hamiltonian to {} terms", backend.len());
        debug!("Finished loop\n\n\n");
        if unassigned_modes.is_empty() {
            break 'assign;
        }
    }
    debug!("TOPPHATT Complete");
    debug!("Restrictions {:?}", restrictions);
    debug!("Dependencies {:?}", node_dependencies);
    debug!("Unassigned {:?}", unassigned_modes);
    debug!("Total weight: {:}", total_weight);
    debug!("Tree {:?}", tree);

    debug!("Update tree");
    restrictions.update_tree(&mut tree)?;
    debug!("Tree {:?}", tree);
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use tinyvec::array_vec;

    fn weight_via_store<S: Backend>(store: &S, comb: &[u16]) -> usize {
        let bound = AtomicUsize::new(usize::MAX);
        let comb = normalise_comb(comb);
        if let Some(comb) = comb {
            store.evaluate_combination(comb, &bound)
        } else {
            usize::MAX
        }
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
        let av =
            MajoranaSparse::new(terms.clone(), vec![Complex64::ONE; terms.len()], 0.0).unwrap();
        let bits = DenseTransposeBackend::from_arrayvecs(&terms, 4);
        let sparse = SparseTransposeBackend::from_arrayvecs(&terms, 4);

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
        let av =
            MajoranaSparse::new(terms.clone(), vec![Complex64::ONE; terms.len()], 0.0).unwrap();
        let bits = DenseTransposeBackend::from_arrayvecs(&terms, 4);
        let sparse = SparseTransposeBackend::from_arrayvecs(&terms, 4);

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
        let av =
            MajoranaSparse::new(terms.clone(), vec![Complex64::ONE; terms.len()], 0.0).unwrap();
        let sliced = DenseTransposeBackend::from_arrayvecs(&terms, 4);

        for comb in [[0u16, 1, 2], [0, 1, 3], [2, 3, 4], [1, 4, 5]] {
            assert_eq!(
                weight_via_store(&av, &comb),
                weight_via_store(&sliced, &comb),
                "bit-sliced weights differ for comb {comb:?}"
            );
        }
    }

    #[test]
    fn dense_transpose_weight_no_mode_ceiling() {
        // Indices well past any native-word ceiling: the transposed store has no
        // limit because words slice terms, not indices.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 200),
            array_vec!([u16; 7] => 100u16, 200, 250),
        ];
        let av =
            MajoranaSparse::new(terms.clone(), vec![Complex64::ONE; terms.len()], 0.0).unwrap();
        let sliced = DenseTransposeBackend::from_arrayvecs(&terms, 130);
        for comb in [[0u16, 100, 200], [0, 200, 250], [100, 200, 250]] {
            assert_eq!(
                weight_via_store(&av, &comb),
                weight_via_store(&sliced, &comb),
                "bit-sliced weight differs for comb {comb:?}"
            );
        }
    }

    #[test]
    fn dense_transpose_reduce_parity_matches() {
        // Same reduce case as `bit_reduce_parity_matches`, checked via the
        // post-reduction weights (the transposed store has no flat term list).
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1, 2, 3),
            array_vec!([u16; 7] => 0u16, 2, 3, 4),
        ];
        let n_leaves = 57;
        let mut sliced = DenseTransposeBackend::from_arrayvecs(&terms, 28);
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
        let n_terms = reduced.len();
        let expected = MajoranaSparse::new(reduced, vec![Complex64::ONE; n_terms], 0.0).unwrap();
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
        let av =
            MajoranaSparse::new(terms.clone(), vec![Complex64::ONE; terms.len()], 0.0).unwrap();
        let sparse = SparseTransposeBackend::from_arrayvecs(&terms, 4);

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
        let av =
            MajoranaSparse::new(terms.clone(), vec![Complex64::ONE; terms.len()], 0.0).unwrap();
        let sparse = SparseTransposeBackend::from_arrayvecs(&terms, 130);
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
        // Same reduce case as `dense_transpose_reduce_parity_matches`, checked via the
        // post-reduction weights.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1, 2, 3),
            array_vec!([u16; 7] => 0u16, 2, 3, 4),
        ];
        let n_leaves = 57;
        let mut sparse = SparseTransposeBackend::from_arrayvecs(&terms, 28);
        let repr = sparse.reduce(0, [2, 3, 55], n_leaves);
        assert_eq!(repr, n_leaves as u16);

        let reduced = vec![
            array_vec!([u16; 7] => 0u16, 1),
            array_vec!([u16; 7] => 0u16, 4),
        ];
        let n_terms = reduced.len();
        let expected = MajoranaSparse::new(reduced, vec![Complex64::ONE; n_terms], 0.0).unwrap();
        for comb in [[0u16, 1, 4], [1, 4, 5], [0, 1, 2]] {
            assert_eq!(
                weight_via_store(&expected, &comb),
                weight_via_store(&sparse, &comb),
                "reduced sparse-list weight differs for comb {comb:?}"
            );
        }
    }
}

#[cfg(test)]
mod test_topphatt {
    use super::Edge::{X, Y, Z};
    use super::NodeRestriction::{ChildNode, Empty, EvenLeaf, OddLeaf};
    use super::*;
    use crate::encode::majorana::MajoranaEncoding;
    use crate::encode::ternarytree::TTFlatpack;
    use crate::encode::ternarytree::TernaryTree;
    use crate::optimise::ternarytree::hatt::{qubit_term_weight, reduce_hamiltonian};
    use log::debug;
    use ndarray::arr1;
    use num_complex::Complex64;
    use tinyvec::array_vec;

    #[test]
    fn test_qubit_term_weight() {
        assert_eq!(qubit_term_weight(&array_vec!(0u16), &[0u16, 1u16, 2u16]), 1);
        assert_eq!(qubit_term_weight(&array_vec!(1u16), &[0u16, 1u16, 2u16]), 1);
        assert_eq!(qubit_term_weight(&array_vec!(2u16), &[0u16, 1u16, 2u16]), 1);
        assert_eq!(
            qubit_term_weight(&array_vec!(0u16, 0u16), &[0u16, 1u16, 2u16]),
            0
        );
        assert_eq!(
            qubit_term_weight(&array_vec!(0u16, 1u16, 2u16), &[0u16, 1u16, 2u16]),
            0
        );
        assert_eq!(
            qubit_term_weight(&array_vec!(0u16, 1u16), &[0u16, 1u16, 2u16]),
            1
        );
        assert_eq!(
            qubit_term_weight(&array_vec!(0u16, 3u16, 4u16, 5u16), &[0u16, 1u16, 2u16]),
            1
        );
        assert_eq!(
            qubit_term_weight(&array_vec!(0u16, 0u16, 0u16, 0u16), &[0u16, 1u16, 2u16]),
            0
        );
    }

    #[test]
    fn test_jw_restrictions() {
        let jw_tree = TernaryTree::naive_jordan_wigner(4);
        let jw_restrictions = TreeRestrictions::new(&jw_tree);
        debug!("{:?}", jw_restrictions);
        let mut expected_pairs: HashMap<LeafLocation, LeafLocation> = HashMap::new();
        expected_pairs.insert(LeafLocation(0, X), LeafLocation(0, Y));
        expected_pairs.insert(LeafLocation(0, Y), LeafLocation(0, X));
        expected_pairs.insert(LeafLocation(1, X), LeafLocation(1, Y));
        expected_pairs.insert(LeafLocation(1, Y), LeafLocation(1, X));
        expected_pairs.insert(LeafLocation(2, X), LeafLocation(2, Y));
        expected_pairs.insert(LeafLocation(2, Y), LeafLocation(2, X));
        expected_pairs.insert(LeafLocation(3, X), LeafLocation(3, Y));
        expected_pairs.insert(LeafLocation(3, Y), LeafLocation(3, X));

        let expected = TreeRestrictions {
            x: vec![EvenLeaf, EvenLeaf, EvenLeaf, EvenLeaf],
            y: vec![OddLeaf, OddLeaf, OddLeaf, OddLeaf],
            z: vec![ChildNode(1), ChildNode(2), ChildNode(3), Empty],
            pairs: expected_pairs,
        };
        assert_eq!(expected, jw_restrictions, "Test JW(4) Restrictions.");
    }

    #[test]
    fn test_pe_restrictions() {
        let tree = TernaryTree::naive_parity(3);
        let restrictions = TreeRestrictions::new(&tree);
        debug!("{:?}", restrictions);
        let mut expected_pairs: HashMap<LeafLocation, LeafLocation> = HashMap::new();
        let pairs = [((1, Z), (0, Y)), ((2, Z), (1, Y)), ((2, X), (2, Y))];
        pairs.iter().for_each(|&(k, v)| {
            let k = LeafLocation(k.0, k.1);
            let v = LeafLocation(v.0, v.1);
            expected_pairs.insert(k, v);
            expected_pairs.insert(v, k);
        });

        let expected = TreeRestrictions {
            x: vec![ChildNode(1), ChildNode(2), EvenLeaf],
            y: vec![OddLeaf, OddLeaf, OddLeaf],
            z: vec![Empty, EvenLeaf, EvenLeaf],
            pairs: expected_pairs,
        };
        assert_eq!(expected, restrictions, "Test Parity(4) Restrictions.");
    }

    #[test]
    fn test_jkmn_restrictions() {
        let tree = TernaryTree::naive_jkmn(6);
        let restrictions = TreeRestrictions::new(&tree);
        debug!("{:?}", restrictions);
        let mut expected_pairs = HashMap::new();
        let pairs = [
            ((1, Z), (2, Z)),
            ((4, Z), (5, Z)),
            ((2, Y), (2, X)),
            ((3, X), (3, Y)),
            ((4, X), (4, Y)),
            ((5, X), (5, Y)),
        ];
        pairs.iter().for_each(|&(k, v)| {
            let k = LeafLocation(k.0, k.1);
            let v = LeafLocation(v.0, v.1);
            expected_pairs.insert(k, v);
            expected_pairs.insert(v, k);
        });

        let expected = TreeRestrictions {
            x: vec![
                ChildNode(1),
                ChildNode(4),
                EvenLeaf,
                EvenLeaf,
                EvenLeaf,
                EvenLeaf,
            ],
            y: vec![
                ChildNode(2),
                ChildNode(5),
                OddLeaf,
                OddLeaf,
                OddLeaf,
                OddLeaf,
            ],
            z: vec![ChildNode(3), EvenLeaf, OddLeaf, Empty, EvenLeaf, OddLeaf],
            pairs: expected_pairs,
        };
        assert_eq!(restrictions, expected, "Test JKMN(6) Restrictions.");
    }

    #[test]
    fn test_node_dependencies_jw_pe() {
        let mut expected_dists = BTreeMap::new();
        expected_dists.insert(0, 0);
        expected_dists.insert(1, 1);
        expected_dists.insert(2, 2);
        expected_dists.insert(3, 3);
        let mut expected_children = BTreeMap::new();
        expected_children.insert(0, array_vec!(1));
        expected_children.insert(1, array_vec!(2));
        expected_children.insert(2, array_vec!(3));
        expected_children.insert(3, array_vec!());
        let jw_tree = TernaryTree::naive_jordan_wigner(4);
        let pe_tree = TernaryTree::naive_parity(4);
        let jw_deps = NodeDependencies::new(&jw_tree);
        let pe_deps = NodeDependencies::new(&pe_tree);
        assert_eq!(expected_dists, jw_deps.root_distances);
        assert_eq!(expected_children, jw_deps.children_without_leaves);
        assert_eq!(jw_deps, pe_deps);
    }

    #[test]
    fn test_node_dependencies_bk() {
        let mut expected_dists = BTreeMap::new();
        expected_dists.insert(0, 0);
        expected_dists.insert(1, 1);
        expected_dists.insert(2, 2);
        expected_dists.insert(3, 2);
        let mut expected_children = BTreeMap::new();
        expected_children.insert(0, array_vec!(1));
        expected_children.insert(1, array_vec!(2, 3));
        expected_children.insert(2, array_vec!());
        expected_children.insert(3, array_vec!());
        let tree = TernaryTree::naive_bravyi_kitaev(4);
        let deps = NodeDependencies::new(&tree);
        assert_eq!(expected_dists, deps.root_distances);
        assert_eq!(expected_children, deps.children_without_leaves);
    }
    #[test]
    fn test_node_dependencies_jkmn() {
        let mut expected_dists = BTreeMap::new();
        expected_dists.insert(0, 0);
        expected_dists.insert(1, 1);
        expected_dists.insert(2, 1);
        expected_dists.insert(3, 1);
        expected_dists.insert(4, 2);
        expected_dists.insert(5, 2);
        expected_dists.insert(6, 2);
        let mut expected_children = BTreeMap::new();
        expected_children.insert(0, array_vec!(1, 2, 3));
        expected_children.insert(1, array_vec!(4, 5, 6));
        expected_children.insert(2, array_vec!());
        expected_children.insert(3, array_vec!());
        expected_children.insert(4, array_vec!());
        expected_children.insert(5, array_vec!());
        expected_children.insert(6, array_vec!());
        let tree = TernaryTree::naive_jkmn(7);
        let deps = NodeDependencies::new(&tree);
        assert_eq!(expected_dists, deps.root_distances);
        assert_eq!(expected_children, deps.children_without_leaves);
    }

    #[test]
    fn test_drop_node_dependency() {
        let jw_tree = TernaryTree::naive_jordan_wigner(4);
        let mut jw_deps = NodeDependencies::new(&jw_tree);
        // assert!(jw_deps.drop_node(0).is_err());
        let mut expected_dists = BTreeMap::new();
        expected_dists.insert(0, 0);
        expected_dists.insert(1, 1);
        expected_dists.insert(2, 2);
        expected_dists.insert(3, 3);
        let mut expected_children = BTreeMap::new();
        expected_children.insert(0, array_vec!(1));
        expected_children.insert(1, array_vec!(2));
        expected_children.insert(2, array_vec!(3));
        expected_children.insert(3, array_vec!());

        assert_eq!(jw_deps.root_distances, expected_dists);
        assert_eq!(jw_deps.children_without_leaves, expected_children);
        jw_deps.drop_node(3);

        let mut expected_dists = BTreeMap::new();
        expected_dists.insert(0, 0);
        expected_dists.insert(1, 1);
        expected_dists.insert(2, 2);
        let mut expected_children = BTreeMap::new();
        expected_children.insert(0, array_vec!(1));
        expected_children.insert(1, array_vec!(2));
        expected_children.insert(2, array_vec!());
        assert_eq!(jw_deps.root_distances, expected_dists);
        assert_eq!(jw_deps.children_without_leaves, expected_children);
    }

    #[test]
    fn test_topphatt() {
        let hamiltonian = MajoranaSparse::new(
            vec![array_vec!([u16; 7]=> 2,3)],
            vec![Complex64::new(1., 0.)],
            0.,
        )
        .unwrap();
        let tree = TernaryTree::naive_jordan_wigner(3);

        let jw_topphatt = topphatt(hamiltonian, tree, true, NodeOrderHeuristic::MinWeight).unwrap();
        let encoding: MajoranaEncoding = jw_topphatt.build_encoding(3).unwrap();
        assert_eq!(*encoding.operators.ipowers(), arr1(&[0, 1, 0, 1, 0, 1]));
        // assert_eq!(
        //     encoding.symplectics,
        //     arr2(&[
        //         [false, false, true, true, true, false],
        //         [false, false, true, true, true, true],
        //         [true, false, false, false, false, false],
        //         [true, false, false, true, false, false],
        //         [false, true, false, true, false, false],
        //         [false, true, false, true, true, false],
        //     ])
        // );
    }

    #[test]
    fn test_with_qubit_labels() {
        let hamiltonian = MajoranaSparse::new(
            vec![array_vec!([u16; 7]=> 2,3)],
            vec![Complex64::new(1., 0.)],
            0.,
        )
        .unwrap();
        let flatpack: TTFlatpack = vec![
            (1, (None, None, Some(2))),
            (2, (None, None, Some(3))),
            (3, (None, None, None)),
        ];

        let tree = TernaryTree::from_flatpack_naive(&flatpack).unwrap();
        let jw_topphatt = topphatt(hamiltonian, tree, true, NodeOrderHeuristic::MinWeight).unwrap();
        let encoding = jw_topphatt.build_encoding(4).unwrap();
        assert_eq!(*encoding.operators.ipowers(), arr1(&[0, 1, 0, 1, 0, 1]));
        // assert_eq!(
        //     encoding.symplectics,
        //     arr2(&[
        //         [false, false, false, true, false, true, true, false],
        //         [false, false, false, true, false, true, true, true],
        //         [false, true, false, false, false, false, false, false],
        //         [false, true, false, false, false, true, false, false],
        //         [false, false, true, false, false, true, false, false],
        //         [false, false, true, false, false, true, true, false],
        //     ])
        // );
    }

    /// Multi-term Hamiltonian on a JKMN(7) tree. JKMN has four leaf-only nodes
    /// at the deepest level on the first assignment iteration, so the heuristic
    /// has a non-trivial choice to make.
    fn multi_active_fixture() -> (MajoranaSparse, TernaryTree) {
        let hamiltonian = MajoranaSparse::new(
            vec![
                array_vec!([u16; 7] => 0, 1, 2, 3),
                array_vec!([u16; 7] => 4, 5, 6, 7),
                array_vec!([u16; 7] => 2, 3, 8, 9),
                array_vec!([u16; 7] => 10, 11, 12, 13),
            ],
            vec![
                Complex64::new(1., 0.),
                Complex64::new(1., 0.),
                Complex64::new(1., 0.),
                Complex64::new(1., 0.),
            ],
            0.,
        )
        .unwrap();
        let tree = TernaryTree::naive_jkmn(7);
        (hamiltonian, tree)
    }

    #[test]
    fn test_topphatt_x_first_and_z_first_diverge() {
        let (h_x, tree_x) = multi_active_fixture();
        let (h_z, tree_z) = multi_active_fixture();

        let x_tree = topphatt(h_x, tree_x, false, NodeOrderHeuristic::XFirst).unwrap();
        let z_tree = topphatt(h_z, tree_z, false, NodeOrderHeuristic::ZFirst).unwrap();

        let x_enc = x_tree.build_encoding(7).unwrap();
        let z_enc = z_tree.build_encoding(7).unwrap();

        // Both heuristics still produce valid 7-mode encodings.
        assert_eq!(x_enc.operators.ipowers().len(), 14);
        assert_eq!(z_enc.operators.ipowers().len(), 14);

        // The two heuristics walk active_nodes from opposite ends, so on a
        // branched tree the resulting symplectic matrix should differ.
        assert_ne!(
            x_enc.operators.x_bools(),
            z_enc.operators.x_bools(),
            "XFirst and ZFirst should produce distinct encodings on JKMN(7)"
        );
    }

    #[test]
    fn test_topphatt_random_reproducible() {
        let (h_a, tree_a) = multi_active_fixture();
        let (h_b, tree_b) = multi_active_fixture();

        let tree_first =
            topphatt(h_a, tree_a, false, NodeOrderHeuristic::Random { seed: 42 }).unwrap();
        let tree_second =
            topphatt(h_b, tree_b, false, NodeOrderHeuristic::Random { seed: 42 }).unwrap();

        let enc_first = tree_first.build_encoding(7).unwrap();
        let enc_second = tree_second.build_encoding(7).unwrap();

        assert_eq!(
            enc_first.operators.ipowers(),
            enc_second.operators.ipowers()
        );
        assert_eq!(
            enc_first.operators.x_bools(),
            enc_second.operators.x_bools()
        );
        assert_eq!(
            enc_first.operators.z_bools(),
            enc_second.operators.z_bools()
        );
    }

    #[test]
    fn test_topphatt_random_seeds_can_differ() {
        // With four active leaf nodes per step on JKMN(7), distinct seeds
        // should pick different active-node sequences and yield different
        // encodings for at least one of these probe seeds.
        let (h_ref, tree_ref) = multi_active_fixture();
        let reference = topphatt(
            h_ref,
            tree_ref,
            false,
            NodeOrderHeuristic::Random { seed: 0 },
        )
        .unwrap();
        let ref_enc = reference.build_encoding(7).unwrap();

        let probe_seeds = [1u64, 7, 13, 42, 99, 1234];
        let mut found_difference = false;
        for seed in probe_seeds {
            let (h, tree) = multi_active_fixture();
            let other = topphatt(h, tree, false, NodeOrderHeuristic::Random { seed }).unwrap();
            let other_enc = other.build_encoding(7).unwrap();
            if other_enc.operators.x_bools() != ref_enc.operators.x_bools() {
                found_difference = true;
                break;
            }
        }
        assert!(
            found_difference,
            "At least one of the probe seeds should diverge from seed=0"
        );
    }

    #[test]
    fn test_reduce_hamiltonian_substitutes_inplace() {
        let mut hamiltonian = vec![
            array_vec!([u16;7] => 0,1,2,3),
            array_vec!([u16;7] => 0,2,3,4),
        ];

        hamiltonian = reduce_hamiltonian(hamiltonian, 999, [2, 3, 55]);

        let expected = vec![
            array_vec!([u16;7] => 0,1,999,999),
            array_vec!([u16;7] => 0,4,999,999),
        ];

        assert_eq!(hamiltonian, expected);
    }

    /// Run the index-list and bit-sliced backends on the same input and assert
    /// they produce identical encodings. `make_tree` is called per run because
    /// [`TernaryTree`] is not `Clone` and each run consumes its tree. (The
    /// bit-sliced backend deduplicates whole terms on the same multiset rule as
    /// the index-list backend, so they match exactly.)
    fn assert_backends_agree(
        hamiltonian: MajoranaSparse,
        make_tree: impl Fn() -> TernaryTree,
        n_modes: usize,
        heuristic: NodeOrderHeuristic,
    ) {
        let sliced = DenseTransposeBackend::from_arrayvecs(&hamiltonian.indices, n_modes);

        let av_tree = topphatt(hamiltonian, make_tree(), false, heuristic).unwrap();
        let sliced_tree = topphatt(sliced, make_tree(), false, heuristic).unwrap();

        let av = av_tree.build_encoding(n_modes).unwrap();
        let bs = sliced_tree.build_encoding(n_modes).unwrap();

        assert_eq!(
            av.operators.x_bools(),
            bs.operators.x_bools(),
            "x_block differs"
        );
        assert_eq!(
            av.operators.z_bools(),
            bs.operators.z_bools(),
            "z_block differs"
        );
        assert_eq!(
            av.operators.ipowers(),
            bs.operators.ipowers(),
            "ipowers differ"
        );
    }

    #[test]
    fn test_bit_backend_matches_arrayvec_on_fixture() {
        let (hamiltonian, _tree) = multi_active_fixture();
        assert_backends_agree(
            hamiltonian,
            || TernaryTree::naive_jkmn(7),
            7,
            NodeOrderHeuristic::MinWeight,
        );
    }

    /// Deterministic random Majorana Hamiltonian for `n_modes` modes: `n_terms`
    /// terms of length 2 or 4 with distinct, sorted indices in `0..2*n_modes`.
    fn random_majorana(n_modes: usize, n_terms: usize, seed: u64) -> MajoranaSparse {
        use rand::Rng;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let n_majoranas = 2 * n_modes;
        let mut indices = Vec::with_capacity(n_terms);
        let mut coefficients = Vec::with_capacity(n_terms);
        while indices.len() < n_terms {
            let len = if rng.random_bool(0.5) { 2 } else { 4 };
            let mut chosen: BTreeSet<u16> = BTreeSet::new();
            while chosen.len() < len {
                chosen.insert(rng.random_range(0..n_majoranas) as u16);
            }
            let term: ArrayVec<[u16; 7]> = chosen.into_iter().collect();
            indices.push(term);
            coefficients.push(Complex64::new(1.0, 0.0));
        }
        // De-duplicate so the index-list backend's terms are unique, matching
        // how real Majorana Hamiltonians are prepared.
        let mut seen: HashSet<ArrayVec<[u16; 7]>> = HashSet::new();
        let mut uniq_indices = Vec::new();
        let mut uniq_coeffs = Vec::new();
        for (t, c) in indices.into_iter().zip(coefficients) {
            if seen.insert(t) {
                uniq_indices.push(t);
                uniq_coeffs.push(c);
            }
        }
        MajoranaSparse::new(uniq_indices, uniq_coeffs, 0.0).unwrap()
    }

    /// All three backends must produce **identical** valid encodings on random
    /// inputs: the transposed backends deduplicate whole terms on the same
    /// multiset rule as the index-list backend, so `index_list == dense_transpose ==
    /// sparse_list` (x/z blocks and ipowers) for every instance.
    #[test]
    fn test_dense_transpose_valid_encodings_random() {
        for n_modes in [4usize, 6, 8, 10] {
            for seed in 0..20u64 {
                let hamiltonian = random_majorana(n_modes, 6 * n_modes, seed);
                let n_majoranas = 2 * n_modes;
                let sliced = DenseTransposeBackend::from_arrayvecs(&hamiltonian.indices, n_modes);
                let sparse = SparseTransposeBackend::from_arrayvecs(&hamiltonian.indices, n_modes);
                let av_tree = topphatt(
                    hamiltonian,
                    TernaryTree::naive_jkmn(n_modes),
                    false,
                    NodeOrderHeuristic::MinWeight,
                )
                .unwrap();
                let t_sliced = topphatt(
                    sliced,
                    TernaryTree::naive_jkmn(n_modes),
                    false,
                    NodeOrderHeuristic::MinWeight,
                )
                .unwrap();
                let t_sparse = topphatt(
                    sparse,
                    TernaryTree::naive_jkmn(n_modes),
                    false,
                    NodeOrderHeuristic::MinWeight,
                )
                .unwrap();
                let av = av_tree.build_encoding(n_modes).unwrap();
                let e_sliced = t_sliced.build_encoding(n_modes).unwrap();
                let e_sparse = t_sparse.build_encoding(n_modes).unwrap();

                // A valid n-mode encoding has 2*n Majorana operators.
                assert_eq!(av.operators.ipowers().len(), n_majoranas);

                // The transposed backends now deduplicate whole terms on the same
                // multiset rule as the index-list backend, so all three produce
                // identical encodings.
                assert_eq!(e_sliced.operators.x_bools(), av.operators.x_bools());
                assert_eq!(e_sliced.operators.z_bools(), av.operators.z_bools());
                assert_eq!(e_sliced.operators.ipowers(), av.operators.ipowers());
                assert_eq!(e_sparse.operators.x_bools(), av.operators.x_bools());
                assert_eq!(e_sparse.operators.z_bools(), av.operators.z_bools());
                assert_eq!(e_sparse.operators.ipowers(), av.operators.ipowers());
            }
        }
    }

    /// `dense_transpose` must produce valid encodings on *every* tree topology, not
    /// just JKMN. The Jordan-Wigner chain in particular has node z-edges whose
    /// representative previously tripped the orchestration's magnitude-based edge
    /// classification (panic: "All leaves should have pairs"); the upper-range
    /// representative fixes it. The index-list backend is checked alongside as a
    /// control.
    #[test]
    fn test_dense_transpose_valid_on_all_topologies() {
        let n_modes = 6;
        for name in ["jordan_wigner", "parity", "bravyi_kitaev", "jkmn"] {
            let build = |n: usize| match name {
                "jordan_wigner" => TernaryTree::naive_jordan_wigner(n),
                "parity" => TernaryTree::naive_parity(n),
                "bravyi_kitaev" => TernaryTree::naive_bravyi_kitaev(n),
                _ => TernaryTree::naive_jkmn(n),
            };
            for seed in 0..5u64 {
                let hamiltonian = random_majorana(n_modes, 6 * n_modes, seed);

                let il = topphatt(
                    hamiltonian.clone(),
                    build(n_modes),
                    false,
                    NodeOrderHeuristic::MinWeight,
                )
                .unwrap()
                .build_encoding(n_modes)
                .unwrap();
                assert_eq!(
                    il.operators.ipowers().len(),
                    2 * n_modes,
                    "index_list {name} seed {seed}"
                );

                let store = DenseTransposeBackend::from_arrayvecs(&hamiltonian.indices, n_modes);
                let sliced = topphatt(store, build(n_modes), false, NodeOrderHeuristic::MinWeight)
                    .unwrap()
                    .build_encoding(n_modes)
                    .unwrap();
                // Multiset dedup makes dense_transpose match index_list exactly on
                // every topology.
                assert_eq!(
                    sliced.operators.x_bools(),
                    il.operators.x_bools(),
                    "dense_transpose vs index_list x {name} seed {seed}"
                );
                assert_eq!(
                    sliced.operators.z_bools(),
                    il.operators.z_bools(),
                    "dense_transpose vs index_list z {name} seed {seed}"
                );

                // The sparse inverted-index store must agree exactly too.
                let sparse_store =
                    SparseTransposeBackend::from_arrayvecs(&hamiltonian.indices, n_modes);
                let sparse = topphatt(
                    sparse_store,
                    build(n_modes),
                    false,
                    NodeOrderHeuristic::MinWeight,
                )
                .unwrap()
                .build_encoding(n_modes)
                .unwrap();
                assert_eq!(
                    sparse.operators.x_bools(),
                    il.operators.x_bools(),
                    "sparse vs index_list x {name} seed {seed}"
                );
                assert_eq!(
                    sparse.operators.z_bools(),
                    il.operators.z_bools(),
                    "sparse vs index_list z {name} seed {seed}"
                );
            }
        }
    }
}
