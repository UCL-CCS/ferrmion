//! Hamming-cube geometry of ternary-tree Pauli weight.
//!
//! Each Hamiltonian term is a parity set `A` of Majorana indices. For any set
//! `B` of Majoranas we write `col_B ∈ F₂^T` for the vector whose `t`-th bit is
//! `|A_t ∩ B| mod 2`. For an even term this bit is 1 exactly when `γ_A`
//! anticommutes with `γ_B`. For a single Majorana `a`, `col_a` marks the terms
//! that contain it.
//!
//! In a ternary tree every subtree `v` is then a point
//! `π_v = ⊕_{leaves ℓ under v} col_{σ(ℓ)}` of the Hamming cube. The Pauli
//! weight that the encoded Hamiltonian places on the qubit of node `v` is the
//! **half-perimeter** of the triangle formed by its three children's points:
//!
//! ```text
//! wt_v = ½ (d(π_X, π_Y) + d(π_Y, π_Z) + d(π_X, π_Z)) = |(π_X ∨ π_Y ∨ π_Z) ∧ ¬(π_X ∧ π_Y ∧ π_Z)|
//! ```
//!
//! The total Pauli weight is the sum of these over the nodes. Equivalently,
//! single-qubit Paulis pull back to Majorana monomials γ_B, so the weight is
//! half the number of anticommuting (term, single-qubit Pauli) pairs.
//!
//! This module provides
//! - [`parity_canonical_terms`]: the exact set of distinct Pauli strings the
//!   encoded Hamiltonian will contain. The set is encoding independent,
//!   because every Majorana encoding is an algebra isomorphism.
//! - [`HammingColumns`] and [`tree_pauli_weight`]: an exact, encoding-free
//!   evaluation of the Pauli weight of a tree.
//! - [`SlotModel`]: the vacuum-preserving, real-coefficient search space of a
//!   fixed topology, written as a permutation of modes over leaf-pair "slots".
//! - [`topdown`] and [`exhaustive`]: a top-down hierarchical partition
//!   optimiser, and an exact optimiser for small systems.
use std::collections::{BTreeMap, HashMap, VecDeque};

use num_complex::Complex64;
use thiserror::Error;
use tinyvec::ArrayVec;

use super::hatt::MAX_MAJORANAS;
use crate::encode::ternarytree::{Child, Edge, TernaryTree};
use crate::operators::MajoranaSparse;
use crate::utils::COEFFICIENT_TOLERANCE;

/// Errors from the geometric optimisers.
#[derive(Debug, Error)]
pub enum GeometryError {
    #[error(
        "Tree has no leaf for Majorana {0}; the input tree must carry a complete leaf assignment."
    )]
    MissingLeaf(usize),
    #[error("Exhaustive search is limited to {max} modes, got {got}.")]
    TooManyModes { max: usize, got: usize },
}

type Term = ArrayVec<[u16; MAX_MAJORANAS]>;

/// Reduce every term to its sorted odd-multiplicity index set. Coefficients
/// are merged with the sign from reordering, and terms whose net coefficient
/// vanishes are dropped, as is the identity.
///
/// The output is exactly the set of Pauli strings that
/// `MajoranaEncoding::encode` produces (apart from the identity), under *any*
/// valid encoding. Sorting and deduplicating once, up front, is the only
/// deduplication that is exact. Merging terms that only look identical
/// after a partial contraction undercounts weight on ancestor qubits.
pub fn parity_canonical_terms(hamiltonian: &MajoranaSparse) -> Vec<Term> {
    let mut merged: HashMap<Term, Complex64> = HashMap::with_capacity(hamiltonian.indices.len());
    for (term, coeff) in hamiltonian.indices.iter().zip(&hamiltonian.coefficients) {
        // Insertion sort, counting transpositions of distinct elements: each one
        // flips the sign because distinct Majoranas anticommute.
        let mut t: Term = *term;
        let mut sign = 1.0;
        for i in 1..t.len() {
            let mut j = i;
            while j > 0 && t[j - 1] > t[j] {
                t.swap(j - 1, j);
                sign = -sign;
                j -= 1;
            }
        }
        // Remove adjacent equal pairs (γ² = I).
        let mut reduced: Term = ArrayVec::new();
        for &x in t.iter() {
            if reduced.last() == Some(&x) {
                reduced.pop();
            } else {
                reduced.push(x);
            }
        }
        if reduced.is_empty() {
            continue;
        }
        *merged.entry(reduced).or_insert(Complex64::ZERO) += coeff * sign;
    }
    let mut out: Vec<Term> = merged
        .into_iter()
        .filter(|(_, c)| c.norm() > COEFFICIENT_TOLERANCE)
        .map(|(t, _)| t)
        .collect();
    out.sort_unstable();
    out
}

/// One bit-vector `col_a ∈ F₂^T` per Majorana index.
#[derive(Debug, Clone)]
pub struct HammingColumns {
    pub n_terms: usize,
    pub n_words: usize,
    pub cols: Vec<Vec<u64>>,
}

impl HammingColumns {
    /// Build columns for `2 * n_modes` Majoranas from parity-canonical terms.
    pub fn new(terms: &[Term], n_modes: usize) -> Self {
        let n_terms = terms.len();
        let n_words = n_terms.div_ceil(64).max(1);
        let mut cols = vec![vec![0u64; n_words]; 2 * n_modes];
        for (t, term) in terms.iter().enumerate() {
            for &a in term.iter() {
                cols[a as usize][t / 64] ^= 1u64 << (t % 64);
            }
        }
        Self {
            n_terms,
            n_words,
            cols,
        }
    }

    /// Convenience constructor straight from a Hamiltonian.
    pub fn from_hamiltonian(hamiltonian: &MajoranaSparse, n_modes: usize) -> Self {
        Self::new(&parity_canonical_terms(hamiltonian), n_modes)
    }
}

/// Weight on one qubit: the half-perimeter of the triangle `(a, b, c)`.
#[inline]
pub fn half_perimeter(a: &[u64], b: &[u64], c: &[u64]) -> usize {
    a.iter()
        .zip(b)
        .zip(c)
        .map(|((&x, &y), &z)| ((x | y | z) & !(x & y & z)).count_ones() as usize)
        .sum()
}

#[inline]
fn xor_into(acc: &mut [u64], v: &[u64]) {
    acc.iter_mut().zip(v).for_each(|(a, b)| *a ^= b);
}

/// What hangs from one edge of a node in a fixed topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hang {
    Node(usize),
    /// Leaf belonging to `slot`; `which = 0` holds the even Majorana `2m`,
    /// `which = 1` the odd Majorana `2m + 1` of the mode `m` placed in it.
    Leaf {
        slot: usize,
        which: usize,
    },
    Empty,
}

/// The search space of a fixed tree topology.
///
/// The leaves of the input tree come in vacuum-preserving pairs, one pair per
/// mode: the leaves that carry `γ_{2k}` and `γ_{2k+1}` form *slot* `k`. Each
/// slot respects the Y-parity rule, so its coefficients are real. Placing mode
/// `m` in slot `s` puts `γ_{2m}` on the slot's even leaf and `γ_{2m+1}` on its
/// odd leaf. Every permutation of modes over slots is therefore a valid
/// encoding with the same topology and qubit labels, which preserves the vacuum
/// and keeps coefficients real. In group language this is the subgroup of
/// Majorana permutations, which are fermionic Gaussian Cliffords, that
/// preserves the mode pairing.
#[derive(Debug, Clone)]
pub struct SlotModel {
    pub n_modes: usize,
    root: usize,
    /// `hang[v][e]` for e in X, Y, Z.
    hang: Vec<[Hang; 3]>,
    /// Nodes in BFS order from the root.
    bfs: Vec<usize>,
    /// `slot_leaf[s][which] = (node, edge index)`.
    slot_leaf: Vec<[(usize, usize); 2]>,
    /// Ancestor path of each slot leaf: `(node, edge index)` pairs from the
    /// leaf's own parent up to the root, giving the edge taken out of each node.
    paths: Vec<[Vec<(usize, usize)>; 2]>,
    /// Depth of each node (root = 0).
    pub depth: Vec<usize>,
}

fn edge_index(e: Edge) -> usize {
    match e {
        Edge::X => 0,
        Edge::Y => 1,
        Edge::Z => 2,
    }
}

impl SlotModel {
    /// Build the slot model from a tree carrying a complete leaf assignment.
    /// The tree's current assignment is the identity permutation.
    pub fn new(tree: &TernaryTree) -> Result<Self, GeometryError> {
        let n = tree.n_nodes;
        let mut hang = vec![[Hang::Empty; 3]; n];
        let mut slot_leaf: Vec<[Option<(usize, usize)>; 2]> = vec![[None, None]; n];
        for (e, child_of) in [&tree.x_child_of, &tree.y_child_of, &tree.z_child_of]
            .into_iter()
            .enumerate()
        {
            for (v, c) in child_of.iter().enumerate() {
                hang[v][e] = match c {
                    Some(Child::Node(c)) => Hang::Node(*c as usize),
                    Some(Child::XLeaf(k)) => {
                        slot_leaf[*k as usize][0] = Some((v, e));
                        Hang::Leaf {
                            slot: *k as usize,
                            which: 0,
                        }
                    }
                    Some(Child::YLeaf(k)) => {
                        slot_leaf[*k as usize][1] = Some((v, e));
                        Hang::Leaf {
                            slot: *k as usize,
                            which: 1,
                        }
                    }
                    None => Hang::Empty,
                };
            }
        }
        let slot_leaf: Vec<[(usize, usize); 2]> = slot_leaf
            .into_iter()
            .enumerate()
            .map(|(k, [a, b])| {
                Ok([
                    a.ok_or(GeometryError::MissingLeaf(2 * k))?,
                    b.ok_or(GeometryError::MissingLeaf(2 * k + 1))?,
                ])
            })
            .collect::<Result<_, GeometryError>>()?;

        let root = (0..n).find(|&v| tree.parent_of[v].is_none()).unwrap_or(0);
        let mut depth = vec![0usize; n];
        let mut bfs = Vec::with_capacity(n);
        let mut queue = VecDeque::from([root]);
        while let Some(v) = queue.pop_front() {
            bfs.push(v);
            for h in hang[v] {
                if let Hang::Node(c) = h {
                    depth[c] = depth[v] + 1;
                    queue.push_back(c);
                }
            }
        }

        let path_of = |(v, e): (usize, usize)| {
            let mut p = vec![(v, e)];
            let mut node = v;
            while let Some(parent) = tree.parent_of[node] {
                p.push((parent.node_index(), edge_index(parent.edge())));
                node = parent.node_index();
            }
            p
        };
        let paths = slot_leaf
            .iter()
            .map(|&[a, b]| [path_of(a), path_of(b)])
            .collect();

        Ok(Self {
            n_modes: n,
            root,
            hang,
            bfs,
            slot_leaf,
            paths,
            depth,
        })
    }

    /// Recover the permutation (`mode_of_slot`) that `tree` applies to this
    /// model's slots. Returns `None` if `tree` places a Majorana pair in a way
    /// that is not a slot permutation (for example with flipped orientation).
    pub fn mode_of_slot_from(&self, tree: &TernaryTree) -> Option<Vec<usize>> {
        let child = |(v, e): (usize, usize)| match e {
            0 => tree.x_child_of[v],
            1 => tree.y_child_of[v],
            _ => tree.z_child_of[v],
        };
        self.slot_leaf
            .iter()
            .map(|&[a, b]| match (child(a), child(b)) {
                (Some(Child::XLeaf(m)), Some(Child::YLeaf(m2))) if m == m2 => Some(m as usize),
                _ => None,
            })
            .collect()
    }

    /// Write a permutation onto (a clone of) `tree`.
    pub fn apply(&self, tree: &TernaryTree, mode_of_slot: &[usize]) -> TernaryTree {
        let mut out = tree.clone();
        for (s, &[(v0, e0), (v1, e1)]) in self.slot_leaf.iter().enumerate() {
            let m = mode_of_slot[s] as u8;
            for ((v, e), leaf) in [((v0, e0), Child::XLeaf(m)), ((v1, e1), Child::YLeaf(m))] {
                match e {
                    0 => out.x_child_of[v] = Some(leaf),
                    1 => out.y_child_of[v] = Some(leaf),
                    _ => out.z_child_of[v] = Some(leaf),
                }
            }
        }
        out
    }

    /// Exact Pauli weight of the permutation `mode_of_slot`, as a sum of
    /// per-node half-perimeters.
    pub fn weight(&self, cols: &HammingColumns, mode_of_slot: &[usize]) -> usize {
        let w = cols.n_words;
        let zero = vec![0u64; w];
        let mut pi: Vec<Vec<u64>> = vec![Vec::new(); self.n_modes];
        let mut total = 0;
        for &v in self.bfs.iter().rev() {
            let child_val = |h: Hang, pi: &Vec<Vec<u64>>| -> Vec<u64> {
                match h {
                    Hang::Node(c) => pi[c].clone(),
                    Hang::Leaf { slot, which } => cols.cols[2 * mode_of_slot[slot] + which].clone(),
                    Hang::Empty => zero.clone(),
                }
            };
            let a = child_val(self.hang[v][0], &pi);
            let b = child_val(self.hang[v][1], &pi);
            let c = child_val(self.hang[v][2], &pi);
            total += half_perimeter(&a, &b, &c);
            let mut p = a;
            xor_into(&mut p, &b);
            xor_into(&mut p, &c);
            pi[v] = p;
        }
        total
    }

    /// Route of a slot at node `v`: for each of its two leaves, the edge out of
    /// `v` that leads to it (`3` if the leaf is not below `v`).
    fn route(&self, slot: usize, v: usize) -> [u8; 2] {
        let mut r = [3u8; 2];
        for (which, p) in self.paths[slot].iter().enumerate() {
            if let Some(&(_, e)) = p.iter().find(|(n, _)| *n == v) {
                r[which] = e as u8;
            }
        }
        r
    }
}

/// Exact Pauli weight of any tree with a complete leaf assignment.
pub fn tree_pauli_weight(
    cols: &HammingColumns,
    tree: &TernaryTree,
) -> Result<usize, GeometryError> {
    let model = SlotModel::new(tree)?;
    let identity: Vec<usize> = (0..model.n_modes).collect();
    Ok(model.weight(cols, &identity))
}

/// Exact optimum over all slot permutations, by exhaustive enumeration
/// (Heap's algorithm). Practical up to about 10 modes. Returns the optimal
/// `mode_of_slot` and its weight.
pub fn exhaustive(
    model: &SlotModel,
    cols: &HammingColumns,
) -> Result<(Vec<usize>, usize), GeometryError> {
    const MAX: usize = 10;
    let n = model.n_modes;
    if n > MAX {
        return Err(GeometryError::TooManyModes { max: MAX, got: n });
    }
    let mut perm: Vec<usize> = (0..n).collect();
    let mut best = (perm.clone(), model.weight(cols, &perm));
    let mut c = vec![0usize; n];
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                perm.swap(0, i);
            } else {
                perm.swap(c[i], i);
            }
            let w = model.weight(cols, &perm);
            if w < best.1 {
                best = (perm.clone(), w);
            }
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    Ok(best)
}

/// Top-down hierarchical partitioning ("parity-cut tri-section").
///
/// The weight on node `v`'s qubit depends only on *which child subtree* each
/// Majorana below `v` ends up in, not on the assignment deeper down. So nodes
/// are visited root first. At each node, the modes are split among the
/// child-subtree classes of slots (sizes are fixed by the topology) so as to
/// minimise that node's half-perimeter. A Kernighan–Lin style swap refinement
/// is used, with each swap evaluated exactly in `O(T/64)` by XOR updates of the
/// three child points. Decisions at a node constrain only its descendants, so
/// the qubits that see the most terms are optimised with the most freedom.
/// Bottom-up greedy (HATT, TOPP-HATT) does the opposite.
///
/// `seed` gives the initial `mode_of_slot`. The search only moves modes
/// between classes at each node, so a better seed gives a better start.
/// `max_passes` bounds the swap passes per node.
///
/// `lookahead` (μ) adds `μ · |π_c|` for every child *node* `c` of `v`. By the
/// edge decomposition `W = Σ_e |π_e| − const − 2N₄`, a term that is odd on the
/// edge into `c` must pay for at least the qubit of `c`, so `|π_c|` is a lower
/// bound on the weight still to come inside that subtree. `μ = 0` optimises
/// each node's qubit alone.
pub fn topdown(
    model: &SlotModel,
    cols: &HammingColumns,
    seed: &[usize],
    max_passes: usize,
    lookahead: f64,
) -> Vec<usize> {
    const SCALE: usize = 64;
    let mu = (lookahead * SCALE as f64).round().max(0.0) as usize;
    let n = model.n_modes;
    let w = cols.n_words;
    // slot_of_mode is the working assignment.
    let mut slot_of_mode = vec![0usize; n];
    for (s, &m) in seed.iter().enumerate() {
        slot_of_mode[m] = s;
    }
    // Current classes of slots (each class's modes are those sitting in it).
    let mut classes: Vec<Vec<usize>> = vec![(0..n).collect()];

    for &v in &model.bfs {
        // Split every class by its route at v.
        let mut groups: Vec<Vec<Vec<usize>>> = Vec::with_capacity(classes.len());
        for class in &classes {
            let mut by_route: BTreeMap<[u8; 2], Vec<usize>> = BTreeMap::new();
            for &s in class {
                by_route.entry(model.route(s, v)).or_default().push(s);
            }
            groups.push(by_route.into_values().collect());
        }

        // Child points of v under the current assignment.
        let mut pts = vec![vec![0u64; w]; 3];
        let mut route_of_slot: HashMap<usize, [u8; 2]> = HashMap::new();
        for g in &groups {
            for sub in g {
                for &s in sub {
                    let r = model.route(s, v);
                    route_of_slot.insert(s, r);
                }
            }
        }
        let mut mos = vec![0usize; n];
        for (m, &s) in slot_of_mode.iter().enumerate() {
            mos[s] = m;
        }
        for (&s, r) in &route_of_slot {
            for which in 0..2 {
                if r[which] < 3 {
                    xor_into(&mut pts[r[which] as usize], &cols.cols[2 * mos[s] + which]);
                }
            }
        }
        let toggle = |pts: &mut Vec<Vec<u64>>, m: usize, r: [u8; 2]| {
            for which in 0..2 {
                if r[which] < 3 {
                    xor_into(&mut pts[r[which] as usize], &cols.cols[2 * m + which]);
                }
            }
        };
        let node_edges: Vec<usize> = (0..3)
            .filter(|&e| matches!(model.hang[v][e], Hang::Node(_)))
            .collect();
        let score = |pts: &Vec<Vec<u64>>| -> usize {
            let mut s = SCALE * half_perimeter(&pts[0], &pts[1], &pts[2]);
            if mu > 0 {
                for &e in &node_edges {
                    s += mu
                        * pts[e]
                            .iter()
                            .map(|x| x.count_ones() as usize)
                            .sum::<usize>();
                }
            }
            s
        };
        let mut cost = score(&pts);

        // Swap refinement within each split class.
        for _ in 0..max_passes {
            let mut improved = false;
            for g in &groups {
                if g.len() < 2 {
                    continue;
                }
                for gi in 0..g.len() {
                    for gj in gi + 1..g.len() {
                        let ri = route_of_slot[&g[gi][0]];
                        let rj = route_of_slot[&g[gj][0]];
                        for &si in &g[gi] {
                            for &sj in &g[gj] {
                                let (mi, mj) = (mos[si], mos[sj]);
                                // Move mi from route ri to rj and mj from rj to ri.
                                toggle(&mut pts, mi, ri);
                                toggle(&mut pts, mi, rj);
                                toggle(&mut pts, mj, rj);
                                toggle(&mut pts, mj, ri);
                                let new_cost = score(&pts);
                                if new_cost < cost {
                                    cost = new_cost;
                                    slot_of_mode[mi] = sj;
                                    slot_of_mode[mj] = si;
                                    mos[si] = mj;
                                    mos[sj] = mi;
                                    improved = true;
                                } else {
                                    toggle(&mut pts, mi, ri);
                                    toggle(&mut pts, mi, rj);
                                    toggle(&mut pts, mj, rj);
                                    toggle(&mut pts, mj, ri);
                                }
                            }
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        classes = groups.into_iter().flatten().collect();
    }
    let mut mos = vec![0usize; n];
    for (m, &s) in slot_of_mode.iter().enumerate() {
        mos[s] = m;
    }
    mos
}

/// Largest `s` (unpaired Majoranas) and `d` (complete mode pairs) tracked in
/// the expected-cost tables; larger counts are clamped.
const MAX_S: usize = 6;
const MAX_D: usize = 3;

/// Per-term structure used by [`topdown_expected`]: each term as a list of
/// `(mode, mask)` where bit 0 / bit 1 of `mask` mark `γ_{2m}` / `γ_{2m+1}`.
struct ModeTerms {
    terms: Vec<Vec<(usize, u8)>>,
    of_mode: Vec<Vec<usize>>,
}

impl ModeTerms {
    fn new(terms: &[Term], n_modes: usize) -> Self {
        let mut out = Vec::with_capacity(terms.len());
        let mut of_mode = vec![Vec::new(); n_modes];
        for (t, term) in terms.iter().enumerate() {
            let mut modes: Vec<(usize, u8)> = Vec::new();
            for &a in term.iter() {
                let (m, bit) = (a as usize / 2, 1u8 << (a % 2));
                match modes.iter_mut().find(|(mm, _)| *mm == m) {
                    Some((_, mask)) => *mask |= bit,
                    None => modes.push((m, bit)),
                }
            }
            for &(m, _) in &modes {
                of_mode[m].push(t);
            }
            out.push(modes);
        }
        Self {
            terms: out,
            of_mode,
        }
    }
}

impl SlotModel {
    /// Leaves and slots below node `c`.
    fn subtree(&self, c: usize) -> (Vec<usize>, Vec<(usize, usize)>, Vec<usize>) {
        let mut nodes = vec![c];
        let mut i = 0;
        let mut leaves = Vec::new();
        while i < nodes.len() {
            let v = nodes[i];
            for h in self.hang[v] {
                match h {
                    Hang::Node(u) => nodes.push(u),
                    Hang::Leaf { slot, which } => leaves.push((slot, which)),
                    Hang::Empty => {}
                }
            }
            i += 1;
        }
        let mut full_slots: Vec<usize> = leaves
            .iter()
            .filter(|(s, w)| *w == 0 && leaves.contains(&(*s, 1)))
            .map(|(s, _)| *s)
            .collect();
        full_slots.sort_unstable();
        (nodes, leaves, full_slots)
    }

    /// Expected weight inside the subtree of `c` of a term with `s` unpaired
    /// Majoranas and `d` complete mode pairs, all placed uniformly at random in
    /// it (pairs on whole slots). Monte Carlo with a fixed seed.
    fn expected_cost_table(&self, c: usize, samples: usize, seed: u64) -> Vec<[f64; MAX_D + 1]> {
        use rand::seq::IndexedRandom;
        use rand::SeedableRng;
        let (nodes, leaves, full_slots) = self.subtree(c);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed ^ (c as u64) << 20);
        let mut table = vec![[0.0f64; MAX_D + 1]; MAX_S + 1];
        let mut parity: HashMap<(usize, usize), bool> = HashMap::new();
        let mut node_par = vec![false; self.n_modes];
        for s in 0..=MAX_S {
            for d in 0..=MAX_D {
                if s + d == 0 || leaves.is_empty() {
                    continue;
                }
                let mut total = 0usize;
                for _ in 0..samples {
                    parity.clear();
                    for _ in 0..s {
                        let &(slot, which) = leaves.choose(&mut rng).unwrap();
                        *parity.entry((slot, which)).or_insert(false) ^= true;
                    }
                    for _ in 0..d {
                        if let Some(&slot) = full_slots.choose(&mut rng) {
                            *parity.entry((slot, 0)).or_insert(false) ^= true;
                            *parity.entry((slot, 1)).or_insert(false) ^= true;
                        } else {
                            for _ in 0..2 {
                                let &(slot, which) = leaves.choose(&mut rng).unwrap();
                                *parity.entry((slot, which)).or_insert(false) ^= true;
                            }
                        }
                    }
                    for &v in nodes.iter().rev() {
                        let mut ps = [false; 3];
                        for (e, h) in self.hang[v].iter().enumerate() {
                            ps[e] = match *h {
                                Hang::Node(u) => node_par[u],
                                Hang::Leaf { slot, which } => {
                                    *parity.get(&(slot, which)).unwrap_or(&false)
                                }
                                Hang::Empty => false,
                            };
                        }
                        let odd = ps.iter().filter(|&&p| p).count();
                        total += (odd == 1 || odd == 2) as usize;
                        node_par[v] = odd % 2 == 1;
                    }
                }
                table[s][d] = total as f64 / samples as f64;
            }
        }
        table
    }
}

/// Top-down partitioning with an *expected-completion* objective.
///
/// Like [`topdown`], nodes are visited from the root and modes are swapped
/// between the child classes of each node. A split at `v` is scored by the
/// exact weight on `v`'s qubit **plus the expected weight of a uniformly random
/// completion inside each child subtree**. For each child node `c` that
/// expectation depends on a term only through how many unpaired Majoranas
/// (`s`) and complete mode pairs (`d`) it has below `c`. It is precomputed per
/// node as `h_c(s, d)`. Unlike the lower bound `|π_c|`, this sees terms that are
/// even on the edge into `c` but still cost weight inside it, and the shape
/// of the leaf-pair slots, e.g. whether a mode's two leaves are siblings. Each
/// swap is evaluated incrementally over the terms touching the two modes.
pub fn topdown_expected(
    model: &SlotModel,
    terms: &[Term],
    seed: &[usize],
    max_passes: usize,
    samples: usize,
) -> Vec<usize> {
    let n = model.n_modes;
    let mt = ModeTerms::new(terms, n);
    let h: Vec<Vec<[f64; MAX_D + 1]>> = (0..n)
        .map(|c| model.expected_cost_table(c, samples, 0x5eed))
        .collect();
    let mut slot_of_mode = vec![0usize; n];
    let mut mos = vec![0usize; n];
    for (s, &m) in seed.iter().enumerate() {
        slot_of_mode[m] = s;
        mos[s] = m;
    }
    let mut classes: Vec<Vec<usize>> = vec![(0..n).collect()];
    let mut stamp = vec![0usize; terms.len()];
    let mut clock = 0usize;

    for &v in &model.bfs {
        let mut groups: Vec<Vec<Vec<usize>>> = Vec::with_capacity(classes.len());
        for class in &classes {
            let mut by_route: BTreeMap<[u8; 2], Vec<usize>> = BTreeMap::new();
            for &s in class {
                by_route.entry(model.route(s, v)).or_default().push(s);
            }
            groups.push(by_route.into_values().collect());
        }
        if groups.iter().all(|g| g.len() < 2) {
            classes = groups.into_iter().flatten().collect();
            continue;
        }
        let route_of_slot: Vec<[u8; 2]> = (0..n).map(|s| model.route(s, v)).collect();
        let child_node: [Option<usize>; 3] = [0, 1, 2].map(|e| match model.hang[v][e] {
            Hang::Node(u) => Some(u),
            _ => None,
        });
        // Cost of term t at v given each mode's current slot.
        let term_cost = |t: usize, slot_of_mode: &Vec<usize>| -> f64 {
            let mut s_cnt = [0usize; 3];
            let mut d_cnt = [0usize; 3];
            for &(m, mask) in &mt.terms[t] {
                let r = route_of_slot[slot_of_mode[m]];
                if mask == 3 && r[0] == r[1] {
                    if r[0] < 3 {
                        d_cnt[r[0] as usize] += 1;
                    }
                } else {
                    for which in 0..2 {
                        if mask & (1 << which) != 0 && r[which] < 3 {
                            s_cnt[r[which] as usize] += 1;
                        }
                    }
                }
            }
            let odd = s_cnt.iter().filter(|&&x| x % 2 == 1).count();
            let mut c = (odd == 1 || odd == 2) as usize as f64;
            for e in 0..3 {
                if let Some(u) = child_node[e] {
                    let (s, d) = (s_cnt[e].min(MAX_S), d_cnt[e].min(MAX_D));
                    c += h[u][s][d];
                }
            }
            c
        };
        let mut cache: Vec<f64> = (0..terms.len())
            .map(|t| term_cost(t, &slot_of_mode))
            .collect();

        for _ in 0..max_passes {
            let mut improved = false;
            for g in &groups {
                if g.len() < 2 {
                    continue;
                }
                for gi in 0..g.len() {
                    for gj in gi + 1..g.len() {
                        for &si in &g[gi] {
                            for &sj in &g[gj] {
                                let (mi, mj) = (mos[si], mos[sj]);
                                clock += 1;
                                let mut affected: Vec<usize> = Vec::new();
                                for &t in mt.of_mode[mi].iter().chain(&mt.of_mode[mj]) {
                                    if stamp[t] != clock {
                                        stamp[t] = clock;
                                        affected.push(t);
                                    }
                                }
                                slot_of_mode[mi] = sj;
                                slot_of_mode[mj] = si;
                                let mut delta = 0.0;
                                let new: Vec<f64> = affected
                                    .iter()
                                    .map(|&t| {
                                        let c = term_cost(t, &slot_of_mode);
                                        delta += c - cache[t];
                                        c
                                    })
                                    .collect();
                                if delta < -1e-9 {
                                    for (&t, c) in affected.iter().zip(new) {
                                        cache[t] = c;
                                    }
                                    mos[si] = mj;
                                    mos[sj] = mi;
                                    improved = true;
                                } else {
                                    slot_of_mode[mi] = si;
                                    slot_of_mode[mj] = sj;
                                }
                            }
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        classes = groups.into_iter().flatten().collect();
    }
    mos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::majorana::Encode;
    use crate::operators::PauliWeight;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn random_hamiltonian(n_modes: usize, n_terms: usize, seed: u64) -> MajoranaSparse {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut terms = Vec::new();
        for _ in 0..n_terms {
            let len = if rng.random_bool(0.4) { 2 } else { 4 };
            let mut t: Term = ArrayVec::new();
            while t.len() < len {
                let x = rng.random_range(0..2 * n_modes) as u16;
                if !t.contains(&x) {
                    t.push(x);
                }
            }
            t.sort_unstable();
            terms.push(t);
        }
        let n = terms.len();
        MajoranaSparse::new(terms, vec![Complex64::ONE; n], 0.0).unwrap()
    }

    fn true_weight(tree: &TernaryTree, h: &MajoranaSparse) -> usize {
        let enc = tree.build_encoding(tree.n_nodes).unwrap();
        let q = enc.encode(h);
        q.pauli_weight()
    }

    fn trees(n: usize) -> Vec<TernaryTree> {
        vec![
            TernaryTree::naive_jordan_wigner(n),
            TernaryTree::naive_parity(n),
            TernaryTree::naive_bravyi_kitaev(n),
            TernaryTree::naive_jkmn(n),
        ]
    }

    #[test]
    fn canonical_terms_merge_sign_and_parity() {
        let terms = vec![
            ArrayVec::from_iter([0u16, 0, 2, 3]),
            ArrayVec::from_iter([3u16, 2]), // = -γ2γ3: cancels the first
            ArrayVec::from_iter([1u16, 4]),
            ArrayVec::from_iter([5u16, 5]), // identity
        ];
        let h = MajoranaSparse::new(terms, vec![Complex64::ONE; 4], 0.0).unwrap();
        let c = parity_canonical_terms(&h);
        assert_eq!(c, vec![ArrayVec::from_iter([1u16, 4])]);
    }

    #[test]
    fn half_perimeter_sum_equals_true_pauli_weight() {
        for seed in 0..4 {
            for n in [3usize, 5, 8] {
                let h = random_hamiltonian(n, 60, seed);
                let cols = HammingColumns::from_hamiltonian(&h, n);
                for tree in trees(n) {
                    let model = SlotModel::new(&tree).unwrap();
                    let mut perm: Vec<usize> = (0..n).collect();
                    for k in 0..3 {
                        perm.rotate_left(k);
                        let t = model.apply(&tree, &perm);
                        assert_eq!(
                            model.weight(&cols, &perm),
                            true_weight(&t, &h),
                            "seed {seed} n {n} perm {perm:?}"
                        );
                        assert_eq!(tree_pauli_weight(&cols, &t).unwrap(), true_weight(&t, &h));
                    }
                }
            }
        }
    }

    /// Edge / T-join form: W = Σ_e |π_e| − Σ_t |A_t|/2 − 2·N₄, where the sum
    /// runs over every tree edge (leaf edges included, no edge above the root)
    /// and N₄ counts (term, node) pairs with all three children odd.
    #[test]
    fn edge_tjoin_identity() {
        let pop = |v: &[u64]| v.iter().map(|x| x.count_ones() as usize).sum::<usize>();
        for seed in 0..4 {
            let n = 7;
            let h = random_hamiltonian(n, 80, seed);
            let terms = parity_canonical_terms(&h);
            let cols = HammingColumns::new(&terms, n);
            for tree in trees(n) {
                let model = SlotModel::new(&tree).unwrap();
                let perm: Vec<usize> = (0..n).rev().collect();
                let w = cols.n_words;
                let mut pi: Vec<Vec<u64>> = vec![vec![0; w]; n];
                let (mut edges, mut n4) = (0usize, 0usize);
                for &v in model.bfs.iter().rev() {
                    let mut ch: Vec<Vec<u64>> = Vec::new();
                    for h in model.hang[v] {
                        ch.push(match h {
                            Hang::Node(c) => pi[c].clone(),
                            Hang::Leaf { slot, which } => cols.cols[2 * perm[slot] + which].clone(),
                            Hang::Empty => vec![0; w],
                        });
                    }
                    for c in &ch {
                        edges += pop(c);
                    }
                    let all: Vec<u64> = (0..w).map(|i| ch[0][i] & ch[1][i] & ch[2][i]).collect();
                    n4 += pop(&all);
                    pi[v] = (0..w).map(|i| ch[0][i] ^ ch[1][i] ^ ch[2][i]).collect();
                }
                let half_len: usize = terms.iter().map(|t| t.len() / 2).sum();
                assert_eq!(model.weight(&cols, &perm), edges - half_len - 2 * n4);
            }
        }
    }

    #[test]
    fn slot_permutation_roundtrip() {
        let tree = TernaryTree::naive_bravyi_kitaev(6);
        let model = SlotModel::new(&tree).unwrap();
        let perm = vec![3, 1, 5, 0, 2, 4];
        let t = model.apply(&tree, &perm);
        assert_eq!(model.mode_of_slot_from(&t), Some(perm));
    }

    #[test]
    fn exhaustive_is_lower_bound_and_topdown_valid() {
        for seed in 0..3 {
            let n = 6;
            let h = random_hamiltonian(n, 50, seed);
            let cols = HammingColumns::from_hamiltonian(&h, n);
            for tree in trees(n) {
                let model = SlotModel::new(&tree).unwrap();
                let identity: Vec<usize> = (0..n).collect();
                let (best, best_w) = exhaustive(&model, &cols).unwrap();
                assert_eq!(model.weight(&cols, &best), best_w);
                assert!(best_w <= model.weight(&cols, &identity));
                let td = topdown(&model, &cols, &identity, 10, 0.5);
                let canon = parity_canonical_terms(&h);
                let te = topdown_expected(&model, &canon, &identity, 10, 64);
                let mut sorted_te = te.clone();
                sorted_te.sort_unstable();
                assert_eq!(sorted_te, identity);
                assert!(best_w <= model.weight(&cols, &te));
                let mut sorted = td.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, identity, "top-down must return a permutation");
                let td_w = model.weight(&cols, &td);
                assert!(best_w <= td_w);
                assert_eq!(td_w, true_weight(&model.apply(&tree, &td), &h));
            }
        }
    }
}
