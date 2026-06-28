use itertools::Itertools;
use log::debug;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::iter::zip;
use std::sync::atomic::AtomicUsize;
use thiserror::Error;
use tinyvec::ArrayVec;

use super::term_store::{combine, ArrayVecTermStore, MajoranaTermStore, ToppHattSelection};
use crate::encode::ternarytree::{Child, Edge, TernaryTree, YParity};
use crate::operators::MajoranaSparse;

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
    InvalidRestriction(Restriction),
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
pub enum Restriction {
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

impl Restriction {
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
            Restriction::EvenLeaf => unassigned.iter().map(|v| (2 * v) as u16).collect(),
            Restriction::OddLeaf => unassigned.iter().map(|v| ((2 * v) + 1) as u16).collect(),
            Restriction::Any => {
                let mut allowed: Vec<u16> = unassigned
                    .iter()
                    .map(|v| (2 * v) as u16)
                    .collect::<Vec<u16>>();
                allowed.extend(unassigned.iter().map(|v| (2 * v + 1) as u16));
                allowed
            }
            // Completed selections.
            Restriction::ChildNode(child_index) => {
                vec![representatives[*child_index as usize]]
            }
            Restriction::Empty => vec![(2 * n_nodes) as u16],
            Restriction::Majorana(index) => vec![*index],
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
    x: Vec<Restriction>,
    y: Vec<Restriction>,
    z: Vec<Restriction>,
    pairs: HashMap<LeafLocation, LeafLocation>,
}

impl TreeRestrictions {
    /// Create a set of [`TreeRestrictons`] for  a [`TernaryTree`].
    fn new(tree: &TernaryTree) -> Self {
        let x: Vec<Restriction> = vec![Restriction::Any; tree.n_nodes];
        let y: Vec<Restriction> = vec![Restriction::Any; tree.n_nodes];
        let z: Vec<Restriction> = vec![Restriction::Any; tree.n_nodes];
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
        self.z[all_z_index] = Restriction::Empty;
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
                    *r = Restriction::ChildNode(*child_index)
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
                        *r = Restriction::EvenLeaf;
                    }
                    Some(Child::YLeaf(_)) => {
                        *r = Restriction::OddLeaf;
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
                    Restriction::Majorana(index) => {
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
                    Restriction::ChildNode(_) => {
                        debug_assert!(matches!(c, Some(Child::Node(_))));
                    }
                    Restriction::Empty => {
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
/// [`ArrayVecTermStore`] backend. See [`topphatt_impl`] to run it over an
/// alternative [`MajoranaTermStore`] (e.g. the bit-packed prototype).
pub fn topphatt(
    hamiltonian: MajoranaSparse,
    tree: TernaryTree,
    parallelize: bool,
    heuristic: NodeOrderHeuristic,
) -> Result<TernaryTree, ToppHattError> {
    let store = ArrayVecTermStore::new(hamiltonian.indices);
    topphatt_impl(store, tree, parallelize, heuristic)
}

/// Toplogy-Preserving Hamiltonian-Adaptive Ternary Tree, generic over the
/// Majorana-term storage backend.
///
/// The restriction/dependency scaffolding and the selection loop are identical
/// across backends; only the per-term weight evaluation and Hamiltonian
/// reduction differ, and those live behind [`MajoranaTermStore`]. The node
/// representative chosen by [`MajoranaTermStore::reduce`] is threaded back into
/// the restriction system via the `representatives` table.
pub fn topphatt_impl<S: MajoranaTermStore + Sync>(
    mut store: S,
    mut tree: TernaryTree,
    parallelize: bool,
    heuristic: NodeOrderHeuristic,
) -> Result<TernaryTree, ToppHattError> {
    let mut restrictions = TreeRestrictions::new(&tree);
    let mut node_dependencies = NodeDependencies::new(&tree);

    // Rough threshold at which parallelism is worth the overhead. When enabled the
    // weight search runs on rayon's global thread pool.
    let mut use_parallel = parallelize && store.len() > 1000;

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
    debug!("Number of hamiltonian terms {:?}", store.len());
    'assign: for loop_index in 0..tree.n_nodes {
        debug!("loop {:}", loop_index);
        debug!("Restrictions {:?}", restrictions);
        debug!("Dependencies {:?}", node_dependencies);
        debug!("Unassigned Modes {:?}", unassigned_modes);
        let n_leaves = 2 * tree.n_nodes + 1;

        // Best (lowest-weight) candidate found across all active nodes this step.
        let mut best = ToppHattSelection::WORST;
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
            let mut unique_choices: HashSet<(&Restriction, &Restriction, &Restriction)> =
                HashSet::with_capacity(active_nodes.len());

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
                    Restriction::EvenLeaf | Restriction::OddLeaf,
                    Restriction::EvenLeaf | Restriction::OddLeaf,
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
                    .map(|comb| store.evaluate_combination(comb, active, &bound))
                    .reduce(|| ToppHattSelection::WORST, combine)
            } else {
                combos
                    .iter()
                    .map(|comb| store.evaluate_combination(comb, active, &bound))
                    .fold(ToppHattSelection::WORST, combine)
            };
            best = combine(best, node_best);
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
                res[selection.min_parent] = Restriction::Majorana(sel);
            } else if (sel as usize) == n_leaves {
                res[selection.min_parent] = Restriction::Empty;
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
                Edge::X => restrictions.x[partner_location.0] = Restriction::Majorana(pair_index),
                Edge::Y => restrictions.y[partner_location.0] = Restriction::Majorana(pair_index),
                Edge::Z => restrictions.z[partner_location.0] = Restriction::Majorana(pair_index),
            }
        }

        // Check for nods which are now complete thanks to assigning leaf pairs.
        let complete_nodes: Vec<usize> = (0..tree.n_nodes)
            .filter(|&ind| {
                matches!(
                    restrictions.x[ind],
                    Restriction::Majorana(_) | Restriction::ChildNode(_)
                ) & matches!(
                    restrictions.y[ind],
                    Restriction::Majorana(_) | Restriction::ChildNode(_)
                ) & matches!(
                    restrictions.z[ind],
                    Restriction::Majorana(_) | Restriction::ChildNode(_) | Restriction::Empty
                )
            })
            .collect();
        debug!("Complete nodes {:?}", complete_nodes);
        complete_nodes
            .iter()
            .for_each(|&ind| node_dependencies.drop_node(ind));

        let representative = store.reduce(selection.min_parent, selection.leaf_indices, n_leaves);
        representatives[selection.min_parent] = representative;
        debug!(
            "Node {} represented by index {representative}.",
            selection.min_parent
        );
        if store.len() < 1000 {
            use_parallel = false;
        }
        debug!("Reduced Hamiltonian to {} terms", store.len());
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
mod test_topphatt {
    use super::Edge::{X, Y, Z};
    use super::Restriction::{ChildNode, Empty, EvenLeaf, OddLeaf};
    use super::*;
    use crate::encode::majorana::MajoranaEncoding;
    use crate::encode::ternarytree::TTFlatpack;
    use crate::encode::ternarytree::TernaryTree;
    use crate::optimise::ternarytree::hatt::{qubit_term_weight, reduce_hamiltonian};
    use crate::optimise::ternarytree::term_store::{BitSlicedTermStore, SparseListTermStore};
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
        assert_eq!(encoding.operators.ipowers, arr1(&[0, 1, 0, 1, 0, 1]));
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
        assert_eq!(encoding.operators.ipowers, arr1(&[0, 1, 0, 1, 0, 1]));
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
        assert_eq!(x_enc.operators.ipowers.len(), 14);
        assert_eq!(z_enc.operators.ipowers.len(), 14);

        // The two heuristics walk active_nodes from opposite ends, so on a
        // branched tree the resulting symplectic matrix should differ.
        assert_ne!(
            x_enc.operators.x_block, z_enc.operators.x_block,
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

        assert_eq!(enc_first.operators.ipowers, enc_second.operators.ipowers);
        assert_eq!(enc_first.operators.x_block, enc_second.operators.x_block);
        assert_eq!(enc_first.operators.z_block, enc_second.operators.z_block);
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
            if other_enc.operators.x_block != ref_enc.operators.x_block {
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
        let sliced = BitSlicedTermStore::from_arrayvecs(&hamiltonian.indices, n_modes);

        let av_tree = topphatt(hamiltonian, make_tree(), false, heuristic).unwrap();
        let sliced_tree = topphatt_impl(sliced, make_tree(), false, heuristic).unwrap();

        let av = av_tree.build_encoding(n_modes).unwrap();
        let bs = sliced_tree.build_encoding(n_modes).unwrap();

        assert_eq!(
            av.operators.x_block, bs.operators.x_block,
            "x_block differs"
        );
        assert_eq!(
            av.operators.z_block, bs.operators.z_block,
            "z_block differs"
        );
        assert_eq!(av.operators.ipowers, bs.operators.ipowers, "ipowers differ");
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
    /// multiset rule as the index-list backend, so `index_list == bit_sliced ==
    /// sparse_list` (x/z blocks and ipowers) for every instance.
    #[test]
    fn test_bit_sliced_valid_encodings_random() {
        for n_modes in [4usize, 6, 8, 10] {
            for seed in 0..20u64 {
                let hamiltonian = random_majorana(n_modes, 6 * n_modes, seed);
                let n_majoranas = 2 * n_modes;
                let sliced = BitSlicedTermStore::from_arrayvecs(&hamiltonian.indices, n_modes);
                let sparse = SparseListTermStore::from_arrayvecs(&hamiltonian.indices, n_modes);
                let av_tree = topphatt(
                    hamiltonian,
                    TernaryTree::naive_jkmn(n_modes),
                    false,
                    NodeOrderHeuristic::MinWeight,
                )
                .unwrap();
                let t_sliced = topphatt_impl(
                    sliced,
                    TernaryTree::naive_jkmn(n_modes),
                    false,
                    NodeOrderHeuristic::MinWeight,
                )
                .unwrap();
                let t_sparse = topphatt_impl(
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
                assert_eq!(av.operators.ipowers.len(), n_majoranas);

                // The transposed backends now deduplicate whole terms on the same
                // multiset rule as the index-list backend, so all three produce
                // identical encodings.
                assert_eq!(e_sliced.operators.x_block, av.operators.x_block);
                assert_eq!(e_sliced.operators.z_block, av.operators.z_block);
                assert_eq!(e_sliced.operators.ipowers, av.operators.ipowers);
                assert_eq!(e_sparse.operators.x_block, av.operators.x_block);
                assert_eq!(e_sparse.operators.z_block, av.operators.z_block);
                assert_eq!(e_sparse.operators.ipowers, av.operators.ipowers);
            }
        }
    }

    /// `bit_sliced` must produce valid encodings on *every* tree topology, not
    /// just JKMN. The Jordan-Wigner chain in particular has node z-edges whose
    /// representative previously tripped the orchestration's magnitude-based edge
    /// classification (panic: "All leaves should have pairs"); the upper-range
    /// representative fixes it. The index-list backend is checked alongside as a
    /// control.
    #[test]
    fn test_bit_sliced_valid_on_all_topologies() {
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
                    il.operators.ipowers.len(),
                    2 * n_modes,
                    "index_list {name} seed {seed}"
                );

                let store = BitSlicedTermStore::from_arrayvecs(&hamiltonian.indices, n_modes);
                let sliced =
                    topphatt_impl(store, build(n_modes), false, NodeOrderHeuristic::MinWeight)
                        .unwrap()
                        .build_encoding(n_modes)
                        .unwrap();
                // Multiset dedup makes bit_sliced match index_list exactly on
                // every topology.
                assert_eq!(
                    sliced.operators.x_block, il.operators.x_block,
                    "bit_sliced vs index_list x {name} seed {seed}"
                );
                assert_eq!(
                    sliced.operators.z_block, il.operators.z_block,
                    "bit_sliced vs index_list z {name} seed {seed}"
                );

                // The sparse inverted-index store must agree exactly too.
                let sparse_store =
                    SparseListTermStore::from_arrayvecs(&hamiltonian.indices, n_modes);
                let sparse = topphatt_impl(
                    sparse_store,
                    build(n_modes),
                    false,
                    NodeOrderHeuristic::MinWeight,
                )
                .unwrap()
                .build_encoding(n_modes)
                .unwrap();
                assert_eq!(
                    sparse.operators.x_block, il.operators.x_block,
                    "sparse vs index_list x {name} seed {seed}"
                );
                assert_eq!(
                    sparse.operators.z_block, il.operators.z_block,
                    "sparse vs index_list z {name} seed {seed}"
                );
            }
        }
    }
}
