use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use log::debug;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::iter::zip;
use std::ops::BitXorAssign;
use std::sync::{Mutex, RwLock};
use thiserror::Error;
use tinyvec::ArrayVec;
const MAJORANA_MAX: usize = 7;

use crate::encode::ternarytree::{Child, Edge, TernaryTree, YParity};
use crate::operators::MajoranaSparse;

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

/// The result of a single TOPP-HATT assignment step.
///
/// Stores the minimum Pauli weight found, the parent node chosen,
/// and the three leaf indices assigned to that node's edges.
#[derive(Debug)]
pub struct ToppHattSelection {
    min_weight: usize,
    min_parent: usize,
    leaf_indices: [u16; 3],
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
    /// As the procedure progresses, the set of unassigned indices will become more restrictive.
    fn get_index_subset(&self, unassigned: &BTreeSet<usize>, n_nodes: usize) -> Vec<u16> {
        match self {
            Restriction::EvenLeaf => unassigned.iter().map(|v| (2 * v) as u16).collect(),
            Restriction::OddLeaf => unassigned.iter().map(|v| ((2 * v) + 1) as u16).collect(),
            Restriction::ChildNode(child_index) => {
                vec![(*child_index as u16 + 2 * n_nodes as u16 + 1)]
            }
            Restriction::Any => {
                let mut allowed: Vec<u16> = unassigned
                    .iter()
                    .map(|v| (2 * v) as u16)
                    .collect::<Vec<u16>>();
                allowed.extend(unassigned.iter().map(|v| (2 * v + 1) as u16));
                allowed
            }
            Restriction::Empty => vec![(2 * n_nodes) as u16],
            Restriction::Majorana(index) => vec![*index],
        }
    }
}

/// Type alias for the location of a leaf
///
/// The first field is the node index of its parent node.
/// The second field is the edge on that parent node.
type LeafLocation = (usize, Edge);

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
                x: (v, Edge::X),
                y: (v, Edge::Y),
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
                            pair.x = (parent_index, edge)
                        }
                        YParity::Odd => {
                            let pair = &mut leaf_pairs[leaf_index];
                            pair.y = (parent_index, edge)
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
                assert!(children_without_leaves
                    .insert(node, ArrayVec::new())
                    .is_none());
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

/// Find the weight of a term on the qubit of a single node.
///
/// This function is used to assess the cost of each possible choice
/// of outward edges of a given node. Each outward edge has an associated
/// index. Either a  Majorana-index, or a Node-index.
///
/// Each term is composed of some number of Majorana operators.
///
/// Where a Majorana operator is included in the _children_ of a given node,
/// the Majorana operator acts on that node's qubit with a non-Identity operator.
///
/// Additionally, using [`reduce_hamiltonian`] we guarantee that for [`TernaryTree`]s,
/// no two distinct indices represent Majorana operators which
/// act with the same Pauli operator.
///
/// We wish to find out whether the product of Majorana operators in a given
/// Hamiltonian term require the application of non-Identity operator.
///
/// For each Majorana operator in a term:
/// - if it is not in _children_,  it acts with the  Identity.
/// - if it appears an even number of times, it acts with the Identity, as: PP=I forall P in {X,Y,Z,I}
///
/// if three Majorana operators appear in both the term and _children_ with odd parity,
/// together, they act with the identity as XYZ=-iI
#[inline(always)]
fn qubit_term_weight(term: &ArrayVec<[u16; MAJORANA_MAX]>, sorted_children: &[u16; 3]) -> usize {
    let mut even_parity_paulis = [true, true, true];
    unsafe {
        for t in term {
            even_parity_paulis
                .get_unchecked_mut(0)
                .bitxor_assign(t == sorted_children.get_unchecked(0));
            even_parity_paulis
                .get_unchecked_mut(1)
                .bitxor_assign(t == sorted_children.get_unchecked(1));
            even_parity_paulis
                .get_unchecked_mut(2)
                .bitxor_assign(t == sorted_children.get_unchecked(2));
        }
    }
    let odd_parity_paulis = 3
        - (even_parity_paulis[0] as usize
            + even_parity_paulis[1] as usize
            + even_parity_paulis[2] as usize);

    !odd_parity_paulis.is_multiple_of(3) as usize
}

/// Simplify the Majorana operator Hamiltonian
///
/// As we traverse from leaves to root, we can simplify the Hamiltonian.
/// For each node we pass, we can guarantee that all Majorana operators
/// passing through that node have taken the same path to that node.
///
/// They therefore act with the same Pauli operator on every node which
/// is on that path.
///
/// We can therefore substitute a single index, representing the node, in place of
/// all the individual Majorana operator indices.
fn reduce_hamiltonian(
    majorana_terms: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
    parent_majorana_index: u16,
    selection: [u16; 3],
) -> Vec<ArrayVec<[u16; MAJORANA_MAX]>> {
    // could also filter here by terms that
    // only contain indices in pairs.
    let mut result: Vec<ArrayVec<[u16; MAJORANA_MAX]>> = majorana_terms
        .into_iter()
        .map(|mut term| {
            let original_len = term.len();
            term.retain(|&ind| !selection.contains(&ind));
            while term.len() < original_len {
                term.push(parent_majorana_index);
            }
            term.sort_unstable();
            term
        })
        .filter(|term| !term.iter().all(|&ind| ind == parent_majorana_index))
        .collect();
    // Use sort + dedup instead of BTreeSet for deduplication:
    // avoids per-element tree insertion overhead.
    result.sort_unstable();
    result.dedup();
    result
}

/// Toplogy-Preserving Hamiltonian-Adaptive Ternary Tree
///
/// Optimises a given [`TernaryTree`] to minimise the Pauli-weight
/// of the qubit hamiltonian obtained by encoding the input [`MajoranaSparse`] hamiltonian.
pub fn topphatt(
    mut hamiltonian: MajoranaSparse,
    mut tree: TernaryTree,
    parallelize: bool,
) -> Result<TernaryTree, ToppHattError> {
    let mut restrictions = TreeRestrictions::new(&tree);
    let mut node_dependencies = NodeDependencies::new(&tree);

    // Rough threshold at which it's worth the cost.
    let mut n_threads: usize = if parallelize && hamiltonian.indices.len() > 1000 {
        num_cpus::get()
    } else {
        1
    };

    // Reversing the direction tends to give better results for molecules
    let mut unassigned_modes: BTreeSet<usize> = BTreeSet::from_iter(0..tree.n_nodes);
    let mut total_weight = 0;
    debug!(
        "Number of hamiltonian terms {:?}",
        hamiltonian.indices.len()
    );
    debug!("Hamiltonian indices\n{:?}", &hamiltonian.indices);
    'assign: for loop_index in 0..tree.n_nodes {
        debug!("loop {:}", loop_index);
        debug!("Restrictions {:?}", restrictions);
        debug!("Dependencies {:?}", node_dependencies);
        debug!("Unassigned Modes {:?}", unassigned_modes);
        let n_leaves = 2 * tree.n_nodes + 1;

        let selection = RwLock::new(ToppHattSelection {
            min_weight: usize::MAX,
            min_parent: usize::MAX,
            leaf_indices: [u16::MAX; 3],
        });

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
            let mut allowed_x =
                restrictions.x[active].get_index_subset(&unassigned_modes, tree.n_nodes);
            // Optimisation:
            // Reversing x, y but leaving z increadsing order reduces the runtime for
            // for hamiltonians in tests.
            allowed_x.reverse();
            let mut allowed_y =
                restrictions.y[active].get_index_subset(&unassigned_modes, tree.n_nodes);
            allowed_y.reverse();
            let allowed_z =
                restrictions.z[active].get_index_subset(&unassigned_modes, tree.n_nodes);

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

            // Find the combination of possible assignments
            // which has the minimum Pauli weight
            // Good target for concurrency.
            // product.into_par_iter().for_each(|comb| {
            //     update_selection(&comb, active, selection, &hamiltonian)
            //         .expect("Threads should not panic in update selection.")
            // });
            let product = Mutex::new(product);
            const BATCH_SIZE: usize = 64;
            std::thread::scope(|s| {
                for _ in 0..n_threads {
                    s.spawn(|| {
                        let mut batch: Vec<Vec<u16>> = Vec::with_capacity(BATCH_SIZE);
                        loop {
                            // Grab a batch of combinations to reduce Mutex contention.
                            {
                                let mut product_guard =
                                    product.lock().expect("Product should not be poisoned.");
                                batch.clear();
                                batch.extend(product_guard.by_ref().take(BATCH_SIZE));
                            }
                            if batch.is_empty() {
                                break;
                            }

                            for comb in &batch {
                                let comb: [u16; 3] = if comb.len() == 3 {
                                    [comb[0], comb[1], comb[2]]
                                } else {
                                    let pair = if comb[0] % 2 == 0 {
                                        comb[0] + 1
                                    } else {
                                        comb[0] - 1
                                    };
                                    [comb[0], pair, comb[1]]
                                };

                                if comb[0] == comb[2] || comb[1] == comb[2] {
                                    continue;
                                };
                                let mut sorted_comb: [u16; 3] = comb;
                                sorted_comb.sort_unstable();
                                let comb_min = unsafe { sorted_comb.get_unchecked(0) };
                                let comb_max = unsafe { sorted_comb.get_unchecked(2) };

                                let read_guard = selection
                                    .read()
                                    .expect("Selection should not be poisoned before read.");
                                let min_weight = read_guard.min_weight;
                                drop(read_guard);

                                // We expect that the hamiltonian terms are sorted!
                                let weight = hamiltonian
                                    .indices
                                    .iter()
                                    .fold_while(0, |acc, inds| {
                                        if acc > min_weight {
                                            Done(acc)
                                        } else {
                                            debug_assert!(inds.is_sorted());
                                            let inds_max =
                                                unsafe { inds.last().unwrap_unchecked() };
                                            let inds_min =
                                                unsafe { inds.first().unwrap_unchecked() };

                                            if (comb_min > inds_max) | (comb_max < inds_min) {
                                                Continue(acc)
                                            } else {
                                                Continue(acc + qubit_term_weight(inds, &comb))
                                            }
                                        }
                                    })
                                    .into_inner();
                                // For most trees, using < gives the best results.
                                // counter example: JKMN(14), benefits from setting <=
                                // This part interacts with the ordering of active nodes,
                                // which is X-most to Z-Most

                                if weight <= min_weight {
                                    debug!("Selection {:?}", selection);
                                    debug!("Min Weight {:?}", min_weight);
                                    debug!("Min Parent {:?}", active);
                                    let mut write_guard = selection
                                        .write()
                                        .expect("Rwlock should not be poisoned before write.");
                                    if weight < write_guard.min_weight {
                                        write_guard.min_weight = weight;
                                        write_guard.leaf_indices = comb;
                                        write_guard.min_parent = active;
                                    } else if weight == write_guard.min_weight {
                                        let li = write_guard.leaf_indices;
                                        // Safety:
                                        // The the only use of these values is to compare u64 values below.
                                        // This allows us to make multi-threaded topp-hatt deterministic,
                                        // enforcing a specific ordering for leaf-index selection.
                                        // These values are immediately dropped.
                                        unsafe {
                                            let current = std::mem::transmute::<[u16; 4], u64>([
                                                0, li[0], li[1], li[2],
                                            ]);
                                            let this = std::mem::transmute::<[u16; 4], u64>([
                                                0, comb[0], comb[1], comb[2],
                                            ]);
                                            if this > current {
                                                write_guard.min_weight = weight;
                                                write_guard.leaf_indices = comb;
                                                write_guard.min_parent = active;
                                            }
                                        }
                                    };
                                };
                            }
                        }
                    });
                }
            });
        }
        // debug!("Selection {:?}", &selection);
        let selection = selection
            .into_inner()
            .expect("Should not have poisoned threads.");
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
                    .get(&(selection.min_parent, Edge::Z))
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

        let parent_majorana_index = selection.min_parent + n_leaves;
        debug!("Parent Majorana Index {parent_majorana_index}.");
        hamiltonian.indices = reduce_hamiltonian(
            hamiltonian.indices,
            parent_majorana_index as u16,
            selection.leaf_indices,
        );
        if hamiltonian.indices.len() < 1000 {
            n_threads = 1;
        }
        debug!("Reduced Hamiltonian {:#?}", hamiltonian.indices);
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
    use crate::encode::encoding::MajoranaEncoding;
    use crate::encode::ternarytree::TTFlatpack;
    use crate::encode::ternarytree::TernaryTree;
    use crate::optimise::topphatt::NodeDependencies;
    use crate::optimise::topphatt::TreeRestrictions;
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
        expected_pairs.insert((0, X), (0, Y));
        expected_pairs.insert((0, Y), (0, X));
        expected_pairs.insert((1, X), (1, Y));
        expected_pairs.insert((1, Y), (1, X));
        expected_pairs.insert((2, X), (2, Y));
        expected_pairs.insert((2, Y), (2, X));
        expected_pairs.insert((3, X), (3, Y));
        expected_pairs.insert((3, Y), (3, X));

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

        let jw_topphatt = topphatt(hamiltonian, tree, true).unwrap();
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
        let jw_topphatt = topphatt(hamiltonian, tree, true).unwrap();
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
}
