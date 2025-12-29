use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use log::debug;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::iter::zip;
use std::usize;
use thiserror::Error;
use tinyvec::ArrayVec;
const MAJORANA_MAX: usize = 4;

use crate::operators::MajoranaSparse;
use crate::ternarytree::{Child, Edge, TernaryTree, YParity};

#[derive(Debug, Error)]
pub enum ToppHattError {
    #[error("Found invalid restriction: {0:?}.")]
    RestrictionError(Restriction),
    #[error("Combination {0:?} cannot be used.")]
    InvalidCombinationError(Vec<u16>),
    #[error("No selection made for loop index {0}.")]
    NoSelectionError(usize),
    #[error("No min parent for loop index {0}.")]
    NoMinParentError(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Restriction {
    OddLeaf,
    EvenLeaf,
    Empty,
    ChildNode(u8),
    Any,
    Majorana(u16),
}

impl Restriction {
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

type LeafLocation = (usize, Edge);

#[derive(Debug, PartialEq)]
struct LeafPair {
    x: LeafLocation,
    y: LeafLocation,
}

#[derive(Debug, PartialEq)]
struct TreeRetrictions {
    x: Vec<Restriction>,
    y: Vec<Restriction>,
    z: Vec<Restriction>,
    pairs: HashMap<LeafLocation, LeafLocation>,
}

impl TreeRetrictions {
    fn new(tree: &TernaryTree) -> Self {
        let x: Vec<Restriction> = vec![Restriction::Any; tree.n_nodes];
        let y: Vec<Restriction> = vec![Restriction::Any; tree.n_nodes];
        let z: Vec<Restriction> = vec![Restriction::Any; tree.n_nodes];
        let pairs: HashMap<LeafLocation, LeafLocation> = HashMap::new();

        let mut output = Self { x, y, z, pairs };

        output.apply_all_z(tree);
        output.apply_retain_children(tree);
        output.apply_leaf_parity(tree);
        output.find_leaf_pairs(tree);

        output
    }

    fn apply_all_z(&mut self, tree: &TernaryTree) {
        let all_z_index = tree
            .z_child_of
            .iter()
            .position(|&v| v.is_none())
            .expect("Input tree should not have all-z leaf assigned.");
        self.z[all_z_index] = Restriction::Empty;
    }

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

    fn apply_leaf_parity(&mut self, tree: &TernaryTree) {
        for (restriction, children) in zip(
            [&mut self.x, &mut self.y, &mut self.z],
            [&tree.x_child_of, &tree.y_child_of, &tree.z_child_of],
        ) {
            for (r, c) in zip(restriction, children) {
                let parity: YParity;
                match c {
                    Some(Child::XLeaf(leaf_index)) => {
                        parity = tree.y_parity_of[*leaf_index as usize];
                    }
                    Some(Child::YLeaf(leaf_index)) => {
                        parity = !tree.y_parity_of[*leaf_index as usize];
                    }
                    _ => {
                        continue;
                    }
                }
                match parity {
                    YParity::Even => {
                        *r = Restriction::EvenLeaf;
                    }
                    YParity::Odd => {
                        *r = Restriction::OddLeaf;
                    }
                }
            }
        }
    }

    fn find_leaf_pairs(&mut self, tree: &TernaryTree) {
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

impl TreeRetrictions {
    fn update_tree(self, tree: &mut TernaryTree) -> Result<(), ToppHattError> {
        let n_nodes = &self.x.len();
        debug!("Updatign tree {self:?}");
        assert_eq!(
            &self.y.len(),
            n_nodes,
            "XYZ restrictions should be same length."
        );
        assert_eq!(
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
                        assert!(matches!(c, Some(Child::Node(_))));
                    }
                    Restriction::Empty => {
                        assert!(c.is_none())
                    }
                    _ => return Err(ToppHattError::RestrictionError(*r)),
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
struct NodeDependencies {
    root_distances: BTreeMap<usize, usize>,
    children_without_leaves: BTreeMap<usize, ArrayVec<[usize; 3]>>,
}

impl NodeDependencies {
    fn new(tree: &TernaryTree) -> Self {
        // find the root node by traversing up
        // it will usually be the 0th position so start there
        let mut parent_index: usize = 0;
        while let Some(parent) = tree.parent_of[parent_index] {
            parent_index = parent.node_index();
        }
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
        debug!("{children_without_leaves:?}");

        Self {
            root_distances,
            children_without_leaves,
        }
    }

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

#[inline(always)]
fn qubit_term_weight(term: &ArrayVec<[u16; MAJORANA_MAX]>, children: &[u16; 3]) -> usize {
    let mut odd_parity_paulis: u8 = 0;
    for c in children {
        let occurances: usize = term.iter().filter(|&t| t == c).count();
        if occurances % 2 == 1 {
            odd_parity_paulis += 1;
        }
    }
    if odd_parity_paulis % 3 != 0 {
        1
    } else {
        0
    }
}

fn reduce_hamiltonian(
    mut majorana_terms: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
    parent_majorana_index: u16,
    selection: [u16; 3],
) -> Vec<ArrayVec<[u16; MAJORANA_MAX]>> {
    // could also filter here by terms that
    // only contain indices in pairs.
    majorana_terms
        .iter_mut()
        .map(|&mut term| {
                term.iter().map(|&ind| if selection.contains(&ind) {parent_majorana_index} else {ind}).collect()
        })
        .filter(|&term| term != ArrayVec::<[u16; MAJORANA_MAX]>::new())
        .collect::<BTreeSet<ArrayVec<[u16; MAJORANA_MAX]>>>()
        .into_iter()
        .collect::<Vec<ArrayVec<[u16; MAJORANA_MAX]>>>()
}

pub fn topphatt(
    mut hamiltonian: MajoranaSparse,
    mut tree: TernaryTree,
) -> Result<TernaryTree, ToppHattError> {
    let mut restrictions = TreeRetrictions::new(&tree);
    let mut node_dependencies = NodeDependencies::new(&tree);

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
        let mut selection: [u16; 3] = [u16::MAX, u16::MAX, u16::MAX];
        let mut min_parent: usize = usize::MAX;
        let mut min_weight = usize::MAX;

        let max_root_distance: &usize = node_dependencies
            .root_distances
            .values()
            .max()
            .expect("Root distances should not be empty.");
        debug!("Max root distance {:?}", max_root_distance);

        let active_nodes: Vec<usize> = node_dependencies
            .root_distances
            .iter()
            .zip(node_dependencies.children_without_leaves.values())
            .filter(|&((_, rd), &uc)| (rd == max_root_distance) & (uc == ArrayVec::new()))
            .map(|((&ind, _), _)| ind)
            .collect();

        debug!("Active Nodes {:?}", active_nodes);
        for active in active_nodes {
            debug!("Active {:?}", active);

            let mut allowed_x =
                restrictions.x[active].get_index_subset(&unassigned_modes, tree.n_nodes);
            allowed_x.reverse();
            let mut allowed_y =
                restrictions.y[active].get_index_subset(&unassigned_modes, tree.n_nodes);
            allowed_y.reverse();
            let mut allowed_z =
                restrictions.z[active].get_index_subset(&unassigned_modes, tree.n_nodes);
            allowed_z.reverse();

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
            for comb in product {
                // debug!("Comb {:?}", &comb);
                let comb: [u16; 3] = match comb.len() {
                    2 => {
                        let pair = if comb[0] % 2 == 0 {
                            comb[0] + 1
                        } else {
                            comb[0] - 1
                        };
                        [comb[0], pair, comb[1]]
                    }
                    3 => [comb[0], comb[1], comb[2]],
                    _ => return Err(ToppHattError::InvalidCombinationError(comb)),
                };
                if comb[0] == comb[2] {
                    continue;
                }
                // We expect that the hamiltonian terms are sorted!
                let weight = hamiltonian
                    .indices
                    .iter()
                    .fold_while(0, |acc, inds| {
                        let inds_max = inds.iter().max().expect("Hamiltonian terms should not be empty.");
                        let inds_min = inds
                            .iter().min()
                            .expect("Hamiltonian terms should not be empty.");

                        let comb_min = comb.iter().min().expect("Combination should not be empty.");
                        let comb_max = comb.iter().max().expect("Combination should not be empty.");

                        if (comb_min > inds_max) | (comb_max < inds_min) {
                            Continue(acc)
                        } else if acc > min_weight {
                            Done(acc)
                        } else {
                            Continue(acc + qubit_term_weight(inds, &comb))
                        }
                    })
                    .into_inner();
                // For most trees, using < gives the best results.
                // counter example: JKMN(14), benefits from setting <=
                // This part interacts with the ordering of active nodes, 
                // which is X-most to Z-Most
                if weight < min_weight {
                    min_weight = weight;
                    selection = comb;
                    min_parent = active;
                    debug!("Min Weight {:?}", min_weight);
                    debug!("Selection {:?}", selection);
                    debug!("Min Parent {:?}", min_parent);
                }
            }
            debug!("Finished active\n");
        }

        debug!("Selection {:?}", selection);
        debug!("Min Parent {:?}", min_parent);
        match selection {
            [u16::MAX, u16::MAX, u16::MAX] => {
                return Err(ToppHattError::NoSelectionError(loop_index))
            }
            _ => {
                debug!("Removing selection from unassigned");
                selection
                    .into_iter()
                    .filter(|&v| n_leaves > v as usize)
                    .inspect(|v| {
                        debug!("1st inspect {:?}", v);
                    })
                    .map(|v| if v % 2 == 0 { v / 2 } else { (v - 1) / 2 })
                    .inspect(|v| {
                        debug!("2nd inspect {:?}", v);
                    })
                    .for_each(|v| {
                        unassigned_modes.remove(&(v as usize));
                    });
            }
        }
        debug!("Unassigned {:?}", unassigned_modes);
        total_weight += min_weight;
        debug!("Total weight {:?}", total_weight);

        match min_parent {
            usize::MAX => return Err(ToppHattError::NoMinParentError(loop_index)),
            _ => node_dependencies.drop_node(min_parent),
        }

        debug!("Dropped dependencies");
        for (&sel, res) in zip(
            &selection,
            [
                &mut restrictions.x,
                &mut restrictions.y,
                &mut restrictions.z,
            ],
        ) {
            if (sel as usize) < n_leaves - 1 {
                res[min_parent] = Restriction::Majorana(sel);
            } else if (sel as usize) == n_leaves {
                res[min_parent] = Restriction::Empty;
            }
        }

        debug!("Selection {:?}", selection);
        // Need to subtract one so that the all-z leaf
        // which is set at index 2*n_nodes doesn't look for a pair.
        // Be careful about zero indexing here too.
        if (selection[2] as usize) < n_leaves - 1 {
            let pair_index: u16 = if selection[2] % 2 == 0 {
                selection[2] + 1
            } else {
                selection[2] - 1
            };
            debug!("pair index {:?}", pair_index);
            let partner_location: LeafLocation = {
                *restrictions
                    .pairs
                    .get(&(min_parent, Edge::Z))
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

        let parent_majorana_index = min_parent + n_leaves;
        debug!("Parent Majorana Index {parent_majorana_index}.");
        hamiltonian.indices =
            reduce_hamiltonian(hamiltonian.indices, parent_majorana_index as u16, selection);
        debug!("Reduced Hamiltonian {:?}", hamiltonian.indices);
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
    use crate::encoding::MajoranaEncoding;
    use crate::optimise::topphatt::NodeDependencies;
    use crate::ternarytree::TTFlatPack;
    use crate::{optimise::topphatt::TreeRetrictions, ternarytree::TernaryTree};
    use log::debug;
    use ndarray::arr1;
    use numpy::Complex64;
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
        let jw_restrictions = TreeRetrictions::new(&jw_tree);
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

        let expected = TreeRetrictions {
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
        let restrictions = TreeRetrictions::new(&tree);
        debug!("{:?}", restrictions);
        let mut expected_pairs: HashMap<LeafLocation, LeafLocation> = HashMap::new();
        let pairs = [((1, Z), (0, Y)), ((2, Z), (1, Y)), ((2, X), (2, Y))];
        pairs.iter().for_each(|&(k, v)| {
            expected_pairs.insert(k, v);
            expected_pairs.insert(v, k);
        });

        let expected = TreeRetrictions {
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
        let restrictions = TreeRetrictions::new(&tree);
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

        let expected = TreeRetrictions {
            x: vec![
                ChildNode(1),
                ChildNode(4),
                OddLeaf,
                EvenLeaf,
                EvenLeaf,
                OddLeaf,
            ],
            y: vec![
                ChildNode(2),
                ChildNode(5),
                EvenLeaf,
                OddLeaf,
                OddLeaf,
                EvenLeaf,
            ],
            z: vec![ChildNode(3), EvenLeaf, OddLeaf, Empty, EvenLeaf, OddLeaf],
            pairs: expected_pairs,
        };
        assert_eq!(expected, restrictions, "Test JKMN(6) Restrictions.");
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
            vec![array_vec!([u16; 4]=> 2,3)],
            vec![Complex64::new(1., 0.)],
            0.,
        )
        .unwrap();
        let tree = TernaryTree::naive_jordan_wigner(3);

        let jw_topphatt = topphatt(hamiltonian, tree).unwrap();
        let encoding: MajoranaEncoding = jw_topphatt.build_encoding(3).unwrap();
        assert_eq!(encoding.ipowers, arr1(&[0, 1, 0, 1, 0, 1]));
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
            vec![array_vec!([u16; 4]=> 2,3)],
            vec![Complex64::new(1., 0.)],
            0.,
        )
        .unwrap();
        let mut flatpack = TTFlatPack::new();
        flatpack.push((1, (None, None, Some(2))));
        flatpack.push((2, (None, None, Some(3))));
        flatpack.push((3, (None, None, None)));

        let tree = TernaryTree::from_flatpack_naive(flatpack).unwrap();
        let jw_topphatt = topphatt(hamiltonian, tree).unwrap();
        let encoding = jw_topphatt.build_encoding(4).unwrap();
        assert_eq!(encoding.ipowers, arr1(&[0, 1, 0, 1, 0, 1]));
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
            array_vec!([u16;4] => 0,1,2,3),
            array_vec!([u16;4] => 0,2,3,4),
        ];

        hamiltonian = reduce_hamiltonian(hamiltonian, 999, [2, 3, 55]);

        let expected = vec![
            array_vec!([u16;4] => 0,1,999,999),
            array_vec!([u16;4] => 0,999,999, 4),
        ];

        assert_eq!(hamiltonian, expected);
    }
}
