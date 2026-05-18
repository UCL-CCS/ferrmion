//! Hamiltonian-Adaptive Ternary Tree (HATT) construction.
//!
//! Greedy construction of a [`crate::encode::ternarytree::TernaryTree`] that
//! minimises the Pauli weight of the resulting qubit Hamiltonian when applied
//! to a given Majorana-operator Hamiltonian. Mirrors the Python reference in
//! `python/ferrmion/optimize/hatt.py::hamiltonian_adaptive_ternary_tree`.

use itertools::Itertools;
use log::debug;
use std::collections::{BTreeSet, VecDeque};
use std::ops::BitXorAssign;
use thiserror::Error;
use tinyvec::ArrayVec;

use crate::encode::ternarytree::{TTFlatpack, TernaryTree, TernaryTreeError};
pub(crate) const MAJORANA_MAX: usize = 7;

/// Errors produced during HATT construction.
#[derive(Debug, Error)]
pub enum HattError {
    #[error("No valid selection found at iteration {0}.")]
    NoSelectionMade(usize),
    #[error("Expected exactly one unassigned entity after HATT; got {0}.")]
    UnassignedRemainder(usize),
    #[error("Failed to materialise the resulting TernaryTree: {0}")]
    TreeConstruction(#[from] TernaryTreeError),
}

/// Find the weight of a term on the qubit of a single node.
///
/// This function is used to assess the cost of each possible choice
/// of outward edges of a given node. Each outward edge has an associated
/// index. Either a Majorana-index, or a Node-index.
///
/// Each term is composed of some number of Majorana operators.
///
/// Where a Majorana operator is included in the _children_ of a given node,
/// the Majorana operator acts on that node's qubit with a non-Identity operator.
///
/// Additionally, using [`reduce_hamiltonian`] we guarantee that for
/// [`crate::encode::ternarytree::TernaryTree`]s,
/// no two distinct indices represent Majorana operators which
/// act with the same Pauli operator.
///
/// We wish to find out whether the product of Majorana operators in a given
/// Hamiltonian term require the application of non-Identity operator.
///
/// For each Majorana operator in a term:
/// - if it is not in _children_, it acts with the Identity.
/// - if it appears an even number of times, it acts with the Identity, as: PP=I forall P in {X,Y,Z,I}
///
/// if three Majorana operators appear in both the term and _children_ with odd parity,
/// together, they act with the identity as XYZ=-iI
#[inline(always)]
pub(crate) fn qubit_term_weight(
    term: &ArrayVec<[u16; MAJORANA_MAX]>,
    sorted_children: &[u16; 3],
) -> usize {
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

/// Simplify the Majorana operator Hamiltonian.
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
pub(crate) fn reduce_hamiltonian(
    majorana_terms: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
    parent_majorana_index: u16,
    selection: [u16; 3],
) -> Vec<ArrayVec<[u16; MAJORANA_MAX]>> {
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

/// Construct a ternary tree adapted to the given Majorana Hamiltonian.
///
/// Entity IDs used internally:
/// - Leaves: `0..2*n_modes+1`, where `2*n_modes` is the "all-Z" terminator leaf.
/// - Nodes: `2*n_modes+1..3*n_modes+1`. The node at ID `2*n_modes+1+i` has
///   qubit label `i`; the node created in the final iteration is the root.
///
/// Returns the constructed [`TernaryTree`] plus the total Pauli weight.
/// A flatpack is recoverable from the tree via [`TernaryTree::to_flatpack`].
pub fn hatt(
    majorana_terms: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
    n_modes: usize,
) -> Result<(TernaryTree, usize), HattError> {
    let n_leaves = 2 * n_modes + 1;
    let total_entities = n_leaves + n_modes;

    // Node children, indexed by node offset `0..n_modes`.
    let mut x_of: Vec<Option<u16>> = vec![None; n_modes];
    let mut y_of: Vec<Option<u16>> = vec![None; n_modes];
    let mut z_of: Vec<Option<u16>> = vec![None; n_modes];

    // Unassigned entity IDs. `BTreeSet` gives deterministic sorted iteration,
    // matching the Python reference once it is pinned to `sorted(unassigned)`.
    let mut unassigned: BTreeSet<u16> = (0..n_leaves as u16).collect();

    // Z-ancestor / Z-descendant maps, indexed by entity ID. Initially each
    // entity is its own ancestor and descendant.
    let mut ancestor_map: Vec<u16> = (0..total_entities as u16).collect();
    let mut descendant_map: Vec<u16> = (0..total_entities as u16).collect();

    let mut hamiltonian = majorana_terms;
    let mut total_weight: usize = 0;

    let all_z_leaf: u16 = (2 * n_modes) as u16;

    for i in 0..n_modes {
        let parent_id: u16 = (n_leaves + i) as u16;

        let mut min_weight = usize::MAX;
        let mut selection: Option<[u16; 3]> = None;

        let candidates: Vec<u16> = unassigned.iter().copied().collect();
        for perm in candidates.iter().copied().permutations(2) {
            let x_index = perm[0];
            let z_index = perm[1];
            let small_x = descendant_map[x_index as usize];

            // The all-Z terminator leaf cannot be the Majorana pair "anchor".
            if small_x == all_z_leaf {
                continue;
            }

            // Majorana pair: even and odd indices come in pairs (2k, 2k+1).
            let small_y = if small_x.is_multiple_of(2) {
                small_x + 1
            } else {
                small_x - 1
            };

            if small_y == x_index || small_y == z_index {
                continue;
            }

            let y_index = ancestor_map[small_y as usize];

            if y_index == x_index || y_index == z_index {
                continue;
            }

            // Order the triple so that even-Majorana pair members go on the
            // X-edge and odd on the Y-edge (required for real coefficients).
            let comb: [u16; 3] = if small_x.is_multiple_of(2) {
                [x_index, y_index, z_index]
            } else {
                [y_index, x_index, z_index]
            };

            let weight: usize = hamiltonian
                .iter()
                .map(|term| qubit_term_weight(term, &comb))
                .sum();

            if weight < min_weight {
                min_weight = weight;
                selection = Some(comb);
            }
        }

        let selection = selection.ok_or(HattError::NoSelectionMade(i))?;
        total_weight += min_weight;

        // Attach the three chosen children to the new parent node.
        for (slot, child_id) in selection.iter().enumerate() {
            let child_id = *child_id;
            unassigned.remove(&child_id);
            match slot {
                0 => x_of[i] = Some(child_id),
                1 => y_of[i] = Some(child_id),
                2 => z_of[i] = Some(child_id),
                _ => unreachable!(),
            }
        }

        // Update Z-chain bookkeeping so subsequent iterations can find the
        // true Z-descendant / Z-ancestor across the new node.
        let z_index = selection[2];
        let z_desc = descendant_map[z_index as usize];
        descendant_map[parent_id as usize] = z_desc;
        ancestor_map[z_index as usize] = parent_id;
        ancestor_map[z_desc as usize] = parent_id;

        unassigned.insert(parent_id);

        hamiltonian = reduce_hamiltonian(hamiltonian, parent_id, selection);
        debug!(
            "HATT iter {i}: weight={min_weight} selection={selection:?} remaining_terms={}",
            hamiltonian.len()
        );
    }

    if unassigned.len() != 1 {
        return Err(HattError::UnassignedRemainder(unassigned.len()));
    }

    let root_id = *unassigned.iter().next().expect("checked non-empty");
    let root_offset = (root_id as usize) - n_leaves;

    // Strip the all-Z leaf: walk the Z-chain from the root and blank out the
    // final Z-edge (which points to the `2*n_modes` terminator leaf).
    let mut z_tip_offset = root_offset;
    loop {
        match z_of[z_tip_offset] {
            Some(child_id) if (child_id as usize) >= n_leaves => {
                z_tip_offset = (child_id as usize) - n_leaves;
            }
            Some(_) => {
                z_of[z_tip_offset] = None;
                break;
            }
            None => break,
        }
    }

    // BFS from root, emitting each node's (qubit_label, (x, y, z)) entry.
    // Child encoding mirrors Python `TernaryTree.flatpack()`:
    //  - A child node emits its qubit label (`child_id - n_leaves`).
    //  - A child leaf emits `majorana_idx + max_node_index + 1`, which is
    //    `leaf_id + n_modes` here (`max_node_index == n_modes - 1`).
    let mut flatpack: TTFlatpack = Vec::with_capacity(n_modes);
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(root_offset);

    while let Some(node_offset) = queue.pop_front() {
        let qubit_label = node_offset;
        let mut encoded: [Option<usize>; 3] = [None, None, None];
        for (edge_idx, child_slot) in [x_of[node_offset], y_of[node_offset], z_of[node_offset]]
            .iter()
            .enumerate()
        {
            if let Some(child_id) = child_slot {
                let child_id = *child_id as usize;
                if child_id >= n_leaves {
                    let child_offset = child_id - n_leaves;
                    queue.push_back(child_offset);
                    encoded[edge_idx] = Some(child_offset);
                } else {
                    encoded[edge_idx] = Some(child_id + n_modes);
                }
            }
        }
        flatpack.push((qubit_label, (encoded[0], encoded[1], encoded[2])));
    }

    let tree = TernaryTree::from_flatpack(&flatpack)?;
    Ok((tree, total_weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tinyvec::array_vec;

    #[test]
    fn test_hatt_3mode_runs() {
        // Matches python/tests/test_optimize/test_hatt.py::test_hatt fixture.
        // Parity with Python's weight is asserted from the Python side in
        // python/tests/test_optimize/test_hatt_rust.py.
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1),
            array_vec!([u16; 7] => 2u16, 3),
            array_vec!([u16; 7] => 4u16, 5),
            array_vec!([u16; 7] => 2u16, 3, 4, 5),
        ];
        let (tree, _weight) = hatt(terms, 3).unwrap();

        assert_eq!(tree.n_nodes, 3);
        let flatpack = tree.to_flatpack();
        let labels: BTreeSet<usize> = flatpack.iter().map(|(q, _)| *q).collect();
        assert_eq!(labels, (0..3).collect::<BTreeSet<_>>());
    }

    #[test]
    fn test_hatt_strip_all_z_leaf() {
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1),
            array_vec!([u16; 7] => 2u16, 3),
        ];
        let (tree, _weight) = hatt(terms, 2).unwrap();
        // Exactly one (qubit, (x, y, z)) entry in the flatpack must have a
        // None z-child: the all-Z terminator leaf, stripped post-pass.
        let flatpack = tree.to_flatpack();
        let none_z_count = flatpack.iter().filter(|(_, (_, _, z))| z.is_none()).count();
        assert_eq!(none_z_count, 1);
    }
}
