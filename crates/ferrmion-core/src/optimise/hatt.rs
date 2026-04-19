//! Hamiltonian-Adaptive Ternary Tree (HATT) construction.
//!
//! Greedy construction of a [`crate::encode::ternarytree::TernaryTree`] that
//! minimises the Pauli weight of the resulting qubit Hamiltonian when applied
//! to a given Majorana-operator Hamiltonian. Mirrors the Python reference in
//! `python/ferrmion/optimize/hatt.py::hamiltonian_adaptive_ternary_tree`.

use itertools::Itertools;
use log::debug;
use std::collections::{BTreeSet, VecDeque};
use thiserror::Error;
use tinyvec::ArrayVec;

use crate::encode::ternarytree::TTFlatpack;
use crate::optimise::common::{qubit_term_weight, reduce_hamiltonian, MAJORANA_MAX};

/// Errors produced during HATT construction.
#[derive(Debug, Error)]
pub enum HattError {
    #[error("No valid selection found at iteration {0}.")]
    NoSelectionMade(usize),
    #[error("Expected exactly one unassigned entity after HATT; got {0}.")]
    UnassignedRemainder(usize),
}

/// Construct a ternary tree adapted to the given Majorana Hamiltonian.
///
/// Entity IDs used internally:
/// - Leaves: `0..2*n_modes+1`, where `2*n_modes` is the "all-Z" terminator leaf.
/// - Nodes: `2*n_modes+1..3*n_modes+1`. The node at ID `2*n_modes+1+i` has
///   qubit label `i`; the node created in the final iteration is the root.
///
/// Returns the constructed tree as a [`TTFlatpack`] plus the total Pauli weight.
pub fn hatt(
    majorana_terms: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
    n_modes: usize,
) -> Result<(TTFlatpack, usize), HattError> {
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

    Ok((flatpack, total_weight))
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
        let (flatpack, _weight) = hatt(terms, 3).unwrap();

        assert_eq!(flatpack.len(), 3);
        let labels: BTreeSet<usize> = flatpack.iter().map(|(q, _)| *q).collect();
        assert_eq!(labels, (0..3).collect::<BTreeSet<_>>());
    }

    #[test]
    fn test_hatt_strip_all_z_leaf() {
        let terms = vec![
            array_vec!([u16; 7] => 0u16, 1),
            array_vec!([u16; 7] => 2u16, 3),
        ];
        let (flatpack, _weight) = hatt(terms, 2).unwrap();
        // Exactly one (qubit, (x, y, z)) entry in the flatpack must have a
        // None z-child: the all-Z terminator leaf, stripped post-pass.
        let none_z_count = flatpack.iter().filter(|(_, (_, _, z))| z.is_none()).count();
        assert_eq!(none_z_count, 1);
    }
}
