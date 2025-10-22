/*
Ternary tree encodings and methods.
*/

use anyhow::Result;
use itertools::Itertools;
use std::iter::zip;
type NodeIndexArray = [u8; 85];
const MAX_SIZE: usize = 85;

struct FastTernaryTree {
    parent_of: NodeIndexArray,
    x_child_of: NodeIndexArray,
    y_child_of: NodeIndexArray,
    z_child_of: NodeIndexArray,
    z_ancestor_of: NodeIndexArray,
    z_descendant_of: NodeIndexArray,
}

impl FastTernaryTree {
    pub fn new() -> Self {
        let initial_array: NodeIndexArray = core::array::from_fn(|i| { i + 1 } as u8);
        Self {
            parent_of: initial_array.clone(),
            x_child_of: initial_array.clone(),
            y_child_of: initial_array.clone(),
            z_child_of: initial_array.clone(),
            z_ancestor_of: initial_array.clone(),
            z_descendant_of: initial_array.clone(),
        }
    }
}

fn add_child(
    parent_index: u8,
    parent_of: &mut NodeIndexArray,
    child_index: u8,
    child_of: &mut NodeIndexArray,
) -> Result<()> {
    // Child should always be set
    // is set as self index initially.

    // if not self-child
    // change child
    // add parent to new child
    // // remove current childs parent
    let existing_child: u8 = child_of[parent_index as usize];
    if existing_child == parent_index {
        parent_of[existing_child as usize] = existing_child;
    }
    child_of[parent_index as usize] = child_index;
    parent_of[child_index as usize] = parent_index;
    Ok(())
}

fn qubit_term_weight(term: &[u8; 4], children: &[u8; 3]) -> usize {
    0
}

fn reduce_hamiltonian(
    mut majorana_terms: Vec<[u8; 4]>,
    parent_index: u8,
    selection: [u8; 3],
) -> Vec<[u8; 4]> {
    majorana_terms
}

pub fn hatt(n_nodes: usize, mut majorana_terms: Vec<[u8; 4]>) -> Result<FastTernaryTree> {
    let mut tree = FastTernaryTree::new();
    let n_leaves = 2 * n_nodes + 1;
    let mut total_weight: usize = 0;
    let mut unassigned: [Option<u8>; 256] = core::array::from_fn(|i: usize| Some((i + 1) as u8));
    unassigned.map(|v| match v {
        Some(val) if (val as usize) < n_leaves => Some(val),
        _ => None,
    });
    let mut selection: [u8; 3] = [u8::MAX, u8::MAX, u8::MAX];
    for ind in 0..n_nodes {
        let parent_index = (n_leaves + ind) as u8;
        let mut min = usize::MAX;
        for comb in unassigned.iter().flatten().combinations(2) {
            let x_index = *comb[0];
            let z_index = *comb[1];

            let small_x: u8 = tree.z_descendant_of[x_index as usize];
            let small_y: u8 = if small_x % 2 == 0 {
                small_x + 1
            } else {
                small_x - 1
            };

            if small_y == x_index || small_y == z_index {
                continue;
            };

            let y_index = tree.z_ancestor_of[small_y as usize];

            if y_index == x_index || y_index == z_index {
                continue;
            };

            let children: [u8; 3] = if small_x % 2 == 0 {
                [x_index, y_index, z_index]
            } else {
                [y_index, x_index, z_index]
            };
            let weight = majorana_terms
                .iter()
                .fold(0, |acc, term| acc + qubit_term_weight(term, &children));

            if weight < min {
                min = weight;
                selection = children;
            }
        }
        // Initialized to max, but we could only ever use this leaf once at most.
        if selection
            .iter()
            .fold(0, |acc, v| if *v == u8::MAX { acc + 1 } else { acc })
            > 1
        {
            continue;
        }

        total_weight += min;
        for (possible_child, child_of) in zip(
            selection,
            [
                &mut tree.x_child_of,
                &mut tree.y_child_of,
                &mut tree.z_child_of,
            ],
        ) {
            let child_index: u8 = possible_child;
            unassigned[child_index as usize] = None;
            if (child_index as usize) < n_leaves {
                // If the child is a node,
                // we need a few steps to keep things consistent.
                add_child(parent_index, &mut tree.parent_of, child_index, child_of)?;
            } else {
                // If it's a leaf then we just assign the leave
                // number as child
                child_of[parent_index as usize] = child_index;
            }
        }
        let z_index = selection[2];
        let z_desc = tree.z_descendant_of[z_index as usize];
        tree.z_descendant_of[parent_index as usize] = z_desc;
        tree.z_ancestor_of[z_index as usize] = parent_index;
        tree.z_ancestor_of[z_desc as usize] = parent_index;

        unassigned[parent_index as usize] = Some(parent_index);

        majorana_terms = reduce_hamiltonian(majorana_terms, parent_index, selection)
    }

    let remaining_nodes = unassigned.iter().flatten();
    assert_eq!(remaining_nodes.try_len().unwrap(), 1);
    Ok(tree)
}
