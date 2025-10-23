/*
Ternary tree encodings and methods.
*/

use anyhow::Result;
use itertools::Itertools;
use std::iter::zip;
type NodeIndexArray = [u8; 256];
const MAX_SIZE: usize = 85;

pub struct FastTernaryTree {
    parent_of: NodeIndexArray,
    x_child_of: NodeIndexArray,
    y_child_of: NodeIndexArray,
    z_child_of: NodeIndexArray,
    z_ancestor_of: NodeIndexArray,
    z_descendant_of: NodeIndexArray,
}

impl FastTernaryTree {
    pub fn new() -> Self {
        let initial_array: NodeIndexArray = core::array::from_fn(|i| { i } as u8);
        Self {
            parent_of: initial_array,
            x_child_of: initial_array,
            y_child_of: initial_array,
            z_child_of: initial_array,
            z_ancestor_of: initial_array,
            z_descendant_of: initial_array,
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
    let mut odd_parity_paulis: u8 = 0;
    for c in children {
        let occurances: usize = term
            .iter()
            .fold(0, |acc, t| if t == c { acc + 1 } else { acc });
        if occurances % 2 == 1 {
            odd_parity_paulis += 1;
        }
    }
    if odd_parity_paulis % 3 == 0 {
        1
    } else {
        0
    }
}

fn reduce_hamiltonian(
    majorana_terms: Vec<[u8; 4]>,
    parent_index: u8,
    selection: [u8; 3],
) -> Vec<[u8; 4]> {
    // could also filter here by terms that
    // only contain indices in pairs.
    majorana_terms
        .iter()
        .map(|term: &[u8; 4]| {
            term.map(|ind| {
                if selection.contains(&ind) {
                    parent_index
                } else {
                    ind
                }
            })
        })
        // .filter(|term: &[u8; 4]| term[0]!=term[1]||term[2]!=term[3])
        .collect()
}

pub fn hatt(
    n_nodes: usize,
    mut majorana_terms: Vec<[u8; 4]>,
) -> Result<(FastTernaryTree, u8, usize)> {
    assert!(n_nodes < MAX_SIZE);

    let mut tree = FastTernaryTree::new();
    let n_leaves = 2 * n_nodes + 1;
    let mut total_weight: usize = 0;
    let mut unassigned: [Option<u8>; 256] = core::array::from_fn(|i: usize| Some((i) as u8));
    unassigned = unassigned.map(|v| match v {
        Some(val) if (val as usize) < n_leaves => Some(val),
        _ => None,
    });
    for ind in 0..n_nodes {
        let mut selection: [u8; 3] = [u8::MAX, u8::MAX, u8::MAX];
        let parent_index = (n_leaves + ind) as u8;
        let mut min = usize::MAX;
        for comb in unassigned.iter().flatten().combinations(2) {
            let x_index = *comb[0];
            let z_index = *comb[1];

            let small_x: u8 = tree.z_descendant_of[x_index as usize];

            if small_x as usize == 2 * n_nodes {
                continue;
            };

            let small_y: u8 = if small_x % 2 == 0 {
                small_x + 1
            } else {
                small_x - 1
            };

            if small_y == x_index || small_y == z_index {
                // println!("small y cannot be one of the children");
                continue;
            };

            let y_index = tree.z_ancestor_of[small_y as usize];

            if y_index == x_index || y_index == z_index {
                // println!("y index cannot be one of the children");
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
            // println!("Selection constians initialisation values.");
            continue;
        }

        total_weight += min;
        for (child_index, child_of) in zip(
            selection,
            [
                &mut tree.x_child_of,
                &mut tree.y_child_of,
                &mut tree.z_child_of,
            ],
        ) {
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

    let remaining_nodes: Vec<u8> = unassigned.into_iter().flatten().collect::<Vec<u8>>();

    Ok((tree, remaining_nodes[0], total_weight))
}

#[cfg(test)]
mod tests {
    use crate::ternarytree::{hatt, FastTernaryTree};

    #[test]
    fn test_hatt_paper_example() {
        let mut majorana_terms: Vec<[u8; 4]> = Vec::from([[0u8, 0u8, 0u8, 1u8]]);
        majorana_terms.push([0u8, 0u8, 2u8, 3u8]);
        majorana_terms.push([0u8, 0u8, 4u8, 5u8]);
        majorana_terms.push([2u8, 3u8, 4u8, 5u8]);
        let n_nodes = 3;
        let tree: FastTernaryTree;
        let root_node: u8;
        let weight: usize;
        (tree, root_node, weight) = hatt(n_nodes, majorana_terms).unwrap();
        assert_eq!(root_node, 9);
        assert_eq!(weight, 1);
    }
}
