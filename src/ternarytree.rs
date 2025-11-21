/*
Ternary tree encodings and methods.
*/

use crate::types::MajoranaSparse;
use anyhow::Result;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use std::iter::zip;
use tinyvec::array_vec;
use tinyvec::ArrayVec;
type NodeIndexArray = [u8; 256];
const MAX_SIZE: usize = 85;
const MAJORANA_MAX: usize = 4;

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

fn qubit_term_weight(term: &ArrayVec<[u8; MAJORANA_MAX]>, children: &[u8; 3]) -> usize {
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
    mut majorana_terms: Vec<ArrayVec<[u8; MAJORANA_MAX]>>,
    parent_index: u8,
    selection: [u8; 3],
) -> Vec<ArrayVec<[u8; MAJORANA_MAX]>> {
    // could also filter here by terms that
    // only contain indices in pairs.
    majorana_terms
        .iter()
        .map(|&term| {
            let initial_length = term.len();
            let mut new_term: ArrayVec<[u8; MAJORANA_MAX]> = term
                .into_iter()
                .filter(|ind| !selection.contains(&ind))
                .collect();
            if (initial_length - new_term.len()) % 2 == 1 {
                new_term.push(parent_index);
            }
            new_term
        })
        .filter(|&term| term != ArrayVec::<[u8; MAJORANA_MAX]>::new())
        .collect()
}

pub fn hatt(
    n_nodes: usize,
    mut majorana_terms: Vec<ArrayVec<[u8; MAJORANA_MAX]>>,
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
        println!("\nParent {:?}", parent_index);
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
                .fold_while(0, |acc, term| {
                    if acc > min {
                        Done(acc)
                    } else {
                        if term.get(term.len() - 1) < children.get(0)
                            || term.get(0) > children.get(2)
                        {
                            Continue(acc)
                        } else {
                            Continue(acc + qubit_term_weight(term, &children))
                        }
                    }
                })
                .into_inner();

            println!("Children {:?}", children.clone());
            println!("Weight {:}", weight.clone());
            if weight < min {
                min = weight;
                println!("NEW MIN {:}", min.clone());
                selection = children;
                println!("SELECTION {:?}", selection.clone());
            }
        }
        // Initialized to max, but we could only ever use this leaf once at most.
        if selection.contains(&u8::MAX) {
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
        majorana_terms = reduce_hamiltonian(majorana_terms, parent_index, selection);
        println!("Majorana Terms {:?}", majorana_terms);
    }

    let remaining_nodes: Vec<u8> = unassigned.into_iter().flatten().collect::<Vec<u8>>();

    Ok((tree, remaining_nodes[0], total_weight))
}

#[cfg(test)]
mod tests {
    use crate::ternarytree::*;

    #[test]
    fn test_hatt_paper_example() {
        let mut majorana_terms: Vec<ArrayVec<[u8; 4]>> =
            Vec::from([array_vec!([u8;4] => 0u8, 1u8)]);
        majorana_terms.push(array_vec!([u8;4] => 2u8, 3u8));
        majorana_terms.push(array_vec!([u8;4] => 4u8, 5u8));
        majorana_terms.push(array_vec!([u8;4] => 2u8, 3u8, 4u8, 5u8));
        println!("Majorana Terms {:?}", majorana_terms);
        let n_nodes = 3;
        let root_node: u8;
        let weight: usize;
        let tree: FastTernaryTree;
        (tree, root_node, weight) = hatt(n_nodes, majorana_terms).unwrap();
        assert_eq!(root_node, 9);
        assert_eq!(weight, 5);
        assert_eq!(tree.parent_of[9], 9);
    }
}
