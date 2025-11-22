/*
Ternary tree encodings and methods.
*/
use anyhow::Result;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use std::iter::zip;
use tinyvec::ArrayVec;
const MAX_SIZE: usize = 85;
const MAJORANA_MAX: usize = 4;

use crate::ternarytree::{Edge, FastTernaryTree};

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

struct HattResult {
    tree: FastTernaryTree,
    root_node_index: u8,
    pauli_weight: usize,
}

pub fn hatt(
    n_nodes: usize,
    mut majorana_terms: Vec<ArrayVec<[u8; MAJORANA_MAX]>>,
) -> Result<HattResult> {
    assert!(n_nodes < MAX_SIZE);

    let mut tree = FastTernaryTree::new(n_nodes);
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
        for (child_index, branch) in zip(selection, [Edge::X, Edge::Y, Edge::Z]) {
            unassigned[child_index as usize] = None;
            if (child_index as usize) < n_leaves {
                // If the child is a node,
                // we need a few steps to keep things consistent.
                tree.add_child(branch, parent_index, child_index);
            } else {
                // If it's a leaf then we just assign the leaf
                // number as child
                tree.add_leaf(branch, parent_index, child_index);
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

    Ok(HattResult {
        tree: tree,
        root_node_index: remaining_nodes[0],
        pauli_weight: total_weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tinyvec::array_vec;

    #[test]
    fn test_hatt_paper_example() {
        let mut majorana_terms: Vec<ArrayVec<[u8; 4]>> =
            Vec::from([array_vec!([u8;4] => 0u8, 1u8)]);
        majorana_terms.push(array_vec!([u8;4] => 2u8, 3u8));
        majorana_terms.push(array_vec!([u8;4] => 4u8, 5u8));
        majorana_terms.push(array_vec!([u8;4] => 2u8, 3u8, 4u8, 5u8));
        println!("Majorana Terms {:?}", majorana_terms);
        let n_nodes = 3;
        let hr = hatt(n_nodes, majorana_terms).unwrap();
        assert_eq!(hr.root_node_index, 9);
        assert_eq!(hr.pauli_weight, 5);
        assert_eq!(hr.tree.parent_of[9], 9);
    }
}
