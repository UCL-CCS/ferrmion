use anyhow::Result;
/*
Ternary Tree Graph Structures and Encoding.
*/
use crate::types::Pauli;
use anyhow::anyhow;
use ndarray::{Array2, ArrayView1};

#[derive(Debug, Clone)]
pub struct ArrayTT {
    matrix: Array2<Option<usize>>,
}

impl ArrayTT {
    pub fn new(n_nodes: usize) -> ArrayTT {
        // parent: Starts on None
        // x: Starts on Leaf n->n_modes+2*n+1
        // y: Starts on Leaf n->n_modes+2*n+2
        // z: Starts on None, or
        // z_ancestor: Index of self
        // z_descendant: Index of self
        let leaves_begin: usize = n_nodes;
        let mut initial_array = Array2::<Option<usize>>::from_elem((0, 6), None);
        for ind in 0..n_nodes {
            initial_array
                .push_row(ArrayView1::from(&[
                    None,
                    Some(leaves_begin + 2 * ind),
                    Some(leaves_begin + 2 * ind + 1),
                    None,
                    Some(ind),
                    Some(ind),
                ]))
                .unwrap();
        }
        Self {
            matrix: initial_array,
        }
    }

    pub fn remove_parent(&self, parent: usize) -> Result<()> {
        Ok(())
    }

    pub fn add_child(&self, parent: usize, child_index: usize, which_child: Pauli) -> Result<()> {
        let edge_index = match which_child {
            Pauli::I => Err(anyhow!("Pauli.I could not be used to add child.")),
            Pauli::X => Ok(2),
            Pauli::Y => Ok(3),
            Pauli::Z => Ok(4),
        };

        Ok(())
    }
}
