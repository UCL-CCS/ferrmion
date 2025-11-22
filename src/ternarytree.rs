/*
Ternary tree encodings and methods.
*/
use anyhow::Result;
type NodeIndexArray = [u8; 256];
const MAX_SIZE: usize = 85;
const MAJORANA_MAX: usize = 4;

pub struct FastTernaryTree {
    pub parent_of: NodeIndexArray,
    pub x_child_of: NodeIndexArray,
    pub y_child_of: NodeIndexArray,
    pub z_child_of: NodeIndexArray,
    pub z_ancestor_of: NodeIndexArray,
    pub z_descendant_of: NodeIndexArray,
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

    pub fn add_child(
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
}
