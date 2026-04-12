//! Ternary tree encodings and methods.
//!
//! The [`TernaryTree`] struct is made up of a set of vectors.
use crate::encode::encoding::{MajoranaEncoding, MajoranaEncodingError};
use crate::operators::Pauli;
use crate::operators::{SymplecticMatrix, SymplecticOperator};
use crate::states::ZBasisState;
use log::{debug, error};
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use std::collections::HashMap;
use std::fmt;
use std::ops::Not;
use std::result::Result;
use thiserror::Error;

/// Flattened structure of a [`TernaryTree`].
///
/// Beginning with the root node at index 0, each node's children
/// are given as a tuple (X,Y,Z).
pub type TTFlatPack = Vec<(usize, (Option<usize>, Option<usize>, Option<usize>))>;

/// Possible outward edges of nodes.
#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash)]
pub enum Edge {
    X,
    Y,
    Z,
}

impl Edge {
    /// Convert an edge to a char.
    fn as_char(&self) -> char {
        match &self {
            Edge::X => 'X',
            Edge::Y => 'Y',
            Edge::Z => 'Z',
        }
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.as_char())
    }
}

/// Parity of the total number of Y edges which must be taken to reach a given node.
///
/// When creating Majorana encodings, each fermionic operator is mapped to two majorana operators:
/// $f_i \to 0.5(\gamma_{2i} \pm i \gamma_{2i+1})$
///
/// To ensure that every term in the hamiltonian has a real coefficient,
/// when assigning indices to majorana operators, each pair should contain
/// one operator (2i) with an even number of Pauli-Y operators and the other (2i+1)
/// should contain an odd number of Pauli-Y operators.
///
/// We keep track of this in the [`TernaryTree`] by keeping track of
/// the Y-parity of a node, adding 1 for any child it has on the Y-Edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YParity {
    Odd,
    Even,
}

impl YParity {
    /// Used to define the offset of Majorana indices in [`TernaryTree::majorana_index`].
    /// The pair of majorana indices for fermionic mode "i" are:
    /// 2*i(+0) and 2*i+1.
    fn as_u8(&self) -> u8 {
        match self {
            Self::Even => 0,
            Self::Odd => 1,
        }
    }
}

/// Swaps between each [`YParity`].
///
/// # Example
/// ```
/// use ferrmion_core::encode::ternarytree::YParity;
///
/// let yp = YParity::Even;
/// assert_eq!(!yp, YParity::Odd);
/// ```
impl Not for YParity {
    type Output = Self;
    fn not(self) -> Self::Output {
        match self {
            YParity::Even => YParity::Odd,
            YParity::Odd => YParity::Even,
        }
    }
}

///A Parent node.
///
/// As parent_of is stored for eachof the N
/// nodes, a single node can be the parent_of in three
/// different ways.
/// Storing the edge with the parent means that we can
/// build pauli strings by traversing from the leaves to the root
/// without having to look at a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parent {
    /// The edge from parent to child.
    edge: Edge,
    /// The index of the parent node.
    index: u8,
}

impl Parent {
    /// Creates a new Parent.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::{Parent, Edge};
    /// let parent = Parent::new(Edge::X, 0);
    /// ```
    pub fn new(edge: Edge, index: u8) -> Self {
        Parent { edge, index }
    }

    /// Returns the node index of the parent.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::{Parent, Edge};
    /// let parent = Parent::new(Edge::X, 5);
    /// assert_eq!(parent.node_index(), 5);
    /// ```
    pub fn node_index(&self) -> usize {
        self.index as usize
    }
}

/// Possible children of a node.
///
/// A child can either be another node, with a node index,
/// or a leaf with an associated majorana operator index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Child {
    /// Another node with index.
    Node(u8),
    /// X leaf with majorana index.
    XLeaf(u8),
    /// Y leaf with majorana index.
    YLeaf(u8),
}

/// Returns the index of a Child as a usize for indexing into arrays.
impl Child {
    fn usize_index(&self) -> usize {
        match self {
            Child::Node(index) => *index as usize,
            Child::XLeaf(index) => *index as usize,
            Child::YLeaf(index) => *index as usize,
        }
    }
}

/// Represents a ternary tree structure for fermion-to-qubit encodings.
#[derive(Debug, PartialEq, Eq)]
pub struct TernaryTree {
    pub(crate) parent_of: Vec<Option<Parent>>,
    pub(crate) x_child_of: Vec<Option<Child>>,
    pub(crate) y_child_of: Vec<Option<Child>>,
    pub(crate) z_child_of: Vec<Option<Child>>,
    pub(crate) y_parity_of: Vec<YParity>,
    pub n_nodes: usize,
    qubit_index_of: Option<Vec<usize>>,
}

/// Errors that can occur when working with TernaryTree.
#[derive(Debug, Error)]
pub enum TernaryTreeError {
    #[error("Could not build Ternary Tree from Node Map: {0:?}")]
    FlatPackError(TTFlatPack),
    #[error("Child cannot be assigned parent.")]
    InvalidChildError(Parent, Child),
    #[error("Parent cannot be assigned child.")]
    InvalidParentError(Parent, Child),
    #[error("Node cannot be its own child/parent.")]
    SelfChildError(Parent, Child),
    #[error("Could not build symplectic from child of node {1} at {0}.")]
    LeafSymplecticError(Edge, usize),
    #[error("Could not build encoding for {0} qubits with Node:Qubit map {1:?}.")]
    BuildEncodingError(usize, Option<Vec<usize>>),
    #[error("Cannot reassign qubit indices of nodes.")]
    QubitReassignmentError,
    #[error("Encoding validation failed: {0}")]
    EncodingValidationError(#[from] MajoranaEncodingError),
}

// Constructors and input
impl TernaryTree {
    /// Creates a new empty TernaryTree with the specified number of nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// let tree = TernaryTree::new(5);
    /// ```
    pub fn new(n_nodes: usize) -> Self {
        Self {
            parent_of: vec![None; n_nodes],
            x_child_of: vec![None; n_nodes],
            y_child_of: vec![None; n_nodes],
            z_child_of: vec![None; n_nodes],
            y_parity_of: vec![YParity::Even; n_nodes],
            n_nodes,
            qubit_index_of: None,
        }
    }

    /// Creates a new naive TernaryTree with the specified number of nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// let tree = TernaryTree::new_naive(5);
    /// ```
    pub fn new_naive(n_nodes: usize) -> Self {
        Self {
            parent_of: vec![None; n_nodes],
            x_child_of: (0..n_nodes).map(|v| Some(Child::XLeaf(v as u8))).collect(),
            y_child_of: (0..n_nodes).map(|v| Some(Child::YLeaf(v as u8))).collect(),
            z_child_of: vec![None; n_nodes],
            y_parity_of: vec![YParity::Even; n_nodes],
            n_nodes,
            qubit_index_of: None,
        }
    }

    /// Builds a TernaryTree from a flatpack representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::{TernaryTree, TTFlatPack};
    /// let flatpack: TTFlatPack = vec![];
    /// let tree = TernaryTree::from_flatpack(&flatpack).unwrap();
    /// ```
    pub fn from_flatpack(flatpack: &TTFlatPack) -> Result<TernaryTree, TernaryTreeError> {
        let n_nodes = flatpack.len();
        let mut tree = TernaryTree::new(n_nodes);
        tree.add_children_from_flatpack(flatpack, true)?;
        Ok(tree)
    }

    /// Builds a naive enumeration TernaryTree from a flatpack representation.
    ///
    /// Leaf child values in the flatpack are ignored; the naive default leaf
    /// assignment (`XLeaf(i)` / `YLeaf(i)` for each node `i`) is applied as usual.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::{TernaryTree, TTFlatPack};
    /// let flatpack: TTFlatPack = vec![];
    /// let tree = TernaryTree::from_flatpack_naive(&flatpack).unwrap();
    /// ```
    pub fn from_flatpack_naive(flatpack: &TTFlatPack) -> Result<TernaryTree, TernaryTreeError> {
        let n_nodes = flatpack.len();
        let mut tree = TernaryTree::new_naive(n_nodes);
        tree.add_children_from_flatpack(flatpack, false)?;
        Ok(tree)
    }

    /// `process_leaves`: when `true`, child values not found in `qubit_node_map` are
    /// decoded as leaves (`child_value - max_node_index` gives the Majorana index).
    /// When `false` (naive mode), such values are silently ignored and the existing
    /// default leaf assignment is left intact.
    fn add_children_from_flatpack(
        &mut self,
        flatpack: &TTFlatPack,
        process_leaves: bool,
    ) -> Result<(), TernaryTreeError> {
        let n_nodes = self.n_nodes;
        let qubit_index_of: Vec<usize> = flatpack.iter().map(|v| v.0).collect();
        self.set_qubit_indices(qubit_index_of)?;

        let mut qubit_node_map: HashMap<usize, usize> = HashMap::with_capacity(n_nodes);
        flatpack
            .iter()
            .zip(0..n_nodes)
            .for_each(|(flattened_node, node)| {
                let qubit_index = flattened_node.0;
                qubit_node_map.insert(qubit_index, node);
            });

        debug!("Flatpack nodes have qubit indices {:?}", &qubit_node_map);

        // Leaf values are encoded as Majorana_index + max_node_index.
        // Node qubit indices may be non-contiguous (e.g. from the bonsai algorithm).
        let max_node_index = qubit_node_map.keys().copied().max().unwrap_or(0);

        for (qubit_index, children) in flatpack.iter() {
            let parent = *qubit_node_map
                .get(qubit_index)
                .ok_or_else(|| TernaryTreeError::FlatPackError(flatpack.clone()))?
                as u8;
            for (child, edge) in std::iter::zip(
                [children.0, children.1, children.2],
                [Edge::X, Edge::Y, Edge::Z],
            ) {
                if let Some(c) = child {
                    if let Some(&node_idx) = qubit_node_map.get(&c) {
                        // c is a known node qubit index — add as a Node child.
                        self.add_child(Parent::new(edge, parent), Child::Node(node_idx as u8))?;
                    } else if process_leaves {
                        // c is not a node — decode as a leaf.
                        // Majorana index m = c - max_node_index.
                        // Even m → XLeaf(m/2), odd m → YLeaf((m-1)/2).
                        let m = c.checked_sub(max_node_index)
                            .ok_or_else(|| TernaryTreeError::FlatPackError(flatpack.clone()))?;
                        let leaf = if m % 2 == 0 {
                            Child::XLeaf((m / 2) as u8)
                        } else {
                            Child::YLeaf(((m - 1) / 2) as u8)
                        };
                        match edge {
                            Edge::X => self.x_child_of[parent as usize] = Some(leaf),
                            Edge::Y => self.y_child_of[parent as usize] = Some(leaf),
                            Edge::Z => self.z_child_of[parent as usize] = Some(leaf),
                        }
                    }
                    // else: not process_leaves — unknown index silently ignored (naive mode)
                }
            }
        }
        Ok(())
    }

    fn set_qubit_indices(&mut self, qubit_indices: Vec<usize>) -> Result<(), TernaryTreeError> {
        match self.qubit_index_of {
            Some(_) => {
                error!("Qubit indices are already set.");
                return Err(TernaryTreeError::QubitReassignmentError);
            }
            None => {
                debug!("Setting qubit indices {:?}", qubit_indices);
                self.qubit_index_of = Some(qubit_indices);
            }
        }
        Ok(())
    }
}

// Standard Encodings
impl TernaryTree {
    /// Creates a naive Jordan-Wigner TernaryTree.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// let tree = TernaryTree::naive_jordan_wigner(4);
    /// ```
    pub fn naive_jordan_wigner(n_nodes: usize) -> TernaryTree {
        let mut tree = TernaryTree::new_naive(n_nodes);
        let branch: Vec<(Edge, usize)> = (0..n_nodes - 1).map(|v| (Edge::Z, v + 1)).collect();
        debug!("{:?}", branch);
        tree.add_branch(0, branch)
            .expect("Naive JW branch should be valid.");
        debug!("{:?}", tree);
        tree
    }

    /// Creates a naive Parity TernaryTree.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// let tree = TernaryTree::naive_parity(4);
    /// ```
    pub fn naive_parity(n_nodes: usize) -> TernaryTree {
        let mut tree = TernaryTree::new_naive(n_nodes);
        debug!("{:?}", tree);
        let branch: Vec<(Edge, usize)> = (0..n_nodes - 1).map(|v| (Edge::X, v + 1)).collect();
        tree.add_branch(0, branch)
            .expect("Naive Parity branch should be valid.");
        tree
    }

    /// Creates a naive Bravyi-Kitaev TernaryTree.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// let tree = TernaryTree::naive_bravyi_kitaev(4);
    /// ```
    pub fn naive_bravyi_kitaev(n_nodes: usize) -> TernaryTree {
        let mut tree = TernaryTree::new_naive(n_nodes);
        if n_nodes >= 2 {
            tree.add_child(Parent::new(Edge::X, 0), Child::Node(1))
                .expect("BK children should be valid.");
        }
        let n_nodes = n_nodes as u8;
        for ind in 2..n_nodes {
            match ind % 2 == 0 {
                true => tree
                    .add_child(Parent::new(Edge::X, ind / 2), Child::Node(ind))
                    .expect("BK children should be valid."),
                false => tree
                    .add_child(Parent::new(Edge::Z, (ind - 1) / 2), Child::Node(ind))
                    .expect("BK children should be valid."),
            };
        }
        tree
    }

    /// Creates a naive JKMN TernaryTree.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// let tree = TernaryTree::naive_jkmn(4);
    /// ```
    pub fn naive_jkmn(n_nodes: usize) -> TernaryTree {
        let mut tree = TernaryTree::new_naive(n_nodes);

        let mut parent = 0_u8;
        let mut edges = [Edge::X, Edge::Y, Edge::Z].into_iter().cycle();
        for ind in 1..n_nodes {
            if let Some(e) = edges.next() {
                debug!("{:?}", e);
                tree.add_child(Parent::new(e, parent), Child::Node(ind as u8))
                    .expect("Naive JKMN children should be valid.");
                if matches!(e, Edge::Z) {
                    parent += 1;
                };
            }
        }
        debug!("{:?}", tree);
        tree
    }
}

// Output
impl TernaryTree {
    /// Builds a MajoranaEncoding from the TernaryTree.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// let tree = TernaryTree::naive_jordan_wigner(4);
    /// let encoding = tree.build_encoding(4).unwrap();
    /// ```
    pub fn build_encoding(&self, n_qubits: usize) -> Result<MajoranaEncoding, TernaryTreeError> {
        debug!("Build encoding from {self:?}");
        if n_qubits < self.n_nodes {
            return Err(TernaryTreeError::BuildEncodingError(
                n_qubits,
                self.qubit_index_of.clone(),
            ));
        }
        let vacuum_state_fock: Array1<bool> = Array1::from_elem(self.n_nodes, false);
        let mut vacuum_state: ZBasisState = ZBasisState::new(vacuum_state_fock, Complex64::ONE);

        let mut x_block: Array2<bool> = Array2::from_elem((2 * self.n_nodes, self.n_nodes), false);
        let mut z_block: Array2<bool> = Array2::from_elem((2 * self.n_nodes, self.n_nodes), false);
        for final_edge in [Edge::X, Edge::Y, Edge::Z] {
            debug!("\nFinal Edge {:?}", final_edge);
            let child_of = match final_edge {
                Edge::X => &self.x_child_of,
                Edge::Y => &self.y_child_of,
                Edge::Z => &self.z_child_of,
            };
            let leaf_locations: Vec<usize> = child_of
                .iter()
                .enumerate()
                .filter(|(_, v)| matches!(v, Some(Child::XLeaf(_)) | Some(Child::YLeaf(_))))
                .map(|(ind, _)| ind)
                .collect();
            debug!("Leaf locations on edge {:?}", leaf_locations);
            leaf_locations.iter().for_each(|&ind| {
                debug!("ind {:?}", ind);
                debug!("final_edge {:?}", final_edge);
                let (row, op) = self
                    .symplectic_from_leaf(&final_edge, ind)
                    .expect("Leaf locations should have been validated.");
                debug!("leaf_result row={:?}", row);
                x_block
                    .slice_mut(s![row as usize, ..])
                    .assign(&op.x_block());
                z_block
                    .slice_mut(s![row as usize, ..])
                    .assign(&op.z_block());
            });
            debug!("x_block {:?}", x_block);
        }
        if let Some(index) = &self.qubit_index_of {
            debug!("Qubit indices {:?}", &self.qubit_index_of);
            if let Some(&max_qi) = index.iter().max() {
                if max_qi >= n_qubits {
                    error!("Cannot build encoding with {n_qubits} qubits");
                    return Err(TernaryTreeError::BuildEncodingError(
                        n_qubits,
                        self.qubit_index_of.clone(),
                    ));
                }
            }
            let mut padded_x: Array2<bool> = Array2::from_elem((2 * self.n_nodes, n_qubits), false);
            let mut padded_z: Array2<bool> = Array2::from_elem((2 * self.n_nodes, n_qubits), false);
            let mut padded_vacuum_state: ZBasisState = ZBasisState::zeros(n_qubits);
            for (col_idx, &qi) in index.iter().enumerate() {
                padded_x.column_mut(qi).assign(&x_block.column(col_idx));
                padded_z.column_mut(qi).assign(&z_block.column(col_idx));
                padded_vacuum_state.state.slice_mut(s![qi]).fill(
                    *vacuum_state
                        .state
                        .get(col_idx)
                        .expect("Vacuum state should have same dimension as encoding."),
                );
            }
            debug!("Padded x_block {:?}", padded_x);
            x_block = padded_x;
            z_block = padded_z;
            vacuum_state = padded_vacuum_state;
        }
        Ok(MajoranaEncoding::with_vacuum(
            SymplecticMatrix::new(x_block, z_block),
            vacuum_state,
        )?)
    }
}

impl TernaryTree {
    #[allow(dead_code)]
    fn real_eigenvalue_majorana_index(&self, child: Child) -> u8 {
        match child {
            Child::XLeaf(ind) => 2 * ind + self.y_parity_of[child.usize_index()].as_u8(),
            Child::YLeaf(ind) => 2 * ind + (!self.y_parity_of[child.usize_index()]).as_u8(),
            Child::Node(ind) => 2 * self.n_nodes as u8 + 1 + ind,
        }
    }

    fn vacuum_preserving_majorana_index(&self, child: Child) -> u8 {
        match child {
            Child::XLeaf(ind) => 2 * ind,
            Child::YLeaf(ind) => 2 * ind + 1,
            Child::Node(ind) => 2 * self.n_nodes as u8 + 1 + ind,
        }
    }

    fn get_z_descendant_of(&self, node_index: usize) -> usize {
        let mut index = node_index;
        while let Some(Child::Node(z_child)) = self.z_child_of[index] {
            index = z_child as usize;
        }
        index
    }

    // Add child
    // 1. check if there is already a child
    // 1, yes child =>
    // // 2. Attach that child to the z_descendant of the
    //
    //
    //
    // 1, no child =>
    // // set parent_of[child] = parent
    // // set child_of[parent] = child

    fn add_child(&mut self, new_parent: Parent, new_child: Child) -> Result<(), TernaryTreeError> {
        if (new_parent.node_index() == new_child.usize_index())
            & matches!(new_child, Child::Node(_))
        {
            return Err(TernaryTreeError::SelfChildError(new_parent, new_child));
        }

        let current_child: Option<Child> = match new_parent.edge {
            Edge::X => self.x_child_of[new_parent.node_index()],
            Edge::Y => self.y_child_of[new_parent.node_index()],
            Edge::Z => self.z_child_of[new_parent.node_index()],
        };

        if let Some(existing_child) = current_child {
            match existing_child {
                // If parent has a child node it cannot accept a new child node
                Child::Node(_) => {
                    return Err(TernaryTreeError::InvalidChildError(new_parent, new_child));
                }
                // If parent has a leaf we give it to the z_ancestor of the child.
                Child::XLeaf(_) | Child::YLeaf(_) => {
                    let z_anc = self.get_z_descendant_of(new_child.usize_index());
                    self.z_child_of[z_anc] = Some(existing_child);
                }
            }
            // return Err(TernaryTreeError::AddChildError(new_parent, new_child));
        }

        if matches!(new_child, Child::Node(_)) {
            let current_parent = self.parent_of[new_child.usize_index()];

            if current_parent.is_some() {
                return Err(TernaryTreeError::InvalidParentError(new_parent, new_child));
            }
        }

        match new_parent.edge {
            Edge::X => {
                self.x_child_of[new_parent.index as usize] = Some(new_child);
            }
            Edge::Y => {
                self.y_child_of[new_parent.index as usize] = Some(new_child);
            }
            Edge::Z => {
                self.z_child_of[new_parent.index as usize] = Some(new_child);
            }
        }

        // Update the Parent and Yparity of the child.
        if matches!(new_child, Child::Node(_)) {
            self.parent_of[new_child.usize_index()] = Some(new_parent);
            self.y_parity_of[new_child.usize_index()] = self.y_parity_of[new_parent.node_index()];

            if matches!(new_parent.edge, Edge::Y) {
                debug!("Swapping parity of child.");
                self.y_parity_of[new_child.usize_index()] =
                    !self.y_parity_of[new_parent.node_index()];
                debug!("{:?}", self.y_parity_of);
            }
        }
        Ok(())
    }

    fn add_branch(
        &mut self,
        root_node: usize,
        branch: Vec<(Edge, usize)>,
    ) -> Result<(), TernaryTreeError> {
        let mut parent_ind = root_node;
        for (edge, child_ind) in branch {
            if child_ind >= self.n_nodes {
                debug!("Ignoring out of bounds index in add_branch");
            } else {
                self.add_child(
                    Parent::new(edge, parent_ind as u8),
                    Child::Node(child_ind as u8),
                )?;
                parent_ind = child_ind;
            }
        }
        Ok(())
    }

    fn symplectic_from_leaf(
        &self,
        leaf_edge: &Edge,
        parent_index: usize,
    ) -> Result<(u8, SymplecticOperator), TernaryTreeError> {
        let child_of = match leaf_edge {
            Edge::X => &self.x_child_of,
            Edge::Y => &self.y_child_of,
            Edge::Z => &self.z_child_of,
        };
        let mut ipower: u8 = 0;
        let mut x_array: Array1<bool> = Array1::from_elem(self.n_nodes, false);
        let mut z_array: Array1<bool> = Array1::from_elem(self.n_nodes, false);

        let majorana_index: u8;
        debug!("Parent index {parent_index}");
        if let Some(child) = child_of[parent_index] {
            debug!("Child {child:?}");
            match child {
                Child::XLeaf(_) | Child::YLeaf(_) => {
                    majorana_index = self.vacuum_preserving_majorana_index(child);
                }
                Child::Node(_) => {
                    return Err(TernaryTreeError::LeafSymplecticError(
                        *leaf_edge,
                        parent_index,
                    ))
                }
            }
        } else {
            return Err(TernaryTreeError::LeafSymplecticError(
                *leaf_edge,
                parent_index,
            ));
        }
        debug!("Majorana Index - {:?}", majorana_index);

        if matches!(leaf_edge, Edge::Y) {
            ipower += 1
        };

        let bool_term: (bool, bool) = Pauli::from(leaf_edge).into();
        x_array[[parent_index]] = bool_term.0;
        z_array[[parent_index]] = bool_term.1;

        // let parent = self.parent_of[]
        debug!("Parent {:?}", parent_index);
        debug!("parent_of {:?}", self.parent_of);

        let mut parent_index = parent_index;
        while let Some(parent) = self.parent_of[parent_index] {
            parent_index = parent.node_index();
            debug!("Parent index {parent_index}");

            if matches!(parent.edge, Edge::Y) {
                ipower += 1;
            }
            let bool_term: (bool, bool) = Pauli::from(&parent.edge).into();

            debug!("XZ Operator {bool_term:?}");
            x_array[[parent_index]] = bool_term.0;
            z_array[[parent_index]] = bool_term.1;
        }
        debug!("Majorana index {:?}", majorana_index);
        debug!("ipower {:?}", ipower);
        Ok((
            majorana_index,
            SymplecticOperator::new(ipower, x_array, z_array),
        ))
    }
}

#[cfg(test)]
mod tt_tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use Child::{Node, XLeaf, YLeaf};
    use Edge::{X, Y, Z};

    #[test]
    fn test_new() {
        let tt = TernaryTree::new(3);
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(tt.x_child_of, vec![None, None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, None]);
    }

    #[test]
    fn test_new_naive() {
        let tt = TernaryTree::new_naive(3);
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(
            tt.x_child_of,
            vec![Some(XLeaf(0)), Some(XLeaf(1)), Some(XLeaf(2))]
        );
        assert_eq!(
            tt.y_child_of,
            vec![Some(YLeaf(0)), Some(YLeaf(1)), Some(YLeaf(2))]
        );
        assert_eq!(tt.z_child_of, vec![None, None, None]);
    }

    #[test]
    fn test_from_empty_flatpack() {
        let flatpack: TTFlatPack = vec![
            (0, (None, None, None)),
            (1, (None, None, None)),
            (2, (None, None, None)),
        ];
        let tt = TernaryTree::from_flatpack(&flatpack).unwrap();
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(tt.x_child_of, vec![None, None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, None]);
    }

    #[test]
    fn test_from_empty_flatpack_naive() {
        let flatpack: TTFlatPack = vec![
            (0, (None, None, None)),
            (1, (None, None, None)),
            (2, (None, None, None)),
        ];
        let tt = TernaryTree::from_flatpack_naive(&flatpack).unwrap();
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(
            tt.x_child_of,
            vec![Some(XLeaf(0)), Some(XLeaf(1)), Some(XLeaf(2))]
        );
        assert_eq!(
            tt.y_child_of,
            vec![Some(YLeaf(0)), Some(YLeaf(1)), Some(YLeaf(2))]
        );
        assert_eq!(tt.z_child_of, vec![None, None, None]);
    }

    #[test]
    fn test_from_flatpack_naive_standard_encodings() {
        let jw_flatpack: TTFlatPack = vec![
            (0, (None, None, Some(1))),
            (1, (None, None, Some(2))),
            (2, (None, None, None)),
        ];
        let mut expected: TernaryTree = TernaryTree::naive_jordan_wigner(3);
        expected.set_qubit_indices(vec![0, 1, 2]).unwrap();
        assert_eq!(
            TernaryTree::from_flatpack_naive(&jw_flatpack).unwrap(),
            expected
        );
        let pe_flatpack: TTFlatPack = vec![
            (0, (Some(1), None, None)),
            (1, (Some(2), None, None)),
            (2, (None, None, None)),
        ];
        let mut expected: TernaryTree = TernaryTree::naive_parity(3);
        expected.set_qubit_indices(vec![0, 1, 2]).unwrap();
        assert_eq!(
            TernaryTree::from_flatpack_naive(&pe_flatpack).unwrap(),
            expected
        );
        let bk_flatpack: TTFlatPack = vec![
            (0, (Some(1), None, None)),
            (1, (Some(2), None, Some(3))),
            (2, (None, None, None)),
            (3, (None, None, None)),
        ];
        let mut expected: TernaryTree = TernaryTree::naive_bravyi_kitaev(4);
        expected.set_qubit_indices(vec![0, 1, 2, 3]).unwrap();
        assert_eq!(
            TernaryTree::from_flatpack_naive(&bk_flatpack).unwrap(),
            expected
        );
        let jkmn_flatpack: TTFlatPack = vec![
            (0, (Some(1), Some(2), Some(3))),
            (1, (None, None, None)),
            (2, (None, None, None)),
            (3, (None, None, None)),
        ];
        let mut expected: TernaryTree = TernaryTree::naive_jkmn(4);
        expected.set_qubit_indices(vec![0, 1, 2, 3]).unwrap();
        assert_eq!(
            TernaryTree::from_flatpack_naive(&jkmn_flatpack).unwrap(),
            expected
        );
    }

    #[test]
    fn test_from_flatpack_with_qubit_labels() {
        let flatpack: TTFlatPack = vec![
            (9, (Some(10), Some(11), Some(12))),
            (10, (None, None, None)),
            (11, (None, None, None)),
            (12, (None, None, None)),
        ];
        let tree = TernaryTree::from_flatpack_naive(&flatpack).unwrap();

        let mut expected: TernaryTree = TernaryTree::naive_jkmn(4);
        expected.set_qubit_indices(vec![9, 10, 11, 12]).unwrap();

        assert_eq!(tree, expected);
    }

    #[test]
    fn test_add_child() {
        let mut tt = TernaryTree::new(3);
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(tt.x_child_of, vec![None, None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, None]);

        //Add Child
        tt.add_child(Parent::new(Z, 0), Node(1)).unwrap();
        assert_eq!(tt.parent_of, vec![None, Some(Parent::new(Z, 0)), None]);
        assert_eq!(tt.x_child_of, vec![None, None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![Some(Node(1)), None, None]);
    }
    #[test]
    fn test_add_leaf() {
        let mut tt = TernaryTree::new(3);
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(tt.x_child_of, vec![None, None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, None]);

        //Add leaf
        tt.add_child(Parent::new(X, 0), XLeaf(0)).unwrap();
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(tt.x_child_of, vec![Some(XLeaf(0)), None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, None]);
    }
    #[test]
    fn test_replace_leaf_with_child() {
        let mut tt = TernaryTree::new(3);
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(tt.x_child_of, vec![None, None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, None]);

        //Add Leaf
        tt.add_child(Parent::new(X, 0), XLeaf(0)).unwrap();
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(tt.x_child_of, vec![Some(XLeaf(0)), None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, None]);

        // Replace leaf with child
        tt.add_child(Parent::new(X, 0), Node(2)).unwrap();
        assert_eq!(tt.parent_of, vec![None, None, Some(Parent::new(X, 0))]);
        assert_eq!(tt.x_child_of, vec![Some(Node(2)), None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, Some(XLeaf(0))]);
    }

    #[test]
    fn test_naive_add_child_z() {
        let mut tt = TernaryTree::new_naive(3);
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(
            tt.x_child_of,
            vec![Some(XLeaf(0)), Some(XLeaf(1)), Some(XLeaf(2))]
        );
        assert_eq!(
            tt.y_child_of,
            vec![Some(YLeaf(0)), Some(YLeaf(1)), Some(YLeaf(2))]
        );
        assert_eq!(tt.z_child_of, vec![None, None, None]);

        //Add Child
        tt.add_child(Parent::new(Z, 0), Node(1)).unwrap();
        assert_eq!(tt.parent_of, vec![None, Some(Parent::new(Z, 0)), None]);
        assert_eq!(
            tt.x_child_of,
            vec![Some(XLeaf(0)), Some(XLeaf(1)), Some(XLeaf(2))]
        );
        assert_eq!(
            tt.y_child_of,
            vec![Some(YLeaf(0)), Some(YLeaf(1)), Some(YLeaf(2))]
        );
        assert_eq!(tt.z_child_of, vec![Some(Node(1)), None, None]);
    }
    #[test]
    fn test_move_leaf_to_grandchild() {
        let mut tt = TernaryTree::new(3);
        tt.add_child(Parent::new(X, 0), XLeaf(0)).unwrap();
        assert_eq!(tt.parent_of, vec![None, None, None]);
        assert_eq!(tt.x_child_of, vec![Some(XLeaf(0)), None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, None, None]);

        //Add grandchild to child
        tt.add_child(Parent::new(Z, 1), Node(2)).unwrap();
        assert_eq!(tt.parent_of, vec![None, None, Some(Parent::new(Z, 1))]);
        assert_eq!(tt.x_child_of, vec![Some(XLeaf(0)), None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, Some(Node(2)), None]);

        //Add child to parent
        tt.add_child(Parent::new(X, 0), Node(1)).unwrap();
        assert_eq!(
            tt.parent_of,
            vec![None, Some(Parent::new(X, 0)), Some(Parent::new(Z, 1))]
        );
        assert_eq!(tt.x_child_of, vec![Some(Node(1)), None, None]);
        assert_eq!(tt.y_child_of, vec![None, None, None]);
        assert_eq!(tt.z_child_of, vec![None, Some(Node(2)), Some(XLeaf(0))]);
    }

    // How do we enforce that nodes are added in such a way as to not
    // require re-assigning YParity if a complex tree is added as a child
    // of a node on a Y-Branch?
    //
    // For now the plan is to ignore it and assume that the ony way to make trees
    // is to add them ordered by generation.
    //
    // 1. Do not allow any node with a child to be assgined as a child.
    //  -
    // 2. Recalculate each time we add on a child.
    //  - Possibly Expensive, imagine adding JKMN(100) to a siingle node on the Y branch
    // 3. Calculate once at the end of tree build
    //  - requires another step for manual trees
    //
    // #[test]
    // fn test_y_parity_update() -> Result<(), TernaryTreeError> {
    //     let mut tt = TernaryTree::new(3);
    //     tt.add_child(Parent::new(Y, 1), Child::Node(2))?;
    //     tt.add_child(Parent::new(Y, 0), Child::Node(1))?;
    //     assert_eq!(
    //         tt.y_parity_of,
    //         vec![YParity::Even, YParity::Odd, YParity::Even]
    //     );
    //     Ok(())
    // }

    #[test]
    fn test_symplectic_from_leaf() -> Result<(), TernaryTreeError> {
        let mut tt = TernaryTree::new(3);
        tt.add_child(Parent::new(X, 0), Child::XLeaf(0))?;
        tt.add_child(Parent::new(Y, 0), Child::YLeaf(0))?;
        tt.add_child(Parent::new(Z, 0), Child::Node(1))?;
        debug!("{:?}", tt);

        tt.add_child(Parent::new(X, 1), Child::XLeaf(1))?;
        tt.add_child(Parent::new(Y, 1), Child::YLeaf(1))?;
        tt.add_child(Parent::new(Z, 1), Child::Node(2))?;

        tt.add_child(Parent::new(X, 2), Child::XLeaf(2))?;
        tt.add_child(Parent::new(Y, 2), Child::YLeaf(2))?;

        assert_eq!(
            tt.parent_of,
            vec![None, Some(Parent::new(Z, 0)), Some(Parent::new(Z, 1))]
        );
        assert_eq!(
            tt.x_child_of,
            vec![Some(XLeaf(0)), Some(XLeaf(1)), Some(XLeaf(2))]
        );

        let xz_result = tt.symplectic_from_leaf(&Edge::X, 0).unwrap();
        let expected = (
            0,
            SymplecticOperator::new(0, arr1(&[true, false, false]), arr1(&[false, false, false])),
        );
        assert_eq!(xz_result, expected);

        let xz_result = tt.symplectic_from_leaf(&Edge::Y, 2).unwrap();
        let expected = (
            5,
            SymplecticOperator::new(1, arr1(&[false, false, true]), arr1(&[true, true, true])),
        );
        assert_eq!(xz_result, expected);
        Ok(())
    }

    #[test]
    fn test_jw_manual_build_encoding() {
        let mut tt = TernaryTree::new(3);
        tt.add_child(Parent::new(X, 0), Child::XLeaf(0)).unwrap();
        tt.add_child(Parent::new(Y, 0), Child::YLeaf(0)).unwrap();
        tt.add_child(Parent::new(Z, 0), Child::Node(1)).unwrap();

        tt.add_child(Parent::new(X, 1), Child::XLeaf(1)).unwrap();
        tt.add_child(Parent::new(Y, 1), Child::YLeaf(1)).unwrap();
        tt.add_child(Parent::new(Z, 1), Child::Node(2)).unwrap();

        tt.add_child(Parent::new(X, 2), Child::XLeaf(2)).unwrap();
        tt.add_child(Parent::new(Y, 2), Child::YLeaf(2)).unwrap();

        let n_qubits = tt.n_nodes;
        let encoding = tt.build_encoding(n_qubits).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 0, 1]);
        assert_eq!(encoding.operators.ipowers, ipow_expected);
        let x_expected = arr2(&[
            [true, false, false],
            [true, false, false],
            [false, true, false],
            [false, true, false],
            [false, false, true],
            [false, false, true],
        ]);
        assert_eq!(encoding.operators.x_block, x_expected);
        let z_expected = arr2(&[
            [false, false, false],
            [true, false, false],
            [true, false, false],
            [true, true, false],
            [true, true, false],
            [true, true, true],
        ]);
        assert_eq!(encoding.operators.z_block, z_expected);
    }

    #[test]
    fn test_jw_flatpack_build_encoding() {
        let flatpack: TTFlatPack = Vec::from(&[
            (0, (None, None, Some(1))),
            (1, (None, None, Some(2))),
            (2, (None, None, None)),
        ]);
        let tt = TernaryTree::from_flatpack_naive(&flatpack).unwrap();
        assert_eq!(tt.qubit_index_of, Some(vec![0, 1, 2]));
        let n_qubits = tt.n_nodes;
        let encoding = tt.build_encoding(n_qubits).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 0, 1]);
        assert_eq!(encoding.operators.ipowers, ipow_expected);
        let x_expected = arr2(&[
            [true, false, false],
            [true, false, false],
            [false, true, false],
            [false, true, false],
            [false, false, true],
            [false, false, true],
        ]);
        assert_eq!(encoding.operators.x_block, x_expected);
        let z_expected = arr2(&[
            [false, false, false],
            [true, false, false],
            [true, false, false],
            [true, true, false],
            [true, true, false],
            [true, true, true],
        ]);
        assert_eq!(encoding.operators.z_block, z_expected);
    }

    #[test]
    fn test_naive_jw_encoding() {
        let tree = TernaryTree::naive_jordan_wigner(3);
        let encoding = tree.build_encoding(3).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 0, 1]);
        assert_eq!(encoding.operators.ipowers, ipow_expected);
        let x_expected = arr2(&[
            [true, false, false],
            [true, false, false],
            [false, true, false],
            [false, true, false],
            [false, false, true],
            [false, false, true],
        ]);
        assert_eq!(encoding.operators.x_block, x_expected);
        let z_expected = arr2(&[
            [false, false, false],
            [true, false, false],
            [true, false, false],
            [true, true, false],
            [true, true, false],
            [true, true, true],
        ]);
        assert_eq!(encoding.operators.z_block, z_expected);
    }

    #[test]
    fn test_naive_parity_encoding() {
        let tree = TernaryTree::naive_parity(3);
        let encoding = tree.build_encoding(3).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 0, 1]);
        assert_eq!(encoding.operators.ipowers, ipow_expected);
        let symplectic_expected = arr2(&[
            [true, false, false, false, true, false],
            [true, false, false, true, false, false],
            [true, true, false, false, false, true],
            [true, true, false, false, true, false],
            [true, true, true, false, false, false],
            [true, true, true, false, false, true],
        ]);
        let combined = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                encoding.operators.x_block.view(),
                encoding.operators.z_block.view(),
            ],
        )
        .unwrap();
        assert_eq!(combined, symplectic_expected);
    }

    #[test]
    fn test_naive_jkmn_encoding() {
        let tree = TernaryTree::naive_jkmn(3);
        let encoding = tree.build_encoding(3).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 1, 2]);
        assert_eq!(encoding.operators.ipowers, ipow_expected);
        let symplectic_expected = arr2(&[
            [true, false, false, false, true, false],
            [true, false, false, true, false, true],
            [true, true, false, false, false, false],
            [true, true, false, false, true, false],
            [true, false, true, true, false, false],
            [true, false, true, true, false, true],
        ]);
        let combined = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                encoding.operators.x_block.view(),
                encoding.operators.z_block.view(),
            ],
        )
        .unwrap();
        assert_eq!(combined, symplectic_expected);
    }

    #[test]
    fn test_add_branch() {
        let mut branch_tt = TernaryTree::new(3);
        branch_tt
            .add_branch(0, vec![(Edge::Z, 1), (Edge::Z, 2)])
            .unwrap();

        assert_eq!(
            branch_tt.z_child_of,
            vec![Some(Node(1)), Some(Node(2)), None]
        );
        assert_eq!(branch_tt.x_child_of.iter().flatten().count(), 0);
        assert_eq!(branch_tt.y_child_of.iter().flatten().count(), 0);

        assert!(branch_tt.add_branch(1, vec![(Edge::X, 2)]).is_err());
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_new_naive_properties(n in 1..10usize) {
            let tt = TernaryTree::new_naive(n);
            prop_assert_eq!(tt.parent_of.len(), n);
            prop_assert_eq!(tt.x_child_of.len(), n);
            prop_assert_eq!(tt.y_child_of.len(), n);
            prop_assert_eq!(tt.z_child_of.len(), n);
            // Check that x_child_of has XLeaf(i) for i in 0..n
            for i in 0..n {
                prop_assert_eq!(tt.x_child_of[i], Some(XLeaf(i as u8)));
                prop_assert_eq!(tt.y_child_of[i], Some(YLeaf(i as u8)));
                prop_assert_eq!(tt.z_child_of[i], None);
            }
        }

        #[test]
        fn test_jw_encodings_valid(n in 5usize..50) {
            let tt = TernaryTree::naive_jordan_wigner(n);
            prop_assert!(tt.build_encoding(n).is_ok());
        }

        #[test]
        fn test_parity_encodings_valid(n in 5usize..50) {
            let tt = TernaryTree::naive_parity(n);
            prop_assert!(tt.build_encoding(n).is_ok());
        }

        #[test]
        fn test_bk_encodings_valid(n in 5usize..50) {
            let tt = TernaryTree::naive_bravyi_kitaev(n);
            prop_assert!(tt.build_encoding(n).is_ok());
        }

        #[test]
        fn test_jkmn_encodings_valid(n in 5usize..50) {
            let tt = TernaryTree::naive_jkmn(n);
            debug!("{tt:#?}");
            prop_assert!(tt.build_encoding(n).is_ok());
        }

    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::encode::encoding::Encode;
    use crate::hamiltonians::QubitHamiltonian;
    use crate::operators::{FermionMatrix, FermionSparse, LadderOperator, MajoranaSparse};
    use ahash::HashMapExt;
    use ndarray::arr2;
    use num_complex::c64;

    #[test]
    fn test_encode_identity_with_jw() {
        let encoding = TernaryTree::naive_jordan_wigner(2)
            .build_encoding(2)
            .unwrap();
        let coeffs = arr2(&[[1f64, 0f64], [0f64, 1f64]]).into_dyn();
        let fmat = FermionMatrix::new(
            vec![LadderOperator::Creation, LadderOperator::Annihilation],
            coeffs,
        )
        .unwrap();
        let mut expected = QubitHamiltonian::new();
        expected.insert("IZ".to_string(), c64(-0.5, 0.));
        expected.insert("ZI".to_string(), c64(-0.5, 0.));
        expected.insert("II".to_string(), c64(1., 0.));
        let qham = encoding.encode(&MajoranaSparse::from(FermionSparse::from(fmat)));
        assert_eq!(expected, qham)
    }

    #[test]
    fn test_encode_off_diag_with_jw() {
        let encoding = TernaryTree::naive_jordan_wigner(2)
            .build_encoding(2)
            .unwrap();
        let coeffs = arr2(&[[0f64, 0f64], [1f64, 0f64]]).into_dyn();
        let fmat = FermionMatrix::new(
            vec![LadderOperator::Creation, LadderOperator::Annihilation],
            coeffs,
        )
        .unwrap();
        let mut expected = QubitHamiltonian::new();
        expected.insert("XY".to_string(), c64(0., -0.25));
        expected.insert("YX".to_string(), c64(0., 0.25));
        expected.insert("XX".to_string(), c64(0.25, 0.));
        expected.insert("YY".to_string(), c64(0.25, 0.));
        let qham = encoding.encode(&MajoranaSparse::from(FermionSparse::from(fmat)));
        assert_eq!(expected, qham);
    }
}
