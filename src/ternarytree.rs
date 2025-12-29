//! Ternary tree encodings and methods.
//!
//! The ['TernaryTree`] struct is made up of a set of vectors.
use crate::{encoding::MajoranaEncoding, operators::Pauli};
use log::{debug, error, info};
use numpy::ndarray::{s, Array1, Array2, Zip};
use std::collections::HashMap;
use std::ops::Not;
use std::result::Result;
use std::{fmt, usize};
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
/// let yp= YParity::Even;
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
    edge: Edge,
    index: u8,
}

impl Parent {
    pub fn new(edge: Edge, index: u8) -> Self {
        Parent { edge, index }
    }
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
    Node(u8),
    XLeaf(u8),
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

#[derive(Debug, PartialEq, Eq)]
pub struct TernaryTree {
    pub(super) parent_of: Vec<Option<Parent>>,
    pub(super) x_child_of: Vec<Option<Child>>,
    pub(super) y_child_of: Vec<Option<Child>>,
    pub(super) z_child_of: Vec<Option<Child>>,
    pub(super) y_parity_of: Vec<YParity>,
    pub n_nodes: usize,
    qubit_index_of: Option<Vec<usize>>,
}

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
}

// Constructors and input
impl TernaryTree {
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

    pub fn from_flatpack(flatpack: TTFlatPack) -> Result<TernaryTree, TernaryTreeError> {
        let n_nodes = flatpack.len();
        let mut tree = TernaryTree::new(n_nodes);
        tree.add_children_from_flatpack(flatpack)?;
        Ok(tree)
    }

    pub fn from_flatpack_naive(flatpack: TTFlatPack) -> Result<TernaryTree, TernaryTreeError> {
        let n_nodes = flatpack.len();
        let mut tree = TernaryTree::new_naive(n_nodes);
        tree.add_children_from_flatpack(flatpack)?;
        Ok(tree)
    }

    fn add_children_from_flatpack(&mut self, flatpack: TTFlatPack) -> Result<(), TernaryTreeError> {
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

        for &(parent, children) in flatpack.iter() {
            let parent = *qubit_node_map
                .get(&parent)
                .ok_or_else(|| TernaryTreeError::FlatPackError(flatpack.clone()))?
                as u8;
            for (child, edge) in std::iter::zip(
                [children.0, children.1, children.2],
                [Edge::X, Edge::Y, Edge::Z],
            ) {
                if let Some(index) = child {
                    let index = *qubit_node_map
                        .get(&index)
                        .ok_or_else(|| TernaryTreeError::FlatPackError(flatpack.clone()))?;
                    self.add_child(Parent::new(edge, parent), Child::Node(index as u8))?
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
                info!("Setting qubit indices {:?}", qubit_indices);
                self.qubit_index_of = Some(qubit_indices);
            }
        }
        Ok(())
    }
}

// Standard Encodings
impl TernaryTree {
    pub fn naive_jordan_wigner(n_nodes: usize) -> TernaryTree {
        let mut tree = TernaryTree::new_naive(n_nodes);
        let branch: Vec<(Edge, usize)> = (0..n_nodes - 1).map(|v| (Edge::Z, v + 1)).collect();
        debug!("{:?}", branch);
        tree.add_branch(0, branch)
            .expect("Naive JW branch should be valid.");
        debug!("{:?}", tree);
        tree
    }

    pub fn naive_parity(n_nodes: usize) -> TernaryTree {
        let mut tree = TernaryTree::new_naive(n_nodes);
        debug!("{:?}", tree);
        let branch: Vec<(Edge, usize)> = (0..n_nodes - 1).map(|v| (Edge::X, v + 1)).collect();
        tree.add_branch(0, branch)
            .expect("Naive Parity branch should be valid.");
        tree
    }

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
    pub fn build_encoding(
        &self,
        n_qubits: usize,
        // mode_op_map: Option<Vec<usize>>, //TODO
    ) -> Result<MajoranaEncoding, TernaryTreeError> {
        debug!("Build encoding from {self:?}");
        if n_qubits < self.n_nodes {
            return Err(TernaryTreeError::BuildEncodingError(
                n_qubits,
                self.qubit_index_of.clone(),
            ));
        }
        let mut ipowers: Array1<u8> = Array1::zeros(2 * self.n_nodes);
        let mut symplectics: Array2<bool> =
            Array2::from_elem((2 * self.n_nodes, 2 * self.n_nodes), false);
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
                let symplectic_result = self
                    .symplectic_from_leaf(&final_edge, ind)
                    .expect("Leaf locations should have been validated.");
                debug!("leaf_result {:?}", symplectic_result);
                ipowers[symplectic_result.0 as usize] = symplectic_result.1;
                symplectics
                    .slice_mut(s![symplectic_result.0 as usize, ..])
                    .assign(&symplectic_result.2);
            });
            debug!("symplectics {:?}", symplectics);
        }
        if self.qubit_index_of.is_some() {
            let column_indices: Array1<usize> = self
                .qubit_index_of
                .as_ref()
                .unwrap()
                .iter()
                .copied()
                .chain(
                    self.qubit_index_of
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|&v| v + n_qubits),
                )
                .collect();
            debug!("Column indices {:?}", &column_indices);
            if let Some(max_column_label) = column_indices.flatten().iter().max() {
                if *max_column_label > 2 * n_qubits {
                    error!("Cannot build encoding with {n_qubits} qubits");
                    return Err(TernaryTreeError::BuildEncodingError(
                        n_qubits,
                        self.qubit_index_of.clone(),
                    ));
                }
            }
            let mut padded_symplectics: Array2<bool> =
                Array2::from_elem((2 * self.n_nodes, 2 * n_qubits), false);
            debug!("Qubit indices {:?}", &self.qubit_index_of);
            Zip::from(symplectics.columns())
                .and(&column_indices)
                .for_each(|unpadded, &index| {
                    debug!("{:?}", unpadded);
                    debug!("{:?}", index);
                    padded_symplectics.column_mut(index).assign(&unpadded);
                    debug!("",);
                });

            debug!("Halfway Padded symplectics {:?}", padded_symplectics);

            debug!("Padded symplectics {:?}", padded_symplectics);
            symplectics = padded_symplectics;
        }
        Ok(MajoranaEncoding::new(ipowers, symplectics))
    }
}

impl TernaryTree {
    fn majorana_index(&self, child: Child) -> u8 {
        match child {
            Child::XLeaf(ind) => 2 * ind + self.y_parity_of[child.usize_index()].as_u8(),
            Child::YLeaf(ind) => 2 * ind + (!self.y_parity_of[child.usize_index()]).as_u8(),
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
    ) -> Result<(u8, u8, Array1<bool>), TernaryTreeError> {
        let child_of = match leaf_edge {
            Edge::X => &self.x_child_of,
            Edge::Y => &self.y_child_of,
            Edge::Z => &self.z_child_of,
        };
        let mut ipower: u8 = 0;
        let mut xz_array: Array1<bool> = Array1::from_elem(2 * self.n_nodes, false);
        let majorana_index: u8;
        debug!("Parent index {parent_index}");
        if let Some(child) = child_of[parent_index] {
            debug!("Child {child:?}");
            match child {
                Child::XLeaf(_) | Child::YLeaf(_) => {
                    majorana_index = self.majorana_index(child);
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
        xz_array[[parent_index]] = bool_term.0;
        xz_array[[parent_index + self.n_nodes]] = bool_term.1;

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
            xz_array[[parent_index]] = bool_term.0;
            xz_array[[parent_index + self.n_nodes]] = bool_term.1;
        }
        debug!("Majorana index {:?}", majorana_index);
        debug!("ipower {:?}", ipower);
        debug!("symplectic {:?}", xz_array);
        Ok((majorana_index, ipower, xz_array))
    }
}

#[cfg(test)]
mod tt_tests {
    use super::*;
    use numpy::ndarray::{arr1, arr2};
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
        let tt = TernaryTree::from_flatpack(flatpack).unwrap();
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
        let tt = TernaryTree::from_flatpack_naive(flatpack).unwrap();
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
            TernaryTree::from_flatpack_naive(jw_flatpack).unwrap(),
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
            TernaryTree::from_flatpack_naive(pe_flatpack).unwrap(),
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
            TernaryTree::from_flatpack_naive(bk_flatpack).unwrap(),
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
            TernaryTree::from_flatpack_naive(jkmn_flatpack).unwrap(),
            expected
        );
    }

    #[test]
    fn test_from_flatpack_with_qubit_labels() {
        let mut flatpack = TTFlatPack::new();
        flatpack.push((9, (Some(10), Some(11), Some(12))));
        flatpack.push((10, (None, None, None)));
        flatpack.push((11, (None, None, None)));
        flatpack.push((12, (None, None, None)));
        let tree = TernaryTree::from_flatpack_naive(flatpack).unwrap();

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
        let expected = (0, 0, arr1(&[true, false, false, false, false, false]));
        assert_eq!(xz_result, expected);

        let xz_result = tt.symplectic_from_leaf(&Edge::Y, 2).unwrap();
        let expected = (5, 1, arr1(&[false, false, true, true, true, true]));
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
        assert_eq!(encoding.ipowers, ipow_expected);
        let symplectic_expected = arr2(&[
            [true, false, false, false, false, false],
            [true, false, false, true, false, false],
            [false, true, false, true, false, false],
            [false, true, false, true, true, false],
            [false, false, true, true, true, false],
            [false, false, true, true, true, true],
        ]);
        assert_eq!(encoding.symplectics, symplectic_expected);
    }

    #[test]
    fn test_jw_flatpack_build_encoding() {
        let flatpack: TTFlatPack = Vec::from(&[
            (0, (None, None, Some(1))),
            (1, (None, None, Some(2))),
            (2, (None, None, None)),
        ]);
        let tt = TernaryTree::from_flatpack_naive(flatpack).unwrap();
        assert_eq!(tt.qubit_index_of, Some(vec![0, 1, 2]));
        let n_qubits = tt.n_nodes;
        let encoding = tt.build_encoding(n_qubits).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 0, 1]);
        assert_eq!(encoding.ipowers, ipow_expected);
        let symplectic_expected = arr2(&[
            [true, false, false, false, false, false],
            [true, false, false, true, false, false],
            [false, true, false, true, false, false],
            [false, true, false, true, true, false],
            [false, false, true, true, true, false],
            [false, false, true, true, true, true],
        ]);
        assert_eq!(encoding.symplectics, symplectic_expected);
    }

    #[test]
    fn test_naive_jw_encoding() {
        let tree = TernaryTree::naive_jordan_wigner(3);
        let encoding = tree.build_encoding(3).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 0, 1]);
        assert_eq!(encoding.ipowers, ipow_expected);
        let symplectic_expected = arr2(&[
            [true, false, false, false, false, false],
            [true, false, false, true, false, false],
            [false, true, false, true, false, false],
            [false, true, false, true, true, false],
            [false, false, true, true, true, false],
            [false, false, true, true, true, true],
        ]);
        assert_eq!(encoding.symplectics, symplectic_expected);
    }

    #[test]
    fn test_naive_parity_encoding() {
        let tree = TernaryTree::naive_parity(3);
        let encoding = tree.build_encoding(3).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 0, 1]);
        assert_eq!(encoding.ipowers, ipow_expected);
        let symplectic_expected = arr2(&[
            [true, false, false, false, true, false],
            [true, false, false, true, false, false],
            [true, true, false, false, false, true],
            [true, true, false, false, true, false],
            [true, true, true, false, false, false],
            [true, true, true, false, false, true],
        ]);
        assert_eq!(encoding.symplectics, symplectic_expected);
    }

    #[test]
    fn test_naive_jkmn_encoding() {
        let tree = TernaryTree::naive_jkmn(3);
        let encoding = tree.build_encoding(3).unwrap();
        let ipow_expected = arr1(&[0, 1, 0, 1, 2, 1]);
        assert_eq!(encoding.ipowers, ipow_expected);
        let symplectic_expected = arr2(&[
            [true, false, false, false, true, false],
            [true, false, false, true, false, true],
            [true, true, false, false, false, false],
            [true, true, false, false, true, false],
            [true, false, true, true, false, true],
            [true, false, true, true, false, false],
        ]);
        assert_eq!(encoding.symplectics, symplectic_expected);
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
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::encoding::Encode;
    use crate::hamiltonians::QubitHamiltonian;
    use crate::operators::{FermionMatrix, FermionSparse, LadderOperator, MajoranaSparse};
    use ahash::HashMapExt;
    use num_complex::c64;
    use numpy::ndarray::arr2;
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
