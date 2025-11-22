/*
Ternary tree encodings and methods.
*/
use crate::{encoding::MajoranaEncodingOwned, types::Pauli};
use numpy::ndarray::{s, Array1, Array2};
use pyo3::PyTraverseError;
use std::fmt;
use std::result::Result;
use thiserror::Error;
use tinyvec::ArrayVec;

type NodeIndexArray = ArrayVec<[u8; 256]>;
const MAX_SIZE: usize = 85;

#[derive(Debug, PartialEq, Clone)]
pub enum Edge {
    X,
    Y,
    Z,
}

impl Edge {
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

#[derive(Debug)]
pub struct FastTernaryTree {
    pub parent_of: NodeIndexArray,
    pub x_child_of: NodeIndexArray,
    pub y_child_of: NodeIndexArray,
    pub z_child_of: NodeIndexArray,
    pub z_ancestor_of: NodeIndexArray,
    pub z_descendant_of: NodeIndexArray,
    pub n_nodes: u8,
}

impl FastTernaryTree {
    pub fn new(n_nodes: usize) -> Self {
        // let init_child: NodeIndexArray =
        // ArrayVec::from_array_len(core::array::from_fn(|i| { i } as u8), n_nodes);
        let init_parents =
            ArrayVec::from_array_len(core::array::from_fn(|i| { i } as u8), 3 * n_nodes + 1);
        Self {
            parent_of: init_parents,
            x_child_of: init_parents,
            y_child_of: init_parents,
            z_child_of: init_parents,
            z_ancestor_of: init_parents,
            z_descendant_of: init_parents,
            n_nodes: n_nodes as u8,
        }
    }

    pub fn add_child(&mut self, branch: Edge, parent_index: u8, child_index: u8) {
        // Child should always be set
        // is set as self index initially.

        // if not self-child
        // change child
        // add parent to new child
        // // remove current childs parent
        let child_of: &mut ArrayVec<[u8; 256]>;
        let parent_branch_offset: u8;
        match branch {
            Edge::X => {
                child_of = &mut self.x_child_of;
                parent_branch_offset = 0;
            }
            Edge::Y => {
                child_of = &mut self.y_child_of;
                parent_branch_offset = self.n_nodes as u8;
            }
            Edge::Z => {
                child_of = &mut self.z_child_of;
                parent_branch_offset = 2 * self.n_nodes as u8;
            }
        }
        let existing_child: u8 = child_of[parent_index as usize];
        if existing_child != parent_index {
            self.parent_of[existing_child as usize] = existing_child;
        }
        child_of[parent_index as usize] = child_index;
        println!("pbo {:?}", parent_branch_offset);
        self.parent_of[child_index as usize] = parent_index + parent_branch_offset;
    }
    pub fn add_leaf(&mut self, branch: Edge, parent_index: u8, leaf_index: u8) {
        // Child should always be set
        // is set as self index initially.

        // if not self-child
        // change current child to leaf
        // remove current childs parent
        let child_of: &mut ArrayVec<[u8; 256]>;
        match branch {
            Edge::X => {
                child_of = &mut self.x_child_of;
            }
            Edge::Y => {
                child_of = &mut self.y_child_of;
            }
            Edge::Z => {
                child_of = &mut self.z_child_of;
            }
        }
        let existing_child: u8 = child_of[parent_index as usize];
        if existing_child != parent_index {
            self.parent_of[existing_child as usize] = existing_child;
        }
        child_of[parent_index as usize] = leaf_index;
    }
    pub fn get_parent_edge(&self, parent_index: u8) -> Option<(Edge, usize)> {
        if parent_index < self.n_nodes {
            Some((Edge::X, parent_index as usize))
        } else if parent_index > self.n_nodes && parent_index < 2 * self.n_nodes {
            Some((Edge::Y, (parent_index % self.n_nodes) as usize))
        } else if parent_index < 3 * self.n_nodes {
            Some((Edge::Z, (parent_index % self.n_nodes) as usize))
        } else {
            None
        }
    }
}

// impl From<FastTernaryTree> for MajoranaEncodingOwned {
//     fn from(ftt: FastTernaryTree) -> MajoranaEncodingOwned {
//         let mut ipowers: Array1<u8> = Array1::zeros(2 * ftt.n_nodes as usize);
//         let mut symplectics: Array2<bool> =
//             Array2::from_elem((2 * ftt.n_nodes as usize, 2 * ftt.n_nodes as usize), false);
//         for final_edge in [Edge::X, Edge::Y, Edge::Z] {
//             let child_of = match final_edge {
//                 Edge::X => ftt.x_child_of,
//                 Edge::Y => ftt.y_child_of,
//                 Edge::Z => ftt.z_child_of,
//             };
//             let leaf_locations: Vec<usize> = child_of
//                 .into_iter()
//                 .enumerate()
//                 .filter(|(ind, v)| *v >= ftt.n_nodes && *v as usize != *ind)
//                 .map(|(ind, _)| ind as usize)
//                 .collect();
//             leaf_locations.iter().for_each(|&v| {
//                 let majorana_index: usize = (child_of[v as usize] - ftt.n_nodes) as usize;
//                 let mut bool_terms: (bool, bool) = Pauli::from(&final_edge).into();
//                 symplectics[[majorana_index, v]] = bool_terms.0;
//                 symplectics[[majorana_index, v + ftt.n_nodes as usize]] = bool_terms.1;
//                 if final_edge == Edge::Y {
//                     ipowers[majorana_index] += 1_u8;
//                 }
//                 println!("MI,PI,Edge {:?} {:?} {:?}", majorana_index, v, final_edge);
//                 println!("symplectics {:?}", symplectics);
//                 if v == 0 {
//                     return;
//                 }
//                 let mut parent: u8 = ftt.parent_of[v as usize];

//                 while let Some((edge, parent_index)) = ftt.get_parent_edge(parent) {
//                     bool_terms = Pauli::from(&edge).into();
//                     println!("MI,PI {:?} {:?} {:?}", majorana_index, parent_index, edge);
//                     symplectics[[majorana_index, parent_index]] = bool_terms.0;
//                     symplectics[[majorana_index, parent_index + ftt.n_nodes as usize]] =
//                         bool_terms.1;

//                     if edge == Edge::Y {
//                         ipowers[majorana_index] += 1_u8;
//                     }

//                     parent = ftt.parent_of[parent_index as usize];
//                     if parent as usize == parent_index {
//                         break;
//                     }
//                 }
//                 println!("symplectics {:?}", symplectics);
//             });
//         }

//         MajoranaEncodingOwned::new(ipowers, symplectics)
//     }
// }

#[cfg(test)]
mod test_ftt {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_new() {
        let mut ftt = FastTernaryTree::new(10);
        ftt.add_child(Edge::X, 1, 5);
        assert_eq!(ftt.parent_of[5], 1);
        ftt.add_child(Edge::Y, 1, 6);
        assert_eq!(ftt.parent_of[6], 1 + ftt.n_nodes);
        ftt.add_child(Edge::Z, 1, 7);
        assert_eq!(ftt.parent_of[7], 1 + (2 * ftt.n_nodes));
    }

    // #[test]
    // fn test_jw_ops() {
    //     let mut ftt = FastTernaryTree::new(3);
    //     ftt.add_child(Edge::Z, 0, 1);
    //     ftt.add_leaf(Edge::X, 0, 3);
    //     ftt.add_leaf(Edge::Y, 0, 4);
    //     ftt.add_child(Edge::Z, 1, 2);
    //     ftt.add_leaf(Edge::X, 1, 5);
    //     ftt.add_leaf(Edge::Y, 1, 6);
    //     ftt.add_leaf(Edge::X, 2, 7);
    //     ftt.add_leaf(Edge::Y, 2, 8);
    //     // ftt.add_leaf(Edge::Z, 2, 3);
    //     let encoding = MajoranaEncodingOwned::from(ftt);
    //     let expected_ipowers: Array1<u8> = arr1(&[0, 1, 0, 1, 0, 1]);
    //     assert_eq!(encoding.ipowers, expected_ipowers);
    //     let expected_symplectic: Array2<bool> = arr2(&[[]]);
    //     assert_eq!(encoding.symplectics, expected_symplectic);
    // }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Parent {
    X(u8),
    Y(u8),
    Z(u8),
}

impl Parent {
    fn new(e: Edge, ind: u8) -> Self {
        match e {
            Edge::X => Parent::X(ind),
            Edge::Y => Parent::Y(ind),
            Edge::Z => Parent::Z(ind),
        }
    }
}

impl From<Parent> for usize {
    fn from(c: Parent) -> usize {
        match c {
            Parent::X(ind) => ind as usize,
            Parent::Y(ind) => ind as usize,
            Parent::Z(ind) => ind as usize,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Child {
    Node(u8),
    EvenLeaf(u8),
    OddLeaf(u8),
}

impl Child {
    fn majorana_index(self, n_modes: u8) -> u8 {
        match self {
            Child::Node(ind) => ind + 1 + (2 * n_modes),
            Child::EvenLeaf(ind) => 2 * ind,
            Child::OddLeaf(ind) => 2 * ind + 1,
        }
    }
}

impl From<Child> for usize {
    fn from(c: Child) -> usize {
        match c {
            Child::Node(ind) => ind as usize,
            Child::EvenLeaf(ind) => ind as usize,
            Child::OddLeaf(ind) => ind as usize,
        }
    }
}

pub struct TernaryTree {
    pub parent_of: Vec<Option<Parent>>,
    pub x_child_of: Vec<Option<Child>>,
    pub y_child_of: Vec<Option<Child>>,
    pub z_child_of: Vec<Option<Child>>,
    pub z_ancestor_of: Vec<Option<Parent>>,
    pub z_descendant_of: Vec<Option<Child>>,
    pub n_nodes: usize,
    pub n_qubits: usize,
    root_node: usize,
}

#[derive(Debug, Error)]
pub enum TernaryTreeError {
    #[error("Should have root_node{0} <= n_modes {1} <= n_qubits {2}.")]
    ConstructorError(usize, usize, usize),
    #[error("Could not build symplectic from child of node {1} at {0}.")]
    LeafSymplecticError(Edge, usize),
}

impl TernaryTree {
    pub fn new(
        root_node: usize,
        n_nodes: usize,
        n_qubits: usize,
    ) -> Result<Self, TernaryTreeError> {
        if (n_qubits < n_nodes) | (root_node > n_nodes) {
            return Err(TernaryTreeError::ConstructorError(
                root_node, n_nodes, n_qubits,
            ));
        }
        Ok(Self {
            parent_of: vec![None; n_nodes],
            x_child_of: vec![None; n_nodes],
            y_child_of: vec![None; n_nodes],
            z_child_of: vec![None; n_nodes],
            z_ancestor_of: vec![None; n_nodes],
            z_descendant_of: vec![None; n_nodes],
            n_nodes: n_nodes,
            n_qubits: n_qubits,
            root_node,
        })
    }

    fn add_child(&mut self, parent: Parent, child: Child) {
        let child_of: &mut Vec<Option<Child>>;
        let parent_index: u8;

        match parent {
            Parent::X(ind) => {
                parent_index = ind;
                child_of = &mut self.x_child_of;
            }
            Parent::Y(ind) => {
                parent_index = ind;
                child_of = &mut self.y_child_of;
            }
            Parent::Z(ind) => {
                parent_index = ind;
                child_of = &mut self.z_child_of;
            }
        }

        if let Some(existing_child) = child_of[parent_index as usize] {
            match existing_child {
                Child::Node(ec) => {
                    self.parent_of[ec as usize] = None;
                    self.z_ancestor_of[ec as usize] = None;
                    // Could also update all the descendant nodes z-ancestor here
                }
                // For vacuum preservation, we put the replaced leaf on the Z-edge of the Z-descendant of the child node.
                Child::EvenLeaf(_) | Child::OddLeaf(_) => {
                    if let Some(z_desc) = self.z_descendant_of[usize::from(child)] {
                        self.z_child_of[usize::from(z_desc)] = Some(existing_child)
                    } else {
                        self.z_child_of[usize::from(child)] = Some(existing_child)
                    }
                    self.z_descendant_of[usize::from(child)] = Some(existing_child)
                }
            }
        } else {
            child_of[parent_index as usize] = Some(child);
            if matches!(child, Child::Node(_)) {
                self.parent_of[usize::from(child)] = Some(parent);
            }
        }
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
        let mut xz_array: Array1<bool> = Array1::from_elem(2 * self.n_qubits, false);
        let majorana_index: u8;

        if let Some(child) = child_of[usize::from(parent_index)] {
            match child {
                Child::EvenLeaf(_) | Child::OddLeaf(_) => {
                    majorana_index = child.majorana_index(self.n_nodes as u8);
                }
                Child::Node(_) => {
                    return Err(TernaryTreeError::LeafSymplecticError(
                        leaf_edge.clone(),
                        parent_index,
                    ))
                }
            }
        } else {
            return Err(TernaryTreeError::LeafSymplecticError(
                leaf_edge.clone(),
                parent_index,
            ));
        }

        if matches!(leaf_edge, Edge::Y) {
            ipower += 1
        };

        let bool_term: (bool, bool) = Pauli::from(leaf_edge).into();
        xz_array[[parent_index]] = bool_term.0;
        xz_array[[parent_index + self.n_qubits]] = bool_term.1;

        // let parent = self.parent_of[]
        println!("Parent {:?}", parent_index);
        println!("parent_of {:?}", self.parent_of);

        let mut counter = 0;
        let mut parent_index = parent_index;
        while let Some(parent) = self.parent_of[parent_index] {
            println!("{:?}", parent);
            parent_index = usize::from(parent);
            println!("{:?}", parent_index);

            let bool_term: (bool, bool) = match parent {
                Parent::X(_) => Pauli::X.into(),
                Parent::Y(_) => {
                    ipower += 1;
                    Pauli::Y.into()
                }
                Parent::Z(_) => Pauli::Z.into(),
            };
            println!("{:?}", bool_term);
            xz_array[[parent_index]] = bool_term.0;
            xz_array[[parent_index + self.n_qubits]] = bool_term.1;
            println!("{:?}", xz_array);
            if counter > 5 {
                break;
            } else {
                counter += 1;
            }
        }
        Ok((majorana_index, ipower, xz_array))
    }
}

impl TryFrom<TernaryTree> for MajoranaEncodingOwned {
    type Error = TernaryTreeError;
    fn try_from(tree: TernaryTree) -> Result<MajoranaEncodingOwned, Self::Error> {
        let mut ipowers: Array1<u8> = Array1::zeros(2 * tree.n_nodes as usize);
        let mut symplectics: Array2<bool> = Array2::from_elem(
            (2 * tree.n_nodes as usize, 2 * tree.n_nodes as usize),
            false,
        );
        for final_edge in [Edge::X, Edge::Y, Edge::Z] {
            println!("\nFinal Edge {:?}", final_edge);
            let child_of = match final_edge {
                Edge::X => &tree.x_child_of,
                Edge::Y => &tree.y_child_of,
                Edge::Z => &tree.z_child_of,
            };
            let leaf_locations: Vec<usize> = child_of
                .into_iter()
                .flatten()
                .filter(|v| matches!(v, Child::EvenLeaf(_) | Child::OddLeaf(_)))
                .enumerate()
                .map(|(ind, _)| ind as usize)
                .collect();
            println!("Leaf locations on edge {:?}", leaf_locations);
            leaf_locations.iter().for_each(|&ind| {
                let symplectic_result = tree
                    .symplectic_from_leaf(&final_edge, ind)
                    .expect("Leaf locations should have been validated.");
                let majorana_index = symplectic_result.0;
                ipowers[symplectic_result.0 as usize] = symplectic_result.1;
                symplectics
                    .slice_mut(s![symplectic_result.0 as usize, ..])
                    .assign(&symplectic_result.2);
            });
            println!("symplectics {:?}", symplectics);
        }

        Ok(MajoranaEncodingOwned::new(ipowers, symplectics))
    }
}

#[cfg(test)]
mod tt_tests {
    use super::*;
    use numpy::ndarray::{arr1, arr2};
    use Child::{EvenLeaf, Node, OddLeaf};
    use Parent::{X, Y, Z};

    #[test]
    fn test_new() {
        assert!(TernaryTree::new(0, 3, 0).is_err());

        let tt = TernaryTree::new(0, 3, 3).unwrap();
        assert_eq!(tt.n_qubits, 3);
        assert_eq!(tt.parent_of, vec![None; tt.n_nodes]);
    }

    #[test]
    fn test_symplectic_from_leaf() {
        let mut tt = TernaryTree::new(0, 3, 3).unwrap();
        tt.add_child(Parent::X(0), Child::EvenLeaf(0));
        tt.add_child(Parent::Y(0), Child::OddLeaf(0));
        tt.add_child(Parent::Z(0), Child::Node(1));

        tt.add_child(Parent::X(1), Child::EvenLeaf(1));
        tt.add_child(Parent::Y(1), Child::OddLeaf(1));
        tt.add_child(Parent::Z(1), Child::Node(2));

        tt.add_child(Parent::X(2), Child::EvenLeaf(2));
        tt.add_child(Parent::Y(2), Child::OddLeaf(2));

        assert_eq!(tt.parent_of, vec![None, Some(Z(0)), Some(Z(1))]);
        assert_eq!(
            tt.x_child_of,
            vec![Some(EvenLeaf(0)), Some(EvenLeaf(1)), Some(EvenLeaf(2))]
        );

        let xz_result = tt.symplectic_from_leaf(&Edge::X, 0).unwrap();
        let expected = (0, 0, arr1(&[true, false, false, false, false, false]));
        assert_eq!(xz_result, expected);

        let xz_result = tt.symplectic_from_leaf(&Edge::Y, 2).unwrap();
        let expected = (5, 1, arr1(&[false, false, true, true, true, true]));
        assert_eq!(xz_result, expected);
    }

    #[test]
    fn test_majorana_encoding_try_from() {
        let mut tt = TernaryTree::new(0, 3, 3).unwrap();
        tt.add_child(Parent::X(0), Child::EvenLeaf(0));
        tt.add_child(Parent::Y(0), Child::OddLeaf(0));
        tt.add_child(Parent::Z(0), Child::Node(1));

        tt.add_child(Parent::X(1), Child::EvenLeaf(1));
        tt.add_child(Parent::Y(1), Child::OddLeaf(1));
        tt.add_child(Parent::Z(1), Child::Node(2));

        tt.add_child(Parent::X(2), Child::EvenLeaf(2));
        tt.add_child(Parent::Y(2), Child::OddLeaf(2));

        let encoding = MajoranaEncodingOwned::try_from(tt).unwrap();
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
    }
}
