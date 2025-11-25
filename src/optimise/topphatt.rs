use itertools::FoldWhile::{Continue, Done};
use itertools::{izip, Itertools};
use std::iter::zip;
use tinyvec::{array_vec, Array, ArrayVec};
const MAX_SIZE: usize = 85;
const MAJORANA_MAX: usize = 4;

use crate::encoding::{MajoranaEncoding, MajoranaEncodingOwned};
use crate::ternarytree::{Child, Edge, Parent, TernaryTree, TernaryTreeError};
use crate::types::MajoranaSparse;

enum Restriction {
    Free,
    LeafParity,
    Empty,
    ChildNode,
    LeafPair,
}

type LeafLocation = (usize, Edge);

struct LeafPair {
    odd: LeafLocation,
    even: LeafLocation,
}

struct TreeRetrictions {
    x: Vec<Restriction>,
    y: Vec<Restriction>,
    z: Vec<Restriction>,
    pairs: Vec<LeafPair>,
}

impl TreeRetrictions {
    fn new(tree: &TernaryTree) -> Self {
        let x: Vec<Restriction> = Vec::with_capacity(tree.n_nodes);
        let y: Vec<Restriction> = Vec::with_capacity(tree.n_nodes);
        let z: Vec<Restriction> = Vec::with_capacity(tree.n_nodes);
        let pairs: Vec<LeafPair> = Vec::with_capacity(tree.n_nodes);
        Self { x, y, z, pairs }
    }
}

struct NodeDependencies(Vec<ArrayVec<[usize; 3]>>);

// impl NodeDependencies {
//     fn new(tree: &TernaryTree) -> Self {
//         let mut nd: Vec<ArrayVec<[usize; 3]>> =
//             Vec::from_iter((0..tree.n_nodes).into_iter().map(|v| ArrayVec::new()));
//         for (dep, xchild, ychild, zchild) in izip!(
//             &mut nd,
//             &tree.x_child_of,
//             &tree.y_child_of,
//             &tree.z_child_of
//         ) {
//             [xchild, ychild, zchild]
//                 .into_iter()
//                 .flatten()
//                 .filter(|v| matches!(v, Child::Node(_)))
//                 .for_each(|&v| dep.push(v.qubit_index()));
//         }
//         NodeDependencies(nd)
//     }
// }

pub fn topphatt(
    mut tree: TernaryTree,
    hamiltonian: MajoranaSparse,
) -> Result<MajoranaEncodingOwned, TernaryTreeError> {
    let mut restrictions = TreeRetrictions::new(&tree);

    let mut active_nodes: Vec<usize> = Vec::new();
    // let mut node_dependencies = NodeDependencies::new(&tree);

    Ok(MajoranaEncodingOwned::try_from(tree)?)
}
