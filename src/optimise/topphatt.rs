use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use std::iter::zip;
use tinyvec::ArrayVec;
const MAX_SIZE: usize = 85;
const MAJORANA_MAX: usize = 4;

use crate::encoding::{MajoranaEncoding, MajoranaEncodingOwned};
use crate::ternarytree::{Edge, TernaryTree, TernaryTreeError};
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

pub fn topphatt(
    mut tree: TernaryTree,
    hamiltonian: MajoranaSparse,
) -> Result<MajoranaEncodingOwned, TernaryTreeError> {
    let mut restrictions = TreeRetrictions::new(&tree);

    Ok(MajoranaEncodingOwned::try_from(tree)?)
}
