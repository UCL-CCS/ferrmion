//! Bravyi-Kitaev-SuperFast (Interaction) basis
//! Edge and Vertex operators
//!
use ndarray::Array2;
use num_complex::{Complex, Complex64};
use std::ops::Mul;
use thiserror::Error;
use tinyvec::{array_vec, ArrayVec};

use crate::operators::{Majorana, MajoranaProduct, Mode};
const MAX_INTERACTION_OPERATORS: usize = 7;

/// Basis for interaction operators.
///
/// If we restricted to only the Edge and Vertex operators,
/// the it wouldn't be possible to know how many operators
/// are required to represent a set of majorana indices.
///
/// For instance, an unprimed operator followed by a primed operator
/// $\gamma_i \gamma_j'$ cannot be represented by a single
/// `Edge` or `Vertex` operator.
/// In the case i==j then they are the identity, but otherwise
/// it must be represented by `Edge(i,j) Vertex(j,j)`,
/// which we can condense into a single `EdgeVertex`.
/// (Note that we could swap the order of operators by adding a -1 factor).
///
/// Similarly, two primed majorana indices $\gamma_i' \gamma_j'$
/// could be represented by either `Identity` or by
/// `Edge(i,j) Vertex(i,i) Vertex(j,j)`
///
/// Used in cases where multiple sets of indices are appied to
/// the same set of operators, such as [`InteractionSparse`].
pub enum InteractionBasis {
    Edge,
    Vertex,
    /// Compound edge-vertex operator.
    /// $\gamma_i \gamma_j \gamma_j \gamma_{k}'$
    EdgeVertex,
    EdgeVertexVertex,
    Identity,
}

/// Bravyi-Kitaev-SuperFast (Interaction) basis.
///
/// Edge operators must have two different [`Mode`] indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum InteractionOperator {
    Edge(Mode, Mode),
    Vertex(Mode),
    #[default]
    Identity,
}

/// Error type for [`InteractionOperator`] operations.
#[derive(Debug, Error)]
pub enum InteractionOperatorError {
    #[error("Edge operator indices {0:?} and {1:?} are not distinct.")]
    SelfEdgeError(Mode, Mode),
    #[error("Vertex operator indices {0:?} and {1:?} are not equal.")]
    NonSelfVertexError(Mode, Mode),
    #[error("Only even degree Majorana operators can be converted to InteractionOperator.")]
    OddDegreeMajoranaError,
    #[error("Too many interaction operators to create ArrayVec.")]
    ArrayVecCapacityError,
}

impl From<VertexOperator> for InteractionOperator {
    fn from(value: VertexOperator) -> Self {
        InteractionOperator::Vertex(value.0)
    }
}

impl From<EdgeOperator> for InteractionOperator {
    fn from(value: EdgeOperator) -> Self {
        InteractionOperator::Edge(value.0, value.1)
    }
}

/// Edge Operator type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EdgeOperator(Mode, Mode);

impl EdgeOperator {
    fn new(left: Mode, right: Mode) -> Result<Self, InteractionOperatorError> {
        if left == right {
            return Err(InteractionOperatorError::SelfEdgeError(left, right));
        }
        Ok(Self(left, right))
    }
}

/// Vertex Operator type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VertexOperator(Mode);

impl VertexOperator {
    fn new(left: Mode, right: Mode) -> Result<Self, InteractionOperatorError> {
        if left != right {
            return Err(InteractionOperatorError::NonSelfVertexError(left, right));
        }
        Ok(Self(left))
    }
}

/// A product of interaction operators.
pub struct InteractionProduct {
    pub ops: ArrayVec<[InteractionOperator; MAX_INTERACTION_OPERATORS]>,
    pub coeff: Complex64,
}

impl InteractionProduct {
    pub fn identity() -> Self {
        Self {
            ops: array_vec![],
            coeff: Complex::ONE,
        }
    }
}

impl Mul<InteractionProduct> for InteractionProduct {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            ops: self.ops.into_iter().chain(rhs.ops).collect(),
            coeff: self.coeff * rhs.coeff,
        }
    }
}

impl Mul<Complex64> for InteractionProduct {
    type Output = Self;
    fn mul(self, rhs: Complex64) -> Self::Output {
        Self {
            ops: self.ops,
            coeff: self.coeff * rhs,
        }
    }
}

/// Converts a pair of [`Majorana`] indices into an [`InteractionOperator`].
///
/// This cannot be done directly to the [`InteractionOperator`] type,
/// as two majoranas may map to multiple interaction operators.
impl From<(Majorana, Majorana)> for InteractionProduct {
    fn from((left, right): (Majorana, Majorana)) -> Self {
        let mut ops: ArrayVec<[InteractionOperator; MAX_INTERACTION_OPERATORS]> = array_vec![];
        let mut coeff = Complex64::new(1.0, 0.0);
        let sorted: (Majorana, Majorana);
        if left < right {
            sorted = (left, right);
        } else {
            sorted = (right, left);
            coeff *= -1.0;
        };
        let (left, right) = (sorted.0, sorted.1);

        match (left.is_even(), right.is_even()) {
            (true, true) => {
                coeff *= Complex::i();
                if left == right {
                    ops.push(InteractionOperator::Identity);
                } else {
                    ops.push(
                        EdgeOperator::new(left.mode(), right.mode())
                            .expect("Should be able to create edge operator.")
                            .into(),
                    );
                }
            }
            (false, false) => {
                if left == right {
                    ops.push(InteractionOperator::Identity);
                } else {
                    coeff *= Complex64::new(0.0, -1.0);
                    ops.push(
                        EdgeOperator::new(left.mode(), right.mode())
                            .expect("Should be able to create edge operator.")
                            .into(),
                    );
                    ops.push(
                        VertexOperator::new(left.mode(), left.mode())
                            .expect("Should be able to create vertex operator.")
                            .into(),
                    );
                    ops.push(
                        VertexOperator::new(right.mode(), right.mode())
                            .expect("Should be able to create vertex operator.")
                            .into(),
                    );
                }
            }
            (true, false) => {
                if left.index() != right.index() - 1 {
                    coeff *= Complex::i();
                    ops.push(EdgeOperator::new(left.mode(), right.mode()).unwrap().into());
                }
                coeff *= Complex::i();
                ops.push(
                    VertexOperator::new(right.mode(), right.mode())
                        .expect("Should be able to create vertex operator.")
                        .into(),
                );
            }
            // Since we have sorted them already, this cannot be a simple
            // reordering of a Vertex (n+1, n).
            (false, true) => {
                ops.push(EdgeOperator::new(right.mode(), left.mode()).unwrap().into());
                ops.push(
                    VertexOperator::new(left.mode(), left.mode())
                        .expect("Should be able to create vertex operator.")
                        .into(),
                );
            }
        }
        Self { ops, coeff }
    }
}

impl TryFrom<MajoranaProduct> for InteractionProduct {
    type Error = InteractionOperatorError;

    fn try_from(mp: MajoranaProduct) -> Result<Self, Self::Error> {
        if !mp.operators.len().is_multiple_of(2) {
            return Err(InteractionOperatorError::OddDegreeMajoranaError);
        }
        let prod: InteractionProduct = mp.operators.chunks(2).fold(
            InteractionProduct::identity() * mp.coefficient,
            |acc, op| acc * InteractionProduct::from((op[0], op[1])),
        );
        Ok(prod)
    }
}

pub type InteractionArrayVec = ArrayVec<[InteractionOperator; MAX_INTERACTION_OPERATORS]>;

impl TryFrom<InteractionProduct> for InteractionArrayVec {
    type Error = InteractionOperatorError;

    fn try_from(prod: InteractionProduct) -> Result<Self, Self::Error> {
        if prod.ops.len() > MAX_INTERACTION_OPERATORS {
            return Err(InteractionOperatorError::ArrayVecCapacityError);
        }
        Ok(prod.ops.into_iter().collect::<InteractionArrayVec>())
    }
}

/// Sparse representation of an interaction operator.
pub struct InteractionSparse {
    pub ops: Vec<InteractionBasis>,
    pub indices: Array2<Mode>,
}

/// Errors that can occur when constructing an [`InteractionSparse`].
#[derive(Error, Debug)]
pub enum InteractionSparseError {
    #[error("Number of operators ({0}) does not match number of indices ({1}).")]
    OperatorIndexMismatch(usize, usize),
    #[error("Interaction operator has odd number of indices.")]
    OddDegreeOperatorError,
}

impl InteractionSparse {
    pub fn new(
        ops: Vec<InteractionBasis>,
        indices: Array2<Mode>,
    ) -> Result<Self, InteractionSparseError> {
        if ops.len() != 2 * indices.shape()[0] {
            return Err(InteractionSparseError::OperatorIndexMismatch(
                ops.len(),
                indices.shape()[0],
            ));
        } else if !indices.ncols().is_multiple_of(2) {
            return Err(InteractionSparseError::OddDegreeOperatorError);
        }
        Ok(Self { ops, indices })
    }
}
