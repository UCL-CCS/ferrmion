//! Quantum operator types and conversions.
//!
//! `ferrmion` is fundamentally a tool for transforming between fermionic operators and qubit operators.
//! These types underly a lot of the functionality of optimisation methods.
//!
//! Additionally, there are a few different ways in which we may want to describe each of these, primarily:
//! - Single operators
//! - Product operators: Tensor products of individual operators.
//! - Sparse operators: Iterables containing product operator indices and coefficients.
//! - Matrix operators: Matrices of coefficents, with operator indices given by the index of each coefficient.
//!
use crate::ternarytree::Edge;
use itertools::Itertools;
use log::debug;
use ndarray::{arr0, s, Dimension};
use num_complex::{c64, ComplexFloat};
use numpy::ndarray::{
    arr1, arr2, Array1, Array2, ArrayD, ArrayView1, ArrayViewD, Axis, IntoDimension, Zip,
};
use numpy::Complex64;
use std::collections::BTreeMap;
use std::iter::repeat_n;
use std::{result::Result, str::FromStr};
use tinyvec::ArrayVec;

/// Maximum length of majorana indices which are allowed in stack-allocated ArrayVecs.
const MAX_MAJORANAS: usize = 4;

/// Total number of qubits for which
/// non-identity Pauli-operators appear in the operator.
pub trait PauliWeight {
    /// Returns the ['PauliWeight'] of a type.
    fn pauli_weight(&self) -> usize;
}

/// [`PauliWeight`] of a term multiplied its coefficient.
pub trait CoefficientPauliWeight: PauliWeight {
    /// Returns the ['CoefficientPauliWeight'] of a type.
    fn coeff_pauli_weight(&self) -> f64;
}

/// Operators of the Pauli-basis.
#[allow(dead_code)]
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub(super) enum Pauli {
    #[default]
    I,
    X,
    Y,
    Z,
}

impl From<&Edge> for Pauli {
    fn from(e: &Edge) -> Pauli {
        match e {
            Edge::X => Pauli::X,
            Edge::Y => Pauli::Y,
            Edge::Z => Pauli::Z,
        }
    }
}

impl From<Pauli> for String {
    fn from(p: Pauli) -> String {
        match p {
            Pauli::I => "I".to_string(),
            Pauli::X => "X".to_string(),
            Pauli::Y => "Y".to_string(),
            Pauli::Z => "Z".to_string(),
        }
    }
}

impl From<Pauli> for char {
    fn from(p: Pauli) -> char {
        match p {
            Pauli::I => 'I',
            Pauli::X => 'X',
            Pauli::Y => 'Y',
            Pauli::Z => 'Z',
        }
    }
}

impl From<(bool, bool)> for Pauli {
    fn from(xz_bools: (bool, bool)) -> Pauli {
        match xz_bools {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (false, true) => Pauli::Z,
            (true, true) => Pauli::Y,
        }
    }
}

impl From<Pauli> for (bool, bool) {
    fn from(p: Pauli) -> (bool, bool) {
        match p {
            Pauli::I => (false, false),
            Pauli::X => (true, false),
            Pauli::Y => (true, true),
            Pauli::Z => (false, true),
        }
    }
}

impl PauliWeight for Pauli {
    fn pauli_weight(&self) -> usize {
        match self {
            Pauli::I => 0,
            _ => 1,
        }
    }
}

pub(super) type PauliMatrix = Array2<Complex64>;

impl From<Pauli> for PauliMatrix {
    fn from(p: Pauli) -> PauliMatrix {
        match p {
            Pauli::I => arr2(&[[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(1., 0.)]]),
            Pauli::X => arr2(&[[c64(0., 0.), c64(1., 0.)], [c64(1., 0.), c64(0., 0.)]]),
            Pauli::Z => arr2(&[[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(-1., 0.)]]),
            Pauli::Y => arr2(&[[c64(0., 0.), c64(0., -1.)], [c64(0., 1.), c64(0., 0.)]]),
        }
    }
}

#[cfg(test)]
mod test_pauli {
    use super::*;
    use crate::operators::{Pauli, PauliMatrix};
    use ndarray::arr2;

    #[test]
    fn test_matrix_identities() {
        let i = arr2(&[[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(1., 0.)]]);
        let x = Into::<PauliMatrix>::into(Pauli::X);
        let y = Into::<PauliMatrix>::into(Pauli::Y);
        let z = Into::<PauliMatrix>::into(Pauli::Z);
        assert_eq!(&i.dot(&i), i);
        assert_eq!(&x.dot(&x), i);
        assert_eq!(&y.dot(&y), i);
        assert_eq!(&z.dot(&z), i);
        assert_eq!(&x.dot(&z), c64(0., -1.) * y.clone());
        assert_eq!(&y.dot(&z), c64(0., 1.) * x.clone());
    }

    #[test]
    fn test_pauli_weight() {
        assert_eq!(Pauli::I.pauli_weight(), 0);
        assert_eq!(Pauli::X.pauli_weight(), 1);
        assert_eq!(Pauli::Y.pauli_weight(), 1);
        assert_eq!(Pauli::Z.pauli_weight(), 1);
    }

    #[test]
    fn test_pauli_bool_rountrip() {
        for pauli in [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
            let tbool: (bool, bool) = pauli.into();
            let pauli_again: Pauli = tbool.into();
            assert_eq!(pauli, pauli_again);
        }
    }
    #[test]
    fn test_bool_pauli_rountrip() {
        for tbool in [(true, true), (true, false), (false, true), (false, false)] {
            let pauli: Pauli = tbool.into();
            let tbool_again: (bool, bool) = pauli.into();
            assert_eq!(tbool, tbool_again);
        }
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_pauli_bool_roundtrip_prop(x in proptest::bool::ANY, z in proptest::bool::ANY) {
            let pauli: Pauli = (x, z).into();
            let (x2, z2): (bool, bool) = pauli.into();
            prop_assert_eq!((x, z), (x2, z2));
        }
    }
}

/// Pauli operator encoded in symplectic (XZ) form.
#[allow(dead_code)]
#[derive(PartialEq, Eq, Debug, Clone)]
struct SymplecticOperator<'sym> {
    ipower: u8,
    symplectic: ArrayView1<'sym, bool>,
}

impl PauliWeight for SymplecticOperator<'_> {
    fn pauli_weight(&self) -> usize {
        let view = self.symplectic.view();
        let x_block: ArrayView1<bool>;
        let z_block: ArrayView1<bool>;
        (x_block, z_block) = view.split_at(Axis(0), view.len() / 2);
        Zip::from(x_block).and(z_block).fold(0, |acc, x, z| {
            acc + if (x == &false) & (z == &false) { 0 } else { 1 }
        })
    }
}

impl CoefficientPauliWeight for SymplecticOperator<'_> {
    fn coeff_pauli_weight(&self) -> f64 {
        self.pauli_weight() as f64
    }
}

/// Operators for second quantisation.
///
/// These are primarily used in the signatures of fermionic operators, e.g. ['FermionProduct`].
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum LadderOperator {
    /// Particle creation operator.
    Creation,
    /// Particle annihilation operator.
    Annihilation,
}

/// Error for failure to parse ladder operators from strings.
#[derive(Debug, PartialEq, Clone)]
pub struct ParseLadderError;

impl FromStr for LadderOperator {
    type Err = ParseLadderError;
    /// Parse a string as a ladfder operator.
    ///
    ///
    fn from_str(string: &str) -> Result<Self, Self::Err> {
        if string == "+" {
            Ok(LadderOperator::Creation)
        } else if string == "-" {
            Ok(LadderOperator::Annihilation)
        } else {
            Err(ParseLadderError)
        }
    }
}

impl LadderOperator {
    /// Returns the coefficients of a fermionic ladder operator in terms of Majorana operators.
    ///
    /// While ladder operators are general, the fermionic ladder operators can be expressed exactly as
    /// a combination of two majorana operators.
    ///
    /// This function is used when converting from fermionic operators with arbitrary signature, to a mjorana operator.
    pub fn majorana_coefficients(&self) -> Array1<Complex64> {
        match &self {
            LadderOperator::Creation => arr1(&[c64(0.5, 0.0), c64(0., -0.5)]),
            LadderOperator::Annihilation => arr1(&[c64(0.5, 0.0), c64(0., 0.5)]),
        }
    }
}
impl TryFrom<char> for LadderOperator {
    type Error = ParseLadderError;

    fn try_from(string: char) -> Result<Self, Self::Error> {
        if string == '+' {
            Ok(LadderOperator::Creation)
        } else if string == '-' {
            Ok(LadderOperator::Annihilation)
        } else {
            Err(ParseLadderError)
        }
    }
}

#[cfg(test)]
mod ladder_tests {
    use crate::operators::*;

    #[test]
    fn test_ladder_operators() {
        assert_eq!(
            LadderOperator::from_str("+").unwrap(),
            LadderOperator::Creation
        );
        assert_eq!(
            LadderOperator::from_str("-").unwrap(),
            LadderOperator::Annihilation
        );
        assert_eq!(LadderOperator::from_str("+-"), Err(ParseLadderError));
        assert_eq!(LadderOperator::from_str("-+"), Err(ParseLadderError));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_ladder_try_from_char(c in proptest::sample::select(&['+', '-'])) {
            let op = LadderOperator::try_from(c);
            if c == '+' {
                prop_assert_eq!(op, Ok(LadderOperator::Creation));
            } else {
                prop_assert_eq!(op, Ok(LadderOperator::Annihilation));
            }
        }
    }
}

/*
Fermion
*/
/// A product of fermionic ladder operators.
///
/// # Example
/// ```
/// FermionProduct::new(vec![LadderOperator:Creation, LadderOperator::Annihilation], vec![0,1], c64(1.,0.))
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct FermionProduct {
    action: Vec<LadderOperator>,
    indices: Vec<usize>,
    coefficient: Complex64,
}

/// Error type for failure to construct [`FermionProduct`]
#[derive(Debug, PartialEq, Clone)]
pub struct FermionProductError;

impl FermionProduct {
    /// Constructor for [`FermionProduct`]
    pub fn new(
        action: Vec<LadderOperator>,
        indices: Vec<usize>,
        coefficient: Complex64,
    ) -> Result<Self, FermionProductError> {
        if action.len() != indices.len() {
            Err(FermionProductError)
        } else {
            Ok(Self {
                action: action,
                indices,
                coefficient,
            })
        }
    }
}
/// Fermion operator with coefficients in matrix form.
///
/// <div class="warning">
/// Coeffients are in Spin-orbit format.
/// For spatial orbital index "i", the spin-up mode is at $2i$ and the spin down mode is at $2i+1$
/// </div>
///
pub struct FermionMatrix {
    action: Vec<LadderOperator>,
    coefficients: ArrayD<f64>,
}

/// Error raised by failure to contruct [`FermionMatrix`]
#[derive(Debug, PartialEq, Clone)]
pub struct FermionMatrixError;

impl FermionMatrix {
    /// Constructor for [`FermionMatrix`]
    pub fn new(
        action: Vec<LadderOperator>,
        coefficients: ArrayD<f64>,
    ) -> Result<Self, FermionMatrixError> {
        // Check we have enough ladder operators
        // and a square/cube/... matrix
        if action.len() != coefficients.ndim()
            || !coefficients
                .shape()
                .iter()
                .all(|s| *s == coefficients.shape()[0])
        {
            return Err(FermionMatrixError);
        }
        let mut out = Self {
            action,
            coefficients,
        };
        out.zero_disallowed_terms();
        Ok(out)
    }

    /// Zero out terms with disallowed fermion operator index combinations.
    ///
    /// For instance, with action [+,+,-,-] any operators with indices i,j,k,l
    /// where i = j or k == l are zeroed out.
    ///
    /// Note: Current implementation only checks for operators of dimension 4.
    ///
    ///
    // Can be made more general when dim > 4 is added.
    fn zero_disallowed_terms(&mut self) {
        use crate::operators::LadderOperator::{Annihilation, Creation};
        if self.coefficients.ndim() == 4 {
            match self.action.as_slice() {
                [Annihilation, Annihilation, Creation, Creation]
                | [Creation, Creation, Annihilation, Annihilation] => {
                    for i in 0..self.coefficients.shape()[0] {
                        self.coefficients
                            .slice_mut(s![i, i, .., ..])
                            .assign(&arr0(0.0));
                        self.coefficients
                            .slice_mut(s![.., .., i, i])
                            .assign(&arr0(0.0));
                    }
                }
                [Creation, Annihilation, Creation, Annihilation]
                | [Annihilation, Creation, Annihilation, Creation] => {
                    for i in 0..self.coefficients.shape()[0] {
                        // Keeping only [i, i, i, ..]
                        self.coefficients
                            .slice_mut(s![i, ..i, i, ..])
                            .assign(&arr0(0.0));
                        self.coefficients
                            .slice_mut(s![i, i + 1.., i, ..])
                            .assign(&arr0(0.0));

                        // Keeping only [.., i, i, i]
                        self.coefficients
                            .slice_mut(s![.., i, ..i, i])
                            .assign(&arr0(0.0));
                        self.coefficients
                            .slice_mut(s![.., i, i + 1.., i])
                            .assign(&arr0(0.0));
                    }
                }
                _ => {}
            }
        }
    }
}

/// Fermion operator in sparse form.
///
/// Each index is non-empty and each coefficient is non-zero.
#[derive(Debug, PartialEq)]
pub struct FermionSparse {
    action: Vec<LadderOperator>,
    indices: Array2<usize>,
    coefficients: Array1<Complex64>,
}

/// Error type for failure in [`FermionSparse`] constructor.
#[derive(Debug, PartialEq, Clone)]
pub struct FermionSparseError;

impl FermionSparse {
    /// Constructor for [`FermionSparse`]
    pub fn new(
        action: Vec<LadderOperator>,
        indices: Array2<usize>,
        coefficients: Array1<Complex64>,
    ) -> Result<Self, FermionSparseError> {
        if coefficients.len() != indices.len_of(Axis(0)) || action.len() != indices.len_of(Axis(1))
        {
            return Err(FermionSparseError);
        };

        Ok(Self {
            action,
            indices,
            coefficients,
        })
    }
}

impl From<FermionMatrix> for FermionSparse {
    fn from(mft: FermionMatrix) -> FermionSparse {
        let n_nonzero = mft.coefficients.iter().filter(|&v| *v != 0.).count();
        let mut sparse_indices: Array2<usize> = Array2::zeros((n_nonzero, mft.action.len()));
        let mut sparse_coefficients: Array1<Complex64> = Array1::from_elem(n_nonzero, c64(0., 0.));
        mft.coefficients
            .indexed_iter()
            .filter(|(_, &v)| v != 0.)
            .enumerate()
            .for_each(|(count, (ind, &v))| {
                sparse_indices
                    .row_mut(count)
                    .assign(&ind.into_dimension().as_array_view());
                sparse_coefficients[count] += c64(v, 0.);
            });
        FermionSparse::new(mft.action, sparse_indices, sparse_coefficients)
            .expect("Conversion from MatrixFermionTerm should be validated.")
    }
}

#[cfg(test)]
mod fermion_tests {
    use crate::operators::*;
    use crate::vector_kron;
    use ndarray::{arr1, arr2};
    use num_complex::c64;

    #[test]
    fn test_action_conversion() {
        let action = [LadderOperator::Creation, LadderOperator::Annihilation];
        let im_coeffs: Array1<Complex64> = action
            .iter()
            .map(|s| s.majorana_coefficients())
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap();
        assert_eq!(
            im_coeffs,
            arr1(&[
                Complex64 { re: 0.25, im: 0.0 },
                Complex64 { re: 0.0, im: -0.25 },
                Complex64 { re: 0.0, im: 0.25 },
                Complex64 { re: 0.25, im: 0.0 }
            ])
        );
    }

    #[test]
    fn test_sparse_term_creation() {
        let action = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let indices = arr2(&[[0, 1], [2, 3]]);
        let coefficients = arr1(&[c64(1.0, 0.), c64(-1., 0.)]);
        let _term = FermionSparse::new(action, indices, coefficients).unwrap();
    }
    #[test]
    fn test_matrix_term_creation() {
        let action = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let dyn_shape = ndarray::IxDyn(&[2, 2]);
        assert_eq!(dyn_shape.clone().ndim(), 2);
        let coefficients = ArrayD::from_elem(dyn_shape, 1.);
        let _term = FermionMatrix::new(action, coefficients).unwrap();
    }
    #[test]
    fn test_sparse_from_matrix() {
        let action = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let dyn_shape = ndarray::IxDyn(&[2, 2]);
        assert_eq!(dyn_shape.clone().ndim(), 2);
        let mut coefficients = ArrayD::from_elem(dyn_shape, 0.);
        coefficients[[0, 0]] = 1.;
        coefficients[[0, 1]] = 0.5;
        coefficients[[1, 0]] = 2.;
        coefficients[[1, 1]] = 10.;
        let term = FermionMatrix::new(action, coefficients).unwrap();
        let sparse = FermionSparse::from(term);
        assert_eq!(sparse.indices, arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]]));
        assert_eq!(
            sparse.coefficients,
            arr1(&[c64(1., 0.), c64(0.5, 0.), c64(2., 0.), c64(10., 0.)])
        );
    }
}

// /*
// Majorana
// */
/// Product of Majorana operators, with a complex coefficient.
///
/// # Example
/// ```
/// let mp = MajoranaProduct::new(vec![0,1,2,3], c64(0.5,0.5))
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaProduct {
    pub(super) indices: Vec<usize>,
    pub(super) coefficient: Complex64,
}

impl MajoranaProduct {
    /// Constructor for [`MajoranaProduct`]
    pub fn new(indices: Vec<usize>, coefficient: Complex64) -> Self {
        // out.majorise();
        Self {
            indices,
            coefficient,
        }
    }
    /// Sorts indices, applying a -1 coefficient for each swap of unequal indices.
    ///
    /// Majorana operators obey the commutation relation:
    /// $$\{\gamma_i,\gamma_j} = 2\delta_{i,j$$
    ///
    /// So we can combine terms for products of majorana operators composed with the same indices.
    fn majorise(&mut self) {
        if self.indices.is_empty() {
            return;
        }
        let mut counter: usize = 0;
        let mut n = self.indices.len();
        while n > 0 {
            let mut new_n = 0;
            for index in 1..n {
                if self.indices[index - 1] > self.indices[index] {
                    self.indices.swap(index - 1, index);
                    counter += 1;
                    new_n = index;
                }
            }
            n = new_n;
        }
        if counter % 2 == 1 {
            self.coefficient *= -1.
        }
    }
}

/// Order-preserving map from indices to complex coefficients.
///
/// This is used as an intermetiate step when constructing or combining majorana operators.
#[derive(Debug)]
pub(super) struct MajoranaBTree {
    operators: BTreeMap<Vec<usize>, Complex64>,
}

impl MajoranaBTree {
    fn new() -> Self {
        let operators = BTreeMap::<Vec<usize>, Complex64>::new();
        Self { operators }
    }

    /// Append a single product of Fermionic operators to the [`MajoranaBTree`].
    fn append_fermion_product(&mut self, fproduct: FermionProduct) {
        let term_length = fproduct.action.len();
        let offsets = repeat_n(0usize..=1usize, term_length).multi_cartesian_product();
        for offset in offsets {
            let scaler = offset
                .iter()
                .zip(fproduct.action.iter())
                .fold(c64(1., 0.), |acc, (&offset, op)| {
                    acc * op.majorana_coefficients()[offset]
                });
            let mut majorana_term = Array1::zeros(term_length);
            majorana_term += &arr1(fproduct.indices.as_slice());
            majorana_term *= 2;
            majorana_term = majorana_term + Array1::from_vec(offset);

            let mut mp =
                MajoranaProduct::new(majorana_term.to_vec(), fproduct.coefficient * scaler);
            mp.majorise();
            *self
                .operators
                .entry(mp.indices)
                .or_insert(Complex64 { re: 0.0, im: 0.0 }) += mp.coefficient;
        }
    }

    /// Append a Fermionic Hamiltonian in sparse form to the [`MajoranaBTree`].
    fn append_fermion_sparse(&mut self, fsparse: FermionSparse) {
        debug!("FSparse Indices {:?}", &fsparse.indices);

        // Note: This can be generalised
        // +-+- (ijkl) is also invalid if i==k and i!=j
        let double_action: Vec<(usize, usize)> = (0..fsparse.action.len() - 1)
            .filter(|&i| fsparse.action[i] == fsparse.action[i + 1])
            .map(|i| (i, i + 1))
            .collect::<Vec<_>>();

        debug!("Doubled fermonic operators {double_action:?}");

        let term_length = fsparse.action.len();
        Zip::from(fsparse.indices.rows())
            .and(fsparse.coefficients.view())
            .for_each(|ind, coeff| {
                let offsets = repeat_n(0usize..=1usize, term_length).multi_cartesian_product();
                for offset in offsets {
                    let scaler = offset
                        .iter()
                        .zip(fsparse.action.iter())
                        .fold(c64(1., 0.), |acc, (&offset, op)| {
                            acc * op.majorana_coefficients()[offset]
                        });
                    let mut majorana_term = Array1::zeros(term_length);
                    majorana_term += &ind;
                    majorana_term *= 2;
                    majorana_term = majorana_term + Array1::from_vec(offset);

                    let mut mp = MajoranaProduct::new(majorana_term.to_vec(), *coeff * scaler);
                    mp.majorise();
                    *self
                        .operators
                        .entry(mp.indices)
                        .or_insert(Complex64 { re: 0.0, im: 0.0 }) += mp.coefficient;
                }
            });
        debug!("MBTree {:?}\n", &self);
    }
}
/// Sparse represtnation of a set of [`MajoranaProduct`] operators.
///
/// # Panics
/// <div class="warning">
/// This type internally represents indices as stack-allocated ArrayVecs.
/// The maximum size of these is currently restricted to [`MAX_MAJORANAS`].
/// Attempting to create an index of length greater than this will cause a panic.
/// </div>
#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaSparse {
    pub(super) indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
    pub(super) coefficients: Vec<Complex64>,
    pub(super) constant: f64,
}

/// Error type for failed construction of [`MajoranaSparse`]
#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaSparseError;

impl MajoranaSparse {
    /// Constructor for [`MajoranaSparse`]
    pub fn new(
        indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
        coefficients: Vec<Complex64>,
        constant: f64,
    ) -> Result<Self, MajoranaSparseError> {
        if coefficients.len() != indices.len() {
            return Err(MajoranaSparseError);
        };

        let (i, c) = indices
            .iter()
            .zip(&coefficients)
            .filter(|&(_, &coeff)| coeff != Complex64::ZERO)
            .unzip();

        Ok(Self {
            indices: i,
            coefficients: c,
            constant,
        })
    }

    /// Constructor which takes two vectors, one for operator signatures and another for coefficient matrices.
    ///
    /// This is primarily used in the PyO3 interop functions, as the FermionHamiltonian python class
    /// outputs data in this format.
    ///
    /// <div class="warning">
    /// Coeffients should be given in Spin-orbit format.
    /// For spatial orbital index "i", the spin-up mode is at $2i$ and the spin down mode is at $2i+1$
    /// </div>
    ///
    pub fn from_signatures_and_coeffs(
        signatures: Vec<String>,
        coeffs: Vec<ArrayViewD<f64>>,
        constant_energy: f64,
    ) -> MajoranaSparse {
        let mut fsparse_vec: Vec<FermionSparse> = Vec::new();
        for (sig, coeff) in std::iter::zip(signatures, coeffs) {
            let vec_sig: Vec<LadderOperator> = sig
                .chars()
                .map(|v| {
                    LadderOperator::try_from(v).expect("Signature components should be + or -")
                })
                .collect();
            let term_coef = coeff.to_owned();
            fsparse_vec.push(
                FermionMatrix::new(vec_sig, term_coef)
                    .expect("Signature lengths and coeff dimensions must match")
                    .into(),
            );
        }
        debug!("FSparse {:?}", &fsparse_vec);
        debug!("Getting MSparse");
        let mut hamiltonian: MajoranaSparse = MajoranaSparse::from(fsparse_vec);
        hamiltonian.constant += constant_energy;
        debug!("Got MSparse");
        hamiltonian
    }
}

impl From<MajoranaBTree> for MajoranaSparse {
    fn from(mbt: MajoranaBTree) -> MajoranaSparse {
        // debug!("Majoranas {:#?}", majoranas);
        let mut sparse_values: Vec<Complex64> =
            Vec::with_capacity(mbt.operators.values().filter(|&v| v.abs() >= 1e-16).count());
        let mut sparse_indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>> =
            Vec::with_capacity(sparse_values.len());
        // debug!("{:#?}", sparse_values.clone());
        let mut sparse_constant: num_complex::Complex<f64> = c64(0., 0.);
        mbt.operators
            .iter()
            .filter(|(_, &v)| v.abs() >= 1e-16)
            .for_each(|(k, &v)| {
                let mut op: ArrayVec<[u16; MAX_MAJORANAS]> = ArrayVec::new();
                if k.is_empty() {
                    sparse_constant += v;
                } else {
                    for ind in k {
                        op.push(*ind as u16);
                    }
                    sparse_indices.push(op);
                    sparse_values.push(v);
                }
            });
        debug!("Sparse Majorana Indices {:?}", &sparse_indices);
        debug!("Sparse Majorana Coefficients {:?}", &sparse_values);
        MajoranaSparse::new(sparse_indices, sparse_values, sparse_constant.norm())
            .expect("Indices and coefficients should be same length.")
    }
}

impl From<FermionProduct> for MajoranaSparse {
    fn from(fproduct: FermionProduct) -> Self {
        // Start off by creating a BTreeMap as we'll need to add a few fermionic terms
        // to each majorana term
        let mut majoranas: MajoranaBTree = MajoranaBTree::new();
        majoranas.append_fermion_product(fproduct);
        majoranas.into()
    }
}

impl From<FermionSparse> for MajoranaSparse {
    fn from(sft: FermionSparse) -> Self {
        // Start off by creating a BTreeMap as we'll need to add a few fermionic terms
        // to each majorana term
        let mut majoranas: MajoranaBTree = MajoranaBTree::new();
        majoranas.append_fermion_sparse(sft);
        majoranas.into()
    }
}

impl From<Vec<FermionSparse>> for MajoranaSparse {
    fn from(sft: Vec<FermionSparse>) -> Self {
        let mut majoranas: MajoranaBTree = MajoranaBTree::new();
        sft.into_iter().for_each(|term| {
            majoranas.append_fermion_sparse(term);
        });
        majoranas.into()
    }
}

#[cfg(test)]
mod majorana_tests {
    use crate::operators::*;
    use crate::vector_kron;
    use log::debug;
    use ndarray::{arr1, arr2};
    use num_complex::c64;
    use tinyvec::array_vec;

    #[test]
    fn test_ladder_to_complex() {
        // Output should look like
        // [left_0 right_0, left_0 right_1, left_1 right_0, left_1 right_1]
        let ladder_vec = [LadderOperator::Creation, LadderOperator::Annihilation];
        let two_action: Vec<Complex64> = ladder_vec
            .iter()
            .map(|signature| signature.majorana_coefficients())
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap()
            .to_vec();
        assert_eq!(
            two_action,
            vec![c64(0.25, 0.), c64(0., -0.25), c64(0., 0.25), c64(0.25, 0.)]
        );

        let ladder_vec = [
            LadderOperator::Creation,
            LadderOperator::Annihilation,
            LadderOperator::Creation,
        ];
        let three_action: Vec<Complex64> = ladder_vec
            .iter()
            .map(|signature| signature.majorana_coefficients())
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap()
            .to_vec();
        assert_eq!(
            three_action,
            vec![
                c64(0.125, 0.),
                c64(0., -0.125),
                c64(0., 0.125),
                c64(0.125, 0.),
                c64(0., -0.125),
                c64(-0.125, 0.),
                c64(0.125, 0.),
                c64(0., -0.125),
            ]
        );
    }

    #[test]
    fn test_majorise_do_nothing() {
        let indices = vec![0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        assert_eq!(mp.indices, indices.clone());
        assert_eq!(mp.coefficient, coefficient.clone());
    }

    #[test]
    fn test_majorise_single_swap() {
        let indices = vec![1, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);
    }

    #[test]
    fn test_majorise_do_not_simplify_to_empty() {
        let indices = vec![0, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 0]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![0, 1, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 0, 1, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![1, 0, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 0, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise_reverse() {
        let indices = vec![3, 2, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 2, 3]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![4, 3, 2, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 2, 3, 4]);
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise() {
        let indices = vec![1, 1, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 1, 1, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 1, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 1, 1, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1, 1, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![1, 1, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 0, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1, 1, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![0, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);
    }
    #[test]
    fn test_from_fermion_sparse_len_one() {
        let indices = arr2(&[[0]]);
        let coefficients = arr1(&[c64(10.0, 0.)]);
        let action = vec![LadderOperator::Creation];
        debug!("{:#?}", indices.clone());
        debug!("{:#?}", coefficients.clone());
        debug!("{:#?}", action.clone());

        let majorana_term = MajoranaSparse::new(
            vec![array_vec!([u16; 4]=> 0), array_vec!([u16; 4]=> 1)],
            vec![c64(5., 0.), c64(0., -5.)],
            0.,
        )
        .unwrap();
        let fermion_term =
            FermionSparse::new(action.clone(), indices.clone(), coefficients.clone()).unwrap();
        assert_eq!(majorana_term, MajoranaSparse::from(fermion_term));
    }

    #[test]
    fn test_from_fermion_sparse_len_two() {
        let indices = arr2(&[[0, 1]]);
        let coefficients = arr1(&[c64(10.0, 0.)]);
        let action = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        debug!("{:#?}", indices.clone());
        debug!("{:#?}", coefficients.clone());
        debug!("{:#?}", action.clone());

        let majorana_term = MajoranaSparse::new(
            vec![
                array_vec!([u16; 4]=> 0, 2),
                array_vec!([u16; 4]=> 0, 3),
                array_vec!([u16; 4]=> 1,2),
                array_vec!([u16; 4]=> 1,3),
            ],
            vec![c64(2.5, 0.), c64(0., 2.5), c64(0.0, -2.5), c64(2.5, 0.)],
            0.,
        )
        .unwrap();
        let fermion_term =
            FermionSparse::new(action.clone(), indices.clone(), coefficients.clone()).unwrap();
        assert_eq!(majorana_term, MajoranaSparse::from(fermion_term));
    }

    #[test]
    fn test_from_fermion_sparse_len_three() {
        let indices = arr2(&[[0, 1, 2]]);
        let coefficients = arr1(&[c64(10.0, 0.)]);
        let action = vec![
            LadderOperator::Creation,
            LadderOperator::Annihilation,
            LadderOperator::Creation,
        ];
        debug!("{:#?}", indices.clone());
        debug!("{:#?}", coefficients.clone());
        debug!("{:#?}", action.clone());

        let majorana_term = MajoranaSparse::new(
            vec![
                array_vec!([u16; 4]=> 0, 2, 4),
                array_vec!([u16; 4]=> 0, 2, 5),
                array_vec!([u16; 4]=> 0, 3, 4),
                array_vec!([u16; 4]=> 0, 3, 5),
                array_vec!([u16; 4]=> 1,2, 4),
                array_vec!([u16; 4]=> 1,2, 5),
                array_vec!([u16; 4]=> 1,3, 4),
                array_vec!([u16; 4]=> 1,3, 5),
            ],
            vec![
                c64(1.25, 0.),
                c64(0., -1.25),
                c64(0.0, 1.25),
                c64(1.25, 0.),
                c64(0., -1.25),
                c64(-1.25, 0.),
                c64(1.25, 0.),
                c64(0., -1.25),
            ],
            0.,
        )
        .unwrap();
        let fermion_term =
            FermionSparse::new(action.clone(), indices.clone(), coefficients.clone()).unwrap();
        assert_eq!(majorana_term, MajoranaSparse::from(fermion_term));
    }

    // #[test]
    // fn test_msparse_from_vec_fsparse() {
    //     todo!();
    // }
}
