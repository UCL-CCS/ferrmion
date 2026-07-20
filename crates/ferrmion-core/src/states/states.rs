//! Structs representing quantum states.

use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use std::ops::Mul;
use thiserror::Error;

use crate::operators::{DenseBlock, DenseIndex};
use crate::spaces::{Fermion, Qubit};

#[derive(Debug, Clone, PartialEq, Eq)]
enum BraKet {
    Bra,
    Ket,
}

/// Trait for a quantum state.
///
/// Note this is implicitly a Ket vector.
pub trait State: Mul<Complex64> {
    /// Normalize the state so that the coefficient has unit norm.
    fn normalize(&mut self);

    /// Return the dimension of the state space.
    fn dimension(&self) -> usize;

    /// Return the adjoint (dagger) of the state.
    fn adjoint(self) -> Self;

    /// Reindex the state according to the given permutation.
    fn reindex(&mut self, permutation: &[usize]);
}

#[derive(Error, Debug)]
pub enum StateError {
    #[error("Invalid bra/ket combination")]
    InvalidBraKet,
}

/// A quantum state in the computational (Pauli Z) basis.
///
/// # Examples
///
/// ```
/// use ferrmion_core::states::{ZBasisState, State};
/// use ndarray::Array1;
/// use num_complex::Complex64;
///
/// let state = ZBasisState::new(Array1::from_vec(vec![true, false, true]), Complex64::new(2.0, 0.0));
/// // The coefficient is normalised on construction.
/// assert_eq!(state.coefficient, Complex64::new(1.0, 0.0));
/// assert_eq!(state.dimension(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ZBasisState {
    /// Qubit occupation packed one bit per qubit into `u64` words.
    pub(crate) state: DenseBlock,
    pub coefficient: Complex64,
    bra_ket: BraKet,
}

impl ZBasisState {
    /// Construct a new `ZBasisState` ket vector with the given state and coefficient.
    ///
    /// The coefficient is automatically normalised to unit norm.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::states::ZBasisState;
    /// use ndarray::arr1;
    /// use num_complex::Complex64;
    ///
    /// let s = ZBasisState::new(arr1(&[false, true]), Complex64::new(0.0, 3.0));
    /// assert_eq!(s.coefficient.norm(), 1.0);
    /// ```
    pub fn new(state: Array1<bool>, coefficient: Complex64) -> Self {
        Self::from_block(DenseBlock::from_bool_view(state.view()), coefficient)
    }

    /// Construct a `ZBasisState` from an already-packed [`DenseBlock`].
    ///
    /// The coefficient is automatically normalised to unit norm.
    pub fn from_block(state: DenseBlock, coefficient: Complex64) -> Self {
        let mut out = Self {
            state,
            coefficient,
            bra_ket: BraKet::Ket,
        };
        out.normalize();
        out
    }

    /// The qubit occupation as a dense boolean array (Python / test boundary).
    pub fn state_bools(&self) -> Array1<bool> {
        self.state.to_bool_array()
    }

    /// Borrow the packed qubit occupation.
    pub fn state_block(&self) -> DenseBlock<&[DenseIndex]> {
        self.state.as_ref()
    }

    /// Construct a new `ZBasisState` with all qubits set to zero and a unit coefficient.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::states::{State, ZBasisState};
    ///
    /// let s = ZBasisState::zeros(4);
    /// assert_eq!(s.dimension(), 4);
    /// assert!(s.state_bools().iter().all(|&b| !b));
    /// ```
    pub fn zeros(n_qubits: usize) -> Self {
        Self::from_block(DenseBlock::zeros(1, n_qubits), Complex64::new(1., 0.))
    }
}

impl Qubit for ZBasisState {}

impl State for ZBasisState {
    fn normalize(&mut self) {
        let norm = self.coefficient.norm();
        if norm != 0. {
            self.coefficient /= norm;
        }
    }
    fn dimension(&self) -> usize {
        self.state.n_indices()
    }

    fn adjoint(self) -> Self {
        Self {
            state: self.state,
            coefficient: self.coefficient.conj(),
            bra_ket: match self.bra_ket {
                BraKet::Bra => BraKet::Ket,
                BraKet::Ket => BraKet::Bra,
            },
        }
    }

    fn reindex(&mut self, permutation: &[usize]) {
        let mut new_state = DenseBlock::zeros(1, self.state.n_indices());
        for original in self.state.iter_ones() {
            new_state.set_index(0, permutation[original], true);
        }
        self.state = new_state;
    }
}

impl Mul<Complex64> for ZBasisState {
    type Output = Self;

    fn mul(self, rhs: Complex64) -> Self::Output {
        ZBasisState {
            state: self.state,
            coefficient: self.coefficient * rhs,
            bra_ket: self.bra_ket,
        }
    }
}

impl Mul for ZBasisState {
    type Output = Result<Complex64, StateError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.bra_ket, rhs.bra_ket) {
            (BraKet::Bra, BraKet::Ket) => {
                if self.state == rhs.state {
                    Ok(self.coefficient * rhs.coefficient.conj())
                } else {
                    Ok(Complex64::ZERO)
                }
            }
            _ => Err(StateError::InvalidBraKet),
        }
    }
}

#[cfg(test)]
mod zbasis_tests {

    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_dimension() {
        let state = Array1::from_elem(3, false);
        let coefficient = Complex64::new(1., 2.);
        let zbasis_state = ZBasisState::new(state, coefficient);
        assert_eq!(zbasis_state.dimension(), 3);
    }

    #[test]
    fn test_adjoint() {
        let state_vec = arr1(&[true, false, true, false]);
        let coefficient = Complex64::new(1., 2.);
        let zbasis_state = ZBasisState::new(state_vec, coefficient);
        let adjoint_state = zbasis_state.adjoint();
        assert_eq!(
            adjoint_state.state_bools(),
            arr1(&[true, false, true, false])
        );
        assert_eq!(
            adjoint_state.coefficient,
            coefficient.conj() / coefficient.norm()
        );
        assert_eq!(adjoint_state.bra_ket, BraKet::Bra);
    }

    #[test]
    fn test_adjoint_roundtrip() {
        let state = Array1::from_elem(3, false);
        let coefficient = Complex64::new(1., 2.);
        let zbasis_state = ZBasisState::new(state, coefficient);
        let adjoint_state = zbasis_state.clone().adjoint();
        let roundtrip_state = adjoint_state.clone().adjoint();
        assert_eq!(roundtrip_state.state, zbasis_state.state);
        assert_eq!(roundtrip_state.coefficient, zbasis_state.coefficient);
    }

    #[test]
    fn test_zbasis_state() {
        let state = Array1::from_elem(3, false);
        let coefficient = Complex64::new(1., 0.);
        let zbasis_state = ZBasisState::new(state, coefficient);
        let adjoint_state = zbasis_state.clone().adjoint();
        assert_eq!(adjoint_state.state_bools(), Array1::from_elem(3, false));
        assert_eq!(adjoint_state.coefficient, Complex64::new(1., 0.));
    }

    #[test]
    fn test_normalize() {
        let mut zbasis_state =
            ZBasisState::new(Array1::from_elem(3, false), Complex64::new(2., 0.));
        zbasis_state.normalize();
        assert_eq!(zbasis_state.coefficient, Complex64::new(1., 0.));
    }

    #[test]
    fn test_reindex_roundtrip() {
        let state = arr1(&[false, true, false]);
        let coefficient = Complex64::new(1., 0.);
        let zbasis_state = ZBasisState::new(state, coefficient);
        let mut reindexed_state = zbasis_state.clone();
        reindexed_state.reindex(&[1, 2, 0]);
        reindexed_state.reindex(&[2, 0, 1]);
        assert_eq!(reindexed_state.state, zbasis_state.state);
        assert_eq!(reindexed_state.coefficient, zbasis_state.coefficient);
    }
}

/// A collection of quantum states in the computational (Pauli Z) basis.
///
/// Holds `n` states as rows of a boolean matrix, paired with complex coefficients.
/// Intended for efficient batch decoding via [`crate::encode::majorana::MajoranaEncoding::decode_zbasis_ensemble`].
#[derive(Debug, Clone)]
pub struct ZBasisEnsemble {
    /// Each entry is a Z-basis state, packed one bit per qubit into `u64` words.
    pub(crate) states: DenseBlock,
    /// Complex coefficient for each state.
    pub coefficients: Array1<Complex64>,
}

impl ZBasisEnsemble {
    /// Construct a [`ZBasisEnsemble`] from a dense states matrix and coefficient
    /// vector, packing each row into a [`DenseBlock`].
    ///
    /// # Panics
    ///
    /// Panics if `states.nrows() != coefficients.len()`.
    pub fn new(states: Array2<bool>, coefficients: Array1<Complex64>) -> Self {
        assert_eq!(
            states.nrows(),
            coefficients.len(),
            "states and coefficients must have the same length"
        );
        let states = states
            .rows()
            .into_iter()
            .map(DenseBlock::from_bool_view)
            .reduce(|mut acc, row| {
                acc.concat(row).unwrap();
                acc
            })
            .unwrap();
        Self {
            states,
            coefficients,
        }
    }
}

impl From<Vec<ZBasisState>> for ZBasisEnsemble {
    fn from(zbasis_states: Vec<ZBasisState>) -> Self {
        let n = zbasis_states.len();
        let n_indices = zbasis_states
            .iter()
            .map(|s| s.state.n_indices())
            .max()
            .unwrap_or(0);
        let mut states = DenseBlock::zeros(n, n_indices);
        let mut coefficients = Array1::zeros(n);
        for (i, s) in zbasis_states.into_iter().enumerate() {
            coefficients[i] = s.coefficient;
            states.set_term(i, s.state_block());
        }
        Self {
            states,
            coefficients,
        }
    }
}

/// A fermionic Fock (occupation number) state.
///
/// Represents an occupation-number state where each mode is either occupied (`true`)
/// or unoccupied (`false`), with an associated complex coefficient.
///
/// # Examples
///
/// ```
/// use ferrmion_core::states::FockState;
/// use ndarray::arr1;
/// use num_complex::Complex64;
///
/// let fs = FockState::new(arr1(&[true, false, true]), Complex64::new(1.0, 0.0));
/// assert_eq!(fs.state.len(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FockState {
    pub state: Array1<bool>,
    pub coefficient: Complex64,
    bra_ket: BraKet,
}

impl FockState {
    /// Construct a new [`FockState`] from an occupation array and a coefficient.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::states::FockState;
    /// use ndarray::arr1;
    /// use num_complex::Complex64;
    ///
    /// let fs = FockState::new(arr1(&[true, true, false]), Complex64::ONE);
    /// assert_eq!(fs.coefficient, Complex64::ONE);
    /// ```
    pub fn new(state: Array1<bool>, coefficient: Complex64) -> Self {
        Self {
            state,
            coefficient,
            bra_ket: BraKet::Ket,
        }
    }
}

impl Fermion for FockState {}

impl State for FockState {
    fn normalize(&mut self) {
        let norm = self.coefficient.norm();
        if norm != 0. {
            self.coefficient /= norm;
        }
    }
    fn dimension(&self) -> usize {
        self.state.len()
    }

    fn adjoint(self) -> Self {
        Self {
            state: self.state.slice(s![..;-1]).to_owned(),
            coefficient: self.coefficient.conj(),
            bra_ket: match self.bra_ket {
                BraKet::Ket => BraKet::Bra,
                BraKet::Bra => BraKet::Ket,
            },
        }
    }

    fn reindex(&mut self, permutation: &[usize]) {
        let mut new_state = Array1::from_elem(self.dimension(), false);
        for (original, &new) in permutation.iter().enumerate() {
            new_state[new] = self.state[original];
        }
        self.state = new_state;
    }
}

impl Mul<Complex64> for FockState {
    type Output = Self;

    fn mul(self, rhs: Complex64) -> Self::Output {
        Self {
            state: self.state,
            coefficient: self.coefficient * rhs,
            bra_ket: self.bra_ket,
        }
    }
}

#[cfg(test)]
mod fock_tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_new() {
        let state = Array1::from_elem(3, false);
        let coefficient = Complex64::new(1., 0.);
        let fock_state = FockState::new(state, coefficient);
        assert_eq!(fock_state.state, Array1::from_elem(3, false));
        assert_eq!(fock_state.coefficient, Complex64::new(1., 0.));
    }

    #[test]
    fn test_adjoint() {
        let state = arr1(&[true, true, false]);
        let coefficient = Complex64::new(1., 0.);
        let fock_state = FockState::new(state, coefficient);
        let adjoint_state = fock_state.clone();
        assert_eq!(adjoint_state.state, arr1(&[true, true, false]));
        assert_eq!(adjoint_state.coefficient, Complex64::new(1., 0.));
        assert_eq!(adjoint_state.bra_ket, BraKet::Ket);
    }

    #[test]
    fn test_reindex_roundtrip() {
        let state = arr1(&[false, true, false]);
        let coefficient = Complex64::new(1., 0.);
        let fock_state = FockState::new(state, coefficient);
        let mut reindexed_state = fock_state.clone();
        reindexed_state.reindex(&[1, 2, 0]);
        reindexed_state.reindex(&[2, 0, 1]);
        assert_eq!(reindexed_state.state, fock_state.state);
        assert_eq!(reindexed_state.coefficient, fock_state.coefficient);
    }
}
