//! Fermion-to-qubit encoding implementations.
//!
//! Provides the [`Encode`] and [`TryEncode`] traits, and the [`MajoranaEncoding`] struct
//! for transforming fermionic operators into qubit Hamiltonians via Majorana representations.
use crate::hamiltonians::QubitHamiltonian;
use crate::operators::{
    Block, CoefficientPauliWeight, FermionProduct, MajoranaProduct, MajoranaSparse, PauliWeight,
    SymplecticMatrix, SymplecticOperator,
};
use crate::spaces::{Fermion, Qubit};
use crate::states::{FockState, ZBasisEnsemble, ZBasisState};
use crate::utils::COEFFICIENT_TOLERANCE;
use crate::utils::{self, icount_to_sign};
use log::{debug, error};
use ndarray::{Array1, Array2, Axis};
use num_complex::{c64, Complex64};
use rayon::prelude::*;
use thiserror::Error;

/// Trait for encoding fermionic operators into qubit Hamiltonians.
///
/// Implementors define how a particular input type `T` (e.g. [`MajoranaProduct`],
/// [`MajoranaSparse`]) is transformed into a [`QubitHamiltonian`].
///
/// # Examples
///
/// ```
/// use ferrmion_core::encode::majorana::{Encode, MajoranaEncoding};
/// use ferrmion_core::operators::MajoranaProduct;
/// use ferrmion_core::encode::ternarytree::TernaryTree;
/// use ferrmion_core::hamiltonians::QubitHamiltonian;
/// use num_complex::Complex64;
///
/// let tree = TernaryTree::naive_jordan_wigner(2);
/// let encoding = tree.build_encoding(2).unwrap();
/// let mp = MajoranaProduct::new(vec![0, 1], Complex64::new(1.0, 0.0));
/// let QubitHamiltonian(qham) = encoding.encode(mp);
/// assert!(!qham.is_empty());
/// ```
pub trait Encode<In: Fermion, Out: Qubit> {
    /// Encodes the input into a QubitHamiltonian.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::majorana::Encode;
    /// // Example usage would depend on the implementor
    /// ```
    fn encode(&self, input: In) -> Out;
}

/// Fallible encoding trait for inputs that may not have a valid qubit representation.
///
/// # Examples
///
/// ```
/// use ferrmion_core::encode::majorana::{TryEncode, MajoranaEncoding};
/// use ferrmion_core::states::FockState;
/// use ferrmion_core::encode::ternarytree::TernaryTree;
/// use ndarray::arr1;
/// use num_complex::Complex64;
///
/// let tree = TernaryTree::naive_jordan_wigner(3);
/// let encoding = tree.build_encoding(3).unwrap();
/// let fock = FockState::new(arr1(&[true, false, false]), Complex64::ONE);
/// let result = encoding.try_encode(fock);
/// assert!(result.is_ok());
/// ```
pub trait TryEncode<In: Fermion, Out: Qubit> {
    /// Attempt to encode the input, returning an error if encoding is not possible.
    fn try_encode(&self, input: In) -> Result<Out, MajoranaEncodingError>;
}

/// A fermion-to-qubit encoding defined by its Majorana operator representations.
///
/// Stores the symplectic matrix of Majorana operators, the number of fermionic modes,
/// and the vacuum state used for Hartree-Fock state construction.
///
/// The [`SymplecticMatrix`] contains `2 * n_modes` rows. Consecutive pairs of rows
/// define a single fermionic operator: rows `2i` and `2i+1` correspond to the
/// two Majorana operators (γ₂ᵢ and γ₂ᵢ₊₁) that make up fermionic mode `i`.
#[derive(Debug, Clone, PartialEq)]
pub struct MajoranaEncoding {
    pub operators: SymplecticMatrix,
    pub vacuum_state: ZBasisState,
    pub n_modes: usize,
    pub n_qubits: usize,
}

/// Errors that can occur when constructing or using a [`MajoranaEncoding`].
#[derive(Debug, Error)]
pub enum MajoranaEncodingError {
    #[error("Cannot construct Hartree-Fock state with Pauli operators {0:?}-i{1:?}.")]
    HartreeFockError(char, char),
    #[error("Input operators are not a valid Majorana encoding.")]
    InvalidOperatorsError,
    #[error("Vacuum state {0:?} is not a valid vacuum for the given Majorana operators.")]
    InvalidVacuumStateError(Array1<bool>),
    #[error("Cannot determine a valid vacuum state from the given Majorana operators.")]
    NoVacuumStateError,
    #[error("Cannot encode Fock state {0:?} to a valid Z basis state.")]
    NullStateError(Array1<bool>),
}

impl MajoranaEncoding {
    /// Construct a new [`MajoranaEncoding`] from a [`SymplecticMatrix`], automatically
    /// determining the vacuum state via GF(2) constraint solving.
    ///
    /// Returns an error if the operators do not form a valid Majorana encoding or if
    /// no consistent vacuum state can be determined.
    ///
    /// To supply the vacuum state explicitly, use [`Self::with_vacuum`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::majorana::MajoranaEncoding;
    /// use ferrmion_core::operators::SymplecticMatrix;
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    ///
    /// let tree = TernaryTree::naive_jordan_wigner(2);
    /// let enc = MajoranaEncoding::new(tree.build_encoding(2).unwrap().operators).unwrap();
    /// assert_eq!(enc.n_modes, 2);
    /// assert_eq!(enc.n_qubits, 2);
    /// ```
    pub fn new(operators: SymplecticMatrix) -> Result<Self, MajoranaEncodingError> {
        let vacuum_state = Self::determine_vacuum_state(&operators)?;
        Self::with_vacuum(operators, vacuum_state)
    }

    /// Construct a [`MajoranaEncoding`] from a [`SymplecticMatrix`] and an explicit vacuum state.
    ///
    /// Returns an error if the operators do not form a valid Majorana encoding
    /// (mismatched X/Z block shapes or odd number of operator rows), or if the
    /// supplied vacuum state is not compatible.
    ///
    /// To have the vacuum state determined automatically, use [`Self::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::majorana::MajoranaEncoding;
    /// use ferrmion_core::operators::SymplecticMatrix;
    /// use ferrmion_core::states::ZBasisState;
    /// use ndarray::arr2;
    ///
    /// let sym = SymplecticMatrix::new(
    ///     arr2(&[[true, false], [true, false]]),
    ///     arr2(&[[false, false], [true, false]]),
    /// );
    /// let enc = MajoranaEncoding::with_vacuum(sym, ZBasisState::zeros(2)).unwrap();
    /// assert_eq!(enc.n_modes, 1);
    /// assert_eq!(enc.n_qubits, 2);
    /// ```
    pub fn with_vacuum(
        operators: SymplecticMatrix,
        vacuum_state: ZBasisState,
    ) -> Result<Self, MajoranaEncodingError> {
        // Shape
        Self::validate_operator_shape(&operators)?;
        // Overlap
        Self::validate_operator_overlap(&operators)?;
        // Linear independence
        Self::validate_linear_independence(&operators)?;

        let n_modes = operators.n_rows() / 2;
        let n_qubits = operators.n_qubits();
        let encoding = Self {
            operators,
            n_modes,
            n_qubits,
            vacuum_state,
        };
        // Vacuum state
        encoding.validate_vacuum_state()?;
        Ok(encoding)
    }

    /// Automatically determine the vacuum state for the given Majorana operators.
    ///
    /// Solves the GF(2) linear system arising from the requirement that each
    /// creation operator `a†_i = 0.5(γ_{2i} − iγ_{2i+1})` produces a valid
    /// (non-null) Z-basis state when applied to the vacuum.
    ///
    /// Returns [`MajoranaEncodingError::NoVacuumStateError`] if the operators
    /// have no consistent Z-basis vacuum state (e.g. mismatched X-block rows or
    /// contradictory phase constraints).
    pub fn determine_vacuum_state(
        operators: &SymplecticMatrix,
    ) -> Result<ZBasisState, MajoranaEncodingError> {
        let n_modes = operators.n_rows() / 2;
        let n_qubits = operators.n_qubits();

        // Build GF(2) augmented matrix [A | b] of shape [n_modes, n_qubits + 1].
        let mut mat: Vec<Vec<bool>> = Vec::with_capacity(n_modes);

        for i in 0..n_modes {
            let x0 = operators.row_x(2 * i);
            let x1 = operators.row_x(2 * i + 1);

            // Structural check: both γ operators for mode i must flip the same qubits,
            // otherwise no Z-basis state can serve as the vacuum for this mode.
            if x0 != x1 {
                return Err(MajoranaEncodingError::NoVacuumStateError);
            }

            let z0 = operators.row_z(2 * i);
            let z1 = operators.row_z(2 * i + 1);
            let ip0 = operators.ipower(2 * i) as i32;
            let ip1 = operators.ipower(2 * i + 1) as i32;

            // Vacuum condition from try_encode:
            //   γ_{2i}|v⟩ coefficient == -i * γ_{2i+1}|v⟩ coefficient
            // => i^(ip0 + 2*<z0,v>) = -i * i^(ip1 + 2*<z1,v>)
            // => 2*<(z0 XOR z1), v> ≡ ip1 − ip0 + 3  (mod 4)
            let diff = (ip1 - ip0 + 3).rem_euclid(4) as u8;
            // LHS is always even; if RHS is odd there is no solution.
            if !diff.is_multiple_of(2) {
                return Err(MajoranaEncodingError::NoVacuumStateError);
            }
            let b = (diff / 2) != 0;

            let constraint: Vec<bool> = (0..n_qubits).map(|q| z0.get(q) ^ z1.get(q)).collect();

            // Zero constraint with b=true is immediately inconsistent (0 ≠ 1).
            if constraint.iter().all(|&x| !x) && b {
                return Err(MajoranaEncodingError::NoVacuumStateError);
            }

            let mut row = constraint;
            row.push(b);
            mat.push(row);
        }

        // Gaussian elimination over GF(2) on the augmented matrix.
        let mut pivot_cols: Vec<Option<usize>> = vec![None; n_modes];
        let mut pivot_row = 0usize;

        for col in 0..n_qubits {
            if let Some(swap) = (pivot_row..n_modes).find(|&r| mat[r][col]) {
                mat.swap(swap, pivot_row);
                pivot_cols[pivot_row] = Some(col);
                let pivot_copy = mat[pivot_row].clone();
                for (r, row) in mat.iter_mut().enumerate() {
                    if r != pivot_row && row[col] {
                        for c in 0..=n_qubits {
                            row[c] ^= pivot_copy[c];
                        }
                    }
                }
                pivot_row += 1;
            }
        }

        // Check for inconsistent rows of the form [0...0 | 1].
        for row in &mat {
            if row[..n_qubits].iter().all(|&x| !x) && row[n_qubits] {
                return Err(MajoranaEncodingError::NoVacuumStateError);
            }
        }

        // Extract solution; free variables default to false (giving |000...0⟩ for
        // all standard encodings: JW, Parity, Bravyi-Kitaev, JKMN).
        let mut solution = vec![false; n_qubits];
        for (r, &pivot_col) in pivot_cols.iter().enumerate() {
            if let Some(col) = pivot_col {
                solution[col] = mat[r][n_qubits];
            }
        }

        Ok(ZBasisState::new(Array1::from(solution), Complex64::ONE))
    }

    fn validate_operator_shape(operators: &SymplecticMatrix) -> Result<(), MajoranaEncodingError> {
        if operators.x_bools().dim() != operators.z_bools().dim() {
            return Err(MajoranaEncodingError::InvalidOperatorsError);
        }
        if !operators.n_rows().is_multiple_of(2) {
            return Err(MajoranaEncodingError::InvalidOperatorsError);
        }
        Ok(())
    }
    fn validate_operator_overlap(
        operators: &SymplecticMatrix,
    ) -> Result<(), MajoranaEncodingError> {
        // Check all distinct pairs anticommute via the symplectic inner product.
        // Two Pauli operators anticommute iff Σ_q (x_i[q]·z_j[q] ⊕ z_i[q]·x_j[q]) is odd.
        let n_ops = operators.n_rows();
        for i in 0..n_ops {
            for j in i + 1..n_ops {
                // Symplectic inner product ⟨op_i, op_j⟩ = |x_i & z_j| + |z_i & x_j| (mod 2).
                let inner_product = operators.row_x(i).and_count_ones(operators.row_z(j))
                    + operators.row_z(i).and_count_ones(operators.row_x(j));
                if inner_product.is_multiple_of(2) {
                    return Err(MajoranaEncodingError::InvalidOperatorsError);
                }
            }
        }
        Ok(())
    }

    fn validate_linear_independence(
        operators: &SymplecticMatrix,
    ) -> Result<(), MajoranaEncodingError> {
        // Perform Gaussian elimination on the concatenated [x | z] symplectic matrix.
        let mut mat = operators.to_concatenated();
        let n_rows = mat.len_of(Axis(0));
        let n_cols = mat.len_of(Axis(1));
        let mut pivot_row = 0;
        for col in 0..n_cols {
            if let Some(swap_row) = (pivot_row..n_rows).find(|&r| mat[[r, col]]) {
                if swap_row != pivot_row {
                    for c in 0..n_cols {
                        let tmp = mat[[pivot_row, c]];
                        mat[[pivot_row, c]] = mat[[swap_row, c]];
                        mat[[swap_row, c]] = tmp;
                    }
                }
                for r in 0..n_rows {
                    if r != pivot_row && mat[[r, col]] {
                        for c in 0..n_cols {
                            mat[[r, c]] ^= mat[[pivot_row, c]];
                        }
                    }
                }
                pivot_row += 1;
            }
        }
        if pivot_row < n_rows {
            return Err(MajoranaEncodingError::InvalidOperatorsError);
        }
        Ok(())
    }

    fn validate_vacuum_state(&self) -> Result<(), MajoranaEncodingError> {
        if self.vacuum_state.state.len() != self.n_qubits {
            error!(
                "Vacuum state length {0:?} not same as n_qubits {1:?}",
                self.vacuum_state.state, self.n_qubits,
            );
            return Err(MajoranaEncodingError::InvalidVacuumStateError(
                self.vacuum_state.state.clone(),
            ));
        }
        // Check each singly-occupied FockState can be encoded ( a_i†|Ω⟩ is well-defined).
        for i in 0..self.n_modes {
            let mut occ = Array1::from_elem(self.n_modes, false);
            occ[i] = true;
            let _: ZBasisState = self
                .try_encode(FockState::new(occ, Complex64::ONE))
                .map_err(|_| {
                    MajoranaEncodingError::InvalidVacuumStateError(self.vacuum_state.state.clone())
                })?;
        }
        // Check the fully-occupied FockState can also be encoded.
        let all_occ = Array1::from_elem(self.n_modes, true);
        let _: ZBasisState = self
            .try_encode(FockState::new(all_occ, Complex64::ONE))
            .map_err(|_| {
                MajoranaEncodingError::InvalidVacuumStateError(self.vacuum_state.state.clone())
            })?;
        Ok(())
    }
}

impl MajoranaEncoding {
    /// Reorder the fermionic modes according to the given permutation.
    ///
    /// Returns a new [`MajoranaEncoding`] with Majorana operator rows reordered
    /// so that mode `i` maps to the operator pair at `mode_op_map[i]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::majorana::MajoranaEncoding;
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    ///
    /// let tree = TernaryTree::naive_jordan_wigner(3);
    /// let encoding = tree.build_encoding(3).unwrap();
    /// let swapped = encoding.apply_mode_enumeration(vec![2, 0, 1]);
    /// assert_eq!(swapped.n_modes, 3);
    /// ```
    pub fn apply_mode_enumeration(&self, mode_op_map: Vec<usize>) -> MajoranaEncoding {
        assert_eq!(
            2 * mode_op_map.len(),
            self.operators.n_rows(),
            "{}",
            format_args!(
                "Mode op map not same length as ipowers {0:?}",
                self.operators.n_rows()
            )
        );
        let majorana_rows: Vec<usize> = mode_op_map
            .iter()
            .flat_map(|v| [2 * v, 2 * v + 1])
            .collect();

        MajoranaEncoding::with_vacuum(
            self.operators.select_rows(&majorana_rows),
            self.vacuum_state.clone(),
        )
        .expect("Reindexing a valid encoding should never fail.")
    }

    /// Encode a fermionic Hamiltonian with multiple mode permutations and return
    /// both the plain and coefficient-weighted Pauli weights for each.
    ///
    /// Each permutation is encoded independently in parallel using Rayon.
    /// This is significantly faster than calling [`Encode::encode`] and the
    /// weight traits separately for each permutation from Python.
    ///
    /// # Arguments
    ///
    /// * `msparse` - The fermionic Hamiltonian in Majorana sparse form.
    /// * `permutations` - Slice of mode-to-operator-pair permutations.  Each
    ///   inner `Vec<usize>` must have length `n_modes` and contain a permutation
    ///   of `0..n_modes`.
    ///
    /// # Returns
    ///
    /// A pair `(plain, weighted)` where both are `Vec<f64>` of length equal to
    /// the number of input permutations.  `plain[i]` is the plain Pauli weight
    /// and `weighted[i]` is the coefficient-weighted Pauli weight for permutation
    /// `i`, in the same order as the input.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::majorana::MajoranaEncoding;
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// use ferrmion_core::operators::MajoranaSparse;
    /// use num_complex::Complex64;
    /// use tinyvec::array_vec;
    ///
    /// let tree = TernaryTree::naive_jordan_wigner(2);
    /// let encoding = tree.build_encoding(2).unwrap();
    /// let ms = MajoranaSparse::new(
    ///     vec![array_vec!([u16; 7] => 0, 1)],
    ///     vec![Complex64::new(1.0, 0.)],
    ///     0.,
    /// ).unwrap();
    /// let (plain, weighted) = encoding.batch_pauli_weights(&ms, &[vec![0, 1], vec![1, 0]]);
    /// assert_eq!(plain.len(), 2);
    /// assert_eq!(weighted.len(), 2);
    /// ```
    pub fn batch_pauli_weights(
        &self,
        msparse: &MajoranaSparse,
        permutations: &[Vec<usize>],
    ) -> (Vec<f64>, Vec<f64>) {
        permutations
            .par_iter()
            .map(|perm| {
                let enumerated_encoding = self.apply_mode_enumeration(perm.clone());
                let qham: QubitHamiltonian = enumerated_encoding.encode(msparse);
                (qham.pauli_weight() as f64, qham.coeff_pauli_weight())
            })
            .unzip()
    }
}

impl Encode<MajoranaProduct, QubitHamiltonian> for MajoranaEncoding {
    fn encode(&self, input: MajoranaProduct) -> QubitHamiltonian {
        let QubitHamiltonian(mut qham) = QubitHamiltonian::default();
        let operator = input
            .indices
            .iter()
            .fold(SymplecticOperator::identity(self.n_qubits), |acc, &ind| {
                acc * self.operators.view_row(ind)
            });
        debug!("{:#?}", operator);
        debug!("{:#?}", &operator.to_pauli_string());
        let (pauli, ipower) = operator.to_pauli_string();

        qham.insert(
            pauli,
            utils::icount_to_sign(ipower as usize) * input.coefficient,
        );

        QubitHamiltonian(qham)
    }
}

impl Encode<&MajoranaSparse, QubitHamiltonian> for MajoranaEncoding {
    fn encode(&self, input: &MajoranaSparse) -> QubitHamiltonian {
        let QubitHamiltonian(mut qham) = QubitHamiltonian::default();
        let paulis_ipowers: Vec<(String, u8)> = input
            .indices
            .par_iter()
            .map(|&indices| {
                // Use in-place multiplication to avoid heap allocations per multiply.
                // Each mul_assign_view reuses the accumulator's arrays instead of
                // allocating 2 new Array1<bool> per call.
                let mut operator = SymplecticOperator::identity(self.n_qubits);
                for &ind in indices.iter() {
                    let row = self.operators.view_row(ind as usize);
                    operator.mul_assign_view(&row);
                }
                debug!("Operator {:?}", operator);

                operator.to_pauli_string()
            })
            .collect();

        for (pauli, coef) in paulis_ipowers.into_iter().zip(&input.coefficients) {
            *qham.entry(pauli.0).or_insert(Complex64::new(0., 0.)) +=
                coef * icount_to_sign(pauli.1 as usize);
            debug!("Total Ipower {:?}", icount_to_sign(pauli.1 as usize));
        }

        *qham
            .entry(
                (0..self.n_qubits)
                    .map(|_| "I".to_string())
                    .collect::<String>(),
            )
            .or_insert(c64(0., 0.)) += input.constant;
        QubitHamiltonian(
            qham.into_iter()
                .filter(|(_, v)| v.norm() > COEFFICIENT_TOLERANCE)
                .collect(),
        )
    }
}

impl Encode<FermionProduct, QubitHamiltonian> for MajoranaEncoding {
    fn encode(&self, input: FermionProduct) -> QubitHamiltonian {
        let msparse = MajoranaSparse::from(input);
        self.encode(&msparse)
    }
}

impl MajoranaEncoding {
    /// Decode a [`ZBasisState`] into the [`FockState`] that encodes to it.
    ///
    /// Applies the annihilation operators `aᵢ = 0.5*(γ_{2i} + i·γ_{2i+1})` in
    /// reverse mode order, identifying occupied modes as those where the
    /// annihilation coefficient is non-zero.  The returned [`FockState`] has
    /// coefficient [`Complex64::ONE`] (global phase is ignored).
    ///
    /// Returns `None` if the input does not correspond to any valid [`FockState`]
    /// in this encoding (the state cannot be wound back to the vacuum).
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::majorana::{MajoranaEncoding, TryEncode};
    /// use ferrmion_core::states::FockState;
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// use ndarray::arr1;
    /// use num_complex::Complex64;
    ///
    /// let tree = TernaryTree::naive_jordan_wigner(3);
    /// let encoding = tree.build_encoding(3).unwrap();
    /// let fock = FockState::new(arr1(&[true, false, true]), Complex64::ONE);
    /// let encoded = encoding.try_encode(fock.clone()).unwrap();
    /// let decoded = encoding.decode_zbasis_state(encoded).unwrap();
    /// assert_eq!(decoded.state, fock.state);
    /// ```
    pub fn decode_zbasis_state(&self, input: ZBasisState) -> Option<FockState> {
        let mut current = input;
        let mut occupation = vec![false; self.n_modes];

        for i in (0..self.n_modes).rev() {
            let left = self.operators.view_row(2 * i) * current.clone();
            let right = self.operators.view_row(2 * i + 1) * current.clone();

            if left.state != right.state {
                return None;
            }

            let ann_coeff = left.coefficient + Complex64::new(0., 1.) * right.coefficient;

            if ann_coeff.norm() > COEFFICIENT_TOLERANCE {
                occupation[i] = true;
                current = ZBasisState::new(left.state, ann_coeff);
            }
        }

        if current.state != self.vacuum_state.state {
            return None;
        }

        Some(FockState::new(Array1::from(occupation), Complex64::ONE))
    }

    /// Decode all states in a [`ZBasisEnsemble`] into [`FockState`]s.
    ///
    /// Applies each fermionic annihilation operator across all rows of the ensemble
    /// before advancing to the next mode, avoiding per-state array clones.
    /// Input coefficients on the ensemble states are ignored; all are treated as
    /// [`Complex64::ONE`].
    ///
    /// Returns `None` for any row that does not correspond to a valid encoded state.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::encode::majorana::{MajoranaEncoding, TryEncode};
    /// use ferrmion_core::states::{FockState, ZBasisEnsemble};
    /// use ferrmion_core::encode::ternarytree::TernaryTree;
    /// use ndarray::Array1;
    /// use num_complex::Complex64;
    ///
    /// let tree = TernaryTree::naive_jordan_wigner(3);
    /// let encoding = tree.build_encoding(3).unwrap();
    /// let fock = FockState::new(Array1::from(vec![true, false, true]), Complex64::ONE);
    /// let encoded = encoding.try_encode(fock.clone()).unwrap();
    /// let ensemble = ZBasisEnsemble::from(vec![encoded]);
    /// let decoded = encoding.decode_zbasis_ensemble(&ensemble);
    /// assert_eq!(decoded[0].as_ref().unwrap().state, fock.state);
    /// ```
    pub fn decode_zbasis_ensemble(&self, ensemble: &ZBasisEnsemble) -> Vec<Option<FockState>> {
        let n_states = ensemble.states.nrows();

        // Working state as one bitpacked `Block` per state, mutated in-place
        // across modes. Coefficients start at ONE (input ignored). Packing once
        // up front lets the parity and state-update steps run as word-level bit
        // ops, which stays fast even for dense operators (Jordan-Wigner / parity
        // Z-strings span every qubit) instead of iterating each set bit.
        let mut state_blocks: Vec<Block> = ensemble
            .states
            .rows()
            .into_iter()
            .map(Block::from_bool_view)
            .collect();
        let mut current_coeffs = Array1::from_elem(n_states, Complex64::ONE);
        let mut occupations = Array2::<bool>::default((n_states, self.n_modes));

        for i in (0..self.n_modes).rev() {
            let x_l = self.operators.row_x(2 * i);
            let z_l = self.operators.row_z(2 * i);
            let ip_l = self.operators.ipower(2 * i);
            let x_r = self.operators.row_x(2 * i + 1);
            let z_r = self.operators.row_z(2 * i + 1);
            let ip_r = self.operators.ipower(2 * i + 1);

            // Validity: left and right new states must agree; this is an encoding
            // property (x_l == x_r), not per-state.
            if x_l != x_r {
                return vec![None; n_states];
            }

            // Precompute the two possible phases for each operator (parity even/odd).
            let phase_l = [
                c64(0., 1.).powi(ip_l as i32 % 4),
                c64(0., 1.).powi((ip_l as i32 + 2) % 4),
            ];
            let phase_r = [
                c64(0., 1.).powi(ip_r as i32 % 4),
                c64(0., 1.).powi((ip_r as i32 + 2) % 4),
            ];

            // Compute annihilation coefficients for all states in parallel.
            // Parity is a word-level popcount of `z & state`.
            let ann_coeffs: Vec<Complex64> = (0..n_states)
                .into_par_iter()
                .map(|j| {
                    let state = state_blocks[j].as_ref();
                    let coeff = current_coeffs[j];
                    let par_l = z_l.and_count_ones(state) % 2;
                    let par_r = z_r.and_count_ones(state) % 2;
                    let lc = coeff * phase_l[par_l];
                    let rc = coeff * phase_r[par_r];
                    lc + Complex64::new(0., 1.) * rc
                })
                .collect();

            // Determine which states have mode i occupied.
            let occupied: Vec<bool> = ann_coeffs
                .iter()
                .map(|c: &Complex64| c.norm() > COEFFICIENT_TOLERANCE)
                .collect();

            // Update occupations, accumulated coefficients, and — for occupied
            // states — the state itself (XOR with x_l, word-level).
            for (j, (&occ, &ann)) in occupied.iter().zip(ann_coeffs.iter()).enumerate() {
                if occ {
                    occupations[[j, i]] = true;
                    current_coeffs[j] = ann;
                    state_blocks[j].xor_assign(x_l);
                }
            }
        }

        // Validate final state matches encoding vacuum.
        let vacuum_block = Block::from_bool_view(self.vacuum_state.state.view());
        (0..n_states)
            .map(|j| {
                if state_blocks[j] == vacuum_block {
                    Some(FockState::new(
                        occupations.row(j).to_owned(),
                        Complex64::ONE,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Attempt to encode a [`FockState`] into a [`ZBasisState`] using the Majorana encoding.
/// Only returns a result if the encoding is able to map a single [`FockState`] to a single [`ZBasisState`]
/// so only number-preserving encodings will work.
///
/// Note this does not prepare a slater determinant, but rather a single [`ZBasisState`].
/// This function can therefore be used to convert from a [`MajoranaEncoding`] to an
/// encoding as a linear combination of fock states.
impl TryEncode<FockState, ZBasisState> for MajoranaEncoding {
    fn try_encode(&self, input: FockState) -> Result<ZBasisState, MajoranaEncodingError> {
        debug!("\nFock state: {input:?}");
        // Working state as a bitpacked `Block`, mutated across occupied modes, so
        // each operator's parity is a word-level popcount of `z & state` rather
        // than iterating every set bit of a (possibly dense) Z-string.
        let mut state_block = Block::from_bool_view(self.vacuum_state.state.view());
        let mut coefficient = self.vacuum_state.coefficient;

        // Applying in reverse order ensures JW maps to all +1 states
        // https://arxiv.org/abs/2412.07578v1
        for idx in (0..input.state.len()).rev() {
            if !input.state[idx] {
                continue;
            }
            let x_l = self.operators.row_x(2 * idx);
            let z_l = self.operators.row_z(2 * idx);
            let ip_l = self.operators.ipower(2 * idx);
            let x_r = self.operators.row_x(2 * idx + 1);
            let z_r = self.operators.row_z(2 * idx + 1);
            let ip_r = self.operators.ipower(2 * idx + 1);

            // Both Majorana operators of a mode flip the same qubits (x_l == x_r);
            // otherwise the two candidate states disagree and there is no Z-basis image.
            if x_l != x_r {
                return Err(MajoranaEncodingError::NullStateError(input.state));
            }

            // Phase applied by each operator: i^(ipower + 2·⟨z, state⟩). These
            // factors are unit-norm, so multiplying the (unit-norm) running
            // coefficient needs no rescale.
            let left_coefficient = coefficient
                * c64(0., 1.).powi(
                    ((ip_l as usize + 2 * z_l.and_count_ones(state_block.as_ref())) % 4) as i32,
                );
            let right_coefficient = coefficient
                * c64(0., 1.).powi(
                    ((ip_r as usize + 2 * z_r.and_count_ones(state_block.as_ref())) % 4) as i32,
                );

            // Apply the shared X flip to the working state (word-level XOR).
            state_block.xor_assign(x_l);

            if left_coefficient == Complex64::new(0., -1.) * right_coefficient {
                // Real-eigenvalued encodings.
                coefficient = left_coefficient;
            } else if left_coefficient == Complex64::new(0., 1.) * right_coefficient {
                // Coeffs cancel, null state.
                error!("Fock state encoded to Null state in Z basis.");
                return Err(MajoranaEncodingError::NullStateError(input.state));
            } else {
                // Coeffs don't cancel; renormalise to unit norm (matching
                // `ZBasisState::new`, whose `combined` here is always non-zero).
                let combined = left_coefficient - Complex64::new(0., -1.) * right_coefficient;
                coefficient = combined / combined.norm();
            }
        }
        Ok(ZBasisState::new(state_block.to_bool_array(), coefficient))
    }
}

#[cfg(test)]
mod owned_tests {
    use super::*;
    use crate::encode::ternarytree::TernaryTree;
    use crate::operators::LadderOperator;
    use crate::states::{State, ZBasisEnsemble};
    use ndarray::{arr1, Array1};
    use num_complex::c64;
    use num_complex::Complex64;
    use tinyvec::array_vec;

    #[test]
    fn test_encode_fermion_product() {
        let fprod = FermionProduct::new(
            vec![LadderOperator::Creation, LadderOperator::Annihilation],
            vec![1, 0],
            c64(1., 0.),
        )
        .unwrap();
        let encoding = TernaryTree::naive_jordan_wigner(2)
            .build_encoding(2)
            .unwrap();

        let qham = encoding.encode(fprod);

        let QubitHamiltonian(mut expected) = QubitHamiltonian::default();
        expected.insert("YX".to_string(), c64(0., 0.25));
        expected.insert("XY".to_string(), c64(0., -0.25));
        expected.insert("XX".to_string(), c64(0.25, 0.));
        expected.insert("YY".to_string(), c64(0.25, 0.));
        assert_eq!(qham, QubitHamiltonian(expected));
    }

    #[test]
    fn test_encode_majorana_product() {
        // JW encoding for 2 modes on 3 qubits: γ₀=XII, γ₁=YII, γ₂=ZXI, γ₃=ZYI
        let x_block = ndarray::arr2(&[
            [true, false, false],
            [true, false, false],
            [false, true, false],
            [false, true, false],
        ]);
        let z_block = ndarray::arr2(&[
            [false, false, false],
            [true, false, false],
            [true, false, false],
            [true, true, false],
        ]);
        let n_qubits = x_block.len_of(Axis(1));
        let sym = SymplecticMatrix::new(x_block, z_block);
        let encoding: MajoranaEncoding =
            MajoranaEncoding::with_vacuum(sym, ZBasisState::zeros(n_qubits)).unwrap();

        let mp = MajoranaProduct::new(vec![0], Complex64::new(1.0, 0.));
        let QubitHamiltonian(qham) = encoding.encode(mp);
        assert_eq!(qham.get("XII").unwrap(), &Complex64::new(1., 0.));

        let mp = MajoranaProduct::new(vec![0, 0], Complex64::new(1.0, 0.));
        let QubitHamiltonian(qham) = encoding.encode(mp);
        assert_eq!(qham.get("III").unwrap(), &Complex64::new(1.0, 0.));

        let mp = MajoranaProduct::new(vec![1, 1], Complex64::new(1.0, 0.));
        let QubitHamiltonian(qham) = encoding.encode(mp);
        assert_eq!(qham.get("III").unwrap(), &Complex64::new(1.0, 0.));

        let mp = MajoranaProduct::new(vec![2, 3], Complex64::new(1.0, 0.));
        let QubitHamiltonian(qham) = encoding.encode(mp);
        assert_eq!(qham.get("IZI").unwrap(), &Complex64::new(0., 1.));

        let mp = MajoranaProduct::new(vec![3, 2], Complex64::new(1.0, 0.));
        let QubitHamiltonian(qham) = encoding.encode(mp);
        assert_eq!(qham.get("IZI").unwrap(), &Complex64::new(0., -1.));

        let mp = MajoranaProduct::new(vec![3, 2, 2, 2], Complex64::new(1.0, 0.));
        let QubitHamiltonian(qham) = encoding.encode(mp);
        assert_eq!(qham.get("IZI").unwrap(), &Complex64::new(0., -1.));
    }

    #[test]
    fn test_encode_sparse_xz() {
        // γ₀=XII, γ₁=YII — 1-mode JW on 3 qubits; vacuum |000⟩ is valid
        let x_block = ndarray::arr2(&[[true, false, false], [true, false, false]]);
        let z_block = ndarray::arr2(&[[false, false, false], [true, false, false]]);
        let encoding: MajoranaEncoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::new(x_block, z_block),
            ZBasisState::zeros(3),
        )
        .unwrap();
        let ms = MajoranaSparse::new(
            vec![array_vec!([u16; 7] =>0, 1), array_vec!([u16; 7] =>1,0)],
            vec![Complex64::new(1.0, 0.), Complex64::new(1.0, 0.)],
            0.,
        )
        .unwrap();
        let _qham = encoding.encode(&ms);
    }

    #[test]
    fn test_encode_sparse_iy() {
        // γ₀=XII, γ₁=YII — 1-mode JW (swapped) on 3 qubits; vacuum |000⟩ is valid
        let x_block = ndarray::arr2(&[[true, false, false], [true, false, false]]);
        let z_block = ndarray::arr2(&[[false, false, false], [true, false, false]]);
        let encoding: MajoranaEncoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::new(x_block, z_block),
            ZBasisState::zeros(3),
        )
        .unwrap();
        let ms = MajoranaSparse::new(
            vec![array_vec!([u16; 7] =>0, 1), array_vec!([u16; 7] =>1,0)],
            vec![Complex64::new(1.0, 0.), Complex64::new(-1.0, 0.)],
            0.,
        )
        .unwrap();
        debug!("{:#?}", ms);
        let qham = encoding.encode(&ms);
        debug!("{:#?}", qham);
    }
    #[test]
    fn test_encode_sparse_long() {
        // JW encoding for 2 modes on 3 qubits: γ₀=XII, γ₁=YII, γ₂=ZXI, γ₃=ZYI
        let x_block = ndarray::arr2(&[
            [true, false, false],
            [true, false, false],
            [false, true, false],
            [false, true, false],
        ]);
        let z_block = ndarray::arr2(&[
            [false, false, false],
            [true, false, false],
            [true, false, false],
            [true, true, false],
        ]);
        let encoding: MajoranaEncoding = MajoranaEncoding::with_vacuum(
            SymplecticMatrix::new(x_block, z_block),
            ZBasisState::zeros(3),
        )
        .unwrap();
        let ms = MajoranaSparse::new(
            vec![
                array_vec!([u16; 7] =>0,0),
                array_vec!([u16; 7] =>1,1),
                array_vec!([u16; 7] =>2,3),
                array_vec!([u16; 7] =>3,2),
            ],
            vec![
                Complex64::new(1.0, 0.),
                Complex64::new(1.0, 0.),
                Complex64::new(1.0, 0.),
                Complex64::new(1.0, 0.),
            ],
            0.,
        )
        .unwrap();
        debug!("{:#?}", ms);
        let QubitHamiltonian(qham) = encoding.encode(&ms);
        debug!("{:#?}", qham);
        // γ₀²=I and γ₁²=I both contribute III with coeff 1 → total 2
        assert_eq!(qham.get("III").unwrap(), &Complex64::new(2., 0.));
        // γ₂γ₃ gives IZI with coeff i, γ₃γ₂ gives IZI with coeff -i → cancel to zero (filtered out)
        assert!(!qham.contains_key("IZI"));
    }

    #[test]
    fn test_encode_fock() {
        let fermionic_hf_state: Array1<bool> =
            Array1::from(vec![true, true, true, false, false, false]);
        let fockstate = FockState::new(fermionic_hf_state, Complex64::ONE);

        let tree = TernaryTree::naive_jordan_wigner(6);
        let encoding: MajoranaEncoding = tree.build_encoding(6).unwrap();
        let result = encoding.try_encode(fockstate);
        assert!(matches!(result, Ok(_)));
        assert!(result.unwrap().state == arr1(&[true, true, true, false, false, false]));
    }

    #[test]
    fn test_hartree_fock() {
        let tree = TernaryTree::naive_jordan_wigner(6);
        let encoding: MajoranaEncoding = tree.build_encoding(6).unwrap();

        let state1 = Array1::from(vec![true, true, true, false, false, false]);
        let result = encoding
            .try_encode(FockState::new(state1, Complex64::ONE))
            .unwrap();
        assert!(result.state == arr1(&[true, true, true, false, false, false]));

        let state2 = Array1::from(vec![true, true, true, true, false, false]);
        let result2 = encoding
            .try_encode(FockState::new(state2, Complex64::ONE))
            .unwrap();
        assert!(result2.state == arr1(&[true, true, true, true, false, false]));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_naive_jw_hf_state_unchanged(hf_state in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
            let n = hf_state.len();
            let tree = TernaryTree::naive_jordan_wigner(n);
            let encoding = tree.build_encoding(n).unwrap();
            let qubit_hf = encoding.try_encode(FockState::new(Array1::from(hf_state.clone()), Complex64::ONE)).unwrap().state;
            let expected: Array1<bool> = hf_state.into_iter().collect();
            prop_assert_eq!(qubit_hf, expected);
        }

        #[test]
        fn test_naive_parity_hf_state_is_occupation_parity(hf_state in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
            let n = hf_state.len();
            let tree = TernaryTree::naive_parity(n);
            let encoding = tree.build_encoding(n).unwrap();
            let qubit_hf = encoding.try_encode(FockState::new(Array1::from(hf_state.clone()), Complex64::ONE)).unwrap().state;
            // expected_parity = cumsum(reversed) % 2, then reverse back
            let mut reversed: Vec<bool> = hf_state.into_iter().rev().collect();
            let mut cumsum: usize = 0;
            for rev in reversed.iter_mut() {
                if *rev { cumsum += 1; }
                *rev = !cumsum.is_multiple_of(2);
            }
            let expected: Array1<bool> = reversed.into_iter().rev().collect();
            prop_assert_eq!(qubit_hf, expected);
        }

        #[test]
        fn test_enumerated_jw_hf_state_match_reordered_naive(mode_op_map in proptest::sample::subsequence((0..10).collect::<Vec<usize>>(), 10), n_electrons in 1..11usize) {
            let n = 10;
            let mut hf_state: Vec<bool> = vec![true; n_electrons];
            hf_state.extend(vec![false; n - n_electrons]);
            let tree = TernaryTree::naive_jordan_wigner(n);
            let encoding = tree.build_encoding(n).unwrap();
            let naive_qubit_hf = encoding.try_encode(FockState::new(Array1::from(hf_state.clone()), Complex64::ONE)).unwrap().state;
            prop_assert_eq!(naive_qubit_hf, Array1::from(hf_state.clone()));
            let mut enumerated_fockstate = FockState::new(Array1::from(hf_state.clone()), Complex64::ONE);
            enumerated_fockstate.reindex(&mode_op_map);
            let enumerated_qubit_hf = encoding.try_encode(enumerated_fockstate).unwrap().state;
            let mut expected = vec![false; n];
            for &i in &mode_op_map[..n_electrons] {
                if i < n {
                    expected[i] = true;
                }
            }
            prop_assert_eq!(enumerated_qubit_hf, Array1::from(expected));
        }
        #[test]
        fn test_naive_jw_tt_hf_state_unchanged(hf_state in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
            let n = hf_state.len();
            let tree = TernaryTree::naive_jordan_wigner(n);
            let encoding = tree.build_encoding(n).unwrap();
            let qubit_hf = encoding.try_encode(FockState::new(Array1::from(hf_state.clone()), Complex64::ONE)).unwrap().state;
            let expected: Array1<bool> = hf_state.into_iter().collect();
            prop_assert_eq!(qubit_hf, expected);
        }

        #[test]
        fn test_naive_parity_tt_hf_state_is_occupation_parity(hf_state in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
            let n = hf_state.len();
            let tree = TernaryTree::naive_parity(n);
            let encoding = tree.build_encoding(n).unwrap();
            let qubit_hf = encoding.try_encode(FockState::new(Array1::from(hf_state.clone()), Complex64::ONE)).unwrap().state;
            // expected_parity = cumsum(reversed) % 2, then reverse back
            let mut reversed: Vec<bool> = hf_state.into_iter().rev().collect();
            let mut cumsum:usize = 0;
            for rev in reversed.iter_mut() {
                if *rev { cumsum += 1; }
                *rev = !cumsum.is_multiple_of(2);
            }
            let expected: Array1<bool> = reversed.into_iter().rev().collect();
            prop_assert_eq!(qubit_hf, expected);
        }

        #[test]
        fn test_enumerated_jw_tt_hf_state_match_reordered_naive(mode_op_map in proptest::sample::subsequence((0..10).collect::<Vec<usize>>(), 10), n_electrons in 1..11usize) {
            let n = 10;
            let mut hf_state: Vec<bool> = vec![true; n_electrons];
            hf_state.extend(vec![false; n - n_electrons]);
            let tree = TernaryTree::naive_jordan_wigner(n);
            let encoding = tree.build_encoding(n).unwrap();
            let naive_qubit_hf = encoding.try_encode(FockState::new(Array1::from(hf_state.clone()), Complex64::ONE)).unwrap().state;
            prop_assert_eq!(naive_qubit_hf, Array1::from(hf_state.clone()));
            let mut enumerated_fockstate = FockState::new(Array1::from(hf_state.clone()), Complex64::ONE);
            enumerated_fockstate.reindex(&mode_op_map);
            let enumerated_qubit_hf = encoding.try_encode(enumerated_fockstate).unwrap().state;
            let mut expected = vec![false; n];
            for &i in &mode_op_map[..n_electrons] {
                if i < n {
                    expected[i] = true;
                }
            }
            prop_assert_eq!(enumerated_qubit_hf, Array1::from(expected));
        }
    }

    proptest! {
        #[test]
        fn test_all_standard_encodings_vacuum_is_all_false(
            n in 1..=8usize,
            encoding_idx in 0usize..4,
        ) {
            let tree = match encoding_idx {
                0 => TernaryTree::naive_jordan_wigner(n),
                1 => TernaryTree::naive_parity(n),
                2 => TernaryTree::naive_bravyi_kitaev(n),
                _ => TernaryTree::naive_jkmn(n),
            };
            let enc = tree.build_encoding(n).expect("valid encoding");
            let vacuum = MajoranaEncoding::determine_vacuum_state(&enc.operators)
                .expect("should determine vacuum state");
            prop_assert!(vacuum.state.iter().all(|&b| !b));
        }
    }

    #[test]
    fn test_determine_vacuum_state_jw() {
        let tree = TernaryTree::naive_jordan_wigner(2);
        let enc = tree.build_encoding(2).unwrap();
        let vacuum = MajoranaEncoding::determine_vacuum_state(&enc.operators).unwrap();
        assert!(vacuum.state.iter().all(|&b| !b));
    }

    #[test]
    fn test_from_operators_matches_build_encoding() {
        let tree = TernaryTree::naive_jordan_wigner(3);
        let enc = tree.build_encoding(3).unwrap();
        let enc2 = MajoranaEncoding::new(enc.operators.clone()).unwrap();
        assert_eq!(enc.vacuum_state.state, enc2.vacuum_state.state);
    }

    #[test]
    fn test_determine_vacuum_state_mismatched_x_blocks_rejected() {
        let x_block = ndarray::arr2(&[[true, false], [false, true]]);
        let z_block = ndarray::arr2(&[[false, false], [false, false]]);
        let sym = SymplecticMatrix::new(x_block, z_block);
        assert!(matches!(
            MajoranaEncoding::determine_vacuum_state(&sym),
            Err(MajoranaEncodingError::NoVacuumStateError)
        ));
    }

    #[test]
    fn test_linearly_dependent_operators_rejected() {
        // Row 2 = Row 0 XOR Row 1, so these are linearly dependent over GF(2)
        let x_block = ndarray::arr2(&[[true, false], [false, true], [true, true], [true, true]]);
        let z_block = ndarray::arr2(&[[false, true], [true, false], [true, true], [true, true]]);
        let sym = SymplecticMatrix::new(x_block, z_block);
        assert!(MajoranaEncoding::with_vacuum(sym, ZBasisState::zeros(2)).is_err());
    }

    #[test]
    fn test_invalid_vacuum_state_rejected() {
        let tree = TernaryTree::naive_jordan_wigner(2);
        let operators = tree.build_encoding(2).unwrap().operators;
        // Wrong number of qubits: encoding has 2 qubits but vacuum has 1.
        let result = MajoranaEncoding::with_vacuum(operators, ZBasisState::zeros(1));
        assert!(matches!(
            result,
            Err(MajoranaEncodingError::InvalidVacuumStateError(_))
        ));
    }

    #[test]
    fn test_decode_jw_roundtrip() {
        let tree = TernaryTree::naive_jordan_wigner(4);
        let encoding = tree.build_encoding(4).unwrap();
        for bits in 0u8..16 {
            let occ: Vec<bool> = (0..4).map(|i| (bits >> i) & 1 != 0).collect();
            let fock = FockState::new(Array1::from(occ.clone()), Complex64::ONE);
            let encoded = encoding.try_encode(fock).unwrap();
            let decoded = encoding.decode_zbasis_state(encoded).unwrap();
            assert_eq!(decoded.state, Array1::from(occ));
        }
    }

    #[test]
    fn test_decode_parity_roundtrip() {
        let tree = TernaryTree::naive_parity(4);
        let encoding = tree.build_encoding(4).unwrap();
        for bits in 0u8..16 {
            let occ: Vec<bool> = (0..4).map(|i| (bits >> i) & 1 != 0).collect();
            let fock = FockState::new(Array1::from(occ.clone()), Complex64::ONE);
            let encoded = encoding.try_encode(fock).unwrap();
            let decoded = encoding.decode_zbasis_state(encoded).unwrap();
            assert_eq!(decoded.state, Array1::from(occ));
        }
    }

    proptest! {
        #[test]
        fn test_decode_roundtrip_all_standard_encodings(
            hf_state in proptest::collection::vec(proptest::bool::ANY, 1..8usize),
            encoding_idx in 0usize..4,
        ) {
            let n = hf_state.len();
            let tree = match encoding_idx {
                0 => TernaryTree::naive_jordan_wigner(n),
                1 => TernaryTree::naive_parity(n),
                2 => TernaryTree::naive_bravyi_kitaev(n),
                _ => TernaryTree::naive_jkmn(n),
            };
            let encoding = tree.build_encoding(n).expect("valid encoding");
            let fock = FockState::new(Array1::from(hf_state.clone()), Complex64::ONE);
            let encoded = encoding.try_encode(fock).unwrap();
            let decoded = encoding.decode_zbasis_state(encoded).unwrap();
            let expected: Array1<bool> = hf_state.into_iter().collect();
            prop_assert_eq!(decoded.state, expected);
        }
    }

    #[test]
    fn test_decode_ensemble_matches_single_decode() {
        let trees = [
            TernaryTree::naive_jordan_wigner(4),
            TernaryTree::naive_parity(4),
            TernaryTree::naive_bravyi_kitaev(4),
            TernaryTree::naive_jkmn(4),
        ];
        for tree in trees {
            let encoding = tree.build_encoding(4).unwrap();
            let encoded_states: Vec<ZBasisState> = (0u8..16)
                .map(|bits| {
                    let occ: Vec<bool> = (0..4).map(|i| (bits >> i) & 1 != 0).collect();
                    let fock = FockState::new(Array1::from(occ), Complex64::ONE);
                    encoding.try_encode(fock).unwrap()
                })
                .collect();
            let ensemble = ZBasisEnsemble::from(encoded_states.clone());
            let batch_results = encoding.decode_zbasis_ensemble(&ensemble);
            for (single_state, batch_result) in encoded_states.into_iter().zip(batch_results) {
                let single_result = encoding.decode_zbasis_state(single_state);
                assert_eq!(
                    single_result.map(|s| s.state),
                    batch_result.map(|s| s.state)
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_decode_ensemble_matches_single_decode_all_encodings(
            hf_states in proptest::collection::vec(
                proptest::collection::vec(proptest::bool::ANY, 1..8usize),
                1..20usize,
            ),
            encoding_idx in 0usize..4,
        ) {
            let n = hf_states[0].len();
            // Filter to same length so they form a valid ensemble matrix.
            let hf_states: Vec<Vec<bool>> = hf_states.into_iter().filter(|s| s.len() == n).collect();
            prop_assume!(!hf_states.is_empty());
            let tree = match encoding_idx {
                0 => TernaryTree::naive_jordan_wigner(n),
                1 => TernaryTree::naive_parity(n),
                2 => TernaryTree::naive_bravyi_kitaev(n),
                _ => TernaryTree::naive_jkmn(n),
            };
            let encoding = tree.build_encoding(n).expect("valid encoding");
            let encoded_states: Vec<ZBasisState> = hf_states.iter()
                .map(|occ| {
                    let fock = FockState::new(Array1::from(occ.clone()), Complex64::ONE);
                    encoding.try_encode(fock).unwrap()
                })
                .collect();
            let ensemble = ZBasisEnsemble::from(encoded_states.clone());
            let batch_results = encoding.decode_zbasis_ensemble(&ensemble);
            for (single_input, batch_result) in encoded_states.into_iter().zip(batch_results) {
                let single_result = encoding.decode_zbasis_state(single_input);
                prop_assert_eq!(
                    single_result.map(|s| s.state),
                    batch_result.map(|s| s.state)
                );
            }
        }
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::encode::ternarytree::TernaryTree;
    use crate::operators::{CoefficientPauliWeight, PauliWeight};
    use num_complex::Complex64;
    use proptest::prelude::*;
    use tinyvec::array_vec;

    /// Generates a permutation of `0..n` by sorting indices by random keys.
    ///
    /// Using keys from `0..n*2` reduces collisions so more orderings are explored,
    /// while still always producing a valid permutation regardless of ties.
    fn arb_perm(n: usize) -> impl Strategy<Value = Vec<usize>> {
        proptest::collection::vec(0usize..n * 2, n).prop_map(move |keys| {
            let mut perm: Vec<usize> = (0..n).collect();
            perm.sort_by_key(|&i| keys[i]);
            perm
        })
    }

    /// Generates a [`MajoranaSparse`] with 1–3 pair-terms whose indices lie in `0..2*n_modes`.
    ///
    /// Coefficients are non-zero integers (±1..=5) to avoid the zero-filtering in
    /// [`MajoranaSparse::new`].
    fn arb_majorana_sparse(n_modes: usize) -> impl Strategy<Value = MajoranaSparse> {
        let n_ops = 2 * n_modes;
        proptest::collection::vec(
            (0..n_ops, 0..n_ops, 1i32..=5i32, any::<bool>())
                .prop_filter("indices must be distinct", |(a, b, _, _)| a != b)
                .prop_map(move |(a, b, abs_coeff, neg)| {
                    let coeff = if neg {
                        -(abs_coeff as f64)
                    } else {
                        abs_coeff as f64
                    };
                    let (lo, hi) = if a < b {
                        (a as u16, b as u16)
                    } else {
                        (b as u16, a as u16)
                    };
                    (array_vec!([u16; 7] => lo, hi), Complex64::new(coeff, 0.))
                }),
            1..=3usize,
        )
        .prop_map(|terms| {
            let (indices, coeffs): (Vec<_>, Vec<_>) = terms.into_iter().unzip();
            MajoranaSparse::new(indices, coeffs, 0.).unwrap()
        })
    }

    proptest! {
        /// Both output vectors always have length equal to the number of input permutations.
        #[test]
        fn test_batch_length_matches_permutation_count(
            n_modes in 2usize..=5,
            n_perms in 0usize..=5,
        ) {
            let tree = TernaryTree::naive_jordan_wigner(n_modes);
            let encoding = tree.build_encoding(n_modes).unwrap();
            let ms = MajoranaSparse::new(
                vec![array_vec!([u16; 7] => 0, 1)],
                vec![Complex64::new(1.0, 0.)],
                0.,
            )
            .unwrap();
            // Repeat the identity permutation n_perms times to decouple from n_modes.
            let perms: Vec<Vec<usize>> = (0..n_perms)
                .map(|_| (0..n_modes).collect())
                .collect();
            let (plain, weighted) = encoding.batch_pauli_weights(&ms, &perms);
            prop_assert_eq!(plain.len(), n_perms);
            prop_assert_eq!(weighted.len(), n_perms);
        }

        /// Each plain weight equals `pauli_weight()` and each coefficient-weighted weight
        /// equals `coeff_pauli_weight()`, both computed individually, for all permutations.
        #[test]
        fn test_batch_both_weights_match_individual(
            (n_modes, perms, ms) in (2usize..=4).prop_flat_map(|n| {
                (
                    Just(n),
                    proptest::collection::vec(arb_perm(n), 1..=4usize),
                    arb_majorana_sparse(n),
                )
            }),
        ) {
            let tree = TernaryTree::naive_jordan_wigner(n_modes);
            let encoding = tree.build_encoding(n_modes).unwrap();

            let (plain, weighted) = encoding.batch_pauli_weights(&ms, &perms);

            prop_assert_eq!(plain.len(), perms.len());
            prop_assert_eq!(weighted.len(), perms.len());
            for (perm, (&plain_weight, &coeff_weight)) in
                perms.iter().zip(plain.iter().zip(weighted.iter()))
            {
                let enumerated = encoding.apply_mode_enumeration(perm.clone());
                let qham = enumerated.encode(&ms);
                prop_assert_eq!(qham.pauli_weight() as f64, plain_weight);
                prop_assert!(
                    (qham.coeff_pauli_weight() - coeff_weight).abs() < crate::utils::WEIGHT_TOLERANCE,
                    "expected coeff weight {}, got {}", qham.coeff_pauli_weight(), coeff_weight
                );
            }
        }
    }
}
