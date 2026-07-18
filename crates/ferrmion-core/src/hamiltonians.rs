use crate::operators::{
    CoefficientPauliWeight, FermionMatrix, FermionSparse, LadderOperator, MajoranaSparse,
    PauliWeight, SymplecticMatrix,
};
use crate::spaces::{Fermion, Qubit};
use crate::utils::{icount_to_sign, pauli_to_symplectic, COEFFICIENT_TOLERANCE};
use ahash::RandomState;
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis};
use num_complex::Complex64;
use std::collections::HashMap;
use thiserror::Error;

/// A qubit Hamiltonian represented as a sparse mapping from Pauli strings to complex coefficients.
///
/// Each key is a Pauli string (e.g. `"XYZII"`), and each value is the corresponding
/// complex coefficient. Uses a randomised hash state for performance.
///
/// # Examples
///
/// ```
/// use ferrmion_core::hamiltonians::QubitHamiltonian;
/// use num_complex::Complex64;
///
/// let QubitHamiltonian(mut ham) = QubitHamiltonian::default();
/// ham.insert("XYZ".to_string(), Complex64::new(1.0, 0.0));
/// assert_eq!(ham.len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct QubitHamiltonian(pub HashMap<String, Complex64, RandomState>);

impl Qubit for QubitHamiltonian {}

impl Default for QubitHamiltonian {
    fn default() -> Self {
        Self(HashMap::with_capacity_and_hasher(0, RandomState::default()))
    }
}

impl PauliWeight for QubitHamiltonian {
    fn pauli_weight(&self) -> usize {
        self.0.keys().fold(0, |acc, term: &String| {
            let n_identity = term.chars().filter(|c| c == &'I').count();
            acc + (term.len() - n_identity)
        })
    }
}

impl CoefficientPauliWeight for QubitHamiltonian {
    fn coeff_pauli_weight(&self) -> f64 {
        self.0.iter().fold(0., |acc, (term, coeff)| {
            let n_identity = term.chars().filter(|c| c == &'I').count();
            acc + (term.len() - n_identity) as f64 * coeff.norm()
        })
    }
}

/// A fermionic Hamiltonian: an insertion-ordered collection of dense
/// [`FermionMatrix`] terms plus a constant energy offset.
///
/// Each term pairs a ladder-operator signature (e.g. `"+-"`, `"++--"`) with a
/// dense coefficient tensor whose dimension equals the signature length.
///
/// <div class="warning">
/// Coefficients are in spin-orbit format: for spatial orbital index `i`, the
/// spin-up mode is at `2i` and the spin-down mode is at `2i+1`.
/// </div>
///
/// # Examples
///
/// ```
/// use ferrmion_core::hamiltonians::FermionHamiltonian;
/// use ndarray::arr2;
///
/// let mut fham = FermionHamiltonian::new(0.5);
/// fham.set_term("+-", arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn()).unwrap();
/// assert_eq!(fham.n_modes(), 2);
/// assert_eq!(fham.signatures(), vec!["+-".to_string()]);
/// let msparse = fham.to_majorana_sparse();
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FermionHamiltonian {
    terms: Vec<FermionMatrix>,
    pub constant_energy: f64,
}

impl Fermion for FermionHamiltonian {}

/// Errors raised when constructing a [`FermionHamiltonian`].
#[derive(Debug, Error, PartialEq, Clone)]
pub enum FermionHamiltonianError {
    #[error("Invalid signature character '{0}'; signature components must be '+' or '-'.")]
    InvalidSignature(char),
    #[error("Coefficient tensor must have one square dimension per signature character.")]
    InvalidTerm,
    #[error("Coefficient tensor with {found} modes does not match existing terms with {expected} modes.")]
    InconsistentModes { expected: usize, found: usize },
}

impl FermionHamiltonian {
    /// Construct an empty [`FermionHamiltonian`] with the given constant energy offset.
    pub fn new(constant_energy: f64) -> Self {
        Self {
            terms: Vec::new(),
            constant_energy,
        }
    }

    /// Set the coefficient tensor for a signature, replacing any existing term
    /// with the same signature.
    ///
    /// Validation (signature characters, tensor dimension and squareness,
    /// antisymmetry zeroing) is delegated to [`FermionMatrix::new`]; the mode
    /// count must additionally agree with any terms already present.
    pub fn set_term(
        &mut self,
        signature: &str,
        coefficients: ArrayD<f64>,
    ) -> Result<(), FermionHamiltonianError> {
        let action: Vec<LadderOperator> = signature
            .chars()
            .map(|c| {
                LadderOperator::try_from(c)
                    .map_err(|_| FermionHamiltonianError::InvalidSignature(c))
            })
            .collect::<Result<_, _>>()?;
        let term = FermionMatrix::new(action, coefficients)
            .map_err(|_| FermionHamiltonianError::InvalidTerm)?;
        if !self.terms.is_empty() && term.n_modes() != self.n_modes() {
            return Err(FermionHamiltonianError::InconsistentModes {
                expected: self.n_modes(),
                found: term.n_modes(),
            });
        }
        match self
            .terms
            .iter_mut()
            .find(|existing| existing.action() == term.action())
        {
            Some(existing) => *existing = term,
            None => self.terms.push(term),
        }
        Ok(())
    }

    /// The number of fermionic modes, or 0 if no terms have been set.
    pub fn n_modes(&self) -> usize {
        self.terms.first().map(FermionMatrix::n_modes).unwrap_or(0)
    }

    /// Signatures of all terms in insertion order.
    pub fn signatures(&self) -> Vec<String> {
        self.terms.iter().map(FermionMatrix::signature).collect()
    }

    /// View of the coefficient tensor for the given signature, if present.
    pub fn term(&self, signature: &str) -> Option<ArrayViewD<'_, f64>> {
        self.terms
            .iter()
            .find(|t| t.signature() == signature)
            .map(FermionMatrix::coefficients)
    }

    /// Iterate over the terms in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &FermionMatrix> {
        self.terms.iter()
    }

    /// Add to the constant energy offset.
    pub fn add_constant(&mut self, constant_energy: f64) {
        self.constant_energy += constant_energy;
    }

    /// Convert to the sparse Majorana representation consumed by encodings.
    ///
    /// Goes through the existing `FermionMatrix -> FermionSparse -> MajoranaSparse`
    /// conversion chain and then adds the constant energy offset.
    pub fn to_majorana_sparse(&self) -> MajoranaSparse {
        let sparse_terms: Vec<FermionSparse> = self
            .terms
            .iter()
            .cloned()
            .map(FermionSparse::from)
            .collect();
        let mut hamiltonian = MajoranaSparse::from(sparse_terms);
        hamiltonian.constant += self.constant_energy;
        hamiltonian
    }
}

/// Qubit hamiltonian with operators in symplectic form.
#[derive(Clone, Debug)]
pub struct SymplecticHamiltonian {
    pub operators: SymplecticMatrix,
    pub coefficients: Array1<f64>,
}

impl Qubit for SymplecticHamiltonian {}

impl SymplecticHamiltonian {
    pub fn new(operators: SymplecticMatrix, coefficients: Array1<f64>) -> Self {
        Self {
            operators,
            coefficients,
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.operators.n_qubits()
    }

    /// Build a [`SymplecticHamiltonian`] from an encoded [`QubitHamiltonian`].
    ///
    /// `pauli_to_symplectic` returns `ipower = n_Y % 4`, which is exactly the
    /// `stored_ipower` needed so that `view_row(i).to_pauli_string()` returns
    /// `total_ipower == 0` before any circuit is applied — meaning
    /// `actual_coeff == stored_coeff` without further phase bookkeeping.
    pub fn from_qubit_hamiltonian(qham: &QubitHamiltonian, n_qubits: usize) -> Self {
        let qham = &qham.0;
        let n = qham.len();
        let mut x_data = Vec::with_capacity(n * n_qubits);
        let mut z_data = Vec::with_capacity(n * n_qubits);
        let mut ipowers = Array1::<u8>::zeros(n);
        let mut coeffs = Array1::<f64>::zeros(n);

        for (i, (pauli_str, coeff)) in qham.iter().enumerate() {
            // returns ([x_block | z_block], n_Y % 4)
            let (symplectic, ipower) = pauli_to_symplectic(pauli_str.clone(), 0);
            x_data.extend(symplectic.iter().take(n_qubits).copied());
            z_data.extend(symplectic.iter().skip(n_qubits).copied());
            ipowers[i] = ipower as u8;
            coeffs[i] = coeff.re;
        }

        let x_block = Array2::from_shape_vec((n, n_qubits), x_data).unwrap();
        let z_block = Array2::from_shape_vec((n, n_qubits), z_data).unwrap();
        Self::new(
            SymplecticMatrix::from_arrays_with_ipowers(x_block, z_block, ipowers),
            coeffs,
        )
    }

    /// Convert back to a [`QubitHamiltonian`].
    ///
    /// `view_row(i).to_pauli_string()` accumulates Y-convention corrections and
    /// sign changes from H gates into `total_ipower`. For a Hermitian molecular
    /// Hamiltonian this is always 0 or 2 (±1), keeping all coefficients real.
    /// Terms mapping to the same Pauli string are summed; results below tolerance
    /// are dropped.
    pub fn to_qubit_hamiltonian(&self) -> QubitHamiltonian {
        let mut result = QubitHamiltonian::default();
        for i in 0..self.coefficients.len() {
            let (pauli_str, total_ipower) = self.operators.view_row(i).to_pauli_string();
            let phase = icount_to_sign(total_ipower as usize);
            let actual_coeff = Complex64::new(self.coefficients[i], 0.) * phase;
            if actual_coeff.norm() > COEFFICIENT_TOLERANCE {
                result
                    .0
                    .entry(pauli_str)
                    .and_modify(|e| *e += actual_coeff)
                    .or_insert(actual_coeff);
            }
        }
        result
    }

    /// Sort rows in-place, keeping each coefficient paired with its operator row.
    ///
    /// Rows are ordered lexicographically by x_block, then z_block, then ipower.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticMatrix;
    /// use ferrmion_core::hamiltonians::SymplecticHamiltonian;
    /// use ndarray::{arr1, arr2};
    ///
    /// // Row 0 is "XI" (coefficient 1.0), row 1 is "IX" (coefficient 2.0).
    /// // After sorting, "IX" < "XI" so the rows swap.
    /// let ops = SymplecticMatrix::from_arrays(
    ///     arr2(&[[true, false], [false, true]]),
    ///     arr2(&[[false, false], [false, false]]),
    /// );
    /// let mut ham = SymplecticHamiltonian::new(ops, arr1(&[1.0, 2.0]));
    /// ham.sort_rows();
    /// let (first, _) = ham.operators.view_row(0).to_pauli_string();
    /// assert_eq!(first, "IX");
    /// assert_eq!(ham.coefficients[0], 2.0);
    /// ```
    pub fn sort_rows(&mut self) {
        let n = self.operators.n_rows();
        let mut indices: Vec<usize> = (0..n).collect();
        indices
            .sort_unstable_by(|&a, &b| self.operators.view_row(a).cmp(&self.operators.view_row(b)));
        self.operators = self.operators.select_rows(&indices);
        self.coefficients = self.coefficients.select(Axis(0), &indices);
    }

    /// Deduplicate consecutive equal operator rows by summing their coefficients.
    ///
    /// Rows must be sorted (e.g. via [`sort_rows`]) before calling this method.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticMatrix;
    /// use ferrmion_core::hamiltonians::SymplecticHamiltonian;
    /// use ndarray::{arr1, arr2};
    ///
    /// let ops = SymplecticMatrix::from_arrays(
    ///     arr2(&[[true, false], [true, false], [false, true]]),
    ///     arr2(&[[false, false], [false, false], [false, false]]),
    /// );
    /// let mut ham = SymplecticHamiltonian::new(ops, arr1(&[1.0, 2.0, 3.0]));
    /// // already sorted; first two rows are identical
    /// ham.dedup();
    /// assert_eq!(ham.coefficients.len(), 2);
    /// assert_eq!(ham.coefficients[0], 3.0);
    /// assert_eq!(ham.coefficients[1], 3.0);
    /// ```
    pub fn dedup(&mut self) {
        let n = self.operators.n_rows();
        if n == 0 {
            return;
        }
        let mut keep: Vec<usize> = vec![0];
        let mut summed: Vec<f64> = vec![self.coefficients[0]];
        for i in 1..n {
            if self.operators.view_row(i)
                == self.operators.view_row(
                    *keep
                        .last()
                        .expect("keep is non-empty; initialised with index 0"),
                )
            {
                *summed
                    .last_mut()
                    .expect("summed is non-empty; initialised with coefficients[0]") +=
                    self.coefficients[i];
            } else {
                keep.push(i);
                summed.push(self.coefficients[i]);
            }
        }
        self.operators = self.operators.select_rows(&keep);
        self.coefficients = Array1::from_vec(summed);
    }
}

impl PauliWeight for SymplecticHamiltonian {
    fn pauli_weight(&self) -> usize {
        self.operators.pauli_weight()
    }
}

impl CoefficientPauliWeight for SymplecticHamiltonian {
    fn coeff_pauli_weight(&self) -> f64 {
        self.operators
            .iter_rows()
            .zip(&self.coefficients)
            .fold(0., |acc, (row, coeff)| {
                acc + (coeff.abs() * row.pauli_weight() as f64)
            })
    }
}

#[cfg(test)]
mod fermion_hamiltonian_tests {
    use super::*;
    use ndarray::{arr2, Array};

    #[test]
    fn test_set_term_and_accessors() {
        let mut fham = FermionHamiltonian::new(0.5);
        let ones = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
        fham.set_term("+-", ones.clone()).unwrap();
        assert_eq!(fham.n_modes(), 2);
        assert_eq!(fham.signatures(), vec!["+-".to_string()]);
        assert_eq!(fham.term("+-").unwrap(), ones.view());
        assert!(fham.term("++--").is_none());
        assert_eq!(fham.constant_energy, 0.5);
        fham.add_constant(1.0);
        assert_eq!(fham.constant_energy, 1.5);
    }

    #[test]
    fn test_set_term_replaces_existing_signature() {
        let mut fham = FermionHamiltonian::new(0.0);
        fham.set_term("+-", arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn())
            .unwrap();
        fham.set_term("+-", arr2(&[[2.0, 0.0], [0.0, 2.0]]).into_dyn())
            .unwrap();
        assert_eq!(fham.signatures().len(), 1);
        assert_eq!(fham.term("+-").unwrap()[[0, 0]], 2.0);
    }

    #[test]
    fn test_set_term_invalid_signature() {
        let mut fham = FermionHamiltonian::new(0.0);
        let err = fham
            .set_term("+x", arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn())
            .unwrap_err();
        assert_eq!(err, FermionHamiltonianError::InvalidSignature('x'));
    }

    #[test]
    fn test_set_term_wrong_dimension() {
        let mut fham = FermionHamiltonian::new(0.0);
        let err = fham
            .set_term("++--", arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn())
            .unwrap_err();
        assert_eq!(err, FermionHamiltonianError::InvalidTerm);
    }

    #[test]
    fn test_set_term_inconsistent_modes() {
        let mut fham = FermionHamiltonian::new(0.0);
        fham.set_term("+-", arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn())
            .unwrap();
        let err = fham
            .set_term("++--", Array::zeros(vec![3, 3, 3, 3]).into_dyn())
            .unwrap_err();
        assert_eq!(
            err,
            FermionHamiltonianError::InconsistentModes {
                expected: 2,
                found: 3
            }
        );
    }

    #[test]
    fn test_to_majorana_sparse_matches_from_signatures_and_coeffs() {
        let ones = arr2(&[[1.0, 0.5], [0.5, 1.0]]).into_dyn();
        let mut twos = Array::zeros(vec![2, 2, 2, 2]).into_dyn();
        twos[[0, 1, 1, 0]] = 0.25;
        twos[[1, 0, 0, 1]] = 0.25;

        let mut fham = FermionHamiltonian::new(0.75);
        fham.set_term("+-", ones.clone()).unwrap();
        fham.set_term("++--", twos.clone()).unwrap();

        let expected = MajoranaSparse::from_signatures_and_coeffs(
            vec!["+-".to_string(), "++--".to_string()],
            vec![ones.view(), twos.view()],
            0.75,
        );
        assert_eq!(fham.to_majorana_sparse(), expected);
    }
}
