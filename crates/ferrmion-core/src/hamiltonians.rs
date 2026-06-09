use crate::operators::{
    CoefficientPauliWeight, PauliWeight, SymplecticMatrix, SymplecticOperatorView,
};
use crate::spaces::Qubit;
use ahash::RandomState;
use ndarray::{Array1, Array2, Axis, Zip};
use num_complex::Complex64;
use std::collections::HashMap;

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
        self.operators.x_block.ncols()
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
            let (symplectic, ipower) = crate::utils::pauli_to_symplectic(pauli_str.clone(), 0);
            x_data.extend(symplectic.iter().take(n_qubits).copied());
            z_data.extend(symplectic.iter().skip(n_qubits).copied());
            ipowers[i] = ipower as u8;
            coeffs[i] = coeff.re;
        }

        let x_block = Array2::from_shape_vec((n, n_qubits), x_data).unwrap();
        let z_block = Array2::from_shape_vec((n, n_qubits), z_data).unwrap();
        Self::new(
            SymplecticMatrix::with_ipowers(x_block, z_block, ipowers),
            coeffs,
        )
    }

    /// Convert back to a [`QubitHamiltonian`].
    ///
    /// `view_row(i).to_pauli_string()` accumulates Y-convention corrections and
    /// sign changes from H gates into `total_ipower`. For a Hermitian molecular
    /// Hamiltonian this is always 0 or 2 (±1), keeping all coefficients real.
    /// Terms mapping to the same Pauli string are summed; results below 1e-12
    /// are dropped.
    pub fn to_qubit_hamiltonian(&self) -> QubitHamiltonian {
        let mut result = QubitHamiltonian::default();
        for i in 0..self.coefficients.len() {
            let (pauli_str, total_ipower) = self.operators.view_row(i).to_pauli_string();
            let phase = crate::utils::icount_to_sign(total_ipower as usize);
            let actual_coeff = Complex64::new(self.coefficients[i], 0.) * phase;
            if actual_coeff.norm() > 1e-12 {
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
    /// let ops = SymplecticMatrix::new(
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
        let n = self.operators.x_block.nrows();
        let mut indices: Vec<usize> = (0..n).collect();
        indices
            .sort_unstable_by(|&a, &b| self.operators.view_row(a).cmp(&self.operators.view_row(b)));
        self.operators.x_block = self.operators.x_block.select(Axis(0), &indices);
        self.operators.z_block = self.operators.z_block.select(Axis(0), &indices);
        self.operators.ipowers = self.operators.ipowers.select(Axis(0), &indices);
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
    /// let ops = SymplecticMatrix::new(
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
        let n = self.operators.x_block.nrows();
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
        self.operators.x_block = self.operators.x_block.select(Axis(0), &keep);
        self.operators.z_block = self.operators.z_block.select(Axis(0), &keep);
        self.operators.ipowers = self.operators.ipowers.select(Axis(0), &keep);
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
        Zip::from(self.operators.x_block.rows())
            .and(self.operators.z_block.rows())
            .and(&self.coefficients)
            .fold(0., |acc, x, y, coeff| {
                acc + (coeff.abs() * SymplecticOperatorView::new(0, x, y).pauli_weight() as f64)
            })
    }
}
