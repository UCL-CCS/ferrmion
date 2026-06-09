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
use crate::spaces::Qubit;
use crate::states::ZBasisState;
use ndarray::{arr2, Array1, Array2, ArrayView1, Axis, Zip};
use ndarray::{s, ArrayViewMut1, ArrayViewMut2};
use num_complex::Complex64;
use num_complex::{c64, ComplexFloat};
use std::ops::{BitAnd, BitOr, BitXor, BitXorAssign, Mul};

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
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub enum Pauli {
    #[default]
    I,
    X,
    Y,
    Z,
}

impl Qubit for Pauli {}

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

/// Pauli operator in symplectic form.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SymplecticOperator {
    ipower: u8,
    x_block: Array1<bool>,
    z_block: Array1<bool>,
}

impl Qubit for SymplecticOperator {}

impl SymplecticOperator {
    /// Construct a new [`SymplecticOperator`] from its components.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticOperator;
    /// use ndarray::arr1;
    ///
    /// let op = SymplecticOperator::new(0, arr1(&[true, false]), arr1(&[false, true]));
    /// assert_eq!(op.ipower(), 0);
    /// ```
    pub fn new(ipower: u8, x_block: Array1<bool>, z_block: Array1<bool>) -> Self {
        Self {
            ipower,
            x_block,
            z_block,
        }
    }

    /// Construct an identity operator acting on `n_modes` qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::{SymplecticOperator, PauliWeight};
    ///
    /// let id = SymplecticOperator::identity(3);
    /// assert_eq!(id.pauli_weight(), 0);
    /// ```
    pub fn identity(n_modes: usize) -> Self {
        Self {
            ipower: 0,
            x_block: Array1::from_elem(n_modes, false),
            z_block: Array1::from_elem(n_modes, false),
        }
    }

    /// Return a borrowed [`SymplecticOperatorView`] of this operator.
    pub fn view(&self) -> SymplecticOperatorView<'_> {
        SymplecticOperatorView {
            ipower: self.ipower,
            x_block: self.x_block.view(),
            z_block: self.z_block.view(),
        }
    }

    /// Return a view of the X block.
    pub fn x_block(&self) -> ArrayView1<'_, bool> {
        self.x_block.view()
    }

    /// Return a view of the Z block.
    pub fn z_block(&self) -> ArrayView1<'_, bool> {
        self.z_block.view()
    }

    /// Return the power of `i` (imaginary unit) for this operator.
    pub fn ipower(&self) -> u8 {
        self.ipower
    }

    /// Convert to a Pauli string representation and its associated `i`-power.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticOperator;
    /// use ndarray::arr1;
    ///
    /// let op = SymplecticOperator::new(0, arr1(&[true, false]), arr1(&[false, true]));
    /// let (pauli, ipower) = op.to_pauli_string();
    /// assert_eq!(pauli, "XZ");
    /// assert_eq!(ipower, 0);
    /// ```
    pub fn to_pauli_string(&self) -> (String, u8) {
        let mut pauli_string = String::new();
        let mut ipower = self.ipower;
        Zip::from(&self.x_block)
            .and(&self.z_block)
            .for_each(|&x, &z| {
                if x && z {
                    ipower += 3;
                };
                pauli_string.push(Pauli::from((x, z)).into());
            });
        (pauli_string, (ipower % 4))
    }
}

impl PauliWeight for SymplecticOperator {
    fn pauli_weight(&self) -> usize {
        Zip::from(&self.x_block)
            .and(&self.z_block)
            .fold(0, |acc, x, z| {
                acc + if (x == &false) & (z == &false) { 0 } else { 1 }
            })
    }
}

impl CoefficientPauliWeight for SymplecticOperator {
    fn coeff_pauli_weight(&self) -> f64 {
        self.pauli_weight() as f64
    }
}

impl<'a> Mul<SymplecticOperatorView<'a>> for SymplecticOperator {
    type Output = SymplecticOperator;

    fn mul(self, rhs: SymplecticOperatorView<'a>) -> Self::Output {
        // bitwise or between two vectors
        let x_product = &self.x_block ^ &rhs.x_block;
        let z_product = &self.z_block ^ &rhs.z_block;

        // bitwise sum of left z and right x
        let mut ipower = self.ipower + rhs.ipower;
        for (&lz, &rx) in self.z_block.iter().zip(&rhs.x_block) {
            if lz && rx {
                ipower += 2;
            };
        }

        Self::Output {
            ipower: ipower % 4,
            x_block: x_product,
            z_block: z_product,
        }
    }
}

impl SymplecticOperator {
    /// In-place multiply: `self = self * rhs`, reusing existing array allocations.
    ///
    /// This avoids 2 heap allocations per call compared to `Mul`, which is critical
    /// in the encoding hot path where we fold over up to 4 Majorana indices per term.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticOperator;
    /// use ndarray::arr1;
    ///
    /// let mut op = SymplecticOperator::identity(2);
    /// let rhs = SymplecticOperator::new(0, arr1(&[true, false]), arr1(&[false, true]));
    /// op.mul_assign_view(&rhs.view());
    /// let (pauli, _) = op.to_pauli_string();
    /// assert_eq!(pauli, "XZ");
    /// ```
    #[inline]
    pub fn mul_assign_view(&mut self, rhs: &SymplecticOperatorView<'_>) {
        // Accumulate ipower from z_block . x_block before mutating z_block
        let mut ipower = self.ipower + rhs.ipower;
        for (&lz, &rx) in self.z_block.iter().zip(rhs.x_block.iter()) {
            if lz && rx {
                ipower += 2;
            }
        }
        self.ipower = ipower % 4;

        // In-place XOR for x_block and z_block
        self.x_block
            .iter_mut()
            .zip(rhs.x_block.iter())
            .for_each(|(l, &r)| *l ^= r);
        self.z_block
            .iter_mut()
            .zip(rhs.z_block.iter())
            .for_each(|(l, &r)| *l ^= r);
    }
}

impl Mul<ZBasisState> for SymplecticOperator {
    type Output = ZBasisState;

    fn mul(self, rhs: ZBasisState) -> Self::Output {
        let mut state = rhs.state;
        let mut phase_factor = c64(-1., 0.).powi(
            self.z_block
                .view()
                .bitand(&state)
                .fold(0, |acc, &n| if n { acc + 2 } else { acc })
                % 4,
        );
        let y_count: i32 = self
            .x_block
            .view()
            .bitand(&self.z_block.view())
            .fold(0, |acc, &n| if n { acc + 1 } else { acc });
        phase_factor *= c64(0., 1.).powi(y_count);

        let coefficient = rhs.coefficient * phase_factor;

        state = self.x_block.bitxor(&state);

        ZBasisState::new(state, coefficient)
    }
}

/// Pauli operator encoded in symplectic (XZ) form.
///
/// # Examples
///
/// ```
/// use ferrmion_core::operators::SymplecticOperatorView;
/// use ndarray::arr1;
///
/// let x = arr1(&[true, false]);
/// let z = arr1(&[false, true]);
/// let view = SymplecticOperatorView::new(0, x.view(), z.view());
/// let (pauli, ipower) = view.to_pauli_string();
/// assert_eq!(pauli, "XZ");
/// assert_eq!(ipower, 0);
/// ```
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SymplecticOperatorView<'sym> {
    ipower: u8,
    x_block: ArrayView1<'sym, bool>,
    z_block: ArrayView1<'sym, bool>,
}

impl Qubit for SymplecticOperatorView<'_> {}

impl<'sym> SymplecticOperatorView<'sym> {
    /// Construct a new [`SymplecticOperatorView`] from borrowed array views.
    pub fn new(
        ipower: u8,
        x_block: ArrayView1<'sym, bool>,
        z_block: ArrayView1<'sym, bool>,
    ) -> Self {
        Self {
            ipower,
            x_block,
            z_block,
        }
    }

    /// Convert to a Pauli string representation and its associated `i`-power.
    pub fn to_pauli_string(self) -> (String, u8) {
        let mut pauli_string = String::new();
        let mut ipower = self.ipower;
        Zip::from(&self.x_block)
            .and(&self.z_block)
            .for_each(|&x, &z| {
                if x && z {
                    ipower += 3;
                };
                pauli_string.push(Pauli::from((x, z)).into());
            });
        (pauli_string, (ipower % 4))
    }
}

impl PauliWeight for SymplecticOperatorView<'_> {
    fn pauli_weight(&self) -> usize {
        Zip::from(&self.x_block)
            .and(self.z_block)
            .fold(0, |acc, x, z| {
                acc + if (x == &false) & (z == &false) { 0 } else { 1 }
            })
    }
}

impl CoefficientPauliWeight for SymplecticOperatorView<'_> {
    fn coeff_pauli_weight(&self) -> f64 {
        self.pauli_weight() as f64
    }
}
impl Mul<ZBasisState> for SymplecticOperatorView<'_> {
    type Output = ZBasisState;

    fn mul(self, rhs: ZBasisState) -> Self::Output {
        let mut state = rhs.state;
        let phase_factor = c64(0., 1.).powi(
            self.z_block
                .bitand(&state)
                .fold(self.ipower as i32, |acc, &n| if n { acc + 2 } else { acc })
                % 4,
        );

        let coefficient = rhs.coefficient * phase_factor;

        state = self.x_block.bitxor(&state);

        ZBasisState::new(state, coefficient)
    }
}

impl Mul<&mut ZBasisState> for SymplecticOperatorView<'_> {
    type Output = ();

    fn mul(self, rhs: &mut ZBasisState) {
        let mut phase_factor = c64(-1., 0.).powi(
            self.z_block
                .bitand(&rhs.state)
                .fold(0, |acc, &n| if n { acc + 2 } else { acc })
                % 4,
        );
        let y_count: i32 =
            self.x_block
                .bitand(&self.z_block)
                .fold(0, |acc, &n| if n { acc + 1 } else { acc });
        phase_factor *= c64(0., 1.).powi(y_count);

        rhs.state = self.x_block.bitxor(&rhs.state);
        rhs.coefficient *= phase_factor;
    }
}

impl PartialOrd for SymplecticOperatorView<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SymplecticOperatorView<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.x_block
            .iter()
            .cmp(other.x_block.iter())
            .then(self.z_block.iter().cmp(other.z_block.iter()))
            .then(self.ipower.cmp(&other.ipower))
    }
}

/// A collection of Pauli operators in symplectic form.
///
/// Each row represents one Pauli operator. The `x_block` and `z_block` matrices
/// encode the Pauli type on each qubit via the symplectic convention:
///
/// | `x_block` | `z_block` | Pauli |
/// |-----------|-----------|-------|
/// | `false`   | `false`   | `I`   |
/// | `true`    | `false`   | `X`   |
/// | `false`   | `true`    | `Z`   |
/// | `true`    | `true`    | `Y`   |
///
/// For example, a row with `x_block = [true, false, true]` and
/// `z_block = [true, true, false]` represents the Pauli string `"YZX"`.
///
/// # Examples
///
/// ```
/// use ferrmion_core::operators::{SymplecticMatrix, PauliWeight};
/// use ndarray::arr2;
///
/// let x = arr2(&[[true, false], [false, true]]);
/// let z = arr2(&[[false, true], [true, false]]);
/// let mat = SymplecticMatrix::new(x, z);
/// assert_eq!(mat.pauli_weight(), 4);
/// ```
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SymplecticMatrix {
    pub x_block: Array2<bool>,
    pub z_block: Array2<bool>,
    pub ipowers: Array1<u8>,
}

impl Qubit for SymplecticMatrix {}

impl SymplecticMatrix {
    /// Construct a new [`SymplecticMatrix`], automatically computing `i`-powers from the X and Z blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticMatrix;
    /// use ndarray::arr2;
    ///
    /// let mat = SymplecticMatrix::new(
    ///     arr2(&[[true, false]]),
    ///     arr2(&[[false, true]]),
    /// );
    /// assert_eq!(mat.ipowers.len(), 1);
    /// assert_eq!(mat.ipowers[0], 0); // no Y operators, so ipower = 0
    /// ```
    pub fn new(x_block: Array2<bool>, z_block: Array2<bool>) -> Self {
        let mut ipowers = Array1::from_elem(x_block.len_of(Axis(0)), 0);
        Zip::from(&mut ipowers)
            .and(x_block.rows())
            .and(z_block.rows())
            .for_each(|ipower, x_row, z_row| {
                let y_count: usize = x_row.bitand(&z_row).map(|b| *b as usize).sum();
                *ipower = (y_count % 4) as u8;
            });
        Self {
            x_block,
            z_block,
            ipowers,
        }
    }

    /// Construct a [`SymplecticMatrix`] with explicitly provided `i`-powers.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticMatrix;
    /// use ndarray::{arr1, arr2};
    ///
    /// let mat = SymplecticMatrix::with_ipowers(
    ///     arr2(&[[true, true]]),
    ///     arr2(&[[true, true]]),
    ///     arr1(&[2u8]),
    /// );
    /// assert_eq!(mat.ipowers[0], 2);
    /// ```
    pub fn with_ipowers(x_block: Array2<bool>, z_block: Array2<bool>, ipowers: Array1<u8>) -> Self {
        Self {
            x_block,
            z_block,
            ipowers,
        }
    }

    /// Construct an identity [`SymplecticMatrix`] with `n_modes` rows and `n_qubits` columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::{SymplecticMatrix, PauliWeight};
    ///
    /// let id = SymplecticMatrix::identity(4, 3);
    /// assert_eq!(id.pauli_weight(), 0);
    /// ```
    pub fn identity(n_modes: usize, n_qubits: usize) -> Self {
        Self {
            ipowers: Array1::from_elem(n_modes, 0),
            x_block: Array2::from_elem((n_modes, n_qubits), false),
            z_block: Array2::from_elem((n_modes, n_qubits), false),
        }
    }

    /// Return a [`SymplecticOperatorView`] for the given row index.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticMatrix;
    /// use ndarray::arr2;
    ///
    /// let mat = SymplecticMatrix::new(
    ///     arr2(&[[true, false], [false, true]]),
    ///     arr2(&[[false, false], [true, true]]),
    /// );
    /// let row0 = mat.view_row(0);
    /// let (pauli, _) = row0.to_pauli_string();
    /// assert_eq!(pauli, "XI");
    /// ```
    pub fn view_row(&self, row: usize) -> SymplecticOperatorView<'_> {
        SymplecticOperatorView {
            x_block: self.x_block.row(row),
            z_block: self.z_block.row(row),
            ipower: self.ipowers[row],
        }
    }

    /// Type safe transpose.
    ///
    /// Used for optimisation routines which use conjugations with
    /// [`Clifford`] gates to update an encoding.
    pub(crate) fn transpose<'a>(&'a mut self) -> SymplecticMatrixTranspose<'a> {
        SymplecticMatrixTranspose {
            x_block: self.x_block.view_mut().reversed_axes(),
            z_block: self.z_block.view_mut().reversed_axes(),
            ipowers: self.ipowers.view_mut(),
        }
    }

    /// Apply a sequence of Clifford gates to this matrix in-place.
    pub fn apply_clifford_chain(&mut self, chain: &[Clifford]) {
        let mut transpose = self.transpose();
        for op in chain {
            match op {
                Clifford::H(idx) => transpose.haddamard(*idx),
                Clifford::S(idx) => transpose.phasegate(*idx),
                Clifford::CNOT { control, target } => transpose.cnot(*control, *target),
            }
        }
    }

    /// Sort rows in-place in lexicographic order.
    ///
    /// Rows are ordered by `x_block` first, then `z_block`, then `ipower`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::SymplecticMatrix;
    /// use ndarray::arr2;
    ///
    /// let mut mat = SymplecticMatrix::new(
    ///     arr2(&[[false, true], [true, false]]),
    ///     arr2(&[[false, false], [false, false]]),
    /// );
    /// mat.sort_rows();
    /// let (first, _) = mat.view_row(0).to_pauli_string();
    /// assert_eq!(first, "IX");
    /// ```
    pub fn sort_rows(&mut self) {
        let n = self.x_block.nrows();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| self.view_row(a).cmp(&self.view_row(b)));
        self.x_block = self.x_block.select(Axis(0), &indices);
        self.z_block = self.z_block.select(Axis(0), &indices);
        self.ipowers = self.ipowers.select(Axis(0), &indices);
    }
}

impl PauliWeight for SymplecticMatrix {
    fn pauli_weight(&self) -> usize {
        Zip::from(self.x_block.rows())
            .and(self.z_block.rows())
            .fold(0, |acc, x_row, z_row| {
                acc + SymplecticOperatorView::new(0, x_row, z_row).pauli_weight()
            })
    }
}

impl Mul<&mut ZBasisState> for SymplecticMatrix {
    type Output = ();

    fn mul(self, rhs: &mut ZBasisState) {
        Zip::from(&self.ipowers)
            .and(self.x_block.rows())
            .and(self.z_block.rows())
            .fold(rhs, |rhs, i, x_row, z_row| {
                let sym = SymplecticOperatorView::new(*i, x_row, z_row);
                #[allow(clippy::let_unit_value)]
                let _ = sym.mul(&mut *rhs);
                rhs
            });
    }
}

#[cfg(test)]
mod symplectic_tests {
    use super::*;

    #[test]
    fn test_symplectic_product() {
        let xxx = SymplecticOperator::new(
            0,
            ndarray::arr1(&[true, true, true]),
            ndarray::arr1(&[false, false, false]),
        );
        let zzz = SymplecticOperator::new(
            0,
            ndarray::arr1(&[false, false, false]),
            ndarray::arr1(&[true, true, true]),
        );
        let result = xxx.clone() * zzz.view();
        assert_eq!(result.ipower(), 0);
        assert_eq!(result.x_block(), ndarray::arr1(&[true, true, true]).view());
        assert_eq!(result.z_block(), ndarray::arr1(&[true, true, true]).view());

        let result = zzz * xxx.view();
        assert_eq!(result.ipower(), 2);
        assert_eq!(result.x_block(), ndarray::arr1(&[true, true, true]).view());
        assert_eq!(result.z_block(), ndarray::arr1(&[true, true, true]).view());
    }

    #[test]
    fn test_symplectic_to_pauli() {
        // YXZI: x=[T,T,F,F], z=[T,F,T,F]
        assert_eq!(
            SymplecticOperator::new(
                0,
                ndarray::arr1(&[true, true, false, false]),
                ndarray::arr1(&[true, false, true, false]),
            )
            .to_pauli_string(),
            (String::from("YXZI"), 3)
        );
        assert_eq!(
            SymplecticOperator::new(0, ndarray::arr1(&[false]), ndarray::arr1(&[false]))
                .to_pauli_string(),
            (String::from("I"), 0)
        );
        assert_eq!(
            SymplecticOperator::new(0, ndarray::arr1(&[false]), ndarray::arr1(&[true]))
                .to_pauli_string(),
            (String::from("Z"), 0)
        );
        assert_eq!(
            SymplecticOperator::new(0, ndarray::arr1(&[true]), ndarray::arr1(&[false]))
                .to_pauli_string(),
            (String::from("X"), 0)
        );
        assert_eq!(
            SymplecticOperator::new(0, ndarray::arr1(&[true]), ndarray::arr1(&[true]))
                .to_pauli_string(),
            (String::from("Y"), 3)
        );
    }
}

/// Transpose of the [`SymplecticMatrix`] type.
///
/// When working with clifford oeprators, they can be most efficiently applied
/// using vectorised operations where the qubit index is kept fixed.
/// The [`SymplecticMatrix`] is backed by two ndarray::Array which are row major
/// with rows being majorana indices and columns being qubit indices.
///
/// To apply clifford operators we therefore want a distinct type which is the
/// transpose of the original.
pub(crate) struct SymplecticMatrixTranspose<'inner> {
    x_block: ArrayViewMut2<'inner, bool>,
    z_block: ArrayViewMut2<'inner, bool>,
    ipowers: ArrayViewMut1<'inner, u8>,
}

// Clifford conjugation functions
//
// Applies P -> CPC
//
// Currently this works given a single qubit index at a time,
// but applied accross all operators in a [`SymplecticMatrix`].
// They could probably be made a little more general and faster by
// allowing a slice input and applying all gates in tandem.
impl<'inner> SymplecticMatrixTranspose<'inner> {
    /// Apply Clifford H Operator
    // $P \to H P H$
    /// -1 for each Y
    /// Z -> X and X -> Z
    pub(crate) fn haddamard(&mut self, qubit: usize) {
        self.ipowers.scaled_add(
            // -1 For each instance
            2,
            &self
                .x_block
                .row(qubit)
                .bitand(&self.z_block.row(qubit))
                .map(|v| *v as u8),
        );
        Zip::from(self.x_block.row_mut(qubit))
            .and(self.z_block.row_mut(qubit))
            .for_each(std::mem::swap);
    }
    /// Apply the Clifford S operator.
    // $P \to S P S$
    /// -1 for each X
    /// Z -> Z ^ X
    pub(crate) fn phasegate(&mut self, qubit: usize) {
        self.ipowers.scaled_add(
            // -1 For each instance
            3,
            &self.x_block.row(qubit).map(|v| *v as u8),
        );
        self.z_block
            .row_mut(qubit)
            .bitxor_assign(&self.x_block.row(qubit));
    }

    // Transform a [`Pauli`] operator by this Clifford operator.
    // $P \to CX P CX$
    pub(crate) fn cnot(&mut self, control: usize, target: usize) {
        // Have tp use multi_slice_mut here to get
        // views into both rows.
        let (cx, mut tx) = self
            .x_block
            .multi_slice_mut((s![control, ..], s![target, ..]));
        tx.bitxor_assign(&cx);

        let (mut cz, tz) = self
            .z_block
            .multi_slice_mut((s![control, ..], s![target, ..]));
        cz.bitxor_assign(&tz);
    }

    pub(crate) fn hamming_weights(&self) -> Array1<usize> {
        // x_block / z_block have shape (n_qubits, n_terms) in the transposed
        // view; folding along axis 1 collapses the term axis to yield one
        // Hamming weight per qubit (the count of non-identity Paulis touching it).
        self.x_block
            .bitor(&self.z_block)
            .fold_axis(Axis(1), 0, |init, v| init + *v as usize)
    }
}

#[cfg(test)]
mod symplictic_transpose_tests {
    use proptest::proptest;

    use super::SymplecticMatrix;

    #[test]
    fn test_HIH() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        let mut transpose = sym.transpose();
        transpose.haddamard(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("I".to_string(), 0));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("I".to_string(), 0));
        assert_eq!(sym.ipowers, SymplecticMatrix::identity(2, 1).ipowers);
    }
    #[test]
    fn test_HXH() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.x_block[[0, 0]] = true;
        sym.x_block[[1, 0]] = true;
        let mut transpose = sym.transpose();
        transpose.haddamard(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("Z".to_string(), 0));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("Z".to_string(), 0));
    }
    #[test]
    fn test_HYH() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.x_block[[0, 0]] = true;
        sym.z_block[[0, 0]] = true;
        sym.ipowers[[0]] = 1;

        sym.x_block[[1, 0]] = true;
        sym.z_block[[1, 0]] = true;
        sym.ipowers[[1]] = 1;

        let mut transpose = sym.transpose();
        transpose.haddamard(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("Y".to_string(), 2));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("Y".to_string(), 2));
    }

    #[test]
    fn test_HZH() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.z_block[[0, 0]] = true;
        sym.z_block[[1, 0]] = true;
        let mut transpose = sym.transpose();
        transpose.haddamard(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("X".to_string(), 0));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("X".to_string(), 0));
    }

    #[test]
    fn test_SIS() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 1);
        let mut transpose = sym.transpose();
        transpose.phasegate(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("I".to_string(), 0));
        assert_eq!(sym.ipowers, SymplecticMatrix::identity(1, 1).ipowers);
    }

    #[test]
    fn test_SXS() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.x_block[[0, 0]] = true;
        sym.x_block[[1, 0]] = true;
        let mut transpose = sym.transpose();
        transpose.phasegate(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("Y".to_string(), 2));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("Y".to_string(), 2));
    }

    #[test]
    fn test_SYS() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.x_block[[0, 0]] = true;
        sym.z_block[[0, 0]] = true;
        sym.ipowers[[0]] = 1;

        sym.x_block[[1, 0]] = true;
        sym.z_block[[1, 0]] = true;
        sym.ipowers[[1]] = 1;
        let mut transpose = sym.transpose();
        transpose.phasegate(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("X".to_string(), 0));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("X".to_string(), 0));
    }

    #[test]
    fn test_SZS() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.z_block[[0, 0]] = true;
        sym.z_block[[1, 0]] = true;
        let mut transpose = sym.transpose();
        transpose.phasegate(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("Z".to_string(), 0));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("Z".to_string(), 0));
    }

    #[test]
    fn test_CX_II_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 2);
        let mut transpose = sym.transpose();
        transpose.cnot(0, 1);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("II".to_string(), 0));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("II".to_string(), 0));
        assert_eq!(sym.ipowers, SymplecticMatrix::identity(2, 2).ipowers);
    }

    #[test]
    fn test_CX_XI_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 2);
        sym.x_block[[0, 0]] = true;
        let mut transpose = sym.transpose();
        transpose.cnot(0, 1);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("XX".to_string(), 0));
        assert_eq!(sym.ipowers, SymplecticMatrix::identity(1, 2).ipowers);
    }

    #[test]
    fn test_CX_IX_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 2);
        sym.x_block[[0, 1]] = true;
        let mut transpose = sym.transpose();
        transpose.cnot(0, 1);
        assert_eq!(
            transpose.x_block.shape(),
            SymplecticMatrix::identity(2, 1).x_block.shape()
        );
        assert_eq!(
            sym.x_block.shape(),
            SymplecticMatrix::identity(1, 2).x_block.shape()
        );
        assert_eq!(sym.view_row(0).to_pauli_string(), ("IX".to_string(), 0));
        assert_eq!(sym.ipowers, SymplecticMatrix::identity(1, 2).ipowers);
    }

    #[test]
    fn test_CX_ZI_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 2);
        sym.z_block[[0, 0]] = true;
        let mut transpose = sym.transpose();
        transpose.cnot(0, 1);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("ZI".to_string(), 0));
        assert_eq!(sym.ipowers, SymplecticMatrix::identity(1, 2).ipowers);
    }

    #[test]
    fn test_CX_IZ_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 2);
        sym.z_block[[0, 1]] = true;
        let mut transpose = sym.transpose();
        transpose.cnot(0, 1);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("ZZ".to_string(), 0));
        assert_eq!(sym.ipowers, SymplecticMatrix::identity(1, 2).ipowers);
    }
}
