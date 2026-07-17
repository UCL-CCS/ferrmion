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
use crate::operators::Clifford;
use crate::operators::{DenseBlock, DenseIndex};
use crate::spaces::Qubit;
use crate::states::ZBasisState;
use ndarray::{arr2, Array1, Array2};
use num_complex::Complex64;
use num_complex::{c64, ComplexFloat};
use std::ops::Mul;

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
///
/// The X and Z blocks are stored bitpacked (one bit per qubit) via [`DenseBlock`].
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SymplecticOperator {
    ipower: u8,
    x: DenseBlock,
    z: DenseBlock,
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
            x: DenseBlock::from_bool_view(x_block.view()),
            z: DenseBlock::from_bool_view(z_block.view()),
        }
    }

    /// Construct a new [`SymplecticOperator`] directly from bitpacked [`DenseBlock`]s.
    pub fn from_blocks(ipower: u8, x: DenseBlock, z: DenseBlock) -> Self {
        Self { ipower, x, z }
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
            x: DenseBlock::zeros(1, n_modes),
            z: DenseBlock::zeros(1, n_modes),
        }
    }

    /// Return a borrowed [`SymplecticOperatorView`] of this operator.
    pub fn view(&self) -> SymplecticOperatorView<'_> {
        SymplecticOperatorView {
            ipower: self.ipower,
            x: self.x.as_ref(),
            z: self.z.as_ref(),
        }
    }

    /// Return a borrowed view of the bitpacked X block.
    pub fn x_bits(&self) -> DenseBlock<&[DenseIndex]> {
        self.x.as_ref()
    }

    /// Return a borrowed view of the bitpacked Z block.
    pub fn z_bits(&self) -> DenseBlock<&[DenseIndex]> {
        self.z.as_ref()
    }

    /// Return the X block as a dense boolean array.
    pub fn x_bools(&self) -> Array1<bool> {
        self.x.to_bool_array()
    }

    /// Return the Z block as a dense boolean array.
    pub fn z_bools(&self) -> Array1<bool> {
        self.z.to_bool_array()
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
        let mut pauli_string = String::with_capacity(self.x.n_indices());
        let mut ipower = self.ipower;
        for i in 0..self.x.n_indices() {
            let x = self.x.get_index(0, i);
            let z = self.z.get_index(0, i);
            if x && z {
                ipower += 3;
            };
            pauli_string.push(Pauli::from((x, z)).into());
        }
        (pauli_string, (ipower % 4))
    }
}

impl PauliWeight for SymplecticOperator {
    fn pauli_weight(&self) -> usize {
        self.x.or_count_ones(&self.z)
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
        // XOR the X and Z blocks (Pauli-group multiplication in symplectic form).
        let x_product = self.x.xor(&rhs.x);
        let z_product = self.z.xor(&rhs.z);

        // Symplectic phase: +2 for each qubit where left-Z and right-X coincide.
        let ipower =
            (self.ipower as usize + rhs.ipower as usize + 2 * self.z.and_count_ones(&rhs.x)) % 4;

        Self::Output {
            ipower: ipower as u8,
            x: x_product,
            z: z_product,
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
        // Accumulate ipower from (self-Z . rhs-X) before mutating the Z block.
        let ipower =
            (self.ipower as usize + rhs.ipower as usize + 2 * self.z.and_count_ones(&rhs.x)) % 4;
        self.ipower = ipower as u8;

        // In-place XOR for the X and Z blocks.
        self.x.xor_assign(&rhs.x);
        self.z.xor_assign(&rhs.z);
    }
}

impl Mul<ZBasisState> for SymplecticOperator {
    type Output = ZBasisState;

    fn mul(self, rhs: ZBasisState) -> Self::Output {
        let mut state = rhs.state;
        let mut phase_factor = c64(-1., 0.).powi(((2 * self.z.and_count_ones(&state)) % 4) as i32);
        let y_count: i32 = self.x.and_count_ones(&self.z) as i32;
        phase_factor *= c64(0., 1.).powi(y_count);

        let coefficient = rhs.coefficient * phase_factor;

        state.xor_assign(&self.x);

        ZBasisState::from_block(state, coefficient)
    }
}

/// Pauli operator encoded in symplectic (XZ) form.
///
/// # Examples
///
/// ```
/// use ferrmion_core::operators::{DenseBlock, SymplecticOperatorView};
/// use ndarray::arr1;
///
/// let x = DenseBlock::from_bool_view(arr1(&[true, false]).view());
/// let z = DenseBlock::from_bool_view(arr1(&[false, true]).view());
/// let view = SymplecticOperatorView::new(0, x.as_ref(), z.as_ref());
/// let (pauli, ipower) = view.to_pauli_string();
/// assert_eq!(pauli, "XZ");
/// assert_eq!(ipower, 0);
/// ```
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub struct SymplecticOperatorView<'sym> {
    ipower: u8,
    x: DenseBlock<&'sym [DenseIndex]>,
    z: DenseBlock<&'sym [DenseIndex]>,
}

impl Qubit for SymplecticOperatorView<'_> {}

impl<'sym> SymplecticOperatorView<'sym> {
    /// Construct a new [`SymplecticOperatorView`] from borrowed bitpacked blocks.
    pub fn new(
        ipower: u8,
        x: DenseBlock<&'sym [DenseIndex]>,
        z: DenseBlock<&'sym [DenseIndex]>,
    ) -> Self {
        Self { ipower, x, z }
    }

    /// Convert to a Pauli string representation and its associated `i`-power.
    pub fn to_pauli_string(self) -> (String, u8) {
        let mut pauli_string = String::with_capacity(self.x.n_terms());
        let mut ipower = self.ipower;
        for i in 0..self.x.n_indices() {
            let x = self.x.get_index(0, i);
            let z = self.z.get_index(0, i);
            if x && z {
                ipower += 3;
            };
            pauli_string.push(Pauli::from((x, z)).into());
        }
        (pauli_string, (ipower % 4))
    }
}

impl PauliWeight for SymplecticOperatorView<'_> {
    fn pauli_weight(&self) -> usize {
        self.x.or_count_ones(&self.z)
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
        let phase_factor = c64(0., 1.)
            .powi(((self.ipower as usize + 2 * self.z.and_count_ones(&state)) % 4) as i32);

        let coefficient = rhs.coefficient * phase_factor;

        state.xor_assign(&self.x);

        ZBasisState::from_block(state, coefficient)
    }
}

impl Mul<&mut ZBasisState> for SymplecticOperatorView<'_> {
    type Output = ();

    fn mul(self, rhs: &mut ZBasisState) {
        let mut phase_factor =
            c64(-1., 0.).powi(((2 * self.z.and_count_ones(&rhs.state)) % 4) as i32);
        let y_count: i32 = self.x.and_count_ones(&self.z) as i32;
        phase_factor *= c64(0., 1.).powi(y_count);

        rhs.state.xor_assign(&self.x);
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
        self.x
            .cmp(&other.x)
            .then(self.z.cmp(&other.z))
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
    /// All X blocks in one contiguous buffer, row-major: row `r` occupies
    /// `x_words[r*words_per_row .. (r+1)*words_per_row]`. Storing the whole
    /// matrix in one allocation (rather than a `Vec` of per-row `DenseBlock`s)
    /// keeps `clone` to a single contiguous copy.
    x_words: Vec<DenseIndex>,
    /// All Z blocks, same row-major layout.
    z_words: Vec<DenseIndex>,
    /// Per-row `i`-power (phase exponent mod 4).
    ipowers: Array1<u8>,
    /// Number of qubits (bits per row).
    n_qubits: usize,
    /// `DenseIndex` words per row = `DenseIndex::words_for(n_qubits)`.
    words_per_row: usize,
    /// Number of operator rows.
    n_rows: usize,
}

impl Qubit for SymplecticMatrix {}

impl SymplecticMatrix {
    /// Pack the rows of a dense boolean matrix into one contiguous word buffer,
    /// `words_per_row` words per row.
    fn pack(matrix: &Array2<bool>, words_per_row: usize) -> Vec<DenseIndex> {
        let bits = DenseIndex::BITS;
        let mut words = vec![DenseIndex::default(); matrix.nrows() * words_per_row];
        for (r, row) in matrix.rows().into_iter().enumerate() {
            let base = r * words_per_row;
            for (i, &b) in row.iter().enumerate() {
                if b {
                    words[base + i / bits].set(i % bits, true);
                }
            }
        }
        words
    }

    #[inline]
    fn x_row(&self, row: usize) -> &[DenseIndex] {
        let base = row * self.words_per_row;
        &self.x_words[base..base + self.words_per_row]
    }

    #[inline]
    fn z_row(&self, row: usize) -> &[DenseIndex] {
        let base = row * self.words_per_row;
        &self.z_words[base..base + self.words_per_row]
    }

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
    /// assert_eq!(mat.n_rows(), 1);
    /// assert_eq!(mat.ipower(0), 0); // no Y operators, so ipower = 0
    /// ```
    pub fn new(x_block: Array2<bool>, z_block: Array2<bool>) -> Self {
        let n_qubits = x_block.ncols();
        let n_rows = x_block.nrows();
        let words_per_row = DenseIndex::words_for(n_qubits);
        let x_words = Self::pack(&x_block, words_per_row);
        let z_words = Self::pack(&z_block, words_per_row);
        let mut ipowers = Array1::from_elem(n_rows, 0u8);
        for (r, ip) in ipowers.iter_mut().enumerate() {
            let base = r * words_per_row;
            let xr = DenseBlock::from_words(&x_words[base..base + words_per_row], n_qubits);
            let zr = DenseBlock::from_words(&z_words[base..base + words_per_row], n_qubits);
            *ip = (xr.and_count_ones(&zr) % 4) as u8;
        }
        Self {
            x_words,
            z_words,
            ipowers,
            n_qubits,
            words_per_row,
            n_rows,
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
    /// assert_eq!(mat.ipower(0), 2);
    /// ```
    pub fn with_ipowers(x_block: Array2<bool>, z_block: Array2<bool>, ipowers: Array1<u8>) -> Self {
        let n_qubits = x_block.ncols();
        let n_rows = x_block.nrows();
        let words_per_row = DenseIndex::words_for(n_qubits);
        Self {
            x_words: Self::pack(&x_block, words_per_row),
            z_words: Self::pack(&z_block, words_per_row),
            ipowers,
            n_qubits,
            words_per_row,
            n_rows,
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
        let words_per_row = DenseIndex::words_for(n_qubits);
        Self {
            ipowers: Array1::from_elem(n_modes, 0),
            x_words: vec![DenseIndex::default(); n_modes * words_per_row],
            z_words: vec![DenseIndex::default(); n_modes * words_per_row],
            n_qubits,
            words_per_row,
            n_rows: n_modes,
        }
    }

    /// Number of qubits (columns) each operator acts on.
    #[inline]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of operators (rows).
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Borrow the bitpacked X block of the given row.
    #[inline]
    pub fn row_x(&self, row: usize) -> DenseBlock<&[DenseIndex]> {
        DenseBlock::from_words(self.x_row(row), self.n_qubits)
    }

    /// Borrow the bitpacked Z block of the given row.
    #[inline]
    pub fn row_z(&self, row: usize) -> DenseBlock<&[DenseIndex]> {
        DenseBlock::from_words(self.z_row(row), self.n_qubits)
    }

    /// Borrow all per-row `i`-powers.
    #[inline]
    pub fn ipowers(&self) -> &Array1<u8> {
        &self.ipowers
    }

    /// The `i`-power of the given row.
    #[inline]
    pub fn ipower(&self, row: usize) -> u8 {
        self.ipowers[row]
    }

    /// Set the X bit at `(row, qubit)`.
    #[inline]
    pub fn set_x(&mut self, row: usize, qubit: usize, value: bool) {
        let bits = DenseIndex::BITS;
        let word = row * self.words_per_row + qubit / bits;
        self.x_words[word].set(qubit % bits, value);
    }

    /// Set the Z bit at `(row, qubit)`.
    #[inline]
    pub fn set_z(&mut self, row: usize, qubit: usize, value: bool) {
        let bits = DenseIndex::BITS;
        let word = row * self.words_per_row + qubit / bits;
        self.z_words[word].set(qubit % bits, value);
    }

    /// Set the `i`-power of the given row.
    #[inline]
    pub fn set_ipower(&mut self, row: usize, value: u8) {
        self.ipowers[row] = value;
    }

    /// Return the X block as a dense boolean matrix.
    pub fn x_bools(&self) -> Array2<bool> {
        let mut out = Array2::from_elem((self.n_rows, self.n_qubits), false);
        for r in 0..self.n_rows {
            for i in self.row_x(r).iter_ones() {
                out[[r, i]] = true;
            }
        }
        out
    }

    /// Return the Z block as a dense boolean matrix.
    pub fn z_bools(&self) -> Array2<bool> {
        let mut out = Array2::from_elem((self.n_rows, self.n_qubits), false);
        for r in 0..self.n_rows {
            for i in self.row_z(r).iter_ones() {
                out[[r, i]] = true;
            }
        }
        out
    }

    /// Concatenate the X and Z blocks into a single `[x_block | z_block]` boolean matrix.
    ///
    /// This is the layout used to exchange symplectic matrices with Python.
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
    /// let combined = mat.to_concatenated();
    /// assert_eq!(combined.shape(), &[1, 4]);
    /// ```
    pub fn to_concatenated(&self) -> Array2<bool> {
        let mut out = Array2::from_elem((self.n_rows, 2 * self.n_qubits), false);
        for r in 0..self.n_rows {
            for i in self.row_x(r).iter_ones() {
                out[[r, i]] = true;
            }
            for i in self.row_z(r).iter_ones() {
                out[[r, self.n_qubits + i]] = true;
            }
        }
        out
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
        SymplecticOperatorView::new(self.ipowers[row], self.row_x(row), self.row_z(row))
    }

    /// Iterate over all rows as [`SymplecticOperatorView`]s.
    pub fn iter_rows(&self) -> impl Iterator<Item = SymplecticOperatorView<'_>> {
        (0..self.n_rows).map(move |row| self.view_row(row))
    }

    /// Return a new matrix containing the given rows, in the given order.
    pub fn select_rows(&self, indices: &[usize]) -> Self {
        let wpr = self.words_per_row;
        let mut x_words = vec![DenseIndex::default(); indices.len() * wpr];
        let mut z_words = vec![DenseIndex::default(); indices.len() * wpr];
        for (new_row, &old_row) in indices.iter().enumerate() {
            let (new_base, old_base) = (new_row * wpr, old_row * wpr);
            x_words[new_base..new_base + wpr]
                .copy_from_slice(&self.x_words[old_base..old_base + wpr]);
            z_words[new_base..new_base + wpr]
                .copy_from_slice(&self.z_words[old_base..old_base + wpr]);
        }
        Self {
            x_words,
            z_words,
            ipowers: Array1::from_iter(indices.iter().map(|&i| self.ipowers[i])),
            n_qubits: self.n_qubits,
            words_per_row: wpr,
            n_rows: indices.len(),
        }
    }

    /// Type safe transpose.
    ///
    /// Used for optimisation routines which use conjugations with
    /// [`Clifford`] gates to update an encoding.
    pub(crate) fn transpose(&mut self) -> SymplecticMatrixTranspose<'_> {
        SymplecticMatrixTranspose {
            x_words: &mut self.x_words,
            z_words: &mut self.z_words,
            ipowers: &mut self.ipowers,
            n_qubits: self.n_qubits,
            words_per_row: self.words_per_row,
            n_rows: self.n_rows,
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
        let mut indices: Vec<usize> = (0..self.n_rows).collect();
        indices.sort_unstable_by(|&a, &b| self.view_row(a).cmp(&self.view_row(b)));
        *self = self.select_rows(&indices);
    }
}

impl PauliWeight for SymplecticMatrix {
    fn pauli_weight(&self) -> usize {
        self.iter_rows().map(|row| row.pauli_weight()).sum()
    }
}

impl Mul<&mut ZBasisState> for SymplecticMatrix {
    type Output = ();

    fn mul(self, rhs: &mut ZBasisState) {
        for row in 0..self.n_rows {
            #[allow(clippy::let_unit_value)]
            let _ = self.view_row(row).mul(&mut *rhs);
        }
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
        assert_eq!(result.x_bools(), ndarray::arr1(&[true, true, true]));
        assert_eq!(result.z_bools(), ndarray::arr1(&[true, true, true]));

        let result = zzz * xxx.view();
        assert_eq!(result.ipower(), 2);
        assert_eq!(result.x_bools(), ndarray::arr1(&[true, true, true]));
        assert_eq!(result.z_bools(), ndarray::arr1(&[true, true, true]));
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

/// A qubit-indexed view over a [`SymplecticMatrix`] for Clifford conjugation.
///
/// Clifford gates act on a fixed qubit index across every operator. This borrows
/// the matrix's contiguous X/Z word buffers mutably and applies each gate by
/// touching the relevant qubit bit in every row.
pub(crate) struct SymplecticMatrixTranspose<'inner> {
    x_words: &'inner mut Vec<DenseIndex>,
    z_words: &'inner mut Vec<DenseIndex>,
    ipowers: &'inner mut Array1<u8>,
    n_qubits: usize,
    words_per_row: usize,
    n_rows: usize,
}

// Clifford conjugation functions
//
// Applies P -> CPC
//
// Currently this works given a single qubit index at a time,
// but applied accross all operators in a [`SymplecticMatrix`].
// They could probably be made a little more general and faster by
// allowing a slice input and applying all gates in tandem.
impl SymplecticMatrixTranspose<'_> {
    /// Apply Clifford H Operator
    // $P \to H P H$
    /// -1 for each Y
    /// Z -> X and X -> Z
    pub(crate) fn haddamard(&mut self, qubit: usize) {
        let word_off = qubit / DenseIndex::BITS;
        let local = qubit % DenseIndex::BITS;
        for r in 0..self.n_rows {
            let w = r * self.words_per_row + word_off;
            let x_set = self.x_words[w].get(local);
            let z_set = self.z_words[w].get(local);
            if x_set && z_set {
                // -1 for each Y
                self.ipowers[r] = self.ipowers[r].wrapping_add(2);
            }
            // Z -> X and X -> Z (swap the two bits).
            self.x_words[w].set(local, z_set);
            self.z_words[w].set(local, x_set);
        }
    }
    /// Apply the Clifford S operator.
    // $P \to S P S$
    /// -1 for each X
    /// Z -> Z ^ X
    pub(crate) fn phasegate(&mut self, qubit: usize) {
        let word_off = qubit / DenseIndex::BITS;
        let local = qubit % DenseIndex::BITS;
        for r in 0..self.n_rows {
            let w = r * self.words_per_row + word_off;
            if self.x_words[w].get(local) {
                // -1 for each X
                self.ipowers[r] = self.ipowers[r].wrapping_add(3);
                // Z -> Z ^ X (X is set here, so toggle Z).
                self.z_words[w].toggle(local);
            }
        }
    }

    // Transform a [`Pauli`] operator by this Clifford operator.
    // $P \to CX P CX$
    pub(crate) fn cnot(&mut self, control: usize, target: usize) {
        let bits = DenseIndex::BITS;
        let (control_off, control_local) = (control / bits, control % bits);
        let (target_off, target_local) = (target / bits, target % bits);
        for r in 0..self.n_rows {
            let base = r * self.words_per_row;
            // X: target ^= control
            if self.x_words[base + control_off].get(control_local) {
                self.x_words[base + target_off].toggle(target_local);
            }
            // Z: control ^= target
            if self.z_words[base + target_off].get(target_local) {
                self.z_words[base + control_off].toggle(control_local);
            }
        }
    }

    pub(crate) fn hamming_weights(&self) -> Array1<usize> {
        // One Hamming weight per qubit: the count of non-identity Paulis
        // (x | z set) touching that qubit across all operators.
        let mut weights = Array1::from_elem(self.n_qubits, 0usize);
        let wpr = self.words_per_row;
        for r in 0..self.n_rows {
            let base = r * wpr;
            let xr = DenseBlock::from_words(&self.x_words[base..base + wpr], self.n_qubits);
            let zr = DenseBlock::from_words(&self.z_words[base..base + wpr], self.n_qubits);
            for i in xr.iter_ones() {
                weights[i] += 1;
            }
            for i in zr.iter_ones() {
                if !xr.get_index(0, i) {
                    weights[i] += 1;
                }
            }
        }
        weights
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
        assert_eq!(*sym.ipowers(), *SymplecticMatrix::identity(2, 1).ipowers());
    }
    #[test]
    fn test_HXH() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.set_x(0, 0, true);
        sym.set_x(1, 0, true);
        let mut transpose = sym.transpose();
        transpose.haddamard(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("Z".to_string(), 0));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("Z".to_string(), 0));
    }
    #[test]
    fn test_HYH() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.set_x(0, 0, true);
        sym.set_z(0, 0, true);
        sym.set_ipower(0, 1);

        sym.set_x(1, 0, true);
        sym.set_z(1, 0, true);
        sym.set_ipower(1, 1);

        let mut transpose = sym.transpose();
        transpose.haddamard(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("Y".to_string(), 2));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("Y".to_string(), 2));
    }

    #[test]
    fn test_HZH() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.set_z(0, 0, true);
        sym.set_z(1, 0, true);
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
        assert_eq!(*sym.ipowers(), *SymplecticMatrix::identity(1, 1).ipowers());
    }

    #[test]
    fn test_SXS() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.set_x(0, 0, true);
        sym.set_x(1, 0, true);
        let mut transpose = sym.transpose();
        transpose.phasegate(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("Y".to_string(), 2));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("Y".to_string(), 2));
    }

    #[test]
    fn test_SYS() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.set_x(0, 0, true);
        sym.set_z(0, 0, true);
        sym.set_ipower(0, 1);

        sym.set_x(1, 0, true);
        sym.set_z(1, 0, true);
        sym.set_ipower(1, 1);
        let mut transpose = sym.transpose();
        transpose.phasegate(0);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("X".to_string(), 0));
        assert_eq!(sym.view_row(1).to_pauli_string(), ("X".to_string(), 0));
    }

    #[test]
    fn test_SZS() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(2, 1);
        sym.set_z(0, 0, true);
        sym.set_z(1, 0, true);
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
        assert_eq!(*sym.ipowers(), *SymplecticMatrix::identity(2, 2).ipowers());
    }

    #[test]
    fn test_CX_XI_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 2);
        sym.set_x(0, 0, true);
        let mut transpose = sym.transpose();
        transpose.cnot(0, 1);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("XX".to_string(), 0));
        assert_eq!(*sym.ipowers(), *SymplecticMatrix::identity(1, 2).ipowers());
    }

    #[test]
    fn test_CX_IX_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 2);
        sym.set_x(0, 1, true);
        {
            let mut transpose = sym.transpose();
            transpose.cnot(0, 1);
        }
        // Dimensions are preserved: 1 operator row acting on 2 qubits.
        assert_eq!(sym.n_rows(), 1);
        assert_eq!(sym.n_qubits(), 2);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("IX".to_string(), 0));
        assert_eq!(*sym.ipowers(), *SymplecticMatrix::identity(1, 2).ipowers());
    }

    #[test]
    fn test_CX_ZI_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 2);
        sym.set_z(0, 0, true);
        let mut transpose = sym.transpose();
        transpose.cnot(0, 1);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("ZI".to_string(), 0));
        assert_eq!(*sym.ipowers(), *SymplecticMatrix::identity(1, 2).ipowers());
    }

    #[test]
    fn test_CX_IZ_CX() {
        let mut sym: SymplecticMatrix = SymplecticMatrix::identity(1, 2);
        sym.set_z(0, 1, true);
        let mut transpose = sym.transpose();
        transpose.cnot(0, 1);
        assert_eq!(sym.view_row(0).to_pauli_string(), ("ZZ".to_string(), 0));
        assert_eq!(*sym.ipowers(), *SymplecticMatrix::identity(1, 2).ipowers());
    }
}
