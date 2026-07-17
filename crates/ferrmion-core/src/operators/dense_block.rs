//! Bitpacked storage for dense operators including symplectic Pauli operators.
//!
//!
//! The packed word is the [`DenseIndex`] newtype (one `usize` lane by default);
//! a block is a sequence of `DenseIndex` words. [`DenseBlock`] is generic over
//! its backing:
//! - [`DenseBlock<Vec<DenseIndex>>`] — owned, used for a single
//!   `SymplecticOperator` and for the mutable working state in
//!   `decode`/`try_encode`.
//! - [`DenseBlock<&[DenseIndex]>`] — a borrowed view over a slice. `SymplecticMatrix`
//!   stores all rows in one contiguous `Vec<DenseIndex>` buffer (rows padded to
//!   whole words), so a term is just a sub-slice wrapped as a borrowed `DenseBlock`.
//!   Cloning the matrix is then a single contiguous copy rather than one heap
//!   allocation per term.
//!
//!
//! # Invariant: padding bits are zero
//!
//! A block of `n_bits` qubits is stored in `DenseIndex::<1>::words_for(n_bits)` words;
//! the unused ("padding") bits above `n_bits` in the final word are always zero.
//! Every constructor zero-fills, and every mutator (`set`, `xor_assign`, the
//! Clifford kernels) only ever writes bit indices `< n_bits`, so XOR/AND/OR of
//! two padding-zero blocks stays padding-zero. The word-level popcounts and XOR
//! rely on this so they can operate on whole words without masking the final
//! partial word. Both operands of a binary op always share the same qubit count,
//! hence the same word count.
use ndarray::{Array1, Array2, ArrayView1};
use std::cmp::Ordering;

/// A packed word of a dense block: `WIDTH` `usize` lanes holding `WIDTH *
/// usize::BITS` bits. `WIDTH` defaults to 1 (a single machine word), which is
/// what [`DenseBlock`] stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DenseIndex(usize);

impl Default for DenseIndex {
    fn default() -> Self {
        Self(0)
    }
}

impl DenseIndex {
    /// Number of bits in a single lane (`usize::BITS`).
    pub(crate) const BITS: usize = usize::BITS as usize;

    /// Number of `DenseIndex` words needed to hold `n_bits` bits.
    #[inline]
    pub(crate) fn words_for(n_indices: usize) -> usize {
        n_indices.div_ceil(Self::BITS)
    }

    /// Read the bit at local position `local` (`0 <= local < BITS`).
    #[inline]
    pub(crate) fn get(self, local: usize) -> bool {
        (self.0 >> (local % Self::BITS)) & 1 != 0
    }

    /// Set the bit at local position `local` (`0 <= local < BITS`).
    #[inline]
    pub(crate) fn set(&mut self, local: usize, value: bool) {
        let mask = 1usize << (local % Self::BITS);
        let lane = &mut self.0;
        if value {
            *lane |= mask;
        } else {
            *lane &= !mask;
        }
    }

    /// Flip the bit at local position `local` (`0 <= local < BITS`).
    #[inline]
    pub(crate) fn toggle(&mut self, local: usize) {
        self.0 ^= 1usize << (local % Self::BITS);
    }

    /// Lane-wise `self & other`.
    #[inline]
    pub(crate) fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Lane-wise `self | other`.
    #[inline]
    pub(crate) fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Lane-wise `self ^ other`.
    #[inline]
    pub(crate) fn xor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// Number of set bits across all lanes.
    #[inline]
    pub(crate) fn count_ones(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Iterate the local positions of set bits, lowest first.
    #[inline]
    pub(crate) fn iter_ones(self) -> impl Iterator<Item = usize> {
        let base = Self::BITS;
        let mut bits = self.0;
        std::iter::from_fn(move || {
            if bits == 0 {
                None
            } else {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1; // clear the lowest set bit
                Some(base + i)
            }
        })
    }

    /// Compare two words as LSB-first bit sequences (bit 0 most significant):
    /// the operand whose lowest *differing* bit is `0` sorts first. `None` when
    /// the words are equal.
    #[inline]
    pub(crate) fn cmp_bits(self, other: Self) -> Option<Ordering> {
        let diff = self.0 ^ other.0;
        if diff != 0 {
            let bit = diff & diff.wrapping_neg();
            Some(if self.0 & bit == 0 {
                Ordering::Less
            } else {
                Ordering::Greater
            })
        } else {
            None
        }
    }
}

/// A symplectic block: `dim` qubits packed into [`DenseIndex`] words.
///
/// Generic over the word backing `S`: `Vec<DenseIndex>` for an owned block or
/// `&[DenseIndex]` for a borrowed view (which is `Copy`, replacing the former
/// `DenseBlockRef`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct DenseBlock<S = Vec<DenseIndex>> {
    terms: S,
    n_indices: usize,
}

impl From<Array2<bool>> for DenseBlock<Vec<DenseIndex>> {
    fn from(matrix: Array2<bool>) -> Self {
        let words_per_row = (matrix.ncols() + DenseIndex::BITS - 1) / DenseIndex::BITS;
        let mut words = vec![DenseIndex::default(); matrix.nrows() * words_per_row];
        for (r, row) in matrix.rows().into_iter().enumerate() {
            let base = r * words_per_row;
            for (i, &b) in row.iter().enumerate() {
                if b {
                    words[base + i].set(i % DenseIndex::BITS, true);
                }
            }
        }
        Self {
            terms: words,
            n_indices: matrix.ncols(),
        }
    }
}

impl<S: AsRef<[DenseIndex]>> DenseBlock<S> {
    pub fn n_indices(&self) -> usize {
        self.n_indices
    }

    pub fn n_terms(&self) -> usize {
        self.words().len() / DenseIndex::words_for(self.n_indices())
    }

    pub fn term_width(&self) -> usize {
        DenseIndex::words_for(self.n_indices())
    }

    #[inline]
    fn words(&self) -> &[DenseIndex] {
        self.terms.as_ref()
    }

    /// Whether the block has zero qubits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_indices == 0
    }

    /// Borrow this block as a `DenseBlock<&[DenseIndex]>`.
    #[inline]
    pub fn as_ref(&self) -> DenseBlock<&[DenseIndex]> {
        DenseBlock {
            terms: self.words(),
            n_indices: self.n_indices,
        }
    }

    /// Read the bit at position `index` of the term at position `term`.
    ///
    /// # Panics
    ///
    /// Panics if `term` is out of bounds.
    #[inline]
    pub fn get_index(&self, term: usize, index: usize) -> bool {
        self.words()[term].get(index % DenseIndex::BITS)
    }

    /// Read the term at position `term`.
    ///
    /// # Panics
    ///
    /// Panics if `term` is out of bounds.
    #[inline]
    pub fn get_term(&self, term: usize) -> DenseBlock<&[DenseIndex]> {
        let width = DenseIndex::words_for(self.n_indices);
        DenseBlock {
            terms: &self.words()[term * width..(term + 1) * width],
            n_indices: self.n_indices,
        }
    }

    /// Number of set bits.
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words().iter().map(|w| w.count_ones()).sum()
    }

    /// Iterator over the indices of set bits, lowest first.
    #[inline]
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words()
            .iter()
            .enumerate()
            .flat_map(|(e, w)| w.iter_ones().map(move |local| e * DenseIndex::BITS + local))
    }

    /// Popcount of `self & other` (the `z & x` phase term and the Y count).
    #[inline]
    pub fn and_count_ones<T: AsRef<[DenseIndex]>>(&self, other: &DenseBlock<T>) -> usize {
        self.words()
            .iter()
            .zip(other.words())
            .map(|(a, b)| a.and(*b).count_ones())
            .sum()
    }

    /// Popcount of `self | other` (Pauli weight of a row).
    #[inline]
    pub fn or_count_ones<T: AsRef<[DenseIndex]>>(&self, other: &DenseBlock<T>) -> usize {
        self.words()
            .iter()
            .zip(other.words())
            .map(|(a, b)| a.or(*b).count_ones())
            .sum()
    }
    /// Convert to a dense boolean array (Python / test boundary).
    pub fn to_bool_array(&self) -> Array1<bool> {
        let mut out = Array1::from_elem(self.n_indices(), false);
        for i in self.iter_ones() {
            out[i] = true;
        }
        out
    }

    /// Convert to a dense boolean array (Python / test boundary).
    pub fn to_bool_matrix(&self) -> Array2<bool> {
        self.to_bool_array()
            .into_shape_with_order((self.n_terms(), self.n_indices()))
            .expect("Should be able to reshape bool array.")
    }
}

impl<S: AsRef<[DenseIndex]> + AsMut<[DenseIndex]>> DenseBlock<S> {
    /// Set the bit at position `i`.
    ///
    /// # Panics
    ///
    /// Panics if `term` is out of bounds.
    #[inline]
    pub fn set_index(&mut self, term: usize, index: usize, value: bool) {
        self.terms.as_mut()[term + index / DenseIndex::BITS].set(index % DenseIndex::BITS, value);
    }

    #[inline]
    pub fn set_term(&mut self, term: usize, value: DenseBlock<&[DenseIndex]>) {
        let width = self.term_width();
        self.terms.as_mut()[term..term + width].copy_from_slice(value.terms.as_ref());
    }

    /// In-place XOR: `self ^= other`.
    #[inline]
    pub fn xor_assign<T: AsRef<[DenseIndex]>>(&mut self, other: &DenseBlock<T>) {
        for (d, s) in self.terms.as_mut().iter_mut().zip(other.words()) {
            *d = d.xor(*s);
        }
    }
}

impl DenseBlock<Vec<DenseIndex>> {
    /// Construct an all-`false` block of `n` bits.
    pub fn zeros(n_terms: usize, n_indices: usize) -> Self {
        Self {
            terms: vec![DenseIndex::default(); DenseIndex::words_for(n_indices) * n_terms],
            n_indices,
        }
    }

    /// Build a block from a dense boolean array view (Python / test boundary).
    pub fn from_bool_view(view: ArrayView1<bool>) -> Self {
        let mut block = DenseBlock::zeros(1, view.len());
        for (i, &b) in view.iter().enumerate() {
            if b {
                block.set_index(0, i, true);
            }
        }
        block
    }

    pub fn into_iter_terms(self) -> impl DoubleEndedIterator<Item = DenseIndex> {
        self.terms.into_iter()
    }

    pub fn rev(self) -> Self {
        Self {
            terms: self.terms.into_iter().rev().collect(),
            n_indices: self.n_indices,
        }
    }

    pub fn transpose(self) -> Self {
        let mut transposed = Self::zeros(self.n_terms(), self.n_indices());
        for (old_term, old_index) in (0..self.n_terms()).zip(0..self.n_indices()) {
            todo!();
            // transposed.set_index(old_index, old_term, self.get_index(old_term, old_index));
        }
        transposed
    }

    /// Return a new block equal to `self ^ other`.
    #[inline]
    pub fn xor<T: AsRef<[DenseIndex]>>(
        &self,
        other: &DenseBlock<T>,
    ) -> DenseBlock<Vec<DenseIndex>> {
        let mut out = self.clone();
        out.xor_assign(other);
        out
    }
}

impl<'a> DenseBlock<&'a [DenseIndex]> {
    /// Wrap a word slice as a borrowed block interpreting the first `n_bits` bits.
    ///
    /// `words.len()` must equal `DenseIndex::words_for(n_bits)`.
    #[inline]
    pub(crate) fn from_words(terms: &'a [DenseIndex], n_indices: usize) -> Self {
        debug_assert_eq!(terms.len(), DenseIndex::words_for(n_indices));
        Self { terms, n_indices }
    }
}

impl<S: AsRef<[DenseIndex]> + Eq> PartialOrd for DenseBlock<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<S: AsRef<[DenseIndex]> + Eq> Ord for DenseBlock<S> {
    /// Lexicographic order over the first `an`/`bn` bits (bit 0 first,
    /// `false < true`), matching the previous `x_block.iter().cmp(...)` behaviour.
    ///
    /// Relies on the padding-zero invariant so it can compare whole words: the
    /// padding bits of the shorter operand are zero, so comparing the shared words
    /// and then falling back to the bit-length tiebreak reproduces the lexicographic
    /// "shorter sequence sorts first" rule.
    fn cmp(&self, other: &Self) -> Ordering {
        for (x, y) in self.words().iter().zip(other.words().iter()) {
            if let Some(ord) = x.cmp_bits(*y) {
                return ord;
            }
        }
        self.n_indices.cmp(&other.n_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn dense_index_word_primitives() {
        let mut w = DenseIndex::default();
        w.set(0, true);
        w.set(5, true);
        assert!(w.get(0) && w.get(5) && !w.get(1));
        assert_eq!(w.count_ones(), 2);
        assert_eq!(w.iter_ones().collect::<Vec<_>>(), vec![0, 5]);

        let mut v = DenseIndex::default();
        v.set(5, true);
        v.set(7, true);
        assert_eq!(w.and(v).iter_ones().collect::<Vec<_>>(), vec![5]);
        assert_eq!(w.or(v).iter_ones().collect::<Vec<_>>(), vec![0, 5, 7]);
        assert_eq!(w.xor(v).iter_ones().collect::<Vec<_>>(), vec![0, 7]);

        assert_eq!(DenseIndex::words_for(0), 0);
        assert_eq!(DenseIndex::words_for(1), 1);
        assert_eq!(DenseIndex::words_for(DenseIndex::BITS), 1);
        assert_eq!(DenseIndex::words_for(DenseIndex::BITS + 1), 2);
    }

    #[test]
    fn get_set_roundtrip() {
        let mut b = DenseBlock::zeros(1, 4);
        assert_eq!(b.n_indices, 4);
        b.set_index(0, 1, true);
        b.set_index(0, 3, true);
        assert!(b.get_index(0, 1) && b.get_index(0, 3));
        assert!(!b.get_index(0, 0) && !b.get_index(0, 2));
        assert_eq!(b.count_ones(), 2);
        assert_eq!(b.iter_ones().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn get_set_multiword() {
        // 130 qubits spans three words; exercise the padding tail.
        let mut b = DenseBlock::zeros(1, 130);
        b.set_index(0, 0, true);
        b.set_index(0, 64, true);
        b.set_index(0, 129, true);
        assert_eq!(b.count_ones(), 3);
        assert_eq!(b.iter_ones().collect::<Vec<_>>(), vec![0, 64, 129]);
    }

    #[test]
    fn bool_roundtrip() {
        let arr = arr1(&[true, false, true, true]);
        let b = DenseBlock::from_bool_view(arr.view());
        assert_eq!(b.to_bool_array(), arr);
    }

    #[test]
    fn and_or_counts() {
        let x = DenseBlock::from_bool_view(arr1(&[true, true, false, false]).view());
        let z = DenseBlock::from_bool_view(arr1(&[false, true, true, false]).view());
        assert_eq!(x.and_count_ones(&z), 1); // only position 1 set in both
        assert_eq!(x.or_count_ones(&z), 3); // positions 0,1,2
    }

    #[test]
    fn xor_ops() {
        let a = DenseBlock::from_bool_view(arr1(&[true, false, true]).view());
        let b = DenseBlock::from_bool_view(arr1(&[true, true, false]).view());
        assert_eq!(a.xor(&b).to_bool_array(), arr1(&[false, true, true]));
    }

    #[test]
    fn ordering_matches_bool_iter() {
        let a = DenseBlock::from_bool_view(arr1(&[false, true]).view());
        let b = DenseBlock::from_bool_view(arr1(&[true, false]).view());
        assert!(a < b);
        assert!(a.as_ref() < b.as_ref());
    }

    #[test]
    fn iter_ones_multiword_dense() {
        // A set bit in every word incl. the final partial (130 bits = 3 words),
        // and adjacent bits, to exercise the raw-word lowest-bit clearing.
        let mut b = DenseBlock::zeros(1, 130);
        for &i in &[0usize, 1, 63, 64, 65, 127, 128, 129] {
            b.set_index(0, i, true);
        }
        assert_eq!(
            b.iter_ones().collect::<Vec<_>>(),
            vec![0, 1, 63, 64, 65, 127, 128, 129]
        );
    }

    #[test]
    fn ordering_multiword_and_unequal_length() {
        // Differ only in a high word (bit 70): the operand with 0 there is
        // smaller, matching LSB-first (bit 0 most significant) order.
        let mut lo = DenseBlock::zeros(1, 80);
        lo.set_index(0, 3, true);
        let mut hi = DenseBlock::zeros(1, 80);
        hi.set_index(0, 3, true);
        hi.set_index(0, 70, true);
        assert!(lo < hi);
        assert!(lo.as_ref() < hi.as_ref());

        // Equal on all shared bits but different length: the shorter sorts first.
        let short = DenseBlock::from_bool_view(arr1(&[true, false]).view());
        let long = DenseBlock::from_bool_view(arr1(&[true, false, false]).view());
        assert!(short < long);
        assert!(short.as_ref() < long.as_ref());
    }

    #[test]
    fn block_ref_eq() {
        let a = DenseBlock::from_bool_view(arr1(&[true, false, true]).view());
        let b = DenseBlock::from_bool_view(arr1(&[true, false, true]).view());
        let c = DenseBlock::from_bool_view(arr1(&[true, true, false]).view());
        assert_eq!(a.as_ref(), b.as_ref());
        assert_ne!(a.as_ref(), c.as_ref());
    }
}
