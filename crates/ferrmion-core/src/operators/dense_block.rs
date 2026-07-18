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
        let mut bits = self.0;
        std::iter::from_fn(move || {
            if bits == 0 {
                None
            } else {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1; // clear the lowest set bit
                Some(i)
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
                    words[base + i / DenseIndex::BITS].set(i % DenseIndex::BITS, true);
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
        let width = DenseIndex::words_for(self.n_indices);
        self.words()[term * width + index / DenseIndex::BITS].get(index % DenseIndex::BITS)
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

    /// Return a new owned block equal to `self ^ other` (whole-word XOR).
    #[inline]
    pub fn xor<T: AsRef<[DenseIndex]>>(
        &self,
        other: &DenseBlock<T>,
    ) -> DenseBlock<Vec<DenseIndex>> {
        DenseBlock {
            terms: self
                .words()
                .iter()
                .zip(other.words())
                .map(|(a, b)| a.xor(*b))
                .collect(),
            n_indices: self.n_indices,
        }
    }

    /// Return a new owned block equal to `self & other` (whole-word AND).
    #[inline]
    pub fn and<T: AsRef<[DenseIndex]>>(
        &self,
        other: &DenseBlock<T>,
    ) -> DenseBlock<Vec<DenseIndex>> {
        DenseBlock {
            terms: self
                .words()
                .iter()
                .zip(other.words())
                .map(|(a, b)| a.and(*b))
                .collect(),
            n_indices: self.n_indices,
        }
    }

    /// Clone this (possibly borrowed) block into an owned `DenseBlock<Vec<DenseIndex>>`.
    #[inline]
    pub fn to_owned_block(&self) -> DenseBlock<Vec<DenseIndex>> {
        DenseBlock {
            terms: self.words().to_vec(),
            n_indices: self.n_indices,
        }
    }
    /// Convert to a dense boolean array (Python / test boundary).
    pub fn to_bool_array(&self) -> Array1<bool> {
        let mut out = Array1::from_elem(self.n_indices(), false);
        for i in self.iter_ones() {
            out[i] = true;
        }
        out
    }

    /// Convert to a dense boolean matrix, one row per term (Python / test boundary).
    pub fn to_bool_matrix(&self) -> Array2<bool> {
        let n_terms = self.n_terms();
        let n_indices = self.n_indices();
        let mut out = Array2::from_elem((n_terms, n_indices), false);
        for t in 0..n_terms {
            for j in self.get_term(t).iter_ones() {
                out[[t, j]] = true;
            }
        }
        out
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
        let width = DenseIndex::words_for(self.n_indices);
        self.terms.as_mut()[term * width + index / DenseIndex::BITS]
            .set(index % DenseIndex::BITS, value);
    }

    #[inline]
    pub fn set_term(&mut self, term: usize, value: DenseBlock<&[DenseIndex]>) {
        let width = self.term_width();
        let base = term * width;
        self.terms.as_mut()[base..base + width].copy_from_slice(value.terms.as_ref());
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

    /// Transpose the bit matrix: a block of `T = n_terms()` terms of
    /// `N = n_indices()` bits becomes a block of `N` terms of `T` bits with
    /// `out[i][t] == self[t][i]`.
    ///
    /// Implemented as a 64×64 blocked bit transpose so it only ever touches
    /// whole `usize` words (mask/shift/XOR), never individual bits.
    pub fn transpose(self) -> Self {
        let t = self.n_terms();
        let n = self.n_indices();
        let src_width = DenseIndex::words_for(n); // words per source term
        let dst_width = DenseIndex::words_for(t); // words per output term
        let bits = DenseIndex::BITS;
        let src = self.words();
        let mut out = vec![DenseIndex::default(); n * dst_width];

        // Walk 64×64 tiles: `tr` = first source term (output index) in the tile,
        // `ti` = first source index (output term) in the tile.
        let mut tr = 0;
        while tr < t {
            let rows = (t - tr).min(bits);
            let mut ti = 0;
            while ti < n {
                let cols = (n - ti).min(bits);
                // Load one word per source term for this index-word column.
                let src_word_col = ti / bits;
                let mut tile = [0usize; 64];
                for (r, slot) in tile.iter_mut().enumerate().take(rows) {
                    *slot = src[(tr + r) * src_width + src_word_col].0;
                }
                transpose_bit_tile(&mut tile);
                // After transpose, `tile[c]` holds the bits for output term
                // `ti + c`, with source term `tr + r` at local bit `r`.
                let dst_word_col = tr / bits;
                for (c, &word) in tile.iter().enumerate().take(cols) {
                    out[(ti + c) * dst_width + dst_word_col] = DenseIndex(word);
                }
                ti += bits;
            }
            tr += bits;
        }

        Self {
            terms: out,
            n_indices: t,
        }
    }
}

/// In-place transpose of a 64×64 bit matrix packed one row per `usize`
/// (bit `c` of `tile[r]` moves to bit `r` of `tile[c]`). Divide-and-conquer
/// word-level transpose: at each level `j` the low `j` bits of every `2j`-bit
/// group in `tile[i]` are delta-swapped with `tile[i + j]`, working only on
/// whole words (mask/shift/XOR), never individual bits.
#[inline]
fn transpose_bit_tile(tile: &mut [usize; 64]) {
    // `mask` selects the low `j` bits of each `2j`-bit group.
    const STEPS: [(usize, usize); 6] = [
        (32, 0x0000_0000_FFFF_FFFF),
        (16, 0x0000_FFFF_0000_FFFF),
        (8, 0x00FF_00FF_00FF_00FF),
        (4, 0x0F0F_0F0F_0F0F_0F0F),
        (2, 0x3333_3333_3333_3333),
        (1, 0x5555_5555_5555_5555),
    ];
    for (j, mask) in STEPS {
        let mut i = 0;
        while i < 64 {
            if i & j == 0 {
                let d = ((tile[i] >> j) ^ tile[i + j]) & mask;
                tile[i] ^= d << j;
                tile[i + j] ^= d;
            }
            i += 1;
        }
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

    /// Straightforward bit-by-bit transpose, used only as a test oracle for the
    /// word-level blocked `DenseBlock::transpose`.
    fn transpose_reference(block: &DenseBlock<Vec<DenseIndex>>) -> DenseBlock<Vec<DenseIndex>> {
        let t = block.n_terms();
        let n = block.n_indices();
        let mut out = DenseBlock::zeros(n, t);
        for term in 0..t {
            for index in 0..n {
                if block.get_index(term, index) {
                    out.set_index(index, term, true);
                }
            }
        }
        out
    }

    fn random_block(seed: u64, n_terms: usize, n_indices: usize) -> DenseBlock<Vec<DenseIndex>> {
        // Cheap deterministic xorshift so the test needs no rng dependency.
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut b = DenseBlock::zeros(n_terms, n_indices);
        for term in 0..n_terms {
            for index in 0..n_indices {
                if next() & 1 == 1 {
                    b.set_index(term, index, true);
                }
            }
        }
        b
    }

    #[test]
    fn transpose_matches_reference_and_roundtrips() {
        for (t, n) in [
            (1, 1),
            (3, 5),
            (64, 64),
            (65, 3),
            (3, 65),
            (130, 70),
            (70, 130),
        ] {
            let b = random_block(0x9E37_79B9_7F4A_7C15 ^ (t as u64) << 8 ^ n as u64, t, n);
            let bt = b.clone().transpose();
            assert_eq!(bt.n_terms(), n, "transpose n_terms for ({t},{n})");
            assert_eq!(bt.n_indices(), t, "transpose n_indices for ({t},{n})");
            assert_eq!(bt, transpose_reference(&b), "transpose bits for ({t},{n})");
            assert_eq!(bt.transpose(), b, "roundtrip for ({t},{n})");
        }
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
