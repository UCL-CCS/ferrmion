//! Bitpacked storage for symplectic Pauli operators.
//!
//! A symplectic block stores one bit per qubit for a single X or Z part of a
//! Pauli operator, packed into machine words. This replaces the previous dense
//! `Array1<bool>` / `Array2<bool>` representation, shrinking memory ~8x and
//! letting the hot paths (symplectic product, Pauli weight, phase accumulation)
//! run as word-level bit operations instead of per-`bool` loops.
//!
//! The packed word is the [`DenseIndex`] newtype (one `usize` lane by default);
//! a block is a sequence of `DenseIndex` words. [`DenseBlock`] is generic over
//! its backing:
//! - [`DenseBlock<Vec<DenseIndex>>`] — owned, used for a single
//!   `SymplecticOperator` and for the mutable working state in
//!   `decode`/`try_encode`.
//! - [`DenseBlock<&[DenseIndex]>`] — a borrowed view over a slice. `SymplecticMatrix`
//!   stores all rows in one contiguous `Vec<DenseIndex>` buffer (rows padded to
//!   whole words), so a row is just a sub-slice wrapped as a borrowed `DenseBlock`.
//!   Cloning the matrix is then a single contiguous copy rather than one heap
//!   allocation per row.
//!
//! The symplectic convention is unchanged: a qubit's Pauli is read from the
//! `(x, z)` bit pair — `(false, false) = I`, `(true, false) = X`,
//! `(false, true) = Z`, `(true, true) = Y`.
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
use ndarray::{Array1, ArrayView1};
use std::cmp::Ordering;

/// A packed word of a dense block: `WIDTH` `usize` lanes holding `WIDTH *
/// usize::BITS` bits. `WIDTH` defaults to 1 (a single machine word), which is
/// what [`DenseBlock`] stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DenseIndex<const WIDTH: usize = 1>([usize; WIDTH]);

impl<const WIDTH: usize> Default for DenseIndex<WIDTH> {
    fn default() -> Self {
        Self([0; WIDTH])
    }
}

impl<const WIDTH: usize> DenseIndex<WIDTH> {
    /// Number of bits in a single lane (`usize::BITS`).
    pub(crate) const LANE_BITS: usize = usize::BITS as usize;

    /// Number of bits held by one `DenseIndex` word.
    pub(crate) const BITS: usize = WIDTH * Self::LANE_BITS;

    /// Number of `DenseIndex` words needed to hold `n_bits` bits.
    #[inline]
    pub(crate) fn words_for(n_bits: usize) -> usize {
        n_bits.div_ceil(Self::BITS)
    }

    /// Read the bit at local position `local` (`0 <= local < BITS`).
    #[inline]
    pub(crate) fn get(self, local: usize) -> bool {
        (self.0[local / Self::LANE_BITS] >> (local % Self::LANE_BITS)) & 1 != 0
    }

    /// Set the bit at local position `local` (`0 <= local < BITS`).
    #[inline]
    pub(crate) fn set(&mut self, local: usize, value: bool) {
        let mask = 1usize << (local % Self::LANE_BITS);
        let lane = &mut self.0[local / Self::LANE_BITS];
        if value {
            *lane |= mask;
        } else {
            *lane &= !mask;
        }
    }

    /// Flip the bit at local position `local` (`0 <= local < BITS`).
    #[inline]
    pub(crate) fn toggle(&mut self, local: usize) {
        self.0[local / Self::LANE_BITS] ^= 1usize << (local % Self::LANE_BITS);
    }

    /// Lane-wise `self & other`.
    #[inline]
    pub(crate) fn and(self, other: Self) -> Self {
        let mut out = [0usize; WIDTH];
        for (o, (a, b)) in out.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *o = a & b;
        }
        Self(out)
    }

    /// Lane-wise `self | other`.
    #[inline]
    pub(crate) fn or(self, other: Self) -> Self {
        let mut out = [0usize; WIDTH];
        for (o, (a, b)) in out.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *o = a | b;
        }
        Self(out)
    }

    /// Lane-wise `self ^ other`.
    #[inline]
    pub(crate) fn xor(self, other: Self) -> Self {
        let mut out = [0usize; WIDTH];
        for (o, (a, b)) in out.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *o = a ^ b;
        }
        Self(out)
    }

    /// Number of set bits across all lanes.
    #[inline]
    pub(crate) fn count_ones(self) -> usize {
        self.0.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate the local positions of set bits, lowest first.
    #[inline]
    pub(crate) fn iter_ones(self) -> impl Iterator<Item = usize> {
        (0..WIDTH).flat_map(move |lane| {
            let base = lane * Self::LANE_BITS;
            let mut bits = self.0[lane];
            std::iter::from_fn(move || {
                if bits == 0 {
                    None
                } else {
                    let i = bits.trailing_zeros() as usize;
                    bits &= bits - 1; // clear the lowest set bit
                    Some(base + i)
                }
            })
        })
    }

    /// Compare two words as LSB-first bit sequences (bit 0 most significant):
    /// the operand whose lowest *differing* bit is `0` sorts first. `None` when
    /// the words are equal.
    #[inline]
    pub(crate) fn cmp_bits(self, other: Self) -> Option<Ordering> {
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            let diff = a ^ b;
            if diff != 0 {
                let bit = diff & diff.wrapping_neg();
                return Some(if a & bit == 0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                });
            }
        }
        None
    }
}

/// Lexicographic order over the first `an`/`bn` bits (bit 0 first,
/// `false < true`), matching the previous `x_block.iter().cmp(...)` behaviour.
///
/// Relies on the padding-zero invariant so it can compare whole words: the
/// padding bits of the shorter operand are zero, so comparing the shared words
/// and then falling back to the bit-length tiebreak reproduces the lexicographic
/// "shorter sequence sorts first" rule.
fn cmp_bits(a: &[DenseIndex], an: usize, b: &[DenseIndex], bn: usize) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        if let Some(ord) = x.cmp_bits(*y) {
            return ord;
        }
    }
    an.cmp(&bn)
}

/// A symplectic block: `n_bits` qubits packed into [`DenseIndex`] words.
///
/// Generic over the word backing `S`: `Vec<DenseIndex>` for an owned block or
/// `&[DenseIndex]` for a borrowed view (which is `Copy`, replacing the former
/// `DenseBlockRef`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct DenseBlock<S = Vec<DenseIndex>> {
    words: S,
    n_bits: usize,
}

impl<S: AsRef<[DenseIndex]>> DenseBlock<S> {
    #[inline]
    fn words(&self) -> &[DenseIndex] {
        self.words.as_ref()
    }

    /// Number of qubits (bits) in this block.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_bits
    }

    /// Whether the block has zero qubits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_bits == 0
    }

    /// Borrow this block as a `DenseBlock<&[DenseIndex]>`.
    #[inline]
    pub fn as_ref(&self) -> DenseBlock<&[DenseIndex]> {
        DenseBlock {
            words: self.words(),
            n_bits: self.n_bits,
        }
    }

    /// Read the bit at position `i`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        self.words()[i / DenseIndex::<1>::BITS].get(i % DenseIndex::<1>::BITS)
    }

    /// Number of set bits.
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words().iter().map(|w| w.count_ones()).sum()
    }

    /// Iterator over the indices of set bits, lowest first.
    #[inline]
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words().iter().enumerate().flat_map(|(e, w)| {
            w.iter_ones()
                .map(move |local| e * DenseIndex::<1>::BITS + local)
        })
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
        let mut out = Array1::from_elem(self.n_bits, false);
        for i in self.iter_ones() {
            out[i] = true;
        }
        out
    }
}

impl<S: AsRef<[DenseIndex]> + AsMut<[DenseIndex]>> DenseBlock<S> {
    /// Set the bit at position `i`.
    #[inline]
    pub fn set(&mut self, i: usize, value: bool) {
        self.words.as_mut()[i / DenseIndex::<1>::BITS].set(i % DenseIndex::<1>::BITS, value);
    }

    /// In-place XOR: `self ^= other`.
    #[inline]
    pub fn xor_assign<T: AsRef<[DenseIndex]>>(&mut self, other: &DenseBlock<T>) {
        for (d, s) in self.words.as_mut().iter_mut().zip(other.words()) {
            *d = d.xor(*s);
        }
    }
}

impl DenseBlock<Vec<DenseIndex>> {
    /// Construct an all-`false` block of `n` bits.
    pub fn zeros(n: usize) -> Self {
        Self {
            words: vec![DenseIndex::default(); DenseIndex::<1>::words_for(n)],
            n_bits: n,
        }
    }

    /// Build a block from a dense boolean array view (Python / test boundary).
    pub fn from_bool_view(view: ArrayView1<bool>) -> Self {
        let mut block = DenseBlock::zeros(view.len());
        for (i, &b) in view.iter().enumerate() {
            if b {
                block.set(i, true);
            }
        }
        block
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
    /// `words.len()` must equal `DenseIndex::<1>::words_for(n_bits)`.
    #[inline]
    pub(crate) fn from_words(words: &'a [DenseIndex], n_bits: usize) -> Self {
        debug_assert_eq!(words.len(), DenseIndex::<1>::words_for(n_bits));
        Self { words, n_bits }
    }
}

impl<S: AsRef<[DenseIndex]> + Eq> PartialOrd for DenseBlock<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<S: AsRef<[DenseIndex]> + Eq> Ord for DenseBlock<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_bits(self.words(), self.n_bits, other.words(), other.n_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn dense_index_word_primitives() {
        let mut w = DenseIndex::<1>::default();
        w.set(0, true);
        w.set(5, true);
        assert!(w.get(0) && w.get(5) && !w.get(1));
        assert_eq!(w.count_ones(), 2);
        assert_eq!(w.iter_ones().collect::<Vec<_>>(), vec![0, 5]);

        let mut v = DenseIndex::<1>::default();
        v.set(5, true);
        v.set(7, true);
        assert_eq!(w.and(v).iter_ones().collect::<Vec<_>>(), vec![5]);
        assert_eq!(w.or(v).iter_ones().collect::<Vec<_>>(), vec![0, 5, 7]);
        assert_eq!(w.xor(v).iter_ones().collect::<Vec<_>>(), vec![0, 7]);

        assert_eq!(DenseIndex::<1>::words_for(0), 0);
        assert_eq!(DenseIndex::<1>::words_for(1), 1);
        assert_eq!(DenseIndex::<1>::words_for(DenseIndex::<1>::BITS), 1);
        assert_eq!(DenseIndex::<1>::words_for(DenseIndex::<1>::BITS + 1), 2);
    }

    #[test]
    fn get_set_roundtrip() {
        let mut b = DenseBlock::zeros(4);
        assert_eq!(b.len(), 4);
        b.set(1, true);
        b.set(3, true);
        assert!(b.get(1) && b.get(3));
        assert!(!b.get(0) && !b.get(2));
        assert_eq!(b.count_ones(), 2);
        assert_eq!(b.iter_ones().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn get_set_multiword() {
        // 130 qubits spans three words; exercise the padding tail.
        let mut b = DenseBlock::zeros(130);
        b.set(0, true);
        b.set(64, true);
        b.set(129, true);
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
        let mut b = DenseBlock::zeros(130);
        for &i in &[0usize, 1, 63, 64, 65, 127, 128, 129] {
            b.set(i, true);
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
        let mut lo = DenseBlock::zeros(80);
        lo.set(3, true);
        let mut hi = DenseBlock::zeros(80);
        hi.set(3, true);
        hi.set(70, true);
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
