//! Bitpacked storage for symplectic Pauli operators.
//!
//! A symplectic block stores one bit per qubit for a single X or Z part of a
//! Pauli operator, packed into 64-bit machine words. This replaces the previous
//! dense `Array1<bool>` / `Array2<bool>` representation, shrinking memory ~8x
//! and letting the hot paths (symplectic product, Pauli weight, phase
//! accumulation) run as word-level bit operations instead of per-`bool` loops.
//!
//! Two block flavours are provided:
//! - [`Block`] — owned (`Vec<u64>`), used for a single `SymplecticOperator` and
//!   for the mutable working state in `decode`/`try_encode`.
//! - [`BlockRef`] — a borrowed view over a `&[u64]` slice. `SymplecticMatrix`
//!   stores all rows in one contiguous buffer (rows padded to whole `u64`
//!   words), so a row is just a sub-slice viewed as a `BlockRef`. Cloning the
//!   matrix is then a single contiguous copy rather than one heap allocation
//!   per row.
//!
//! The symplectic convention is unchanged: a qubit's Pauli is read from the
//! `(x, z)` bit pair — `(false, false) = I`, `(true, false) = X`,
//! `(false, true) = Z`, `(true, true) = Y`.
//!
//! # Invariant: padding bits are zero
//!
//! A block of `n_bits` qubits is stored in `words_for(n_bits)` words; the unused
//! ("padding") bits above `n_bits` in the final word are always zero. Every
//! constructor zero-fills, and every mutator (`set`, `xor_assign`, the Clifford
//! kernels) only ever writes bit indices `< n_bits`, so XOR/AND/OR of two
//! padding-zero blocks stays padding-zero. The word-level popcounts and XOR rely
//! on this so they can operate on the raw `u64` words without masking the final
//! partial word. Both operands of a binary op always share the same qubit count,
//! hence the same word count.
use ndarray::{Array1, ArrayView1};
use std::cmp::Ordering;

/// A `DenseIndex` is a multi-dimensional index into a dense block of qubits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DenseIndex<const WIDTH: usize>([usize; WIDTH]);

impl<const WIDTH: usize> DenseIndex<WIDTH> {
    /// Number of `u64` words needed to hold `n_bits` bits.
    #[inline]
    pub(crate) fn words_for(n_bits: usize) -> usize {
        n_bits.div_ceil(usize::BITS as usize)
    }
}

impl<const WIDTH: usize> Default for DenseIndex<WIDTH> {
    fn default() -> Self {
        Self([0; WIDTH])
    }
}

/// Iterate the indices of set bits in `words`, lowest first.
///
/// The `n_bits` argument is unused: the padding bits above `n_bits` are always
/// zero (see the module-level invariant), so they never yield an index and no
/// bound is needed. It is kept for call-site symmetry with the other helpers.
fn iter_ones(words: &[u64], _n_bits: usize) -> impl Iterator<Item = usize> + '_ {
    words.iter().enumerate().flat_map(|(w, &word)| {
        let mut bits = word;
        std::iter::from_fn(move || {
            if bits == 0 {
                None
            } else {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1; // clear the lowest set bit
                Some(w * 64 + i)
            }
        })
    })
}

/// Compare two words as LSB-first bit sequences: the operand whose lowest
/// *differing* bit is `0` sorts first. Returns `None` when the words are equal.
#[inline]
fn cmp_word(a: u64, b: u64) -> Option<Ordering> {
    let diff = a ^ b;
    if diff == 0 {
        None
    } else {
        // Isolate the lowest differing bit; whichever operand has `0` there is
        // the smaller bit sequence (bit 0 is most significant for this order).
        let bit = diff & diff.wrapping_neg();
        if a & bit == 0 {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

/// Lexicographic order over the first `an`/`bn` bits (bit 0 first,
/// `false < true`), matching the previous `x_block.iter().cmp(...)` behaviour.
///
/// Relies on the padding-zero invariant so it can compare whole `u64` words;
/// only the partial word at the `min(an, bn)` boundary is masked.
fn cmp_bits(a: &[u64], an: usize, b: &[u64], bn: usize) -> Ordering {
    let l = an.min(bn);
    let full = l / 64;
    for i in 0..full {
        if let Some(ord) = cmp_word(a[i], b[i]) {
            return ord;
        }
    }
    let rem = l % 64;
    if rem != 0 {
        let mask = (1u64 << rem) - 1;
        if let Some(ord) = cmp_word(a[full] & mask, b[full] & mask) {
            return ord;
        }
    }
    // All shared bits equal; the shorter bit sequence sorts first.
    an.cmp(&bn)
}

/// A borrowed view over one symplectic block (`n_bits` qubits packed into a
/// `&[u64]` word slice). Cheap to copy.
#[derive(Clone, Copy, Debug)]
pub struct DenseBlockRef<'a> {
    words: &'a [u64],
    n_bits: usize,
}

impl<'a> DenseBlockRef<'a> {
    /// Construct a view over `words`, interpreting the first `n_bits` bits.
    ///
    /// `words.len()` must equal `words_for(n_bits)`.
    #[inline]
    pub(crate) fn new(words: &'a [u64], n_bits: usize) -> Self {
        debug_assert_eq!(words.len(), words_for(n_bits));
        Self { words, n_bits }
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

    /// Read the bit at position `i`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        (self.words[i >> 6] >> (i & 63)) & 1 != 0
    }

    /// Number of set bits.
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|x| x.count_ones() as usize).sum()
    }

    /// Iterator over the indices of set bits.
    #[inline]
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + 'a {
        iter_ones(self.words, self.n_bits)
    }

    /// Popcount of `self & other` (the `z & x` phase term and the Y count).
    #[inline]
    pub fn and_count_ones(&self, other: DenseBlockRef) -> usize {
        self.words
            .iter()
            .zip(other.words)
            .map(|(x, y)| (x & y).count_ones() as usize)
            .sum()
    }

    /// Popcount of `self | other` (Pauli weight of a row).
    #[inline]
    pub fn or_count_ones(&self, other: DenseBlockRef) -> usize {
        self.words
            .iter()
            .zip(other.words)
            .map(|(x, y)| (x | y).count_ones() as usize)
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

impl PartialEq for DenseBlockRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.n_bits == other.n_bits && self.words == other.words
    }
}
impl Eq for DenseBlockRef<'_> {}

impl PartialOrd for DenseBlockRef<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DenseBlockRef<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_bits(self.words, self.n_bits, other.words, other.n_bits)
    }
}

pub struct DenseProduct<const CHUNKS: usize>([usize; CHUNKS]);

/// An owned symplectic block: `n_bits` qubits packed into `Vec<u64>` words.
#[derive(Clone, PartialEq, Eq, Debug, Default, Hash)]
pub struct DenseBlock {
    words: Vec<u64>,
    n_bits: usize,
}

impl DenseBlock {
    /// Construct an all-`false` block of `n` bits.
    pub fn zeros(n: usize) -> Self {
        Self {
            words: vec![0u64; words_for(n)],
            n_bits: n,
        }
    }

    /// Borrow this block as a [`BlockRef`].
    #[inline]
    pub fn as_ref(&self) -> DenseBlockRef<'_> {
        DenseBlockRef {
            words: &self.words,
            n_bits: self.n_bits,
        }
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

    /// Read the bit at position `i`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        (self.words[i >> 6] >> (i & 63)) & 1 != 0
    }

    /// Set the bit at position `i`.
    #[inline]
    pub fn set(&mut self, i: usize, value: bool) {
        let mask = 1u64 << (i & 63);
        let word = &mut self.words[i >> 6];
        if value {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    /// Number of set bits.
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|x| x.count_ones() as usize).sum()
    }

    /// Iterator over the indices of set bits.
    #[inline]
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        iter_ones(&self.words, self.n_bits)
    }

    /// Popcount of `self & other`.
    #[inline]
    pub fn and_count_ones(&self, other: DenseBlockRef) -> usize {
        self.words
            .iter()
            .zip(other.words)
            .map(|(x, y)| (x & y).count_ones() as usize)
            .sum()
    }

    /// Popcount of `self | other`.
    #[inline]
    pub fn or_count_ones(&self, other: DenseBlockRef) -> usize {
        self.words
            .iter()
            .zip(other.words)
            .map(|(x, y)| (x | y).count_ones() as usize)
            .sum()
    }

    /// In-place XOR: `self ^= other`.
    #[inline]
    pub fn xor_assign(&mut self, other: DenseBlockRef) {
        for (d, s) in self.words.iter_mut().zip(other.words) {
            *d ^= s;
        }
    }

    /// Return a new block equal to `self ^ other`.
    #[inline]
    pub fn xor(&self, other: DenseBlockRef) -> DenseBlock {
        let mut out = self.clone();
        out.xor_assign(other);
        out
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

    /// Convert to a dense boolean array (Python / test boundary).
    pub fn to_bool_array(&self) -> Array1<bool> {
        self.as_ref().to_bool_array()
    }
}

impl PartialOrd for DenseBlock {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DenseBlock {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_bits(&self.words, self.n_bits, &other.words, other.n_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

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
        // 130 qubits spans three u64 words; exercise the padding tail.
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
        assert_eq!(x.and_count_ones(z.as_ref()), 1); // only position 1 set in both
        assert_eq!(x.or_count_ones(z.as_ref()), 3); // positions 0,1,2
    }

    #[test]
    fn xor_ops() {
        let a = DenseBlock::from_bool_view(arr1(&[true, false, true]).view());
        let b = DenseBlock::from_bool_view(arr1(&[true, true, false]).view());
        assert_eq!(
            a.xor(b.as_ref()).to_bool_array(),
            arr1(&[false, true, true])
        );
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
        // and adjacent bits, to exercise the raw-u64 lowest-bit clearing.
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
