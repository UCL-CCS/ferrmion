//! Bitpacked storage for symplectic Pauli operators.
//!
//! A [`Block`] stores one bit per qubit for a single X or Z symplectic block,
//! packed into 64-bit machine words via [`bitvec`]. This replaces the previous
//! dense `Array1<bool>` / `Array2<bool>` representation, shrinking memory ~8x
//! and letting the hot paths (symplectic product, Pauli weight, phase
//! accumulation) run as word-level bit operations instead of per-`bool` loops.
//!
//! The symplectic convention is unchanged: a qubit's Pauli is read from the
//! `(x, z)` bit pair — `(false, false) = I`, `(true, false) = X`,
//! `(false, true) = Z`, `(true, true) = Y`.
use bitvec::prelude::*;
use ndarray::{Array1, ArrayView1};
use std::cmp::Ordering;

/// Underlying bit store: little-endian bits packed into `u64` words.
type Store = BitVec<u64, Lsb0>;

/// A single symplectic block (the X or Z part of one Pauli operator), one bit
/// per qubit, bitpacked into `u64` words.
///
/// # Invariant: dead bits are zero
///
/// The unused ("dead") bits above `len()` in the final storage word are always
/// zero. Every constructor starts from `BitVec::repeat(false, _)`, and every
/// mutator (`set`, `swap_bit`, `xor_assign`) only ever touches live indices or
/// preserves zeros (`0 ^ 0 == 0`). The word-level `and_count_ones`,
/// `or_count_ones`, and `xor_assign` rely on this so they can operate on the raw
/// `u64` storage without masking the final partial word.
#[derive(Clone, PartialEq, Eq, Debug, Default, Hash)]
pub struct Block {
    bits: Store,
}

impl Block {
    /// Construct an all-`false` block of length `n` bits.
    pub fn zeros(n: usize) -> Self {
        Self {
            bits: Store::repeat(false, n),
        }
    }

    /// Number of qubits (bits) in this block.
    #[inline]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Whether the block has zero qubits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Read the bit at position `i`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        self.bits[i]
    }

    /// Set the bit at position `i` to `value`.
    #[inline]
    pub fn set(&mut self, i: usize, value: bool) {
        self.bits.set(i, value);
    }

    /// Number of set bits (the Hamming weight of this block).
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.bits.count_ones()
    }

    /// Iterator over the indices of set bits.
    #[inline]
    pub fn iter_ones(&self) -> bitvec::slice::IterOnes<'_, u64, Lsb0> {
        self.bits.iter_ones()
    }

    /// Popcount of `self & other` — the number of positions set in both blocks.
    ///
    /// Used for the symplectic phase term (`z & x`) and the Y count (`x & z`).
    ///
    /// Operates directly on the underlying `u64` storage words. This is sound
    /// because every `Block` keeps its unused "dead" bits (above `len()`) at zero
    /// (see the module-level invariant), so `a & b` in the final partial word can
    /// never count stray bits. Both operands share the same length, hence the same
    /// word count.
    #[inline]
    pub fn and_count_ones(&self, other: &Block) -> usize {
        self.bits
            .as_raw_slice()
            .iter()
            .zip(other.bits.as_raw_slice())
            .map(|(a, b)| (a & b).count_ones() as usize)
            .sum()
    }

    /// Popcount of `self | other` — the number of positions set in either block.
    ///
    /// Used for the Pauli weight of a row (`x | z`, i.e. non-identity qubits).
    /// Word-level for the same reason as [`Block::and_count_ones`].
    #[inline]
    pub fn or_count_ones(&self, other: &Block) -> usize {
        self.bits
            .as_raw_slice()
            .iter()
            .zip(other.bits.as_raw_slice())
            .map(|(a, b)| (a | b).count_ones() as usize)
            .sum()
    }

    /// In-place XOR: `self ^= other`.
    ///
    /// Word-level over the raw storage; preserves the dead-bits-zero invariant
    /// since `0 ^ 0 == 0`.
    #[inline]
    pub fn xor_assign(&mut self, other: &Block) {
        self.bits
            .as_raw_mut_slice()
            .iter_mut()
            .zip(other.bits.as_raw_slice())
            .for_each(|(a, &b)| *a ^= b);
    }

    /// Return a new block equal to `self ^ other`.
    #[inline]
    pub fn xor(&self, other: &Block) -> Block {
        let mut out = self.clone();
        out.xor_assign(other);
        out
    }

    /// Swap the bits at position `i` between `self` and `other`.
    ///
    /// Used by the Clifford Hadamard kernel to exchange X and Z on a qubit.
    #[inline]
    pub fn swap_bit(&mut self, other: &mut Block, i: usize) {
        let a = self.bits[i];
        let b = other.bits[i];
        self.bits.set(i, b);
        other.bits.set(i, a);
    }

    /// Build a block from a dense boolean array view (Python / test boundary).
    pub fn from_bool_view(view: ArrayView1<bool>) -> Self {
        let mut bits = Store::repeat(false, view.len());
        for (i, &b) in view.iter().enumerate() {
            if b {
                bits.set(i, true);
            }
        }
        Self { bits }
    }

    /// Convert to a dense boolean array (Python / test boundary).
    pub fn to_bool_array(&self) -> Array1<bool> {
        let mut out = Array1::from_elem(self.bits.len(), false);
        for i in self.bits.iter_ones() {
            out[i] = true;
        }
        out
    }

    /// Popcount of `self & bools` for a dense boolean array `bools`.
    ///
    /// Bridges the bitpacked operator against a `ZBasisState`, whose state is a
    /// dense `Array1<bool>`.
    #[inline]
    pub fn and_count_bools(&self, bools: &Array1<bool>) -> usize {
        self.bits.iter_ones().filter(|&i| bools[i]).count()
    }

    /// Toggle `bools` in place at every position where `self` is set
    /// (i.e. `bools ^= self`), for a dense boolean array `bools`.
    #[inline]
    pub fn xor_into_bools(&self, bools: &mut Array1<bool>) {
        for i in self.bits.iter_ones() {
            bools[i] = !bools[i];
        }
    }
}

impl PartialOrd for Block {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Block {
    /// Lexicographic order over the bits from index 0, matching the previous
    /// `x_block.iter().cmp(...)` behaviour (`false < true`).
    fn cmp(&self, other: &Self) -> Ordering {
        self.bits.iter().by_vals().cmp(other.bits.iter().by_vals())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn get_set_roundtrip() {
        let mut b = Block::zeros(4);
        assert_eq!(b.len(), 4);
        b.set(1, true);
        b.set(3, true);
        assert!(b.get(1) && b.get(3));
        assert!(!b.get(0) && !b.get(2));
        assert_eq!(b.count_ones(), 2);
        assert_eq!(b.iter_ones().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn bool_roundtrip() {
        let arr = arr1(&[true, false, true, true]);
        let b = Block::from_bool_view(arr.view());
        assert_eq!(b.to_bool_array(), arr);
    }

    #[test]
    fn and_or_counts() {
        let x = Block::from_bool_view(arr1(&[true, true, false, false]).view());
        let z = Block::from_bool_view(arr1(&[false, true, true, false]).view());
        assert_eq!(x.and_count_ones(&z), 1); // only position 1 set in both
        assert_eq!(x.or_count_ones(&z), 3); // positions 0,1,2
    }

    #[test]
    fn xor_ops() {
        let a = Block::from_bool_view(arr1(&[true, false, true]).view());
        let b = Block::from_bool_view(arr1(&[true, true, false]).view());
        assert_eq!(a.xor(&b).to_bool_array(), arr1(&[false, true, true]));
    }

    #[test]
    fn ordering_matches_bool_iter() {
        let a = Block::from_bool_view(arr1(&[false, true]).view());
        let b = Block::from_bool_view(arr1(&[true, false]).view());
        assert!(a < b);
    }

    #[test]
    fn swap_bit_exchanges() {
        let mut a = Block::from_bool_view(arr1(&[true, false]).view());
        let mut b = Block::from_bool_view(arr1(&[false, false]).view());
        a.swap_bit(&mut b, 0);
        assert!(!a.get(0) && b.get(0));
    }
}
