//! Fixed-capacity SIMD-friendly representation of a Majorana operator term.
//!
//! A [`PauliTerm`] holds up to 8 Majorana indices. The first `len` entries
//! of the backing `[u16; 8]` slot array carry the present indices; the
//! remaining slots are always set to `u16::MAX` — the niche value of
//! [`nonmax::NonMaxU16`] — so the same byte pattern serves as both
//! "no entry" and the SIMD-compare sentinel used by `qubit_term_weight`
//! in the TOPP-HATT optimiser.
//!
//! Compared to the previous `tinyvec::ArrayVec<[u16; 7]>` (2-byte length
//! plus 14-byte data = 16 bytes), this widens the data array from 7 to 8
//! lanes so the whole term fits in a single 128-bit SIMD register, at
//! a cost of 2 extra bytes per term.

use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

use nonmax::NonMaxU16;
use wide::i16x8;

/// Stack-allocated, fixed-capacity Majorana index set used as a single
/// Hamiltonian term.
///
/// # Invariants
/// - `len <= CAPACITY`
/// - `slots[len..]` is always [`SENTINEL`](Self::SENTINEL).
///
/// The trailing-sentinel invariant lets [`as_lanes`](Self::as_lanes)
/// hand the whole array to a single SIMD compare without needing a
/// length mask — padding lanes never equal a real Majorana index
/// (`idx < 2 * n_modes < 2^15`).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PauliTerm {
    len: u16,
    slots: [u16; PauliTerm::CAPACITY],
}

impl PauliTerm {
    /// Maximum number of indices a single term can hold.
    pub const CAPACITY: usize = 8;

    /// Sentinel value used to mark absent slots. Real Majorana indices
    /// satisfy `idx < 2 * n_modes`, so they never collide with this.
    pub const SENTINEL: u16 = u16::MAX;

    /// The empty term.
    pub const EMPTY: Self = Self {
        len: 0,
        slots: [Self::SENTINEL; Self::CAPACITY],
    };

    /// Construct an empty term.
    #[inline]
    pub const fn new() -> Self {
        Self::EMPTY
    }

    /// Number of present indices.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` iff no indices are present.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the present indices as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u16] {
        // SAFETY: invariant `len <= CAPACITY` guarantees this index is in bounds.
        unsafe { self.slots.get_unchecked(..self.len as usize) }
    }

    /// Append an index.
    ///
    /// # Panics
    /// - if `idx == u16::MAX` (reserved as the sentinel)
    /// - if the term is already at [`Self::CAPACITY`]
    #[inline]
    pub fn push(&mut self, idx: u16) {
        let nz = NonMaxU16::new(idx).expect("u16::MAX is reserved as the sentinel");
        let len = self.len as usize;
        assert!(
            len < Self::CAPACITY,
            "PauliTerm overflow (capacity = {})",
            Self::CAPACITY
        );
        self.slots[len] = nz.get();
        self.len += 1;
    }

    /// Remove every present index for which `pred` returns `false`,
    /// re-compacting the survivors and restoring the trailing-sentinel
    /// invariant.
    #[inline]
    pub fn retain(&mut self, mut pred: impl FnMut(u16) -> bool) {
        let len = self.len as usize;
        let mut write = 0usize;
        for read in 0..len {
            let v = self.slots[read];
            if pred(v) {
                self.slots[write] = v;
                write += 1;
            }
        }
        for slot in &mut self.slots[write..len] {
            *slot = Self::SENTINEL;
        }
        self.len = write as u16;
    }

    /// Sort present indices ascending. The trailing-sentinel slots are
    /// untouched (they were already at the end and remain `SENTINEL`).
    #[inline]
    pub fn sort_unstable(&mut self) {
        let len = self.len as usize;
        self.slots[..len].sort_unstable();
    }

    /// Iterate over present indices.
    #[inline]
    pub fn iter(&self) -> core::iter::Copied<core::slice::Iter<'_, u16>> {
        self.as_slice().iter().copied()
    }

    /// First (smallest after `sort_unstable`) present index, if any.
    #[inline]
    pub fn first(&self) -> Option<u16> {
        self.as_slice().first().copied()
    }

    /// Last (largest after `sort_unstable`) present index, if any.
    #[inline]
    pub fn last(&self) -> Option<u16> {
        self.as_slice().last().copied()
    }

    /// `true` iff `idx` is present.
    #[inline]
    pub fn contains(&self, idx: u16) -> bool {
        self.as_slice().contains(&idx)
    }

    /// `true` iff present indices are in non-decreasing order.
    #[inline]
    pub fn is_sorted(&self) -> bool {
        self.as_slice().is_sorted()
    }

    /// Load all 8 lanes as a SIMD vector for the
    /// `qubit_term_weight` hot-path.
    ///
    /// The vector is typed as [`i16x8`] because the wide-crate's
    /// `move_mask` helper lives on the signed variant. Bit-patterns are
    /// preserved: `SENTINEL` (= `u16::MAX`) is `-1i16`, and real Majorana
    /// indices satisfy `idx < 2 * n_modes < 2^15`, so they round-trip
    /// through the cast unchanged.
    #[inline(always)]
    pub fn as_lanes(&self) -> i16x8 {
        // SAFETY: `[u16; 8]` and `[i16; 8]` have identical size, alignment,
        // and bit-level representation. The transmute is purely a type
        // reinterpretation.
        let lanes: [i16; 8] = unsafe { core::mem::transmute(self.slots) };
        i16x8::new(lanes)
    }
}

impl Default for PauliTerm {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl PartialEq for PauliTerm {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // The trailing-sentinel invariant means the full slots arrays
        // are equal iff the present indices match.
        self.len == other.len && self.slots == other.slots
    }
}
impl Eq for PauliTerm {}

impl Hash for PauliTerm {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl PartialOrd for PauliTerm {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PauliTerm {
    /// Lexicographic order on the present indices — matches the previous
    /// `tinyvec::ArrayVec` behaviour, so a prefix sorts before its
    /// extension: `[0, 1] < [0, 1, 2]`.
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl core::fmt::Debug for PauliTerm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl FromIterator<u16> for PauliTerm {
    fn from_iter<I: IntoIterator<Item = u16>>(iter: I) -> Self {
        let mut term = Self::EMPTY;
        for idx in iter {
            term.push(idx);
        }
        term
    }
}

impl<'a> IntoIterator for &'a PauliTerm {
    type Item = u16;
    type IntoIter = core::iter::Copied<core::slice::Iter<'a, u16>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Construct a [`PauliTerm`] from a list of integer literals.
///
/// Indices are cast to `u16`. Passing `u16::MAX` is a runtime panic
/// (it is reserved as the sentinel).
#[macro_export]
macro_rules! pauli_term {
    () => { $crate::pauli_term::PauliTerm::EMPTY };
    ($($idx:expr),+ $(,)?) => {{
        let mut __t = $crate::pauli_term::PauliTerm::EMPTY;
        $( __t.push($idx as u16); )+
        __t
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_term() {
        let t = PauliTerm::EMPTY;
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert!(t.first().is_none());
        assert!(t.last().is_none());
        assert!(t.iter().next().is_none());
    }

    #[test]
    fn push_and_iter() {
        let mut t = PauliTerm::EMPTY;
        t.push(3);
        t.push(1);
        t.push(7);
        assert_eq!(t.len(), 3);
        let collected: Vec<u16> = t.iter().collect();
        assert_eq!(collected, vec![3, 1, 7]);
    }

    #[test]
    fn sort_compacts_to_front() {
        let mut t = pauli_term![5, 1, 3];
        t.sort_unstable();
        let collected: Vec<u16> = t.iter().collect();
        assert_eq!(collected, vec![1, 3, 5]);
        assert_eq!(t.first(), Some(1));
        assert_eq!(t.last(), Some(5));
    }

    #[test]
    fn retain_then_push_then_sort() {
        // Mirrors the pattern in reduce_hamiltonian.
        let mut t = pauli_term![0, 1, 2, 3];
        let n = t.len();
        t.retain(|i| ![1u16, 3].contains(&i));
        while t.len() < n {
            t.push(99);
        }
        t.sort_unstable();
        let collected: Vec<u16> = t.iter().collect();
        assert_eq!(collected, vec![0, 2, 99, 99]);
    }

    #[test]
    fn retain_preserves_sentinel_invariant() {
        let mut t = pauli_term![0, 1, 2, 3];
        t.retain(|i| i != 1);
        // slots[..len] are present indices in original order; slots[len..] sentinels.
        let collected: Vec<u16> = t.iter().collect();
        assert_eq!(collected, vec![0, 2, 3]);
        // The slot array used by `as_lanes` must have sentinels in trailing positions.
        let lanes = t.as_lanes().to_array();
        assert_eq!(lanes, [0, 2, 3, -1, -1, -1, -1, -1]);
    }

    #[test]
    fn is_sorted_true_after_sort() {
        let mut t = pauli_term![5, 1, 3];
        t.sort_unstable();
        assert!(t.is_sorted());
    }

    #[test]
    fn contains_ignores_sentinel() {
        let t = pauli_term![0, 1, 2];
        assert!(t.contains(0));
        assert!(t.contains(1));
        assert!(!t.contains(99));
        assert!(!t.contains(u16::MAX));
    }

    #[test]
    fn as_lanes_layout() {
        let mut t = PauliTerm::EMPTY;
        t.push(2);
        t.push(5);
        let arr = t.as_lanes().to_array();
        // The sentinel (u16::MAX) is `-1i16` in two's-complement.
        assert_eq!(arr, [2, 5, -1, -1, -1, -1, -1, -1]);
    }

    #[test]
    #[should_panic]
    fn push_sentinel_panics() {
        let mut t = PauliTerm::EMPTY;
        t.push(u16::MAX);
    }

    #[test]
    #[should_panic]
    fn push_overflow_panics() {
        let mut t = PauliTerm::EMPTY;
        for i in 0..=PauliTerm::CAPACITY as u16 {
            t.push(i);
        }
    }

    #[test]
    fn ordering_matches_lexicographic_on_present_indices() {
        // A shorter prefix sorts before its extension; matches the previous
        // tinyvec semantics.
        let a = pauli_term![0, 1];
        let b = pauli_term![0, 1, 2];
        assert!(a < b);

        let c = pauli_term![0, 1, 3];
        assert!(b < c);
    }
}
