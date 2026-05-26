//! Fixed-capacity SIMD-friendly representation of a Majorana operator term.
//!
//! A [`MajoranaTerm`] holds up to 7 Majorana indices in a 16-byte struct
//! (`len: u16` + `slots: [u16; 7]`). Absent slots carry `u16::MAX` —
//! the niche value of [`nonmax::NonMaxU16`] — so the same byte pattern
//! serves both as the "no entry" marker and as the SIMD-compare sentinel
//! used by `qubit_term_weight` in the TOPP-HATT optimiser.
//!
//! # Invariants
//! - `len <= CAPACITY`.
//! - `slots[..len]` are present indices; `slots[len..]` are
//!   [`SENTINEL`](MajoranaTerm::SENTINEL).
//!
//! # SIMD layout
//! `as_lanes()` transmutes the entire 16 bytes into an `i16x8`.
//! Lane 0 holds `len`; lanes 1..7 hold the slot data. The SIMD kernel
//! in `qubit_term_weight` masks lane 0 out of each `move_mask` result,
//! so the length value never contributes to parity counts.

use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

use nonmax::NonMaxU16;
use wide::i16x8;

/// Stack-allocated, fixed-capacity Majorana index set used as a single
/// Hamiltonian term.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct MajoranaTerm {
    len: u16,
    slots: [u16; MajoranaTerm::CAPACITY],
}

impl MajoranaTerm {
    pub const CAPACITY: usize = 7;

    /// Sentinel value used to mark absent slots. Real Majorana indices
    /// satisfy `idx < 2 * n_modes`, so they never collide with this.
    pub const SENTINEL: u16 = u16::MAX;

    /// The empty term.
    pub const EMPTY: Self = Self {
        len: 0,
        slots: [Self::SENTINEL; Self::CAPACITY],
    };

    #[inline]
    pub const fn new() -> Self {
        Self::EMPTY
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[u16] {
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
        let pos = self.len as usize;
        assert!(
            pos < Self::CAPACITY,
            "MajoranaTerm overflow (capacity = {})",
            Self::CAPACITY
        );
        self.slots[pos] = nz.get();
        self.len += 1;
    }

    /// Remove every present index for which `pred` returns `false`,
    /// re-compacting the survivors and restoring the trailing-sentinel
    /// invariant.
    #[inline]
    pub fn retain(&mut self, mut pred: impl FnMut(u16) -> bool) {
        let old_len = self.len as usize;
        let mut write = 0usize;
        for read in 0..old_len {
            let v = self.slots[read];
            if pred(v) {
                self.slots[write] = v;
                write += 1;
            }
        }
        for slot in &mut self.slots[write..old_len] {
            *slot = Self::SENTINEL;
        }
        self.len = write as u16;
    }

    /// Sort present indices ascending. Trailing sentinels are untouched.
    #[inline]
    pub fn sort_unstable(&mut self) {
        let n = self.len as usize;
        self.slots[..n].sort_unstable();
    }

    #[inline]
    pub fn iter(&self) -> core::iter::Copied<core::slice::Iter<'_, u16>> {
        self.as_slice().iter().copied()
    }

    #[inline]
    pub fn first(&self) -> Option<u16> {
        (self.len > 0).then(|| self.slots[0])
    }

    #[inline]
    pub fn last(&self) -> Option<u16> {
        (self.len > 0).then(|| self.slots[self.len as usize - 1])
    }

    #[inline]
    pub fn contains(&self, idx: u16) -> bool {
        self.as_slice().contains(&idx)
    }

    #[inline]
    pub fn is_sorted(&self) -> bool {
        self.as_slice().is_sorted()
    }

    /// Load the entire 16-byte struct as an `i16x8` SIMD vector.
    ///
    /// Lane 0 holds `len`; lanes 1..7 hold the slot data. Callers
    /// that compare against Majorana indices must mask lane 0 out of
    /// the `move_mask` result (see `qubit_term_weight`).
    #[inline(always)]
    pub fn as_lanes(&self) -> i16x8 {
        // SAFETY: MajoranaTerm is #[repr(C, align(16))] with size 16,
        // identical to [i16; 8].
        let raw: [i16; 8] = unsafe { core::mem::transmute(*self) };
        i16x8::new(raw)
    }
}

impl Default for MajoranaTerm {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl PartialEq for MajoranaTerm {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.slots == other.slots
    }
}
impl Eq for MajoranaTerm {}

impl Hash for MajoranaTerm {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl PartialOrd for MajoranaTerm {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MajoranaTerm {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl core::fmt::Debug for MajoranaTerm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl FromIterator<u16> for MajoranaTerm {
    fn from_iter<I: IntoIterator<Item = u16>>(iter: I) -> Self {
        let mut term = Self::EMPTY;
        for idx in iter {
            term.push(idx);
        }
        term
    }
}

impl<'a> IntoIterator for &'a MajoranaTerm {
    type Item = u16;
    type IntoIter = core::iter::Copied<core::slice::Iter<'a, u16>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

const _: () = assert!(core::mem::size_of::<MajoranaTerm>() == 16);
const _: () = assert!(core::mem::align_of::<MajoranaTerm>() == 16);

/// Construct a [`MajoranaTerm`] from a list of integer literals.
#[macro_export]
macro_rules! majorana_term {
    () => { $crate::majorana_term::MajoranaTerm::EMPTY };
    ($($idx:expr),+ $(,)?) => {{
        let mut __t = $crate::majorana_term::MajoranaTerm::EMPTY;
        $( __t.push($idx as u16); )+
        __t
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_term() {
        let t = MajoranaTerm::EMPTY;
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert!(t.first().is_none());
        assert!(t.last().is_none());
        assert!(t.iter().next().is_none());
    }

    #[test]
    fn push_and_iter() {
        let mut t = MajoranaTerm::EMPTY;
        t.push(3);
        t.push(1);
        t.push(7);
        assert_eq!(t.len(), 3);
        let collected: Vec<u16> = t.iter().collect();
        assert_eq!(collected, vec![3, 1, 7]);
    }

    #[test]
    fn sort_compacts_to_front() {
        let mut t = majorana_term![5, 1, 3];
        t.sort_unstable();
        let collected: Vec<u16> = t.iter().collect();
        assert_eq!(collected, vec![1, 3, 5]);
        assert_eq!(t.first(), Some(1));
        assert_eq!(t.last(), Some(5));
    }

    #[test]
    fn retain_then_push_then_sort() {
        let mut t = majorana_term![0, 1, 2, 3];
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
        let mut t = majorana_term![0, 1, 2, 3];
        t.retain(|i| i != 1);
        let collected: Vec<u16> = t.iter().collect();
        assert_eq!(collected, vec![0, 2, 3]);
        let lanes = t.as_lanes().to_array();
        // Lane 0 = len (3); lanes 1..3 = present; lanes 4..7 = -1 (sentinel).
        assert_eq!(lanes, [3, 0, 2, 3, -1, -1, -1, -1]);
    }

    #[test]
    fn is_sorted_true_after_sort() {
        let mut t = majorana_term![5, 1, 3];
        t.sort_unstable();
        assert!(t.is_sorted());
    }

    #[test]
    fn contains_ignores_sentinel() {
        let t = majorana_term![0, 1, 2];
        assert!(t.contains(0));
        assert!(t.contains(1));
        assert!(!t.contains(99));
        assert!(!t.contains(u16::MAX));
    }

    #[test]
    fn as_lanes_layout() {
        let mut t = MajoranaTerm::EMPTY;
        t.push(2);
        t.push(5);
        let arr = t.as_lanes().to_array();
        // Lane 0 = len (2); lanes 1..2 = slots; lanes 3..7 = sentinel.
        assert_eq!(arr, [2, 2, 5, -1, -1, -1, -1, -1]);
    }

    #[test]
    #[should_panic]
    fn push_sentinel_panics() {
        let mut t = MajoranaTerm::EMPTY;
        t.push(u16::MAX);
    }

    #[test]
    #[should_panic]
    fn push_overflow_panics() {
        let mut t = MajoranaTerm::EMPTY;
        for i in 0..=MajoranaTerm::CAPACITY as u16 {
            t.push(i);
        }
    }

    #[test]
    fn ordering_matches_lexicographic_on_present_indices() {
        let a = majorana_term![0, 1];
        let b = majorana_term![0, 1, 2];
        assert!(a < b);

        let c = majorana_term![0, 1, 3];
        assert!(b < c);
    }
}
