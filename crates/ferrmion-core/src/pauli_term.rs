//! Fixed-capacity SIMD-friendly representation of a Majorana operator term.
//!
//! A [`PauliTerm`] holds up to 8 Majorana indices in a 128-bit
//! `[u16; 8]` backing array. Absent slots carry `u16::MAX` — the niche
//! value of [`nonmax::NonMaxU16`] — so the same byte pattern serves as
//! both "no entry" and the SIMD-compare sentinel used by
//! `qubit_term_weight` in the TOPP-HATT optimiser.
//!
//! Compared to the previous `tinyvec::ArrayVec<[u16; 7]>` (14-byte data
//! plus 2-byte length = 16 bytes), this drops the runtime length field,
//! keeps the 128-bit footprint, and allows a single aligned SIMD load.

use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

use nonmax::NonMaxU16;
use wide::i16x8;

/// Stack-allocated, fixed-capacity Majorana index set used as a single
/// Hamiltonian term.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct PauliTerm {
    slots: [u16; PauliTerm::CAPACITY],
}

impl PauliTerm {
    /// Maximum number of indices a single term can hold.
    pub const CAPACITY: usize = 8;

    /// Sentinel value used to mark absent slots. Real Majorana indices
    /// satisfy `idx < 2 * n_modes`, so they never collide with this.
    pub const SENTINEL: u16 = u16::MAX;

    /// The empty term (all slots set to [`Self::SENTINEL`]).
    pub const EMPTY: Self = Self {
        slots: [Self::SENTINEL; Self::CAPACITY],
    };

    /// Construct an empty term.
    #[inline]
    pub const fn new() -> Self {
        Self::EMPTY
    }

    /// Number of present (non-sentinel) slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|&&v| v != Self::SENTINEL).count()
    }

    /// `true` iff no indices are present.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|&v| v == Self::SENTINEL)
    }

    /// Append an index into the first sentinel slot.
    ///
    /// # Panics
    /// - if `idx == u16::MAX` (reserved as the sentinel)
    /// - if the term is already at [`Self::CAPACITY`]
    #[inline]
    pub fn push(&mut self, idx: u16) {
        let nz = NonMaxU16::new(idx).expect("u16::MAX is reserved as the sentinel");
        for slot in &mut self.slots {
            if *slot == Self::SENTINEL {
                *slot = nz.get();
                return;
            }
        }
        panic!("PauliTerm overflow (capacity = {})", Self::CAPACITY);
    }

    /// Remove every present index for which `pred` returns `false`.
    ///
    /// Removed slots are set back to [`Self::SENTINEL`]; the relative
    /// order of surviving slots is preserved (a following
    /// [`Self::sort_unstable`] will compact them to the front).
    #[inline]
    pub fn retain(&mut self, mut pred: impl FnMut(u16) -> bool) {
        for slot in &mut self.slots {
            if *slot != Self::SENTINEL && !pred(*slot) {
                *slot = Self::SENTINEL;
            }
        }
    }

    /// Sort present indices ascending.
    ///
    /// Sentinel slots automatically come last because `u16::MAX` is the
    /// largest value, so after the sort the layout is
    /// `[idx_0, idx_1, ..., idx_{n-1}, MAX, ..., MAX]`.
    #[inline]
    pub fn sort_unstable(&mut self) {
        self.slots.sort_unstable();
    }

    /// Iterate over present indices.
    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            inner: self.slots.iter(),
        }
    }

    /// Smallest present index, if any.
    #[inline]
    pub fn first(&self) -> Option<u16> {
        self.iter().next()
    }

    /// Largest present index, if any.
    #[inline]
    pub fn last(&self) -> Option<u16> {
        self.iter().next_back()
    }

    /// `true` iff `idx` is present.
    #[inline]
    pub fn contains(&self, idx: u16) -> bool {
        idx != Self::SENTINEL && self.slots.contains(&idx)
    }

    /// `true` iff the present indices are in non-decreasing order. The
    /// trailing sentinels never violate sortedness because they hold
    /// the largest possible `u16`.
    #[inline]
    pub fn is_sorted(&self) -> bool {
        self.slots.is_sorted()
    }

    /// Load all 8 lanes as a SIMD vector for the
    /// `qubit_term_weight` hot-path.
    ///
    /// The vector is typed as [`i16x8`] because the wide-crate's
    /// `move_mask` / `reduce_add` helpers live on the signed variant.
    /// Bit-patterns are preserved: `SENTINEL` (= `u16::MAX`) is `-1i16`,
    /// and real Majorana indices satisfy `idx < 2 * n_modes < 2^15`, so
    /// they round-trip through the cast unchanged.
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
        self.slots == other.slots
    }
}
impl Eq for PauliTerm {}

impl Hash for PauliTerm {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.slots.hash(state);
    }
}

impl PartialOrd for PauliTerm {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PauliTerm {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.slots.cmp(&other.slots)
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
        for (i, idx) in iter.into_iter().enumerate() {
            assert!(
                i < Self::CAPACITY,
                "PauliTerm overflow (capacity = {})",
                Self::CAPACITY
            );
            assert!(
                idx != Self::SENTINEL,
                "u16::MAX is reserved as the sentinel"
            );
            term.slots[i] = idx;
        }
        term
    }
}

impl<'a> IntoIterator for &'a PauliTerm {
    type Item = u16;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over the present (non-sentinel) indices of a [`PauliTerm`].
pub struct Iter<'a> {
    inner: core::slice::Iter<'a, u16>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = u16;
    #[inline]
    fn next(&mut self) -> Option<u16> {
        self.inner
            .by_ref()
            .find(|&&v| v != PauliTerm::SENTINEL)
            .copied()
    }
}

impl<'a> DoubleEndedIterator for Iter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<u16> {
        self.inner
            .by_ref()
            .rfind(|&&v| v != PauliTerm::SENTINEL)
            .copied()
    }
}

// Compile-time sanity: the struct must be exactly one 128-bit register.
const _: () = assert!(core::mem::size_of::<PauliTerm>() == 16);
const _: () = assert!(core::mem::align_of::<PauliTerm>() == 16);

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
    fn is_sorted_true_for_compacted_sentinels() {
        let mut t = PauliTerm::EMPTY;
        t.push(1);
        t.push(3);
        t.push(5);
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
    fn ordering_matches_lexicographic_on_slots() {
        let a = pauli_term![0, 1, 2];
        let b = pauli_term![0, 1, 3];
        assert!(a < b);
    }
}
