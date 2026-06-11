//! Fermionic Operators
use crate::operators::ladder::LadderOperator;
use crate::spaces::Fermion;
use crate::utils::COEFFICIENT_TOLERANCE;
use ahash::{HashMap, HashMapExt};
use log::debug;
use ndarray::{arr0, s, Dimension};
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis, IntoDimension};
use num_complex::Complex64;
use num_complex::{c64, ComplexFloat};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::result::Result;
use tinyvec::ArrayVec;

/// Maximum length of majorana indices which are allowed in stack-allocated ArrayVecs.
const MAX_MAJORANAS: usize = 7;

/// Minimum number of independent terms in a [`FermionSparse`] before
/// `append_fermion_sparse` expands them across rayon worker threads. Below this,
/// the serial path is used, avoiding rayon's scheduling overhead on small inputs.
const PARALLEL_TERM_THRESHOLD: usize = 256;

/// Minimum number of terms a single rayon task expands in the parallel path
/// (`with_min_len`), so tiny per-term work is batched rather than scheduled
/// individually.
const PARALLEL_CHUNK: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mode(u16);

impl Mode {
    pub fn new(index: u16) -> Self {
        Self(index)
    }

    pub fn as_usize(self) -> usize {
        (self.0 & 0x7FFF) as usize
    }
}

impl Default for Mode {
    fn default() -> Self {
        Self(u16::MAX)
    }
}

/*
Fermion
*/
/// A product of fermionic ladder operators.
///
/// # Example
///
/// ```
/// use ferrmion_core::operators::{FermionProduct, LadderOperator};
/// use num_complex::Complex64;
///
/// let fp = FermionProduct::new(
///     vec![LadderOperator::Creation, LadderOperator::Annihilation],
///     vec![0, 1],
///     Complex64::new(1.0, 0.0),
/// ).unwrap();
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct FermionProduct {
    action: Vec<LadderOperator>,
    indices: Vec<usize>,
    coefficient: Complex64,
}

/// Error type for failure to construct [`FermionProduct`]
#[derive(Debug, PartialEq, Clone)]
pub struct FermionProductError;

impl Fermion for FermionProduct {}

impl FermionProduct {
    /// Constructor for [`FermionProduct`]
    pub fn new(
        action: Vec<LadderOperator>,
        indices: Vec<usize>,
        coefficient: Complex64,
    ) -> Result<Self, FermionProductError> {
        if action.len() != indices.len() {
            Err(FermionProductError)
        } else {
            Ok(Self {
                action,
                indices,
                coefficient,
            })
        }
    }
}
/// Fermion operator with coefficients in matrix form.
///
/// <div class="warning">
/// Coeffients are in Spin-orbit format.
/// For spatial orbital index "i", the spin-up mode is at $2i$ and the spin down mode is at $2i+1$
/// </div>
///
#[derive(Debug, PartialEq, Clone)]
pub struct FermionMatrix {
    action: Vec<LadderOperator>,
    coefficients: ArrayD<f64>,
}

/// Error raised by failure to contruct [`FermionMatrix`]
#[derive(Debug, PartialEq, Clone)]
pub struct FermionMatrixError;

impl Fermion for FermionMatrix {}

impl FermionMatrix {
    /// Constructor for [`FermionMatrix`]
    pub fn new(
        action: Vec<LadderOperator>,
        coefficients: ArrayD<f64>,
    ) -> Result<Self, FermionMatrixError> {
        // Check we have enough ladder operators
        // and a square/cube/... matrix
        if action.len() != coefficients.ndim()
            || !coefficients
                .shape()
                .iter()
                .all(|s| *s == coefficients.shape()[0])
        {
            return Err(FermionMatrixError);
        }
        let mut out = Self {
            action,
            coefficients,
        };
        out.zero_disallowed_terms();
        Ok(out)
    }

    /// The ladder operator sequence of this term.
    pub fn action(&self) -> &[LadderOperator] {
        &self.action
    }

    /// The signature string of this term, e.g. `"+-"` or `"++--"`.
    pub fn signature(&self) -> String {
        self.action
            .iter()
            .map(|op| match op {
                LadderOperator::Creation => '+',
                LadderOperator::Annihilation => '-',
            })
            .collect()
    }

    /// View of the dense coefficient tensor.
    pub fn coefficients(&self) -> ArrayViewD<'_, f64> {
        self.coefficients.view()
    }

    /// The number of fermionic modes (the length of each coefficient dimension).
    pub fn n_modes(&self) -> usize {
        self.coefficients.shape().first().copied().unwrap_or(0)
    }

    /// Zero out terms with disallowed fermion operator index combinations.
    ///
    /// For instance, with action [+,+,-,-] any operators with indices i,j,k,l
    /// where i = j or k == l are zeroed out.
    ///
    /// Note: Current implementation only checks for operators of dimension 4.
    ///
    ///
    // Can be made more general when dim > 4 is added.
    fn zero_disallowed_terms(&mut self) {
        use crate::operators::LadderOperator::{Annihilation, Creation};
        if self.coefficients.ndim() == 4 {
            match self.action.as_slice() {
                [Annihilation, Annihilation, Creation, Creation]
                | [Creation, Creation, Annihilation, Annihilation] => {
                    for i in 0..self.coefficients.shape()[0] {
                        self.coefficients
                            .slice_mut(s![i, i, .., ..])
                            .assign(&arr0(0.0));
                        self.coefficients
                            .slice_mut(s![.., .., i, i])
                            .assign(&arr0(0.0));
                    }
                }
                [Creation, Annihilation, Creation, Annihilation]
                | [Annihilation, Creation, Annihilation, Creation] => {
                    for i in 0..self.coefficients.shape()[0] {
                        // Keeping only [i, i, i, ..]
                        self.coefficients
                            .slice_mut(s![i, ..i, i, ..])
                            .assign(&arr0(0.0));
                        self.coefficients
                            .slice_mut(s![i, i + 1.., i, ..])
                            .assign(&arr0(0.0));

                        // Keeping only [.., i, i, i]
                        self.coefficients
                            .slice_mut(s![.., i, ..i, i])
                            .assign(&arr0(0.0));
                        self.coefficients
                            .slice_mut(s![.., i, i + 1.., i])
                            .assign(&arr0(0.0));
                    }
                }
                _ => {}
            }
        }
    }
}

/// Fermion operator in sparse form.
///
/// Each index is non-empty and each coefficient is non-zero.
#[derive(Debug, PartialEq)]
pub struct FermionSparse {
    action: Vec<LadderOperator>,
    indices: Array2<usize>,
    coefficients: Array1<Complex64>,
}

/// Error type for failure in [`FermionSparse`] constructor.
#[derive(Debug, PartialEq, Clone)]
pub struct FermionSparseError;

impl Fermion for FermionSparse {}

impl FermionSparse {
    /// Constructor for [`FermionSparse`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::{FermionSparse, LadderOperator};
    /// use ndarray::{arr1, arr2};
    /// use num_complex::Complex64;
    ///
    /// let fs = FermionSparse::new(
    ///     vec![LadderOperator::Creation, LadderOperator::Annihilation],
    ///     arr2(&[[0, 1], [2, 3]]),
    ///     arr1(&[Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]),
    /// ).unwrap();
    /// ```
    pub fn new(
        action: Vec<LadderOperator>,
        indices: Array2<usize>,
        coefficients: Array1<Complex64>,
    ) -> Result<Self, FermionSparseError> {
        if coefficients.len() != indices.len_of(Axis(0)) || action.len() != indices.len_of(Axis(1))
        {
            return Err(FermionSparseError);
        };

        Ok(Self {
            action,
            indices,
            coefficients,
        })
    }
}

impl From<FermionMatrix> for FermionSparse {
    fn from(mft: FermionMatrix) -> FermionSparse {
        let n_nonzero = mft.coefficients.iter().filter(|&v| *v != 0.).count();
        let mut sparse_indices: Array2<usize> = Array2::zeros((n_nonzero, mft.action.len()));
        let mut sparse_coefficients: Array1<Complex64> = Array1::from_elem(n_nonzero, c64(0., 0.));
        mft.coefficients
            .indexed_iter()
            .filter(|(_, &v)| v != 0.)
            .enumerate()
            .for_each(|(count, (ind, &v))| {
                sparse_indices
                    .row_mut(count)
                    .assign(&ind.into_dimension().as_array_view());
                sparse_coefficients[count] += v;
            });
        FermionSparse::new(mft.action, sparse_indices, sparse_coefficients)
            .expect("Conversion from MatrixFermionTerm should be validated.")
    }
}

#[cfg(test)]
mod fermion_tests {
    use super::*;
    use crate::utils::vector_kron;
    use ndarray::{arr1, arr2};
    use num_complex::c64;

    #[test]
    fn test_ladder_to_complex() {
        // Output should look like
        // [left_0 right_0, left_0 right_1, left_1 right_0, left_1 right_1]
        let ladder_vec = [LadderOperator::Creation, LadderOperator::Annihilation];
        let two_action: Vec<Complex64> = ladder_vec
            .iter()
            .map(|signature| arr1(&signature.majorana_coefficients()))
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap()
            .to_vec();
        assert_eq!(
            two_action,
            vec![c64(0.25, 0.), c64(0., -0.25), c64(0., 0.25), c64(0.25, 0.)]
        );

        let ladder_vec = [
            LadderOperator::Creation,
            LadderOperator::Annihilation,
            LadderOperator::Creation,
        ];
        let three_action: Vec<Complex64> = ladder_vec
            .iter()
            .map(|signature| arr1(&signature.majorana_coefficients()))
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap()
            .to_vec();
        assert_eq!(
            three_action,
            vec![
                c64(0.125, 0.),
                c64(0., -0.125),
                c64(0., 0.125),
                c64(0.125, 0.),
                c64(0., -0.125),
                c64(-0.125, 0.),
                c64(0.125, 0.),
                c64(0., -0.125),
            ]
        );
    }

    #[test]
    fn test_action_conversion() {
        let action = [LadderOperator::Creation, LadderOperator::Annihilation];
        let im_coeffs: Array1<Complex64> = action
            .iter()
            .map(|s| arr1(&s.majorana_coefficients()))
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap();
        assert_eq!(
            im_coeffs,
            arr1(&[
                Complex64 { re: 0.25, im: 0.0 },
                Complex64 { re: 0.0, im: -0.25 },
                Complex64 { re: 0.0, im: 0.25 },
                Complex64 { re: 0.25, im: 0.0 }
            ])
        );
    }

    #[test]
    fn test_sparse_term_creation() {
        let action = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let indices = arr2(&[[0, 1], [2, 3]]);
        let coefficients = arr1(&[c64(1.0, 0.), c64(-1., 0.)]);
        let _term = FermionSparse::new(action, indices, coefficients).unwrap();
    }
    #[test]
    fn test_matrix_term_creation() {
        let action = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let dyn_shape = ndarray::IxDyn(&[2, 2]);
        assert_eq!(dyn_shape.clone().ndim(), 2);
        let coefficients = ArrayD::from_elem(dyn_shape, 1.);
        let _term = FermionMatrix::new(action, coefficients).unwrap();
    }
    #[test]
    fn test_sparse_from_matrix() {
        let action = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let dyn_shape = ndarray::IxDyn(&[2, 2]);
        assert_eq!(dyn_shape.clone().ndim(), 2);
        let mut coefficients = ArrayD::from_elem(dyn_shape, 0.);
        coefficients[[0, 0]] = 1.;
        coefficients[[0, 1]] = 0.5;
        coefficients[[1, 0]] = 2.;
        coefficients[[1, 1]] = 10.;
        let term = FermionMatrix::new(action, coefficients).unwrap();
        let sparse = FermionSparse::from(term);
        assert_eq!(sparse.indices, arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]]));
        assert_eq!(
            sparse.coefficients,
            arr1(&[c64(1., 0.), c64(0.5, 0.), c64(2., 0.), c64(10., 0.)])
        );
    }
}

// /*
// Majorana
// */
//
/// A single Majorana Operator
/// Can have either even or odd index.
#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub struct Majorana(u16);

impl Fermion for Majorana {}

impl Majorana {
    pub fn new(index: u16) -> Self {
        Self(index)
    }

    pub fn mode(&self) -> Mode {
        if self.is_even() {
            Mode(self.0 / 2)
        } else {
            Mode((self.0 - 1) / 2)
        }
    }

    pub fn index(&self) -> usize {
        self.0 as usize
    }

    pub fn is_even(&self) -> bool {
        self.0.is_multiple_of(2)
    }
}

impl PartialOrd for Majorana {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Majorana {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// Product of Majorana operators, with a complex coefficient.
///
/// # Example
///
/// ```
/// use ferrmion_core::operators::MajoranaProduct;
/// use num_complex::Complex64;
///
/// let mp = MajoranaProduct::new(vec![0, 1, 2, 3], Complex64::new(0.5, 0.5));
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaProduct {
    pub operators: Vec<Majorana>,
    pub coefficient: Complex64,
}

impl Fermion for MajoranaProduct {}

impl MajoranaProduct {
    /// Constructor for [`MajoranaProduct`]
    pub fn new(operators: Vec<u16>, coefficient: Complex64) -> Self {
        Self {
            operators: operators.into_iter().map(Majorana::new).collect(),
            coefficient,
        }
    }

    pub fn cannonicalize(&mut self) {
        let sign = cannonicalize(&mut self.operators);
        self.coefficient *= sign;
    }
}

/// Sorts a stack-allocated Majorana index key in place, returning the sign
/// introduced by the swaps (`1.0` for an even number of swaps, `-1.0` for odd).
///
/// Majorana operators obey the commutation relation $\{\gamma_i,\gamma_j\} =
/// 2\delta_{i,j}$, so a product can be reordered into ascending index order by
/// adjacent swaps, picking up a `-1` for each swap of unequal indices.
fn cannonicalize<T: Ord>(indices: &mut [T]) -> f64 {
    if indices.is_empty() {
        return 1.0;
    }
    let mut counter: usize = 0;
    let mut n = indices.len();
    while n > 0 {
        let mut new_n = 0;
        for index in 1..n {
            if indices[index - 1] > indices[index] {
                indices.swap(index - 1, index);
                counter += 1;
                new_n = index;
            }
        }
        n = new_n;
    }
    if counter % 2 == 1 {
        -1.0
    } else {
        1.0
    }
}

/// Map from majorana indices to complex coefficients, used to accumulate and combine like terms.
#[derive(Debug)]
pub(super) struct MajoranaHashMap {
    operators: HashMap<ArrayVec<[u16; MAX_MAJORANAS]>, Complex64>,
}

impl Fermion for MajoranaHashMap {}

impl MajoranaHashMap {
    /// Default constructor, allocating an empty [`MajoranaHashMap`].
    fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    /// Return a new [`MajoranaHashMap`] with the given capacity.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            operators: HashMap::with_capacity(capacity),
        }
    }

    /// Merge the operators from a set of [`MajoranaHashMap`] partials into this map,
    /// distributing them across `n_shards` using a stable hash function.
    fn merge_into(&mut self, partials: &[MajoranaHashMap], shard: usize, n_shards: usize) {
        for local in partials {
            for (key, &value) in local.operators.iter() {
                if n_shards == 1 {
                    *self.operators.entry(*key).or_insert(Complex64::ZERO) += value;
                } else {
                    // Stable (run-independent) hash of a Majorana key used to assign it to a merge
                    // shard. FNV-1a over the `u16` indices — cheap and well-spread. Uses `u64`
                    // (not `usize`) so the 64-bit FNV constants are valid on 32-bit targets too.
                    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
                    for &index in key.iter() {
                        hash = (hash ^ index as u64).wrapping_mul(0x0100_0000_01b3);
                    }
                    if (hash % n_shards as u64) as usize == shard {
                        *self.operators.entry(*key).or_insert(Complex64::ZERO) += value;
                    }
                }
            }
        }
    }

    /// Core accumulation: expand one fermionic term (given by its action, mode indices, and
    /// coefficient) into its 2^n majorana components and insert into the map.
    fn append_term(&mut self, action: &[LadderOperator], indices: &[usize], coeff: Complex64) {
        let term_length = action.len();
        (0u32..(1u32 << term_length))
            .map(move |mask| {
                let mut scaler = c64(1., 0.);
                let mut key: ArrayVec<[u16; MAX_MAJORANAS]> = ArrayVec::new();
                for (j, (op, &idx)) in action.iter().zip(indices).enumerate() {
                    let o = ((mask >> j) & 1) as usize;
                    scaler *= op.majorana_coefficients()[o];
                    key.push((2 * idx + o) as u16);
                }
                let sign = cannonicalize(&mut key);
                (key, coeff * scaler * sign)
            })
            .for_each(|(key, value)| {
                *self.operators.entry(key).or_insert(Complex64::ZERO) += value;
            });
    }

    /// Append a single product of Fermionic operators to the [`MajoranaHashMap`].
    fn append_fermion_product(&mut self, fproduct: FermionProduct) {
        self.append_term(&fproduct.action, &fproduct.indices, fproduct.coefficient);
    }

    /// Append a Fermionic operator in sparse form to the [`MajoranaHashMap`].
    ///
    /// Each row of `fsparse` is an independent term sharing `fsparse.action`. For
    /// large operators (at least [`PARALLEL_TERM_THRESHOLD`] rows) the expansion
    /// runs across rayon worker threads.
    fn append_fermion_sparse(&mut self, fsparse: &FermionSparse) {
        debug!("FSparse Indices {:?}", &fsparse.indices);
        let action = fsparse.action.as_slice();
        let indices = &fsparse.indices;
        let coefficients = &fsparse.coefficients;
        let n_terms = indices.nrows();

        if n_terms < PARALLEL_TERM_THRESHOLD {
            for r in 0..n_terms {
                self.append_term(
                    action,
                    indices
                        .row(r)
                        .as_slice()
                        .expect("Should be able to make slice form FermionSparse row."),
                    coefficients[r],
                );
            }
            debug!("MBTree {:?}\n", &self);
            return;
        }
        // Phase 1 — expand chunks of independent terms into thread-local maps in
        // parallel. Each chunk dedups its own contributions (aHash), so the
        // intermediate stays small.
        let partials: Vec<MajoranaHashMap> = (0..n_terms)
            .into_par_iter()
            .chunks(PARALLEL_CHUNK)
            .map(|rows| {
                let mut local = MajoranaHashMap::with_capacity(PARALLEL_CHUNK);
                for r in rows {
                    let row: ArrayVec<[usize; MAX_MAJORANAS]> =
                        indices.row(r).iter().copied().collect();
                    local.append_term(action, &row, coefficients[r]);
                }
                local
            })
            .collect();

        // Phase 2 — merge the per-chunk maps. The key space is partitioned into
        // shards so the merge itself runs in parallel; each shard scans the chunk
        // maps in chunk order and owns a disjoint set of keys, so a key's value is
        // summed in chunk order regardless of the shard count — the result is
        // deterministic and independent of how rayon schedules the work.
        let n_shards = rayon::current_num_threads().max(1);
        if n_shards == 1 {
            self.merge_into(&partials, 0, 1);
        } else {
            let shards: Vec<MajoranaHashMap> = (0..n_shards)
                .into_par_iter()
                .map(|shard| {
                    let mut out = MajoranaHashMap::new();
                    out.merge_into(&partials, shard, n_shards);
                    out
                })
                .collect();
            for shard in shards {
                for (key, value) in shard.operators {
                    *self.operators.entry(key).or_insert(Complex64::ZERO) += value;
                }
            }
        }
        debug!("MBTree {:?}\n", &self);
    }
}

/// Sparse represtnation of a set of [`MajoranaProduct`] operators.
///
/// # Panics
/// <div class="warning">
/// This type internally represents indices as stack-allocated ArrayVecs.
/// The maximum size of these is currently restricted to [`MAX_MAJORANAS`].
/// Attempting to create an index of length greater than this will cause a panic.
/// </div>
#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaSparse {
    pub indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
    pub coefficients: Vec<Complex64>,
    pub constant: f64,
}

impl Fermion for MajoranaSparse {}

impl Fermion for &MajoranaSparse {}

/// Error type for failed construction of [`MajoranaSparse`]
#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaSparseError;

impl MajoranaSparse {
    /// Constructor for [`MajoranaSparse`].
    ///
    /// Filters out terms with zero coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion_core::operators::MajoranaSparse;
    /// use num_complex::Complex64;
    /// use tinyvec::array_vec;
    ///
    /// let ms = MajoranaSparse::new(
    ///     vec![array_vec!([u16; 7] => 0, 1)],
    ///     vec![Complex64::new(1.0, 0.0)],
    ///     0.0,
    /// ).unwrap();
    /// ```
    pub fn new(
        indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
        coefficients: Vec<Complex64>,
        constant: f64,
    ) -> Result<Self, MajoranaSparseError> {
        if coefficients.len() != indices.len() {
            return Err(MajoranaSparseError);
        };

        let (i, c) = indices
            .iter()
            .zip(&coefficients)
            .filter(|&(_, &coeff)| coeff != Complex64::ZERO)
            .unzip();

        Ok(Self {
            indices: i,
            coefficients: c,
            constant,
        })
    }

    /// Constructor which takes two vectors, one for operator signatures and another for coefficient matrices.
    ///
    /// Prefer building a [`crate::hamiltonians::FermionHamiltonian`] and calling
    /// `to_majorana_sparse`; this constructor remains as an independent
    /// conversion path for loose signature/coefficient data.
    ///
    /// <div class="warning">
    /// Coeffients should be given in Spin-orbit format.
    /// For spatial orbital index "i", the spin-up mode is at $2i$ and the spin down mode is at $2i+1$
    /// </div>
    ///
    pub fn from_signatures_and_coeffs(
        signatures: Vec<String>,
        coeffs: Vec<ArrayViewD<f64>>,
        constant_energy: f64,
    ) -> MajoranaSparse {
        let mut majoranas = MajoranaHashMap::new();
        for (sig, coeff_view) in std::iter::zip(signatures, coeffs) {
            let action: Vec<LadderOperator> = sig
                .chars()
                .map(|v| {
                    LadderOperator::try_from(v).expect("Signature components should be + or -")
                })
                .collect();
            let term_length = action.len();
            // Gather this signature's non-zero, antisymmetry-valid entries into a
            // FermionSparse and expand it (in parallel for large tensors).
            let mut flat_indices: Vec<usize> = Vec::new();
            let mut coefficients: Vec<Complex64> = Vec::new();
            coeff_view
                .indexed_iter()
                .filter(|(_, &v)| v != 0.0)
                .for_each(|(ind, &v)| {
                    let iv = ind.into_dimension();
                    let indices = iv.as_array_view();
                    let slice = indices.as_slice().unwrap();
                    if is_valid_fermion_term(&action, slice) {
                        flat_indices.extend_from_slice(slice);
                        coefficients.push(c64(v, 0.));
                    }
                });
            let indices = Array2::from_shape_vec((coefficients.len(), term_length), flat_indices)
                .expect("Collected indices should form a rectangular array.");
            let fsparse = FermionSparse::new(action, indices, Array1::from(coefficients))
                .expect("Indices and coefficients should be consistent.");
            majoranas.append_fermion_sparse(&fsparse);
        }
        debug!("Getting MSparse");
        let mut hamiltonian = MajoranaSparse::from(majoranas);
        hamiltonian.constant += constant_energy;
        debug!("Got MSparse");
        hamiltonian
    }
}

/// Returns `false` for index combinations that are zeroed out by the antisymmetry constraints
/// of the fermionic action, `true` otherwise.
///
/// Mirrors the logic of [`FermionMatrix::zero_disallowed_terms`] but as a per-term predicate,
/// so it can be used to filter a coefficient array view without needing to copy it.
fn is_valid_fermion_term(action: &[LadderOperator], indices: &[usize]) -> bool {
    use LadderOperator::{Annihilation as Ann, Creation as Cr};
    // Should make this more general!
    if action.len() != 4 {
        return true;
    }
    let (a, b, c, d) = (indices[0], indices[1], indices[2], indices[3]);
    match action {
        [Ann, Ann, Cr, Cr] | [Cr, Cr, Ann, Ann] => a != b && c != d,
        [Cr, Ann, Cr, Ann] | [Ann, Cr, Ann, Cr] => !(a == c && b != a || b == d && c != b),
        _ => true,
    }
}

impl From<MajoranaHashMap> for MajoranaSparse {
    fn from(mbt: MajoranaHashMap) -> MajoranaSparse {
        let mut sparse_constant: num_complex::Complex<f64> = c64(0., 0.);
        let mut pairs: Vec<(ArrayVec<[u16; MAX_MAJORANAS]>, Complex64)> = Vec::new();
        for (k, v) in mbt
            .operators
            .into_iter()
            .filter(|(_, v)| v.abs() >= COEFFICIENT_TOLERANCE)
        {
            if k.is_empty() {
                sparse_constant += v;
            } else {
                pairs.push((k, v));
            }
        }
        // Restore deterministic ordering (equivalent to the prior BTreeMap key order).
        pairs.sort_unstable_by_key(|(a, _)| *a);
        let (sparse_indices, sparse_values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        debug!("Sparse Majorana Indices {:?}", &sparse_indices);
        debug!("Sparse Majorana Coefficients {:?}", &sparse_values);
        MajoranaSparse::new(sparse_indices, sparse_values, sparse_constant.norm())
            .expect("Indices and coefficients should be same length.")
    }
}

impl From<FermionProduct> for MajoranaSparse {
    fn from(fproduct: FermionProduct) -> Self {
        // Start off by creating a BTreeMap as we'll need to add a few fermionic terms
        // to each majorana term
        let mut majoranas: MajoranaHashMap = MajoranaHashMap::new();
        majoranas.append_fermion_product(fproduct);
        majoranas.into()
    }
}

impl From<FermionSparse> for MajoranaSparse {
    fn from(sft: FermionSparse) -> Self {
        let mut majoranas: MajoranaHashMap = MajoranaHashMap::new();
        majoranas.append_fermion_sparse(&sft);
        majoranas.into()
    }
}

impl From<Vec<FermionSparse>> for MajoranaSparse {
    fn from(sft: Vec<FermionSparse>) -> Self {
        let mut majoranas: MajoranaHashMap = MajoranaHashMap::new();
        sft.iter().for_each(|term| {
            majoranas.append_fermion_sparse(term);
        });
        majoranas.into()
    }
}

/// Transposed ("bit-sliced") view of a Majorana Hamiltonian.
///
/// Where [`MajoranaSparse`] keeps one index-list per term, this is column-major:
/// one `u64` bit-vector per **Majorana index**, whose bits correspond to **terms**
/// (`columns[i]` bit `t` set ⇔ term `t` contains index `i`). Scoring a candidate
/// selection then reads only the three relevant vectors and computes a Pauli
/// weight with word-parallel bit ops over `⌈T/64⌉` words, instead of touching
/// every term.
///
/// This is a pure operator representation: it carries no algorithm state. Each
/// term is **parity-canonicalised** (γ²=I), a bit XOR-toggled per index
/// occurrence, so an index appearing an even number of times in a term cancels.
/// There is no mode ceiling — columns are indexed by Majorana index and the `u64`
/// words slice terms.
pub struct MajoranaDenseTranspose {
    pub n_terms: usize,
    pub n_words: usize,
    /// One bit-vector per index (length `3*n_modes + 1`): real Majoranas
    /// `0..2*n_nodes`, the all-Z leaf `2*n_nodes`, and node representatives
    /// `2*n_nodes+1..=3*n_nodes`.
    pub columns: Vec<Vec<u64>>,
}

impl MajoranaDenseTranspose {
    /// Build a dense transpose from Majorana-index terms, sizing the column table
    /// to `3*n_modes + 1` indices (real Majoranas, the all-Z leaf, and node
    /// representatives). Terms are parity-canonicalised by XOR-toggling.
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAX_MAJORANAS]>], n_modes: usize) -> Self {
        let n_terms = terms.len();
        let n_words = n_terms.div_ceil(64);
        let n_cols = 3 * n_modes + 1;
        let mut columns = vec![vec![0u64; n_words]; n_cols];
        for (t, term) in terms.iter().enumerate() {
            let word = t / 64;
            let bit = 1u64 << (t % 64);
            for &idx in term.iter() {
                // XOR (not OR): a repeated index toggles back off — γ²=I parity.
                columns[idx as usize][word] ^= bit;
            }
        }
        Self {
            n_terms,
            n_words,
            columns,
        }
    }
}

/// Sparse inverted-index view of a Majorana Hamiltonian.
///
/// The sparse counterpart of [`MajoranaDenseTranspose`]: instead of a dense `u64`
/// bit-vector per index, each index keeps a **sorted list of the term indices it
/// appears in**. For sparse Hamiltonians (e.g. molecular) the dense bit columns
/// are mostly zero, so these lists are short and scoring a selection — a 3-way
/// merge of three lists — costs `O(|L0|+|L1|+|L2|)` instead of `O(T/64)`.
///
/// Like [`MajoranaDenseTranspose`] this is a pure operator representation. The
/// lists are parity-canonicalised (γ²=I): only indices appearing an odd number of
/// times in a term are recorded. No mode ceiling.
pub struct MajoranaSparseTranspose {
    pub n_terms: usize,
    /// One ascending, duplicate-free list of term indices per index (length
    /// `3*n_modes + 1`): real Majoranas `0..2*n_nodes`, the all-Z leaf
    /// `2*n_nodes`, and node representatives `2*n_nodes+1..=3*n_nodes`.
    pub lists: Vec<Vec<u32>>,
}

impl MajoranaSparseTranspose {
    /// Build a sparse inverted index from Majorana-index terms, sizing the list
    /// table to `3*n_modes + 1` indices. Lists are parity-canonicalised and come
    /// out ascending and duplicate-free (terms are visited in ascending order).
    pub fn from_arrayvecs(terms: &[ArrayVec<[u16; MAX_MAJORANAS]>], n_modes: usize) -> Self {
        let n_terms = terms.len();
        let n_cols = 3 * n_modes + 1;
        let mut lists = vec![Vec::new(); n_cols];
        for (t, term) in terms.iter().enumerate() {
            let mut parity_set: ArrayVec<[u16; MAX_MAJORANAS]> = ArrayVec::new();
            for &idx in term {
                if let Some(pos) = parity_set.iter().position(|&x| x == idx) {
                    parity_set.remove(pos);
                } else {
                    parity_set.push(idx);
                }
            }
            parity_set.sort_unstable();
            for idx in parity_set {
                lists[idx as usize].push(t as u32);
            }
        }
        Self { n_terms, lists }
    }
}

#[cfg(test)]
mod majorana_tests {
    use super::*;
    use crate::utils::vector_kron;
    use log::debug;
    use ndarray::{arr1, arr2};
    use num_complex::c64;
    use tinyvec::array_vec;

    #[test]
    fn test_ladder_to_complex() {
        // Output should look like
        // [left_0 right_0, left_0 right_1, left_1 right_0, left_1 right_1]
        let ladder_vec = [LadderOperator::Creation, LadderOperator::Annihilation];
        let two_action: Vec<Complex64> = ladder_vec
            .iter()
            .map(|signature| arr1(&signature.majorana_coefficients()))
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap()
            .to_vec();
        assert_eq!(
            two_action,
            vec![c64(0.25, 0.), c64(0., -0.25), c64(0., 0.25), c64(0.25, 0.)]
        );

        let ladder_vec = [
            LadderOperator::Creation,
            LadderOperator::Annihilation,
            LadderOperator::Creation,
        ];
        let three_action: Vec<Complex64> = ladder_vec
            .iter()
            .map(|signature| arr1(&signature.majorana_coefficients()))
            .reduce(|acc, s| vector_kron(&acc, &s))
            .unwrap()
            .to_vec();
        assert_eq!(
            three_action,
            vec![
                c64(0.125, 0.),
                c64(0., -0.125),
                c64(0., 0.125),
                c64(0.125, 0.),
                c64(0., -0.125),
                c64(-0.125, 0.),
                c64(0.125, 0.),
                c64(0., -0.125),
            ]
        );
    }

    #[test]
    fn test_cannonicalize_do_nothing() {
        let indices = vec![0, 1];

        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        assert_eq!(mp.operators, vec![Majorana(0), Majorana(1)]);
        assert_eq!(mp.coefficient, coefficient.clone());
    }

    #[test]
    fn test_cannonicalize_single_swap() {
        let indices = vec![1, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(mp.operators, vec![Majorana(0), Majorana(1)]);
        assert_eq!(mp.coefficient, -1. * coefficient);
    }

    #[test]
    fn test_cannonicalize_do_not_simplify_to_empty() {
        let indices = vec![0, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(mp.operators, vec![Majorana(0), Majorana(0)]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![0, 1, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            [0, 0, 1, 1]
                .into_iter()
                .map(|v| Majorana::new(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![1, 0, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            vec![0, 0, 1, 1]
                .into_iter()
                .map(|v| Majorana::new(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_cannonicalize_reverse() {
        let indices = vec![3, 2, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            vec![1, 2, 3]
                .into_iter()
                .map(|v| Majorana::new(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![4, 3, 2, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            vec![1, 2, 3, 4]
                .into_iter()
                .map(|v| Majorana::new(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_cannonicalize() {
        let indices = vec![1, 1, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            vec![1, 1, 1, 1, 1]
                .into_iter()
                .map(|v| Majorana::new(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            vec![1, 1, 1, 1]
                .into_iter()
                .map(|v| Majorana::new(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1u16, 1, 1, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            [0, 1, 1, 1]
                .iter()
                .map(|&v| Majorana::new(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![1u16, 1, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            [0, 1, 1, 1]
                .iter()
                .map(|&v| Majorana::new(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 0, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            vec![Majorana(0), Majorana(1), Majorana(1), Majorana(1)]
        );
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![0, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.cannonicalize();
        // debug!("{:#?}", mp);
        assert_eq!(
            mp.operators,
            vec![Majorana(0), Majorana(1), Majorana(1), Majorana(1)]
        );
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_from_fermion_sparse_len_one() {
        let indices = arr2(&[[0]]);
        let coefficients = arr1(&[c64(10.0, 0.)]);
        let action = vec![LadderOperator::Creation];
        debug!("{:#?}", indices.clone());
        debug!("{:#?}", coefficients.clone());
        debug!("{:#?}", action.clone());

        let majorana_term = MajoranaSparse::new(
            vec![array_vec!([u16; 7]=> 0), array_vec!([u16; 7]=> 1)],
            vec![c64(5., 0.), c64(0., -5.)],
            0.,
        )
        .unwrap();
        let fermion_term =
            FermionSparse::new(action.clone(), indices.clone(), coefficients.clone()).unwrap();
        assert_eq!(majorana_term, MajoranaSparse::from(fermion_term));
    }

    #[test]
    fn test_from_fermion_sparse_len_two() {
        let indices = arr2(&[[0, 1]]);
        let coefficients = arr1(&[c64(10.0, 0.)]);
        let action = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        debug!("{:#?}", indices.clone());
        debug!("{:#?}", coefficients.clone());
        debug!("{:#?}", action.clone());

        let majorana_term = MajoranaSparse::new(
            vec![
                array_vec!([u16; 7]=> 0, 2),
                array_vec!([u16; 7]=> 0, 3),
                array_vec!([u16; 7]=> 1,2),
                array_vec!([u16; 7]=> 1,3),
            ],
            vec![c64(2.5, 0.), c64(0., 2.5), c64(0.0, -2.5), c64(2.5, 0.)],
            0.,
        )
        .unwrap();
        let fermion_term =
            FermionSparse::new(action.clone(), indices.clone(), coefficients.clone()).unwrap();
        assert_eq!(majorana_term, MajoranaSparse::from(fermion_term));
    }

    #[test]
    fn test_from_fermion_sparse_len_three() {
        let indices = arr2(&[[0, 1, 2]]);
        let coefficients = arr1(&[c64(10.0, 0.)]);
        let action = vec![
            LadderOperator::Creation,
            LadderOperator::Annihilation,
            LadderOperator::Creation,
        ];
        debug!("{:#?}", indices.clone());
        debug!("{:#?}", coefficients.clone());
        debug!("{:#?}", action.clone());

        let majorana_term = MajoranaSparse::new(
            vec![
                array_vec!([u16; 7]=> 0, 2, 4),
                array_vec!([u16; 7]=> 0, 2, 5),
                array_vec!([u16; 7]=> 0, 3, 4),
                array_vec!([u16; 7]=> 0, 3, 5),
                array_vec!([u16; 7]=> 1,2, 4),
                array_vec!([u16; 7]=> 1,2, 5),
                array_vec!([u16; 7]=> 1,3, 4),
                array_vec!([u16; 7]=> 1,3, 5),
            ],
            vec![
                c64(1.25, 0.),
                c64(0., -1.25),
                c64(0.0, 1.25),
                c64(1.25, 0.),
                c64(0., -1.25),
                c64(-1.25, 0.),
                c64(1.25, 0.),
                c64(0., -1.25),
            ],
            0.,
        )
        .unwrap();
        let fermion_term =
            FermionSparse::new(action.clone(), indices.clone(), coefficients.clone()).unwrap();
        assert_eq!(majorana_term, MajoranaSparse::from(fermion_term));
    }

    // #[test]
    // fn test_msparse_from_vec_fsparse() {
    //     todo!();
    // }

    /// The parallel `append_fermion_sparse` path (taken once a `FermionSparse`
    /// has at least `PARALLEL_TERM_THRESHOLD` rows) must produce exactly the same
    /// result as accumulating the terms serially. Coefficients are kept as small
    /// dyadic-rational values so the sums are exact in `f64` and the comparison
    /// can be bit-for-bit regardless of accumulation order.
    #[test]
    fn test_append_fermion_sparse_parallel_matches_serial() {
        use LadderOperator::{Annihilation, Creation};

        let action = vec![Creation, Creation, Annihilation, Annihilation];
        let n_terms = PARALLEL_TERM_THRESHOLD * 3 + 7;
        let n_orb = 5;

        let mut indices = Array2::<usize>::zeros((n_terms, 4));
        let mut coefficients = Array1::<Complex64>::zeros(n_terms);
        for t in 0..n_terms {
            indices[[t, 0]] = t % n_orb;
            indices[[t, 1]] = (t / n_orb) % n_orb;
            indices[[t, 2]] = (t / (n_orb * n_orb)) % n_orb;
            indices[[t, 3]] = (t / (n_orb * n_orb * n_orb)) % n_orb;
            coefficients[t] = c64((t % 4 + 1) as f64, 0.0);
        }

        // Parallel path: > PARALLEL_TERM_THRESHOLD rows in a single FermionSparse.
        let fsparse =
            FermionSparse::new(action.clone(), indices.clone(), coefficients.clone()).unwrap();
        let parallel = MajoranaSparse::from(fsparse);

        // Serial reference: accumulate one term at a time via append_term.
        let mut serial_map = MajoranaHashMap::new();
        for t in 0..n_terms {
            let ind: Vec<usize> = indices.row(t).iter().copied().collect();
            serial_map.append_term(&action, &ind, coefficients[t]);
        }
        let serial = MajoranaSparse::from(serial_map);

        assert_eq!(parallel, serial);
    }

    /// `from_signatures_and_coeffs` takes the same parallel `extend_terms` path
    /// once a signature has at least `PARALLEL_TERM_THRESHOLD` valid terms; its
    /// result must match accumulating those terms serially. The dense unit tensor
    /// keeps every coefficient at `1.0` (exact in `f64`) so the comparison is
    /// bit-for-bit regardless of accumulation order.
    #[test]
    fn test_from_signatures_and_coeffs_parallel_matches_serial() {
        use LadderOperator::{Annihilation, Creation};

        let n_orb = 5;
        let coeffs = ArrayD::<f64>::from_elem(ndarray::IxDyn(&[n_orb, n_orb, n_orb, n_orb]), 1.0);

        // Parallel path via the public constructor (> threshold valid terms).
        let parallel = MajoranaSparse::from_signatures_and_coeffs(
            vec!["++--".to_string()],
            vec![coeffs.view()],
            0.0,
        );

        // Serial reference: identical filtering, accumulated one term at a time.
        let action = vec![Creation, Creation, Annihilation, Annihilation];
        let mut serial_map = MajoranaHashMap::new();
        coeffs
            .indexed_iter()
            .filter(|(_, &v)| v != 0.0)
            .for_each(|(ind, &v)| {
                let iv = ind.into_dimension();
                let indices = iv.as_array_view();
                let slice = indices.as_slice().unwrap();
                if is_valid_fermion_term(&action, slice) {
                    serial_map.append_term(&action, slice, c64(v, 0.));
                }
            });
        let serial = MajoranaSparse::from(serial_map);

        assert_eq!(parallel, serial);
    }
}
