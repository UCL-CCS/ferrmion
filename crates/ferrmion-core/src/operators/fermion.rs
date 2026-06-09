//! Fermionic Operators
use crate::operators::ladder::LadderOperator;
use crate::spaces::Fermion;
use itertools::Itertools;
use log::debug;
use ndarray::{arr0, s, Dimension};
use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis, IntoDimension, Zip};
use num_complex::Complex64;
use num_complex::{c64, ComplexFloat};
use std::collections::HashMap;
use std::iter::repeat_n;
use std::result::Result;
use tinyvec::ArrayVec;

/// Maximum length of majorana indices which are allowed in stack-allocated ArrayVecs.
const MAX_MAJORANAS: usize = 7;

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
                sparse_coefficients[count] += c64(v, 0.);
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
    fn test_action_conversion() {
        let action = [LadderOperator::Creation, LadderOperator::Annihilation];
        let im_coeffs: Array1<Complex64> = action
            .iter()
            .map(|s| s.majorana_coefficients())
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
    pub indices: Vec<usize>,
    pub coefficient: Complex64,
}

impl Fermion for MajoranaProduct {}

impl MajoranaProduct {
    /// Constructor for [`MajoranaProduct`]
    pub fn new(indices: Vec<usize>, coefficient: Complex64) -> Self {
        // out.majorise();
        Self {
            indices,
            coefficient,
        }
    }
    /// Sorts indices, applying a -1 coefficient for each swap of unequal indices.
    ///
    /// Majorana operators obey the commutation relation:
    /// $$\{\gamma_i,\gamma_j} = 2\delta_{i,j$$
    ///
    /// So we can combine terms for products of majorana operators composed with the same indices.
    fn majorise(&mut self) {
        if self.indices.is_empty() {
            return;
        }
        let mut counter: usize = 0;
        let mut n = self.indices.len();
        while n > 0 {
            let mut new_n = 0;
            for index in 1..n {
                if self.indices[index - 1] > self.indices[index] {
                    self.indices.swap(index - 1, index);
                    counter += 1;
                    new_n = index;
                }
            }
            n = new_n;
        }
        if counter % 2 == 1 {
            self.coefficient *= -1.
        }
    }
}

/// Map from majorana indices to complex coefficients, used to accumulate and combine like terms.
#[derive(Debug)]
pub(super) struct MajoranaHashMap {
    operators: HashMap<ArrayVec<[u16; MAX_MAJORANAS]>, Complex64>,
}

impl Fermion for MajoranaHashMap {}

impl MajoranaHashMap {
    fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    /// Core accumulation: expand one fermionic term (given by its action, mode indices, and
    /// coefficient) into its 2^n majorana components and insert into the map.
    fn append_term(&mut self, action: &[LadderOperator], indices: &[usize], coeff: Complex64) {
        let term_length = action.len();
        for offset in repeat_n(0usize..=1usize, term_length).multi_cartesian_product() {
            let scaler = offset
                .iter()
                .zip(action.iter())
                .fold(c64(1., 0.), |acc, (&o, op)| {
                    acc * op.majorana_coefficients()[o]
                });
            let raw_indices: Vec<usize> = indices
                .iter()
                .zip(&offset)
                .map(|(&i, &o)| 2 * i + o)
                .collect();
            let mut mp = MajoranaProduct::new(raw_indices, coeff * scaler);
            mp.majorise();
            let key: ArrayVec<[u16; MAX_MAJORANAS]> =
                mp.indices.iter().map(|&i| i as u16).collect();
            *self.operators.entry(key).or_insert(Complex64::ZERO) += mp.coefficient;
        }
    }

    /// Append a single product of Fermionic operators to the [`MajoranaHashMap`].
    fn append_fermion_product(&mut self, fproduct: FermionProduct) {
        self.append_term(&fproduct.action, &fproduct.indices, fproduct.coefficient);
    }

    /// Append a Fermionic Hamiltonian in sparse form to the [`MajoranaHashMap`].
    fn append_fermion_sparse(&mut self, fsparse: FermionSparse) {
        debug!("FSparse Indices {:?}", &fsparse.indices);
        Zip::from(fsparse.indices.rows())
            .and(fsparse.coefficients.view())
            .for_each(|ind, coeff| {
                let ind_slice: Vec<usize> = ind.iter().copied().collect();
                self.append_term(&fsparse.action, &ind_slice, *coeff);
            });
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
    /// This is primarily used in the PyO3 interop functions, as the FermionHamiltonian python class
    /// outputs data in this format.
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
            coeff_view
                .indexed_iter()
                .filter(|(_, &v)| v != 0.0)
                .for_each(|(ind, &v)| {
                    let iv = ind.into_dimension();
                    let indices = iv.as_array_view();
                    if is_valid_fermion_term(&action, indices.as_slice().unwrap()) {
                        majoranas.append_term(&action, indices.as_slice().unwrap(), c64(v, 0.));
                    }
                });
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
        for (k, v) in mbt.operators.into_iter().filter(|(_, v)| v.abs() >= 1e-16) {
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
        // Start off by creating a BTreeMap as we'll need to add a few fermionic terms
        // to each majorana term
        let mut majoranas: MajoranaHashMap = MajoranaHashMap::new();
        majoranas.append_fermion_sparse(sft);
        majoranas.into()
    }
}

impl From<Vec<FermionSparse>> for MajoranaSparse {
    fn from(sft: Vec<FermionSparse>) -> Self {
        let mut majoranas: MajoranaHashMap = MajoranaHashMap::new();
        sft.into_iter().for_each(|term| {
            majoranas.append_fermion_sparse(term);
        });
        majoranas.into()
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
            .map(|signature| signature.majorana_coefficients())
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
            .map(|signature| signature.majorana_coefficients())
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
    fn test_majorise_do_nothing() {
        let indices = vec![0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        assert_eq!(mp.indices, indices.clone());
        assert_eq!(mp.coefficient, coefficient.clone());
    }

    #[test]
    fn test_majorise_single_swap() {
        let indices = vec![1, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);
    }

    #[test]
    fn test_majorise_do_not_simplify_to_empty() {
        let indices = vec![0, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 0]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![0, 1, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 0, 1, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![1, 0, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 0, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise_reverse() {
        let indices = vec![3, 2, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 2, 3]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![4, 3, 2, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 2, 3, 4]);
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise() {
        let indices = vec![1, 1, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 1, 1, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 1, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 1, 1, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1, 1, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![1, 1, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1, 1, 1]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 0, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1, 1, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![0, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient);
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1, 1, 1]);
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
}
