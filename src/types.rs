use crate::ternarytree::Edge;
use ndarray::Dimension;
use tinyvec::ArrayVec;
/*
Shared Types.
*/
use crate::utils::vector_kron;
use itertools::Itertools;
use log::debug;
use num_complex::{c64, ComplexFloat};
use numpy::ndarray::{arr1, arr2, Array1, Array2, ArrayD, Axis, IntoDimension, Zip};
use numpy::Complex64;
use std::collections::BTreeMap;
use std::iter::{repeat_n, zip};
use std::{result::Result, str::FromStr};

const MAX_MAJORANAS: usize = 4;

#[allow(dead_code)]
#[derive(Debug, Default)]
pub enum Pauli {
    #[default]
    I,
    X,
    Y,
    Z,
}
impl From<&Edge> for Pauli {
    fn from(e: &Edge) -> Pauli {
        match e {
            Edge::X => Pauli::X,
            Edge::Y => Pauli::Y,
            Edge::Z => Pauli::Z,
        }
    }
}

impl From<Pauli> for String {
    fn from(p: Pauli) -> String {
        match p {
            Pauli::I => "I".to_string(),
            Pauli::X => "X".to_string(),
            Pauli::Y => "Y".to_string(),
            Pauli::Z => "Z".to_string(),
        }
    }
}

impl From<Pauli> for char {
    fn from(p: Pauli) -> char {
        match p {
            Pauli::I => 'I',
            Pauli::X => 'X',
            Pauli::Y => 'Y',
            Pauli::Z => 'Z',
        }
    }
}

impl From<(bool, bool)> for Pauli {
    fn from(xz_bools: (bool, bool)) -> Pauli {
        match xz_bools {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (false, true) => Pauli::Z,
            (true, true) => Pauli::Y,
        }
    }
}

impl From<Pauli> for (bool, bool) {
    fn from(p: Pauli) -> (bool, bool) {
        match p {
            Pauli::I => (false, false),
            Pauli::X => (true, false),
            Pauli::Y => (true, true),
            Pauli::Z => (false, true),
        }
    }
}

type PauliMatrix = Array2<Complex64>;

impl From<Pauli> for PauliMatrix {
    fn from(p: Pauli) -> PauliMatrix {
        match p {
            Pauli::I => arr2(&[[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(1., 0.)]]),
            Pauli::X => arr2(&[[c64(0., 0.), c64(1., 0.)], [c64(1., 0.), c64(0., 0.)]]),
            Pauli::Z => arr2(&[[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(-1., 0.)]]),
            Pauli::Y => arr2(&[[c64(0., 0.), c64(0., -1.)], [c64(0., 1.), c64(0., 0.)]]),
        }
    }
}

#[cfg(test)]
mod test_pauli {
    use super::*;
    use crate::types::{Pauli, PauliMatrix};
    use ndarray::arr2;

    #[test]
    fn test_matrix_identities() {
        let i = arr2(&[[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(1., 0.)]]);
        let x = Into::<PauliMatrix>::into(Pauli::X);
        let y = Into::<PauliMatrix>::into(Pauli::Y);
        let z = Into::<PauliMatrix>::into(Pauli::Z);
        assert_eq!(&i.dot(&i), i);
        assert_eq!(&x.dot(&x), i);
        assert_eq!(&y.dot(&y), i);
        assert_eq!(&z.dot(&z), i);
        assert_eq!(&x.dot(&z), c64(0., -1.) * y.clone());
        assert_eq!(&y.dot(&z), c64(0., 1.) * x.clone());
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum LadderOperator {
    Creation,
    Annihilation,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ParseLadderError;

impl FromStr for LadderOperator {
    type Err = ParseLadderError;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        if string == "+" {
            Ok(LadderOperator::Creation)
        } else if string == "-" {
            Ok(LadderOperator::Annihilation)
        } else {
            Err(ParseLadderError)
        }
    }
}

impl LadderOperator {
    pub fn fermion_coeff(&self) -> Array1<Complex64> {
        match &self {
            LadderOperator::Creation => arr1(&[c64(0.5, 0.0), c64(0., -0.5)]),
            LadderOperator::Annihilation => arr1(&[c64(0.5, 0.0), c64(0., 0.5)]),
        }
    }
}
impl TryFrom<char> for LadderOperator {
    type Error = ParseLadderError;

    fn try_from(string: char) -> Result<Self, Self::Error> {
        if string == '+' {
            Ok(LadderOperator::Creation)
        } else if string == '-' {
            Ok(LadderOperator::Annihilation)
        } else {
            Err(ParseLadderError)
        }
    }
}

#[cfg(test)]
mod ladder_tests {
    use crate::types::*;

    #[test]
    fn test_ladder_operators() {
        assert_eq!(
            LadderOperator::from_str("+").unwrap(),
            LadderOperator::Creation
        );
        assert_eq!(
            LadderOperator::from_str("-").unwrap(),
            LadderOperator::Annihilation
        );
        assert_eq!(LadderOperator::from_str("+-"), Err(ParseLadderError));
        assert_eq!(LadderOperator::from_str("-+"), Err(ParseLadderError));
    }
}

/*
Fermion
*/

#[derive(Debug, PartialEq, Clone, Copy)]
struct FermionOperator {
    op: LadderOperator,
    index: u32,
}

impl FermionOperator {
    fn new(op: LadderOperator, index: u32) -> Self {
        Self { op, index }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct FermionProduct {
    ops: Vec<LadderOperator>,
    indices: Vec<u32>,
    coefficient: Complex64,
}

#[derive(Debug, PartialEq, Clone)]
struct FermionProductError;

impl FermionProduct {
    pub fn new(
        ops: Vec<LadderOperator>,
        indices: Vec<u32>,
        coefficient: Complex64,
    ) -> Result<Self, FermionProductError> {
        if ops.len() != indices.len() {
            Err(FermionProductError)
        } else {
            Ok(Self {
                ops,
                indices,
                coefficient,
            })
        }
    }
}

pub struct FermionMatrix {
    ops: Vec<LadderOperator>,
    coefficients: ArrayD<f64>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FermionMatrixError;

impl FermionMatrix {
    pub fn new(
        ops: Vec<LadderOperator>,
        coefficients: ArrayD<f64>,
    ) -> Result<Self, FermionMatrixError> {
        // Check we have enough ladder operators
        // and a square/cube/... matrix
        if ops.len() != coefficients.ndim()
            || !coefficients
                .shape()
                .into_iter()
                .all(|s| *s == coefficients.shape()[0])
        {
            return Err(FermionMatrixError);
        }
        Ok(Self { ops, coefficients })
    }
}
#[derive(Debug, PartialEq)]
pub struct FermionSparse {
    ops: Vec<LadderOperator>,
    indices: Array2<usize>,
    coefficients: Array1<Complex64>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FermionSparseError;

impl FermionSparse {
    pub fn new(
        ops: Vec<LadderOperator>,
        indices: Array2<usize>,
        coefficients: Array1<Complex64>,
    ) -> Result<Self, FermionSparseError> {
        if coefficients.len() != indices.len_of(Axis(0)) || ops.len() != indices.len_of(Axis(1)) {
            return Err(FermionSparseError);
        };

        Ok(Self {
            ops,
            indices,
            coefficients,
        })
    }
    fn append_majorana_btree<'map>(
        &self,
        majoranas: &'map mut MajoranaBTree,
    ) -> &'map MajoranaBTree {
        let term_length = self.indices.ncols();
        Zip::from(self.indices.rows())
            .and(self.coefficients.view())
            .for_each(|ind, coeff| {
                let majorana_op_scalers: Vec<Complex64> = self
                    .ops
                    .iter()
                    .map(|signature| signature.fermion_coeff())
                    .reduce(|acc, s| vector_kron(&acc, &s))
                    .unwrap()
                    .to_vec();

                // debug!("{:#?}", ops_coeffs);
                let offset = repeat_n(0usize..2usize, term_length).multi_cartesian_product();
                for (scaler, offset) in zip(majorana_op_scalers, offset) {
                    let mut majorana_term = Array1::zeros(term_length);
                    majorana_term += &ind;
                    majorana_term *= 2;
                    majorana_term = majorana_term + Array1::from_vec(offset);

                    let mut mp = MajoranaProduct::new(majorana_term.to_vec(), *coeff * scaler);
                    mp.majorise();
                    *majoranas
                        .entry(mp.indices)
                        .or_insert(Complex64 { re: 0.0, im: 0.0 }) += mp.coefficient;
                }
            });
        majoranas
    }
}

impl From<FermionMatrix> for FermionSparse {
    fn from(mft: FermionMatrix) -> FermionSparse {
        // let temp_hashmap: HashMap<ArrayView1<usize>, f64, RandomState> =
        //     HashMap::with_hasher(RandomState::new());
        // mft.coefficients
        //     .indexed_iter()
        //     .filter(|(_, &v)| v != 0.)
        //     .for_each(|(ind, &v)| {
        //         *temp_hashmap.entry(ind.as_array_view()).or_default() += v;
        //     });

        let n_nonzero = mft.coefficients.iter().filter(|&v| *v != 0.).count();
        let mut sparse_indices: Array2<usize> = Array2::zeros((n_nonzero, mft.ops.len()));
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
        FermionSparse::new(mft.ops, sparse_indices, sparse_coefficients)
            .expect("Conversion from MatrixFermionTerm should be validated.")
    }
}

#[cfg(test)]
mod fermion_tests {
    use crate::types::*;
    use ndarray::{arr1, arr2};
    use num_complex::c64;

    #[test]
    fn test_operator_creation() {
        let c0 = FermionOperator::new(LadderOperator::Creation, 0);
        let a1 = FermionOperator::new(LadderOperator::Annihilation, 1);
        assert_eq!(
            c0,
            FermionOperator {
                op: LadderOperator::Creation,
                index: 0
            }
        );
        assert_eq!(
            a1,
            FermionOperator {
                op: LadderOperator::Annihilation,
                index: 1
            }
        );
    }

    #[test]
    fn test_product_creation() {
        let ops = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let coefficient = Complex64::default();
        let indices = vec![0, 1];
        let _product = FermionProduct::new(ops, indices, coefficient);
    }

    #[test]
    fn test_ops_conversion() {
        let ops = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let im_coeffs: Array1<Complex64> = ops
            .iter()
            .map(|s| s.fermion_coeff())
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
        let ops = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let indices = arr2(&[[0, 1], [2, 3]]);
        let coefficients = arr1(&[c64(1.0, 0.), c64(-1., 0.)]);
        let _term = FermionSparse::new(ops, indices, coefficients).unwrap();
    }
    #[test]
    fn test_matrix_term_creation() {
        let ops = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let dyn_shape = ndarray::IxDyn(&[2, 2]);
        assert_eq!(dyn_shape.clone().ndim(), 2);
        let coefficients = ArrayD::from_elem(dyn_shape, 1.);
        let _term = FermionMatrix::new(ops, coefficients).unwrap();
    }
    #[test]
    fn test_sparse_from_matrix() {
        let ops = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let dyn_shape = ndarray::IxDyn(&[2, 2]);
        assert_eq!(dyn_shape.clone().ndim(), 2);
        let mut coefficients = ArrayD::from_elem(dyn_shape, 0.);
        coefficients[[0, 0]] = 1.;
        coefficients[[0, 1]] = 0.5;
        coefficients[[1, 0]] = 2.;
        coefficients[[1, 1]] = 10.;
        let term = FermionMatrix::new(ops, coefficients).unwrap();
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

#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaProduct {
    pub indices: Vec<usize>,
    pub coefficient: Complex64,
}

impl MajoranaProduct {
    pub fn new(indices: Vec<usize>, coefficient: Complex64) -> Self {
        let mut out = Self {
            indices,
            coefficient,
        };
        out.majorise();
        out
    }
    fn majorise(&mut self) {
        if self.indices.len() == 0 {
            return;
        }
        let mut counter: usize = 0;
        let mut ind = 0;
        let mut safety = 0;
        'outer: while safety < 10 {
            while ind < (self.indices.len() - 1) && safety < 10 {
                safety += 1;
                let left = &self.indices[ind];
                let right = &self.indices[ind + 1];
                if left == right {
                    self.indices.remove(ind);
                    self.indices.remove(ind);
                    if self.indices.len() <= 1 {
                        break 'outer;
                    };
                    if ind > (self.indices.len() - 1) {
                        ind = self.indices.len() - 1;
                        break;
                    }
                    continue;
                } else if left > right {
                    self.indices.swap(ind, ind + 1);
                    counter += 1;
                }
                ind += 1;
                if self.indices.len() == 0 {
                    break 'outer;
                }
            }
            // ind = if ind >= self.indices.len() {
            //     self.indices.len() - 1
            // } else {
            //     ind
            // };
            ind = if ind > (self.indices.len() - 1) {
                self.indices.len() - 1
            } else {
                ind
            };
            if ind == 0 {
                break 'outer;
            }
            while (ind >= 1) && (ind < self.indices.len()) && (safety < 10) {
                safety += 1;
                let left = &self.indices[ind.checked_sub(1).unwrap()];
                let right = &self.indices[ind];
                if left == right {
                    // things get moved left
                    self.indices.remove(ind - 1);
                    self.indices.remove(ind - 1);
                    if self.indices.len() <= 1 {
                        break 'outer;
                    };
                    ind -= 1;
                    continue;
                } else if left > right {
                    self.indices.swap(ind, ind - 1);
                    counter += 1;
                }
                ind -= 1;
            }
        }
        if counter % 2 == 1 {
            self.coefficient *= -1.
        }
    }
}

type MajoranaBTree = BTreeMap<Vec<usize>, Complex64>;

#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaSparse {
    pub indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
    pub coefficients: Vec<Complex64>,
    pub constant: Complex64,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaSparseError;
impl MajoranaSparse {
    pub fn new(
        indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>>,
        coefficients: Vec<Complex64>,
        constant: Complex64,
    ) -> Result<Self, MajoranaSparseError> {
        if coefficients.len() != indices.len() {
            return Err(MajoranaSparseError);
        };
        let identity_terms_constant: Complex64 = indices
            .iter()
            .zip(&coefficients)
            .filter(|&(ind, _)| ind == &ArrayVec::new())
            .map(|(_, coeff)| coeff)
            .sum();

        let (i, c) = indices
            .iter()
            .zip(&coefficients)
            .filter(|&(&inds, &coeff)| (coeff != Complex64::ZERO) && (inds != ArrayVec::new()))
            .unzip();

        Ok(Self {
            indices: i,
            coefficients: c,
            constant: constant + identity_terms_constant,
        })
    }
}

impl From<FermionSparse> for MajoranaSparse {
    fn from(sft: FermionSparse) -> Self {
        // Start off by creating a BTreeMap as we'll need to add a few fermionic terms
        // to each majorana term
        let mut majoranas: MajoranaBTree = MajoranaBTree::new();
        let majoranas = sft.append_majorana_btree(&mut majoranas);

        // debug!("Majoranas {:#?}", majoranas);
        let mut sparse_values: Vec<Complex64> =
            Vec::with_capacity(majoranas.values().filter(|&v| v.abs() >= 1e-16).count());
        let mut sparse_indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>> =
            Vec::with_capacity(sparse_values.len());
        // debug!("{:#?}", sparse_values.clone());
        let mut sparse_constant: num_complex::Complex<f64> = c64(0., 0.);
        majoranas
            .iter()
            .filter(|(_, &v)| v.abs() >= 1e-16)
            .for_each(|(k, &v)| {
                let mut op: ArrayVec<[u16; MAX_MAJORANAS]> = ArrayVec::new();
                if k.len() == 0 {
                    sparse_constant += v;
                } else {
                    for ind in k {
                        op.push(*ind as u16);
                    }
                    sparse_indices.push(op);
                    sparse_values.push(v);
                }
            });
        debug!("Sparse Majorana Indices {:?}", &sparse_indices);
        debug!("Sparse Majorana Coefficients {:?}", &sparse_values);
        MajoranaSparse::new(sparse_indices, sparse_values, sparse_constant)
            .expect("Indices and coefficients should be same length.")
    }
}

impl From<Vec<FermionSparse>> for MajoranaSparse {
    fn from(sft: Vec<FermionSparse>) -> Self {
        // Start off by creating a BTreeMap as we'll need to add a few fermionic terms
        // to each majorana term
        // let term_length = sft.ops.len();
        //     .flatten()
        //     .collect::<Vec<usize>>();
        // let offset_array: Array2<usize> =
        //     Array2::from_shape_vec((2_i32.pow(term_length as u32), term_length), offset_vec);

        // let mut majoranas: MajoranaBTree = MajoranaBTree::new();
        // for term in sft {
        //     let majoranas = term.append_majorana_btree(&mut majoranas);
        // }
        let mut majoranas: MajoranaBTree = MajoranaBTree::new();
        sft.iter().for_each(|term| {
            term.append_majorana_btree(&mut majoranas);
        });

        let mut sparse_values: Vec<Complex64> =
            Vec::with_capacity(majoranas.values().filter(|&v| v.abs() >= 1e-16).count());
        let mut sparse_indices: Vec<ArrayVec<[u16; MAX_MAJORANAS]>> =
            Vec::with_capacity(sparse_values.len());
        // debug!("{:#?}", sparse_values.clone());
        let mut sparse_constant: num_complex::Complex<f64> = c64(0., 0.);
        majoranas
            .iter()
            .filter(|(_, &v)| v.abs() >= 1e-16)
            .for_each(|(k, &v)| {
                let mut op: ArrayVec<[u16; MAX_MAJORANAS]> = ArrayVec::new();
                if k.len() == 0 {
                    sparse_constant += v;
                } else {
                    for ind in k {
                        op.push(*ind as u16);
                    }
                    sparse_indices.push(op);
                    sparse_values.push(v);
                }
            });
        debug!("Sparse Majorana Indices {:?}", &sparse_indices);
        debug!("Sparse Majorana Coefficients {:?}", &sparse_values);
        MajoranaSparse::new(sparse_indices, sparse_values, sparse_constant)
            .expect("Indices and coefficients should be same length.")
    }
}

#[cfg(test)]
mod majorana_tests {
    use crate::types::*;
    use log::debug;
    use ndarray::{arr1, arr2};
    use num_complex::c64;
    use tinyvec::array_vec;

    #[test]
    fn test_majorise_do_nothing() {
        let indices = vec![0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        assert_eq!(mp.indices, indices.clone());
        assert_eq!(mp.coefficient, coefficient.clone());
    }

    #[test]
    fn test_majorise_single_swap() {
        let indices = vec![1, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);
    }

    #[test]
    fn test_majorise_simplify() {
        let indices = vec![0, 0, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0]);
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise_simplify_to_empty() {
        let indices = vec![0, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, Vec::<usize>::new());
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![0, 1, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, Vec::<usize>::new());
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![1, 0, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, Vec::<usize>::new());
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise_reverse() {
        let indices = vec![3, 2, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 2, 3]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![4, 3, 2, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 2, 3, 4]);
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise() {
        let indices = vec![1, 1, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, Vec::<usize>::new());
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 1, 1, 0];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![1, 1, 0, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1]);
        assert_eq!(mp.coefficient, coefficient);

        let indices = vec![1, 0, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![0, 1, 1, 1];
        let coefficient = c64(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // debug!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1]);
        assert_eq!(mp.coefficient, coefficient);
    }
    #[test]
    fn test_from_fermion_sparse() {
        let indices = arr2(&[[0, 1]]);
        let coefficients = arr1(&[c64(10.0, 0.)]);
        let ops = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        debug!("{:#?}", indices.clone());
        debug!("{:#?}", coefficients.clone());
        debug!("{:#?}", ops.clone());

        let majorana_term = MajoranaSparse::new(
            vec![
                array_vec!([u16; 4]=> 0, 2),
                array_vec!([u16; 4]=> 0, 3),
                array_vec!([u16; 4]=> 1,2),
                array_vec!([u16; 4]=> 1,3),
            ],
            vec![c64(2.5, 0.), c64(0., -2.5), c64(0.0, 2.5), c64(2.5, 0.)],
            Complex64::ZERO,
        )
        .unwrap();
        let fermion_term =
            FermionSparse::new(ops.clone(), indices.clone(), coefficients.clone()).unwrap();
        assert_eq!(majorana_term, MajoranaSparse::from(fermion_term));
    }

    // #[test]
    // fn test_msparse_from_vec_fsparse() {
    //     todo!();
    // }
}
