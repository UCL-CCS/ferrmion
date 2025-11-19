use ndarray::Dimension;
/*
Shared Types.
*/
use crate::utils::vector_kron;
use itertools::Itertools;
use numpy::ndarray::{arr0, arr1, arr2, Array, Array1, Array2, Axis, IntoDimension, Zip};
use numpy::Complex64;
use std::collections::BTreeMap;
use std::iter::{repeat_n, zip};
use std::{result::Result, str::FromStr};

#[allow(dead_code)]
#[derive(Debug, Default)]
pub enum Pauli {
    #[default]
    I,
    X,
    Y,
    Z,
}

type PauliMatrix = Array2<Complex64>;

impl Into<PauliMatrix> for Pauli {
    fn into(self) -> PauliMatrix {
        match self {
            Pauli::I => arr2(&[
                [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                [Complex64::new(0., 0.), Complex64::new(1., 0.)],
            ]),
            Pauli::X => arr2(&[
                [Complex64::new(0., 0.), Complex64::new(1., 0.)],
                [Complex64::new(1., 0.), Complex64::new(0., 0.)],
            ]),
            Pauli::Z => arr2(&[
                [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                [Complex64::new(0., 0.), Complex64::new(-1., 0.)],
            ]),
            Pauli::Y => arr2(&[
                [Complex64::new(0., 0.), Complex64::new(0., -1.)],
                [Complex64::new(0., 1.), Complex64::new(0., 0.)],
            ]),
        }
    }
}

impl Into<String> for Pauli {
    fn into(self) -> String {
        match self {
            Pauli::I => "I".to_string(),
            Pauli::X => "X".to_string(),
            Pauli::Y => "Y".to_string(),
            Pauli::Z => "Z".to_string(),
        }
    }
}
impl Into<char> for Pauli {
    fn into(self) -> char {
        match self {
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

#[cfg(test)]
mod test_pauli {
    use crate::types::{Pauli, PauliMatrix};
    use ndarray::arr2;
    use num_complex::Complex64;

    #[test]
    fn test_matrix_identities() {
        let i = arr2(&[
            [Complex64::new(1., 0.), Complex64::new(0., 0.)],
            [Complex64::new(0., 0.), Complex64::new(1., 0.)],
        ]);
        let x = Into::<PauliMatrix>::into(Pauli::X);
        let y = Into::<PauliMatrix>::into(Pauli::Y);
        let z = Into::<PauliMatrix>::into(Pauli::Z);
        assert_eq!(&i.dot(&i), i);
        assert_eq!(&x.dot(&x), i);
        assert_eq!(&y.dot(&y), i);
        assert_eq!(&z.dot(&z), i);
        assert_eq!(&x.dot(&z), Complex64::new(0., -1.) * y.clone());
        assert_eq!(&y.dot(&z), Complex64::new(0., 1.) * x.clone());
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
            LadderOperator::Creation => arr1(&[Complex64::new(0.5, 0.0), Complex64::new(0., -0.5)]),
            LadderOperator::Annihilation => {
                arr1(&[Complex64::new(0.5, 0.0), Complex64::new(0., 0.5)])
            }
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

// #[derive(Debug, PartialEq)]
// struct FermionHamiltonian<'coeff> {
//     terms: Vec<(Vec<LadderOperator>, Array
//
// View2<'coeff, f64>)>,
// }

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

enum FermionTerm {
    FermionProduct,
    FermionSparse,
    FermionMatrix,
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

struct FermionMatrix<D: Dimension> {
    ops: Vec<LadderOperator>,
    coefficients: Array<f64, D>,
}

#[derive(Debug, PartialEq, Clone)]
struct FermionMatrixError;

impl<D: Dimension> FermionMatrix<D> {
    pub fn new(
        ops: Vec<LadderOperator>,
        coefficients: Array<f64, D>,
    ) -> Result<Self, FermionMatrixError> {
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

struct FermionSparse {
    ops: Vec<LadderOperator>,
    indices: Array2<usize>,
    coefficients: Array1<Complex64>,
}

#[derive(Debug, PartialEq, Clone)]
struct SparseFermionError;

impl FermionSparse {
    pub fn new(
        ops: Vec<LadderOperator>,
        indices: Array2<usize>,
        coefficients: Array1<Complex64>,
    ) -> Result<Self, SparseFermionError> {
        if coefficients.len() != indices.len_of(Axis(0)) || ops.len() != indices.len_of(Axis(1)) {
            return Err(SparseFermionError);
        };

        Ok(Self {
            ops,
            indices,
            coefficients,
        })
    }
}

impl<D: ndarray::Dimension + Copy> From<FermionMatrix<D>> for FermionSparse {
    fn from(mft: FermionMatrix<D>) -> FermionSparse {
        let n_nonzero = mft.coefficients.iter().filter(|&v| *v != 0.).count();
        let mut sparse_indices: Array2<usize> = Array2::zeros((n_nonzero, mft.ops.len()));
        let mut sparse_coefficients: Array1<Complex64> = Array1::zeros(n_nonzero);
        mft.coefficients
            .indexed_iter()
            .filter(|(_, v)| **v != 0.)
            .for_each(|(ind, v)| {
                let _ = sparse_indices.push_row(ind.into_dimension().as_array_view());
                let _ = sparse_coefficients.push(Axis(0), arr0(Complex64::new(*v, 0.)).view());
            });
        FermionSparse::new(mft.ops, sparse_indices, sparse_coefficients)
            .expect("Conversion from MatrixFermionTerm should be validated.")
    }
}

#[cfg(test)]
mod fermion_tests {
    use crate::types::*;
    use ndarray::{arr1, arr2};

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
        let coefficients = arr1(&[Complex64::new(1.0, 0.), Complex64::new(-1., 0.)]);
        let _term = FermionSparse::new(ops, indices, coefficients).unwrap();
    }
    #[test]
    fn test_matrix_term_creation() {
        let ops = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        let coefficients = arr2(&[[0., 0.], [0., 0.]]);
        let _term = FermionMatrix::new(ops, coefficients).unwrap();
    }
}

// /*
// Majorana
// */
//

// #[derive(Debug, PartialEq, Clone)]
// pub enum MajoranaTerm {
//     MajoranaProduct {
//         indices: Vec<usize>,
//         coefficient: Complex64,
//     },
//     MajoranaSparse,
// }

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
        let mut counter: usize = 0;
        let mut lower = 0;
        let mut upper = 0;

        let mut ind = 0;
        let mut safety = 0;
        'outer: while lower + upper < self.indices.len() && safety < 1000 {
            upper += 1;
            while ind < self.indices.len() - upper && safety < 1000 {
                safety += 1;
                let left = &self.indices[ind];
                let right = &self.indices[ind + 1];
                if left == right {
                    self.indices.remove(ind);
                    self.indices.remove(ind);
                    if self.indices.len() <= 1 {
                        break 'outer;
                    };
                } else if left > right {
                    self.indices.swap(ind, ind + 1);
                    counter += 1;
                }
                ind += 1;
            }
            lower += 1;
            while ind >= lower && safety < 1000 {
                safety += 1;
                let left = &self.indices[ind - 1];
                let right = &self.indices[ind];
                if left == right {
                    // things get moved left
                    self.indices.remove(ind - 1);
                    self.indices.remove(ind - 1);
                    if self.indices.len() <= 1 {
                        break 'outer;
                    };
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

#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaSparse {
    pub indices: Array2<usize>,
    pub coefficients: Array1<Complex64>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MajoranaSparseError;

impl MajoranaSparse {
    pub fn new(
        indices: Array2<usize>,
        coefficients: Array1<Complex64>,
    ) -> Result<Self, MajoranaSparseError> {
        if coefficients.len() != indices.nrows() {
            return Err(MajoranaSparseError);
        };
        Ok(Self {
            indices,
            coefficients,
        })
    }
}

impl From<FermionSparse> for MajoranaSparse {
    fn from(sft: FermionSparse) -> Self {
        // Start off by creating a BTreeMap as we'll need to add a few fermionic terms
        // to each majorana term
        let term_length = sft.ops.len();
        //     .flatten()
        //     .collect::<Vec<usize>>();
        // let offset_array: Array2<usize> =
        //     Array2::from_shape_vec((2_i32.pow(term_length as u32), term_length), offset_vec);
        let mut majoranas: BTreeMap<Vec<usize>, Complex64> = BTreeMap::new();
        Zip::from(sft.indices.rows())
            .and(sft.coefficients.view())
            .for_each(|ind, coeff| {
                let ops_coeffs: Vec<Complex64> = sft
                    .ops
                    .iter()
                    .map(|s| s.fermion_coeff())
                    .reduce(|acc, s| vector_kron(&acc, &s))
                    .unwrap()
                    .to_vec();

                // println!("{:#?}", ops_coeffs);
                let offset = repeat_n(0usize..2usize, term_length).multi_cartesian_product();
                for (sc, offset) in zip(ops_coeffs, offset) {
                    let mut majorana_term = Array1::zeros(term_length);
                    majorana_term += &ind;
                    majorana_term *= 2;
                    majorana_term = majorana_term + Array1::from_vec(offset);

                    let mut mp = MajoranaProduct::new(majorana_term.to_vec(), *coeff);
                    mp.majorise();
                    *majoranas
                        .entry(mp.indices)
                        .or_insert(Complex64 { re: 0.0, im: 0.0 }) += sc * mp.coefficient;
                }
            });

        // println!("Majoranas {:#?}", majoranas);
        let sparse_values: Array1<Complex64> = majoranas.values().cloned().collect();
        let mut sparse_indices: Array2<usize> =
            Array2::zeros((majoranas.keys().len(), term_length));
        // println!("{:#?}", sparse_values.clone());
        for (mut row, k) in zip(sparse_indices.rows_mut(), majoranas.keys()) {
            // let mut row_array: Array1<usize> = Array1::from_vec(k.clone());
            row.scaled_add(1, &Array1::from_vec(k.to_vec()));
        }
        MajoranaSparse::new(sparse_indices, sparse_values)
            .expect("Indices and coefficients should be same length.")
    }
}

#[cfg(test)]
mod majorana_tests {
    use crate::types::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_majorise_do_nothing() {
        let indices = vec![0, 1];
        let coefficient = Complex64::new(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        assert_eq!(mp.indices, indices.clone());
        assert_eq!(mp.coefficient, coefficient.clone());
    }

    #[test]
    fn test_majorise_single_swap() {
        let indices = vec![1, 0];
        let coefficient = Complex64::new(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // println!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0, 1]);
        assert_eq!(mp.coefficient, -1. * coefficient);
    }

    #[test]
    fn test_majorise_simplify() {
        let indices = vec![0, 0, 0];
        let coefficient = Complex64::new(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // println!("{:#?}", mp);
        assert_eq!(mp.indices, vec![0]);
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise_simplify_to_empty() {
        let indices = vec![0, 0];
        let coefficient = Complex64::new(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // println!("{:#?}", mp);
        assert_eq!(mp.indices, Vec::<usize>::new());
        assert_eq!(mp.coefficient, coefficient);
    }

    #[test]
    fn test_majorise_reverse() {
        let indices = vec![3, 2, 1];
        let coefficient = Complex64::new(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // println!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 2, 3]);
        assert_eq!(mp.coefficient, -1. * coefficient);

        let indices = vec![3, 4, 2, 1];
        let coefficient = Complex64::new(10.0, 0.);
        let mut mp = MajoranaProduct::new(indices.clone(), coefficient.clone());
        mp.majorise();
        // println!("{:#?}", mp);
        assert_eq!(mp.indices, vec![1, 2, 3, 4]);
        assert_eq!(mp.coefficient, -1. * coefficient);
    }

    #[test]
    fn test_sparse_term() {
        let indices = arr2(&[[0, 1]]);
        let coefficients = arr1(&[Complex64::new(10.0, 0.)]);
        let ops = vec![LadderOperator::Creation, LadderOperator::Annihilation];
        println!("{:#?}", indices.clone());
        println!("{:#?}", coefficients.clone());
        println!("{:#?}", ops.clone());

        let majorana_term = MajoranaSparse::new(
            arr2(&[[0, 2], [0, 3], [1, 2], [1, 3]]),
            arr1(&[
                Complex64::new(2.5, 0.),
                Complex64::new(0., -2.5),
                Complex64::new(0.0, 2.5),
                Complex64::new(2.5, 0.),
            ]),
        )
        .unwrap();
        let fermion_term =
            FermionSparse::new(ops.clone(), indices.clone(), coefficients.clone()).unwrap();
        assert_eq!(majorana_term, MajoranaSparse::from(fermion_term));
    }
}
