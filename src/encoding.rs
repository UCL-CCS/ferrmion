/*
Functions relating to the FermionQubitEncoding base class.
*/
use crate::hamiltonians::QubitHamiltonian;
use crate::operators::{FermionProduct, MajoranaProduct, MajoranaSparse, Pauli};
use crate::utils::{self, icount_to_sign};
use ahash::RandomState;
use itertools::izip;
use log::debug;
use ndarray::{Axis, Zip};
use num_complex::c64;
use numpy::ndarray::{azip, s, Array1, Array2, Array3, ArrayView1};
use numpy::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;
use thiserror::Error;

pub trait Encode<T> {
    /// Encodes the input into a QubitHamiltonian.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrmion::encoding::Encode;
    /// // Example usage would depend on the implementor
    /// ```
    fn encode(&self, input: T) -> QubitHamiltonian;
}

#[derive(Debug)]
pub struct MajoranaEncoding {
    pub ipowers: Array1<u8>,
    pub symplectics: Array2<bool>,
    pub n_modes: usize,
}

#[derive(Debug, Error)]
pub enum MajoranaEncodingError {
    #[error("Cannot construct Hartree-Fock state with Pauli operators {0:?}-i{1:?}.")]
    HartreeFockError(char, char),
}

// This caches symplectic products so that we don't have to calculate them
// four times for each pair of fermionic operators
// it does require memory scaling On^3 so if that becomes an issue we can be more clever.
impl MajoranaEncoding {
    pub fn new(ipowers: Array1<u8>, symplectics: Array2<bool>) -> Self {
        let n_modes = ipowers.len() / 2;
        Self {
            ipowers,
            symplectics,
            n_modes,
        }
    }

    pub fn symplectic_to_pauli(symplectic: ArrayView1<bool>, ipower: u8) -> (String, u8) {
        let block_width = symplectic.len_of(Axis(0)) / 2;
        let mut ipower: u8 = ipower;
        let (x_block, z_block) = symplectic.split_at(Axis(0), block_width);
        let mut pauli_string = String::new();
        Zip::from(x_block).and(z_block).for_each(|&x, &z| {
            if x && z {
                ipower += 3;
            };
            pauli_string.push(Pauli::from((x, z)).into());
        });
        (pauli_string, (ipower % 4))
    }

    pub fn symplectic_product(
        left: ArrayView1<bool>,
        right: ArrayView1<bool>,
        mut ipower: u8,
    ) -> (Array1<bool>, u8) {
        // bitwise or between two vectors
        let product = &left ^ &right;

        // bitwise sum of left z and right x
        let half_length: usize = left.len() / 2;

        let left_z = left.slice(s![half_length..]);
        let right_x = right.slice(s![..half_length]);
        for index in 0..half_length {
            if left_z[index] & right_x[index] {
                ipower += 2;
            };
        }

        (product, ipower % 4)
    }

    pub fn symplectic_product_map(
        &self,
        // ipowers: ArrayView1<u8>,
        // symplectics: ArrayView2<bool>,
    ) -> (Array2<u8>, Array3<bool>) {
        debug!("Calculating symplectic product map");

        let n_majoranas = self.symplectics.nrows();
        assert_eq!(n_majoranas, self.ipowers.len());

        let mut product_powers: Array2<u8> = Array2::zeros((n_majoranas, n_majoranas));
        let mut product_map: Array3<bool> =
            Array3::from_elem((n_majoranas, n_majoranas, self.symplectics.ncols()), false);
        azip!((index (l, r), pow in &mut product_powers) {
            let left = self.symplectics.slice(s![l,..]);
            let right = self.symplectics.slice(s![r,..]);
            let (term, imaginary) = MajoranaEncoding::symplectic_product(left, right, 0);

            *pow += &((imaginary + self.ipowers[[l]] + self.ipowers[[r]]) % 4);
            product_map.slice_mut(s![l,r,..]).assign(&term);
        });

        debug!("Found symplectic product map.");
        (product_powers, product_map)
    }

    /// Transforms a given fermionic Hartree-Fock to its computational-basis state.
    ///
    /// This will only work for vacuum preserving ['TernaryTree'] encodings.
    /// It assumes that:
    /// - The vacuum state is
    /// - two majorana operators forming a fermionic operator have non-trivial-overlap on exactly one qubit,
    /// - that one applies Pauli X and the other applies Y
    /// - they are ordered so that X is found on even rows and Y on odd rows
    /// - No entangling operators exist, so we can ignore global phase.
    pub fn ternary_tree_hartree_fock_state(
        &self,
        fermionic_hf_state: ArrayView1<bool>,
        mode_op_map: ArrayView1<usize>,
    ) -> Result<Array1<bool>, MajoranaEncodingError> {
        debug!("Calculating Hartree-fock state");

        let mut current_state: Array1<bool> = Array1::from_elem(fermionic_hf_state.len(), false);

        let half_length = self.symplectics.len_of(ndarray::Axis(1)) / 2;
        let (x_block, z_block) = self.symplectics.view().split_at(Axis(1), half_length);

        for (mode, occ) in fermionic_hf_state.into_iter().enumerate() {
            if !occ {
                continue;
            }
            let mode_index = mode_op_map[[mode]];

            // split the left and right operators into x and z sections
            let left_x = x_block.index_axis(ndarray::Axis(0), 2 * mode_index);
            let right_x = x_block.index_axis(ndarray::Axis(0), 2 * mode_index + 1);
            let left_z = z_block.index_axis(ndarray::Axis(0), 2 * mode_index);
            let right_z = z_block.index_axis(ndarray::Axis(0), 2 * mode_index + 1);

            let left_ops: Vec<Pauli> = left_x
                .iter()
                .zip(left_z.iter())
                .map(|(&x, &z)| Pauli::from((x, z)))
                .collect();
            let right_ops: Vec<Pauli> = right_x
                .iter()
                .zip(right_z.iter())
                .map(|(&x, &z)| Pauli::from((x, z)))
                .collect();

            for (s, &left, &right) in izip!(&mut current_state, &left_ops, &right_ops) {
                // Create an operator to act on the state with
                match (left, right) {
                    (Pauli::X, Pauli::Y) => {
                        if s == &false {
                            *s = true;
                        } else {
                            return Err(MajoranaEncodingError::HartreeFockError(
                                left.into(),
                                right.into(),
                            ));
                        }
                    }
                    // If the parent is an Odd Y-parity node
                    // The order of these can be swapped.
                    (Pauli::Y, Pauli::X) => {
                        if s == &false {
                            *s = true;
                        } else {
                            return Err(MajoranaEncodingError::HartreeFockError(
                                left.into(),
                                right.into(),
                            ));
                        }
                    }
                    (Pauli::Z, Pauli::I) => continue,
                    (Pauli::I, Pauli::Z) => continue,
                    (Pauli::X, Pauli::X) => *s = !*s,
                    (Pauli::Y, Pauli::Y) => *s = !*s,
                    (Pauli::Z, Pauli::Z) => continue,
                    (Pauli::I, Pauli::I) => continue,
                    _ => {
                        return Err(MajoranaEncodingError::HartreeFockError(
                            left.into(),
                            right.into(),
                        ))
                    }
                }
            }
        }
        Ok(current_state)
    }
}

impl MajoranaEncoding {
    pub fn apply_mode_enumeration(&self, mode_op_map: Vec<usize>) -> MajoranaEncoding {
        assert_eq!(2 * mode_op_map.len(), self.ipowers.len());
        assert_eq!(2 * mode_op_map.len(), self.symplectics.nrows());
        let majorana_rows: Vec<usize> = mode_op_map
            .iter()
            .flat_map(|v| [2 * v, 2 * v + 1])
            .collect();
        let ipowers: Array1<u8> = self.ipowers.select(Axis(0), &majorana_rows);
        let symplectics: Array2<bool> = self.symplectics.select(Axis(0), &majorana_rows);

        MajoranaEncoding::new(ipowers, symplectics)
    }
}

impl Encode<MajoranaProduct> for MajoranaEncoding {
    fn encode(&self, input: MajoranaProduct) -> HashMap<String, Complex64, RandomState> {
        let mut qham: HashMap<String, Complex64, RandomState> =
            HashMap::with_hasher(RandomState::new());
        let mut ipower: u8 = input
            .indices
            .iter()
            .fold(0_u8, |acc, &ind| acc + self.ipowers[ind])
            % 4;
        let (operator, product_ipower) = input.indices.iter().fold(
            (Array1::from_elem(self.symplectics.ncols(), false), 0_u8),
            |acc, &ind| Self::symplectic_product(acc.0.view(), self.symplectics.row(ind), acc.1),
        );
        ipower += product_ipower;
        debug!("{:#?}", operator);
        debug!(
            "{:#?}",
            MajoranaEncoding::symplectic_to_pauli(operator.view(), ipower)
        );
        let (pauli, ipower) = MajoranaEncoding::symplectic_to_pauli(operator.view(), ipower);

        qham.insert(
            pauli,
            utils::icount_to_sign(ipower as usize) * input.coefficient,
        );

        qham
    }
}

impl Encode<&MajoranaSparse> for MajoranaEncoding {
    fn encode(&self, input: &MajoranaSparse) -> QubitHamiltonian {
        let mut qham: QubitHamiltonian = HashMap::with_hasher(RandomState::new());
        let paulis_ipowers: Vec<(String, u8)> = input
            .indices
            .par_iter()
            .map(|&indices| {
                let (operator, product_ipower) = indices.iter().fold(
                    (Array1::from_elem(self.symplectics.ncols(), false), 0_u8),
                    |acc, &ind| {
                        let (a, i) = Self::symplectic_product(
                            acc.0.view(),
                            self.symplectics.row(ind as usize),
                            acc.1,
                        );
                        debug!("a, i {:#?}", (&a, &i));
                        (a, i)
                    },
                );
                debug!("Operator {:?}", operator);
                debug!("Product Ipower {:?}", product_ipower);
                let ipower: u8 = indices
                    .iter()
                    .fold(product_ipower, |acc, &ind| acc + self.ipowers[ind as usize])
                    % 4;
                // let ipower = product_ipower;
                debug!("Total Ipower {:?}", ipower);
                let (pauli, ipower) =
                    MajoranaEncoding::symplectic_to_pauli(operator.view(), ipower);
                (pauli, ipower)
            })
            .collect();

        for (pauli, coef) in paulis_ipowers.into_iter().zip(&input.coefficients) {
            *qham.entry(pauli.0).or_insert(Complex64::new(0., 0.)) +=
                coef * icount_to_sign(pauli.1 as usize);
            debug!("Total Ipower {:?}", icount_to_sign(pauli.1 as usize));
        }

        *qham
            .entry(
                (0..self.n_modes)
                    .map(|_| "I".to_string())
                    .collect::<String>(),
            )
            .or_insert(c64(0., 0.)) += input.constant;
        qham.into_iter().filter(|(_, v)| v.norm() > 1e-16).collect()
    }
}

impl Encode<FermionProduct> for MajoranaEncoding {
    fn encode(&self, input: FermionProduct) -> QubitHamiltonian {
        let msparse = MajoranaSparse::from(input);
        self.encode(&msparse)
    }
}

#[cfg(test)]
mod owned_tests {
    use super::*;

    use crate::{operators::LadderOperator, ternarytree::TernaryTree};
    use ndarray::{arr1, Array1, ArrayView1};
    use num_complex::c64;
    use numpy::Complex64;
    use tinyvec::array_vec;

    #[test]
    fn test_symplectic_product() {
        let xxx: Array1<bool> = ndarray::arr1(&[true, true, true, false, false, false]);
        let zzz: Array1<bool> = ndarray::arr1(&[false, false, false, true, true, true]);
        let product_result = MajoranaEncoding::symplectic_product(xxx.view(), zzz.view(), 0);
        let expected = (ndarray::arr1(&[true, true, true, true, true, true]), 0);
        assert_eq!(product_result, expected);
        let product_result = MajoranaEncoding::symplectic_product(zzz.view(), xxx.view(), 0);
        let expected = (ndarray::arr1(&[true, true, true, true, true, true]), 2);
        assert_eq!(product_result, expected);
    }

    #[test]
    fn test_symplectic_to_pauli() {
        // YXZI
        let symplectic: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 1]>> =
            ndarray::arr1(&[true, true, false, false, true, false, true, false]);
        assert_eq!(
            MajoranaEncoding::symplectic_to_pauli(symplectic.view(), 0),
            (String::from("YXZI"), 3)
        );
        let symplectic: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 1]>> =
            ndarray::arr1(&[false, false]);
        assert_eq!(
            MajoranaEncoding::symplectic_to_pauli(symplectic.view(), 0),
            (String::from("I"), 0)
        );
        let symplectic: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 1]>> =
            ndarray::arr1(&[false, true]);
        assert_eq!(
            MajoranaEncoding::symplectic_to_pauli(symplectic.view(), 0),
            (String::from("Z"), 0)
        );
        let symplectic: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 1]>> =
            ndarray::arr1(&[true, false]);
        assert_eq!(
            MajoranaEncoding::symplectic_to_pauli(symplectic.view(), 0),
            (String::from("X"), 0)
        );
        let symplectic: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 1]>> =
            ndarray::arr1(&[true, true]);
        assert_eq!(
            MajoranaEncoding::symplectic_to_pauli(symplectic.view(), 0),
            (String::from("Y"), 3)
        );
    }

    #[test]
    fn test_encode_fermion_product() {
        let fprod = FermionProduct::new(
            vec![LadderOperator::Creation, LadderOperator::Annihilation],
            vec![1, 0],
            c64(1., 0.),
        )
        .unwrap();
        let encoding = TernaryTree::naive_jordan_wigner(2)
            .build_encoding(2)
            .unwrap();

        let qham = encoding.encode(fprod);

        let mut expected = QubitHamiltonian::with_hasher(RandomState::new());
        expected.insert("YX".to_string(), c64(0., 0.25));
        expected.insert("XY".to_string(), c64(0., -0.25));
        expected.insert("XX".to_string(), c64(0.25, 0.));
        expected.insert("YY".to_string(), c64(0.25, 0.));
        assert_eq!(qham, expected);
    }

    #[test]
    fn test_encode_majorana_product() {
        let ipowers = ndarray::arr1(&[0, 1, 0, 0]);
        let symplectics = ndarray::arr2(&[
            [false, false, false, false, false, false],
            [true, true, true, true, true, true],
            [true, true, false, false, true, true],
            [true, false, true, false, true, false],
        ]);
        let encoding: MajoranaEncoding = MajoranaEncoding::new(ipowers, symplectics);

        let mp = MajoranaProduct::new(vec![0], Complex64::new(1.0, 0.));
        let qham = encoding.encode(mp);
        assert_eq!(qham.get("III").unwrap(), &Complex64::new(1., 0.));

        let mp = MajoranaProduct::new(vec![0, 0], Complex64::new(1.0, 0.));
        let qham = encoding.encode(mp);
        assert_eq!(qham.get("III").unwrap(), &Complex64::new(1.0, 0.));

        let mp = MajoranaProduct::new(vec![1, 1], Complex64::new(1.0, 0.));
        let qham = encoding.encode(mp);
        assert_eq!(qham.get("III").unwrap(), &Complex64::new(1.0, 0.));

        let mp = MajoranaProduct::new(vec![2, 3], Complex64::new(1.0, 0.));
        let qham = encoding.encode(mp);
        assert_eq!(qham.get("IXY").unwrap(), &Complex64::new(0., 1.));

        let mp = MajoranaProduct::new(vec![3, 2], Complex64::new(1.0, 0.));
        let qham = encoding.encode(mp);
        assert_eq!(qham.get("IXY").unwrap(), &Complex64::new(0.0, 1.));

        let mp = MajoranaProduct::new(vec![3, 2, 2, 2], Complex64::new(1.0, 0.));
        let qham = encoding.encode(mp);
        assert_eq!(qham.get("IXY").unwrap(), &Complex64::new(0.0, -1.));
    }

    #[test]
    fn test_encode_sparse_xz() {
        let ipowers = ndarray::arr1(&[0, 0]);
        let symplectics = ndarray::arr2(&[
            [true, true, true, false, false, false],
            [false, false, false, true, true, true],
        ]);
        let encoding: MajoranaEncoding = MajoranaEncoding::new(ipowers, symplectics);
        let ms = MajoranaSparse::new(
            vec![array_vec!([u16; 4] =>0, 1), array_vec!([u16; 4] =>1,0)],
            vec![Complex64::new(1.0, 0.), Complex64::new(1.0, 0.)],
            0.,
        )
        .unwrap();
        let _qham = encoding.encode(&ms);
    }

    #[test]
    fn test_encode_sparse_iy() {
        let ipowers = ndarray::arr1(&[0, 0]);
        let symplectics = ndarray::arr2(&[
            [false, false, false, false, false, false],
            [true, true, true, true, true, true],
        ]);
        let encoding: MajoranaEncoding = MajoranaEncoding::new(ipowers, symplectics);
        let ms = MajoranaSparse::new(
            vec![array_vec!([u16; 4] =>0, 1), array_vec!([u16; 4] =>1,0)],
            vec![Complex64::new(1.0, 0.), Complex64::new(-1.0, 0.)],
            0.,
        )
        .unwrap();
        debug!("{:#?}", ms);
        let qham = encoding.encode(&ms);
        debug!("{:#?}", qham);
    }
    #[test]
    fn test_encode_sparse_long() {
        let ipowers = ndarray::arr1(&[0, 1, 0, 0]);
        let symplectics = ndarray::arr2(&[
            [false, false, false, false, false, false],
            [true, true, true, true, true, true],
            [true, true, false, false, true, true],
            [true, false, true, false, true, false],
        ]);
        let encoding: MajoranaEncoding = MajoranaEncoding::new(ipowers, symplectics);
        let ms = MajoranaSparse::new(
            vec![
                array_vec!([u16; 4] =>0,0),
                array_vec!([u16; 4] =>1,1),
                array_vec!([u16; 4] =>2,3),
                array_vec!([u16; 4] =>3,2),
            ],
            vec![
                Complex64::new(1.0, 0.),
                Complex64::new(1.0, 0.),
                Complex64::new(1.0, 0.),
                Complex64::new(1.0, 0.),
            ],
            0.,
        )
        .unwrap();
        debug!("{:#?}", ms);
        let qham = encoding.encode(&ms);
        debug!("{:#?}", qham);
        assert_eq!(qham.get("III").unwrap(), &Complex64::new(2., 0.));
        assert_eq!(qham.get("IXY").unwrap(), &Complex64::new(0., 2.));
        // assert_eq!(qham.get("IXY").unwrap(), &Complex64::new(-1., 0.));
    }

    #[test]
    fn test_symplectic_product_map() {
        let ipowers = ndarray::arr1(&[0, 1]);
        let symplectics = ndarray::arr2(&[[true, true, false, false], [true, false, true, false]]);
        let (iproducts, symplectic_products) =
            MajoranaEncoding::new(ipowers, symplectics).symplectic_product_map();
        debug!("{}", iproducts);
        debug!("{}", symplectic_products);
        assert_eq!(iproducts, ndarray::arr2(&[[0, 1], [3, 0]]));
        assert_eq!(
            symplectic_products.view(),
            ndarray::arr3(&[
                [[false, false, false, false], [false, true, true, false]],
                [[false, true, true, false], [false, false, false, false]]
            ])
            .view()
        );
    }

    #[test]
    fn test_tt_hartree_fock() {
        let fermionic_hf_state: ArrayView1<bool> =
            ArrayView1::from(&[true, true, true, false, false, false]);
        let mode_op_map: ArrayView1<usize> = ArrayView1::from(&[0, 1, 2, 3, 4, 5]);
        let tree = TernaryTree::naive_jordan_wigner(6);
        let encoding: MajoranaEncoding = tree.build_encoding(6).unwrap();
        let result = encoding
            .ternary_tree_hartree_fock_state(fermionic_hf_state, mode_op_map)
            .unwrap();
        assert!(result == arr1(&[true, true, true, false, false, false]));

        let result2 = encoding
            .ternary_tree_hartree_fock_state(
                ArrayView1::from(&[true, true, true, true, false, false]),
                mode_op_map,
            )
            .unwrap();
        assert!(result2 == arr1(&[true, true, true, true, false, false]));
    }

    #[test]
    fn test_hartree_fock() {
        let fermionic_hf_state: ArrayView1<bool> =
            ArrayView1::from(&[true, true, true, false, false, false]);
        let mode_op_map: ArrayView1<usize> = ArrayView1::from(&[0, 1, 2, 3, 4, 5]);
        let tree = TernaryTree::naive_jordan_wigner(6);
        let encoding: MajoranaEncoding = tree.build_encoding(6).unwrap();
        let result = encoding
            .ternary_tree_hartree_fock_state(fermionic_hf_state, mode_op_map)
            .unwrap();

        assert!(result == arr1(&[true, true, true, false, false, false]));

        let result2 = encoding
            .ternary_tree_hartree_fock_state(
                ArrayView1::from(&[true, true, true, true, false, false]),
                mode_op_map,
            )
            .unwrap();
        assert!(result2 == arr1(&[true, true, true, true, false, false]));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_naive_jw_hf_state_unchanged(hf_state in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
            let n = hf_state.len();
            let tree = TernaryTree::naive_jordan_wigner(n);
            let encoding = tree.build_encoding(n).unwrap();
            let mode_op_map: Array1<usize> = (0..n).collect();
            let qubit_hf = encoding.ternary_tree_hartree_fock_state(Array1::from(hf_state.clone()).view(), mode_op_map.view()).unwrap();
            let expected: Array1<bool> = hf_state.into_iter().collect();
            prop_assert_eq!(qubit_hf, expected);
        }

        #[test]
        fn test_naive_parity_hf_state_is_occupation_parity(hf_state in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
            let n = hf_state.len();
            let tree = TernaryTree::naive_parity(n);
            let encoding = tree.build_encoding(n).unwrap();
            let mode_op_map: Array1<usize> = (0..n).collect();
            let qubit_hf = encoding.ternary_tree_hartree_fock_state(Array1::from(hf_state.clone()).view(), mode_op_map.view()).unwrap();
            // expected_parity = cumsum(reversed) % 2, then reverse back
            let mut reversed: Vec<bool> = hf_state.into_iter().rev().collect();
            let mut cumsum: usize = 0;
            for rev in reversed.iter_mut() {
                if *rev { cumsum += 1; }
                *rev = !cumsum.is_multiple_of(2);
            }
            let expected: Array1<bool> = reversed.into_iter().rev().collect();
            prop_assert_eq!(qubit_hf, expected);
        }

        #[test]
        fn test_enumerated_jw_hf_state_match_reordered_naive(mode_op_map in proptest::sample::subsequence((0..10).collect::<Vec<usize>>(), 10), n_electrons in 1..11usize) {
            let n = 10;
            let mut hf_state: Vec<bool> = vec![true; n_electrons];
            hf_state.extend(vec![false; n - n_electrons]);
            let tree = TernaryTree::naive_jordan_wigner(n);
            let encoding = tree.build_encoding(n).unwrap();
            let naive_mode_op_map: Array1<usize> = (0..n).collect();
            let naive_qubit_hf = encoding.ternary_tree_hartree_fock_state(Array1::from(hf_state.clone()).view(), naive_mode_op_map.view()).unwrap();
            prop_assert_eq!(naive_qubit_hf, Array1::from(hf_state.clone()));
            let enumerated_qubit_hf = encoding.ternary_tree_hartree_fock_state(Array1::from(hf_state.clone()).view(), Array1::from(mode_op_map.clone()).view()).unwrap();
            let mut expected = vec![false; n];
            for &i in &mode_op_map[..n_electrons] {
                if i < n {
                    expected[i] = true;
                }
            }
            prop_assert_eq!(enumerated_qubit_hf, Array1::from(expected));
        }
        #[test]
        fn test_naive_jw_tt_hf_state_unchanged(hf_state in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
            let n = hf_state.len();
            let tree = TernaryTree::naive_jordan_wigner(n);
            let encoding = tree.build_encoding(n).unwrap();
            let mode_op_map: Array1<usize> = (0..n).collect();
            let qubit_hf = encoding.ternary_tree_hartree_fock_state(Array1::from(hf_state.clone()).view(), mode_op_map.view()).unwrap();
            let expected: Array1<bool> = hf_state.into_iter().collect();
            prop_assert_eq!(qubit_hf, expected);
        }

        #[test]
        fn test_naive_parity_tt_hf_state_is_occupation_parity(hf_state in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
            let n = hf_state.len();
            let tree = TernaryTree::naive_parity(n);
            let encoding = tree.build_encoding(n).unwrap();
            let mode_op_map: Array1<usize> = (0..n).collect();
            let qubit_hf = encoding.ternary_tree_hartree_fock_state(Array1::from(hf_state.clone()).view(), mode_op_map.view()).unwrap();
            // expected_parity = cumsum(reversed) % 2, then reverse back
            let mut reversed: Vec<bool> = hf_state.into_iter().rev().collect();
            let mut cumsum:usize = 0;
            for rev in reversed.iter_mut() {
                if *rev { cumsum += 1; }
                *rev = !cumsum.is_multiple_of(2);
            }
            let expected: Array1<bool> = reversed.into_iter().rev().collect();
            prop_assert_eq!(qubit_hf, expected);
        }

        #[test]
        fn test_enumerated_jw_tt_hf_state_match_reordered_naive(mode_op_map in proptest::sample::subsequence((0..10).collect::<Vec<usize>>(), 10), n_electrons in 1..11usize) {
            let n = 10;
            let mut hf_state: Vec<bool> = vec![true; n_electrons];
            hf_state.extend(vec![false; n - n_electrons]);
            let tree = TernaryTree::naive_jordan_wigner(n);
            let encoding = tree.build_encoding(n).unwrap();
            let naive_mode_op_map: Array1<usize> = (0..n).collect();
            let naive_qubit_hf = encoding.ternary_tree_hartree_fock_state(Array1::from(hf_state.clone()).view(), naive_mode_op_map.view()).unwrap();
            prop_assert_eq!(naive_qubit_hf, Array1::from(hf_state.clone()));
            let enumerated_qubit_hf = encoding.ternary_tree_hartree_fock_state(Array1::from(hf_state.clone()).view(), Array1::from(mode_op_map.clone()).view()).unwrap();
            let mut expected = vec![false; n];
            for &i in &mode_op_map[..n_electrons] {
                if i < n {
                    expected[i] = true;
                }
            }
            prop_assert_eq!(enumerated_qubit_hf, Array1::from(expected));
        }
    }
}
