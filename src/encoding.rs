use crate::hamiltonians::QubitHamiltonian;
/*
Functions relating to the FermionQubitEncoding base class.
*/

use crate::operators::{MajoranaProduct, MajoranaSparse, Pauli};
use crate::utils::{self, icount_to_sign, vector_kron};
use ahash::RandomState;
use log::debug;
use ndarray::{Axis, Zip};
use num_complex::c64;
use numpy::ndarray::{azip, s, Array1, Array2, Array3, ArrayView1};
use numpy::Complex64;
use std::collections::HashMap;

pub trait Encode<T> {
    fn encode(&self, input: T) -> QubitHamiltonian;
}

#[derive(Debug)]
pub struct MajoranaEncoding {
    pub ipowers: Array1<u8>,
    pub symplectics: Array2<bool>,
    pub n_modes: usize,
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

    pub fn hartree_fock_state(
        &self,
        vacuum_state: ArrayView1<f64>,
        fermionic_hf_state: ArrayView1<bool>,
        mode_op_map: ArrayView1<usize>,
    ) -> (Array1<Complex64>, Array2<bool>) {
        debug!("Calculating Hartree-fock state");

        let mut current_state =
            vec![Array1::from(vec![c64(1., 0.), c64(0., 0.)]); vacuum_state.len_of(Axis(0))];

        let half_length = self.symplectics.len_of(ndarray::Axis(1)) / 2;
        let (x_block, z_block) = self.symplectics.view().split_at(Axis(1), half_length);

        for (mode, occ) in fermionic_hf_state.into_iter().enumerate() {
            if !occ {
                continue;
            }
            let mode_index = mode_op_map[[mode]];

            let left_x = x_block.index_axis(ndarray::Axis(0), 2 * mode_index);
            let right_x = x_block.index_axis(ndarray::Axis(0), 2 * mode_index + 1);
            let left_z = z_block.index_axis(ndarray::Axis(0), 2 * mode_index);
            let right_z = z_block.index_axis(ndarray::Axis(0), 2 * mode_index + 1);

            // split the left and righ operators into x and z sections
            Zip::from(&mut current_state)
                .and(&left_x)
                .and(&left_z)
                .and(&right_x)
                .and(&right_z)
                .for_each(|s, &lx, &lz, &rx, &rz| {
                    // Create an operator to act on the state with
                    let left_op: Array2<Complex64> = Pauli::from((lx, lz)).into();
                    let right_op: Array2<Complex64> = Pauli::from((rx, rz)).into();
                    let total_op = left_op - right_op.map(|op| op * Complex64::new(0., 1.));
                    *s = total_op.dot(s);
                });
        }

        let mut vector_state: Array1<Complex64> = Zip::from(&current_state)
            .fold(Array1::from_elem(1, c64(1., 0.)), |acc, c| {
                vector_kron(&acc, c)
            });

        let mut zero_coeffs = Vec::new();
        let mut hf_components: Vec<bool> = Vec::new();
        // According to ndarray docs, when we don't know the final size
        // of a multidimensional array we want to build iteratively
        // the best thing to do is create a flat array and then reshape
        for index in 0..vector_state.len() {
            let coeff = vector_state[index];
            if !(coeff == c64(0., 0.)) {
                let binary = format!("{:0<width$}", format!("{index:b}"), width = (half_length));
                for val in binary.chars() {
                    hf_components.push(val.to_digit(10).unwrap() == 1)
                }
            } else {
                zero_coeffs.push(index);
            }
        }
        for index in zero_coeffs.iter().rev() {
            vector_state.remove_index(Axis(0), *index);
        }

        let coeffs = vector_state.mapv(|c| c / (vector_state[0]));

        let hf_components: ndarray::ArrayBase<ndarray::OwnedRepr<bool>, ndarray::Dim<[usize; 2]>> =
            Array2::from_shape_vec((coeffs.len(), vacuum_state.len()), hf_components)
                .expect("Should be able to make hf components array from vec.");
        debug!(
            "Found Hartree-Fock state: coeffs={:?}, hf_components={:#?}",
            coeffs, hf_components
        );
        (coeffs, hf_components)
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
            .fold(0_u8, |acc, &ind| acc + &self.ipowers[ind])
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
        input
            .indices
            .iter()
            .zip(&input.coefficients)
            .for_each(|(&indices, coef)| {
                debug!("\nindices {:?}", indices);
                debug!("coef {:?}", coef);

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
                let ipower: u8 = indices.iter().fold(product_ipower, |acc, &ind| {
                    acc + &self.ipowers[ind as usize]
                }) % 4;
                // let ipower = product_ipower;
                debug!("Total Ipower {:?}", ipower);
                let (pauli, ipower) =
                    MajoranaEncoding::symplectic_to_pauli(operator.view(), ipower);
                debug!("Pauli term {:?}", pauli.clone());
                *qham.entry(pauli).or_insert(Complex64::new(0., 0.)) +=
                    coef * icount_to_sign(ipower as usize);
                debug!("Total Ipower {:?}", icount_to_sign(ipower as usize));
            });

        *qham
            .entry(
                (0..self.n_modes)
                    .map(|_| "I".to_string())
                    .collect::<String>(),
            )
            .or_insert(c64(0., 0.)) += input.constant;
        // qham.into_iter().filter(|(_, v)| v.abs() > 1e-12).collect()
        qham.into_iter().filter(|(_, v)| v.norm() > 1e-16).collect()
    }
}

#[cfg(test)]
mod owned_tests {
    

    use super::*;
    use ndarray::{arr2, Array1, ArrayView1};
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
    fn test_encode_product() {
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
        let qham = encoding.encode(&ms);
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
    fn test_hartree_fock() {
        let vacuum_state: ArrayView1<f64> = ArrayView1::from(&[0., 0., 0., 0., 0., 0.]);
        let fermionic_hf_state: ArrayView1<bool> =
            ArrayView1::from(&[true, true, true, false, false, false]);
        let mode_op_map: ArrayView1<usize> = ArrayView1::from(&[0, 1, 2, 3, 4, 5, 6]);
        let ipowers = ndarray::arr1(&[0, 0, 0, 0, 0, 0, 0]);
        let symplectics: Array2<bool> = arr2(&[
            [
                true, false, false, false, false, false, false, false, false, false, false, false,
            ],
            [
                true, false, false, false, false, false, true, false, false, false, false, false,
            ],
            [
                false, true, false, false, false, false, true, false, false, false, false, false,
            ],
            [
                false, true, false, false, false, false, true, true, false, false, false, false,
            ],
            [
                false, false, true, false, false, false, true, true, false, false, false, false,
            ],
            [
                false, false, true, false, false, false, true, true, true, false, false, false,
            ],
            [
                false, false, false, true, false, false, true, true, true, false, false, false,
            ],
            [
                false, false, false, true, false, false, true, true, true, true, false, false,
            ],
            [
                false, false, false, false, true, false, true, true, true, true, false, false,
            ],
            [
                false, false, false, false, true, false, true, true, true, true, true, false,
            ],
            [
                false, false, false, false, false, true, true, true, true, true, true, false,
            ],
            [
                false, false, false, false, false, true, true, true, true, true, true, true,
            ],
        ]);
        let encoding = MajoranaEncoding::new(ipowers, symplectics);
        let result = encoding.hartree_fock_state(vacuum_state, fermionic_hf_state, mode_op_map);
        let c1 = c64(1., 0.);
        assert!(result.0 == ndarray::arr1(&[c1]));
        assert!(result.1 == arr2(&[[true, true, true, false, false, false]]));

        let result2 = encoding.hartree_fock_state(
            vacuum_state,
            ArrayView1::from(&[true, true, true, true, false, false]),
            mode_op_map,
        );
        assert!(result2.0 == ndarray::arr1(&[c1]));
        assert!(result2.1 == arr2(&[[true, true, true, true, false, false]]));
    }
}
