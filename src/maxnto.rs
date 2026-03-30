//! MaxNTO (k-NTO) encoding for fermionic operators.
//!
//! Builds the symplectic matrix of Majorana operators for the k-NTO encoding.
//! Requires n_modes such that k = n_modes - 1 is odd.

use ndarray::{Array1, Array2};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MaxNTOError {
    #[error("Only works for Odd k (n_modes - 1 must be odd, got n_modes = {0})")]
    EvenK(usize),
}

/// Build the symplectic matrix of Majorana operators for the k-NTO encoding.
///
/// # Arguments
/// * `n_modes` - Number of fermionic modes. Requires `n_modes - 1` to be odd.
///
/// # Returns
/// A tuple of `(y_count, symplectic_matrix)` where:
/// - `y_count` is a 1D array of length `2 * n_modes` containing the phase exponents (mod 4).
/// - `symplectic_matrix` is a 2D boolean array of shape `(2 * n_modes, 2 * n_modes)`.
pub fn maxnto_symplectic_matrix(
    n_modes: usize,
) -> Result<(Array1<u8>, Array2<bool>), MaxNTOError> {
    let k = n_modes - 1;
    if k % 2 != 1 {
        return Err(MaxNTOError::EvenK(n_modes));
    }

    // x_block: all true except on diagonal (upper-tri + lower-tri off-diagonal)
    let mut x_block = Array2::<bool>::from_elem((n_modes, n_modes), true);
    for i in 0..n_modes {
        x_block[[i, i]] = false;
    }

    // z_block: lower triangular; even-indexed diagonal entries set to true
    let mut z_block = Array2::<bool>::from_elem((n_modes, n_modes), false);
    for i in 0..n_modes {
        for j in 0..i {
            z_block[[i, j]] = true;
        }
        if i % 2 == 0 {
            z_block[[i, i]] = true;
        }
    }

    // z_block[1::2] = z_block.T[1::2, :]
    // Snapshot transposed values before modifying (numpy evaluates RHS first)
    let z_block_t: Array2<bool> = z_block.t().to_owned();
    for i in (1..n_modes).step_by(2) {
        for j in 0..n_modes {
            z_block[[i, j]] = z_block_t[[i, j]];
        }
    }

    // Build output matrix (2*n_modes x 2*n_modes):
    //   even rows (2i)   = odd_majorana[i]  = hstack(x_block[i], z_block[i])
    //   odd rows  (2i+1) = even_majorana[i] = xy_swap(hstack(x_block[i], z_block[i]))
    //
    // xy_swap: X (x=1,z=0) -> Y (x=1,z=1) and Y (x=1,z=1) -> X (x=1,z=0)
    let mut output = Array2::<bool>::from_elem((2 * n_modes, 2 * n_modes), false);
    for i in 0..n_modes {
        for j in 0..n_modes {
            let x = x_block[[i, j]];
            let z = z_block[[i, j]];

            // odd_majorana row
            output[[2 * i, j]] = x;
            output[[2 * i, j + n_modes]] = z;

            // even_majorana row = xy_swap applied element-wise
            if x && !z {
                // X -> Y
                output[[2 * i + 1, j]] = true;
                output[[2 * i + 1, j + n_modes]] = true;
            } else if x && z {
                // Y -> X
                output[[2 * i + 1, j]] = true;
                output[[2 * i + 1, j + n_modes]] = false;
            } else {
                // I or Z: unchanged
                output[[2 * i + 1, j]] = x;
                output[[2 * i + 1, j + n_modes]] = z;
            }
        }
    }

    // y_count[r] = count of positions j where output[r,j] && output[r,j+n_modes], mod 4
    let mut y_count = Array1::<u8>::zeros(2 * n_modes);
    for r in 0..(2 * n_modes) {
        let mut count: u32 = 0;
        for j in 0..n_modes {
            if output[[r, j]] && output[[r, j + n_modes]] {
                count += 1;
            }
        }
        y_count[r] = (count % 4) as u8;
    }

    Ok((y_count, output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxnto_symplectic_matrix_even_k_error() {
        // n_modes = 3 → k = 2, which is even → should error
        assert!(matches!(
            maxnto_symplectic_matrix(3),
            Err(MaxNTOError::EvenK(3))
        ));
        // n_modes = 5 → k = 4, even → error
        assert!(matches!(
            maxnto_symplectic_matrix(5),
            Err(MaxNTOError::EvenK(5))
        ));
    }

    #[test]
    fn test_maxnto_symplectic_matrix_n14() {
        let (y_count, symplectics) = maxnto_symplectic_matrix(14).unwrap();

        let expected_y_count: Vec<u8> =
            vec![0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1];
        assert_eq!(y_count.to_vec(), expected_y_count);

        // Verify shape
        assert_eq!(symplectics.shape(), &[28, 28]);

        // Spot-check first row: x_part = [F,T,T,...,T], z_part = all false
        // Row 0 = odd_majorana[0]: x_block[0] = [F,T,T,...,T], z_block[0] after transpose = [T,F,F,...,F]
        // (i=0 is even, so z_block[0,0]=true; j<0 is nothing; result: [T,F,F,...,F])
        // So row 0 = [F,T,T,...,T, T,F,F,...,F]
        // x_part[0,0] = false (diagonal), x_part[0,1..] = true
        assert!(!symplectics[[0, 0]]);
        for j in 1..14 {
            assert!(symplectics[[0, j]], "row 0 x_part col {j} should be true");
        }
        // z_part[0,0] = true (even diagonal), z_part[0,1..] = false
        assert!(symplectics[[0, 14]]);
        for j in 1..14 {
            assert!(
                !symplectics[[0, 14 + j]],
                "row 0 z_part col {j} should be false"
            );
        }
    }
}
