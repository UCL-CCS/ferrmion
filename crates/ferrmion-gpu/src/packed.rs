//! Flat, device-ready layout of a Pauli-weight problem, the per-thread kernel
//! arithmetic, and the host-side sum of the kernel's partial weights.
//!
//! Nothing here depends on CUDA: [`PackedProblem::emulate`] runs
//! [`lane_weights`] for every thread on the CPU, so the kernel arithmetic can
//! be tested on stable Rust without a GPU.

use ferrmion_core::encode::majorana::MajoranaEncoding;
use ferrmion_core::operators::{MAX_MAJORANAS, MajoranaSparse};
use rayon::prelude::*;

/// Target number of terms each kernel thread sums.
const TERMS_PER_LANE: usize = 8;
/// Upper bound on the lanes (threads) per permutation.
const MAX_LANES: usize = 1024;

/// A [`MajoranaEncoding`] and the [`ferrmion_core::encode::majorana::PauliWeightTerms`]
/// of a Hamiltonian, flattened into the buffers the weight kernel reads.
///
/// Thread `i` handles permutation `i / lanes` and sums the terms
/// `lane, lane + lanes, …` for `lane = i % lanes`, so adjacent threads read
/// adjacent terms. `lanes` depends only on the number of terms, which keeps
/// the summation order, and so the result, independent of the batch.
#[derive(Debug, Clone)]
pub struct PackedProblem {
    pub n_modes: usize,
    pub n_terms: usize,
    /// 64-bit words per qubit row.
    pub words: usize,
    /// Threads per permutation.
    pub lanes: usize,
    /// X block of the encoding, row-major, `2 * n_modes` rows of `words` words.
    pub x_rows: Vec<u64>,
    /// Z block of the encoding, same layout as `x_rows`.
    pub z_rows: Vec<u64>,
    /// Majorana indices of each term, padded to [`MAX_MAJORANAS`] per term.
    pub term_indices: Vec<u32>,
    /// Number of Majorana indices in each term.
    pub term_lens: Vec<u32>,
    /// Merged coefficient norm of each term.
    pub norms: Vec<f64>,
}

impl PackedProblem {
    pub fn new(encoding: &MajoranaEncoding, msparse: &MajoranaSparse) -> Self {
        let terms = encoding.pauli_weight_terms(msparse);
        let n_terms = terms.indices.len();
        let mut term_indices = vec![0u32; n_terms * MAX_MAJORANAS];
        let mut term_lens = Vec::with_capacity(n_terms);
        for (t, indices) in terms.indices.iter().enumerate() {
            for (k, &ind) in indices.iter().enumerate() {
                term_indices[t * MAX_MAJORANAS + k] = ind as u32;
            }
            term_lens.push(indices.len() as u32);
        }
        let operators = &encoding.operators;
        Self {
            n_modes: encoding.n_modes,
            n_terms,
            words: operators.x_block.term_width(),
            lanes: n_terms.div_ceil(TERMS_PER_LANE).clamp(1, MAX_LANES),
            x_rows: operators.x_block.to_u64_words(),
            z_rows: operators.z_block.to_u64_words(),
            term_indices,
            term_lens,
            norms: terms.norms,
        }
    }

    /// Flatten mode permutations (as taken by
    /// [`MajoranaEncoding::batch_pauli_weights`]) into one buffer.
    ///
    /// # Panics
    ///
    /// Panics if a permutation does not have length `n_modes`.
    pub fn pack_permutations(&self, permutations: &[Vec<usize>]) -> Vec<u32> {
        let mut packed = Vec::with_capacity(permutations.len() * self.n_modes);
        for perm in permutations {
            assert_eq!(
                perm.len(),
                self.n_modes,
                "Permutation length is not n_modes"
            );
            packed.extend(perm.iter().map(|&m| m as u32));
        }
        packed
    }

    /// Number of kernel threads (and partial sums) for `n_perms` permutations.
    pub fn n_threads(&self, n_perms: usize) -> usize {
        n_perms * self.lanes
    }

    /// Sum each permutation's per-lane partials, in lane order, into the
    /// `(plain, weighted)` Pauli weights returned by
    /// [`MajoranaEncoding::batch_pauli_weights`].
    pub fn reduce(&self, n_perms: usize, plain: &[u32], weighted: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(plain.len(), self.n_threads(n_perms));
        assert_eq!(weighted.len(), self.n_threads(n_perms));
        plain
            .chunks(self.lanes)
            .zip(weighted.chunks(self.lanes))
            .map(|(p, w)| {
                (
                    p.iter().map(|&x| f64::from(x)).sum::<f64>(),
                    w.iter().sum::<f64>(),
                )
            })
            .unzip()
    }

    /// Run the kernel arithmetic on the CPU and reduce the result.
    ///
    /// Produces exactly the partials the GPU kernel writes, so it is the
    /// reference the device path is tested against.
    pub fn emulate(&self, permutations: &[Vec<usize>]) -> (Vec<f64>, Vec<f64>) {
        let perms = self.pack_permutations(permutations);
        let (plain, weighted): (Vec<u32>, Vec<f64>) = (0..self.n_threads(permutations.len()))
            .into_par_iter()
            .map(|i| {
                lane_weights(
                    i,
                    &self.x_rows,
                    &self.z_rows,
                    &self.term_indices,
                    &self.term_lens,
                    &self.norms,
                    &perms,
                    self.n_terms,
                    self.n_modes,
                    self.words,
                    self.lanes,
                )
            })
            .unzip();
        self.reduce(permutations.len(), &plain, &weighted)
    }
}

/// The work of one weight-kernel thread.
///
/// Returns the plain and coefficient-weighted Pauli weight of this thread's
/// share of the terms (see [`PackedProblem`] for the mapping) under its
/// permutation. A term's Pauli string is the XOR of its permuted encoding
/// rows, so its weight is `popcount(x | z)` summed over the row's words; no
/// phase is needed because the terms were merged in advance.
///
/// This is a plain function so that the cuda-oxide backend compiles it into
/// the kernel and the CPU emulation calls the same code.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn lane_weights(
    thread: usize,
    x_rows: &[u64],
    z_rows: &[u64],
    term_indices: &[u32],
    term_lens: &[u32],
    norms: &[f64],
    perms: &[u32],
    n_terms: usize,
    n_modes: usize,
    words: usize,
    lanes: usize,
) -> (u32, f64) {
    let lane = thread % lanes;
    let perm = &perms[(thread / lanes) * n_modes..(thread / lanes + 1) * n_modes];

    let mut plain = 0u32;
    let mut weighted = 0f64;
    let mut term = lane;
    while term < n_terms {
        let indices =
            &term_indices[term * MAX_MAJORANAS..term * MAX_MAJORANAS + term_lens[term] as usize];
        let mut weight = 0u32;
        let mut word = 0;
        while word < words {
            let mut x = 0u64;
            let mut z = 0u64;
            let mut k = 0;
            while k < indices.len() {
                let ind = indices[k] as usize;
                let row = 2 * perm[ind / 2] as usize + ind % 2;
                x ^= x_rows[row * words + word];
                z ^= z_rows[row * words + word];
                k += 1;
            }
            weight += (x | z).count_ones();
            word += 1;
        }
        plain += weight;
        weighted += weight as f64 * norms[term];
        term += lanes;
    }
    (plain, weighted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrmion_core::encode::ternarytree::TernaryTree;
    use num_complex::Complex64;
    use proptest::prelude::*;
    use tinyvec::ArrayVec;

    fn arb_perm(n: usize) -> impl Strategy<Value = Vec<usize>> {
        Just((0..n).collect::<Vec<usize>>()).prop_shuffle()
    }

    fn arb_majorana_sparse(n_modes: usize) -> impl Strategy<Value = MajoranaSparse> {
        let n_ops = (2 * n_modes) as u16;
        proptest::collection::vec(
            (
                proptest::sample::subsequence((0..n_ops).collect::<Vec<u16>>(), 4),
                any::<bool>(),
                -5i32..=5i32,
            )
                .prop_filter("non-zero coefficient", |(_, _, c)| *c != 0)
                .prop_map(|(inds, quartic, coeff)| {
                    let take = if quartic { 4 } else { 2 };
                    let inds: ArrayVec<[u16; MAX_MAJORANAS]> =
                        inds.into_iter().take(take).collect();
                    (inds, Complex64::new(coeff as f64, 0.))
                }),
            1..=40usize,
        )
        .prop_map(|terms| {
            let (indices, coeffs): (Vec<_>, Vec<_>) = terms.into_iter().unzip();
            MajoranaSparse::new(indices, coeffs, 0.25).unwrap()
        })
    }

    fn tree(builder: usize, n_modes: usize) -> TernaryTree {
        match builder {
            0 => TernaryTree::naive_jordan_wigner(n_modes),
            1 => TernaryTree::naive_parity(n_modes),
            2 => TernaryTree::naive_bravyi_kitaev(n_modes),
            _ => TernaryTree::naive_jkmn(n_modes),
        }
    }

    proptest! {
        /// The emulated kernel reproduces the CPU batch weights, including for
        /// qubit counts that span two 64-bit words.
        #[test]
        fn emulation_matches_cpu_batch(
            (n_modes, perms, ms) in prop_oneof![2usize..=8, 63usize..=66].prop_flat_map(|n| {
                (
                    Just(n),
                    proptest::collection::vec(arb_perm(n), 1..=4usize),
                    arb_majorana_sparse(n),
                )
            }),
            builder in 0usize..4,
        ) {
            let encoding = tree(builder, n_modes).build_encoding(n_modes).unwrap();
            let packed = PackedProblem::new(&encoding, &ms);
            let (plain, weighted) = packed.emulate(&perms);
            let (cpu_plain, cpu_weighted) = encoding.batch_pauli_weights(&ms, &perms);
            prop_assert_eq!(plain, cpu_plain);
            for (a, b) in weighted.iter().zip(&cpu_weighted) {
                prop_assert!((a - b).abs() < 1e-9, "weighted {} vs {}", a, b);
            }
        }
    }
}
