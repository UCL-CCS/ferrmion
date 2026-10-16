//! Pauli weights of one Hamiltonian under a batch of mode permutations: the
//! CPU rayon path (`MajoranaEncoding::batch_pauli_weights`) against the
//! cuda-oxide kernel (`GpuEncoder`, with the `cuda` feature).
//!
//! Run on a GPU machine with `cargo oxide bench`, or the CPU baseline alone
//! with `cargo +stable bench --no-default-features`.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ferrmion_core::encode::majorana::MajoranaEncoding;
use ferrmion_core::encode::ternarytree::TernaryTree;
use ferrmion_core::operators::MajoranaSparse;
use num_complex::Complex64;
use tinyvec::ArrayVec;

/// Small deterministic generator so the bench needs no RNG dependency.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// `n_terms` random four-index terms over `n_modes` modes.
fn random_hamiltonian(n_modes: usize, n_terms: usize, rng: &mut SplitMix) -> MajoranaSparse {
    let n_ops = 2 * n_modes as u64;
    let mut indices = Vec::with_capacity(n_terms);
    let mut coeffs = Vec::with_capacity(n_terms);
    while indices.len() < n_terms {
        let mut term: Vec<u16> = (0..4).map(|_| (rng.next() % n_ops) as u16).collect();
        term.sort_unstable();
        term.dedup();
        if term.len() != 4 {
            continue;
        }
        indices.push(term.into_iter().collect::<ArrayVec<[u16; 7]>>());
        coeffs.push(Complex64::new((rng.next() % 1000) as f64 / 100. + 0.01, 0.));
    }
    MajoranaSparse::new(indices, coeffs, 0.).unwrap()
}

fn random_permutations(n_modes: usize, n_perms: usize, rng: &mut SplitMix) -> Vec<Vec<usize>> {
    (0..n_perms)
        .map(|_| {
            let mut perm: Vec<usize> = (0..n_modes).collect();
            for i in (1..n_modes).rev() {
                perm.swap(i, (rng.next() % (i as u64 + 1)) as usize);
            }
            perm
        })
        .collect()
}

fn bench_batch_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_batch_weights");
    group.sample_size(10);
    let mut rng = SplitMix(7);

    for (n_modes, n_terms, n_perms) in [
        (20usize, 1_000usize, 64usize),
        (20, 1_000, 1_024),
        (50, 10_000, 256),
        (100, 10_000, 256),
        (100, 100_000, 64),
    ] {
        let encoding: MajoranaEncoding = TernaryTree::naive_jordan_wigner(n_modes)
            .build_encoding(n_modes)
            .unwrap();
        let msparse = random_hamiltonian(n_modes, n_terms, &mut rng);
        let perms = random_permutations(n_modes, n_perms, &mut rng);
        let id = format!("{n_modes}m_{n_terms}t_{n_perms}p");

        group.bench_with_input(BenchmarkId::new("cpu", &id), &id, |b, _| {
            b.iter(|| black_box(encoding.batch_pauli_weights(&msparse, &perms)))
        });

        #[cfg(feature = "cuda")]
        {
            let gpu = ferrmion_gpu::GpuEncoder::new(&encoding, &msparse, 0).unwrap();
            assert_eq!(
                gpu.batch_pauli_weights(&perms).unwrap().0,
                encoding.batch_pauli_weights(&msparse, &perms).0
            );
            group.bench_with_input(BenchmarkId::new("gpu", &id), &id, |b, _| {
                b.iter(|| black_box(gpu.batch_pauli_weights(&perms).unwrap()))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_batch_weights);
criterion_main!(benches);
