//! Benchmarks encoding a fermionic Hamiltonian into a qubit (Pauli) Hamiltonian
//! via `MajoranaEncoding::encode(&MajoranaSparse)`. This is the symplectic-product
//! hot path: each Majorana term folds several operator rows together with
//! `SymplecticOperator::mul_assign_view` (XOR of the bitpacked X/Z blocks plus a
//! `z & x` popcount for the phase), so it directly exercises the bitpacked
//! symplectic representation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrmion_core::encode::majorana::{Encode, MajoranaEncoding};
use ferrmion_core::encode::ternarytree::TernaryTree;
use ferrmion_core::hamiltonians::QubitHamiltonian;
use ferrmion_core::operators::{CoefficientPauliWeight, MajoranaSparse, PauliWeight};
use ndarray::{ArrayD, IxDyn};

/// Build a dense coefficient tensor of rank `term_length` over `n_modes` modes.
///
/// Values vary with the flat index so the tensor is *not* permutation-symmetric:
/// a symmetric (e.g. all-ones) tensor antisymmetrises to zero for the two-body
/// `"++--"` signature, which would leave an empty Hamiltonian and measure nothing.
fn varied_coeffs(n_modes: usize, term_length: usize) -> ArrayD<f64> {
    let mut tensor = ArrayD::from_elem(IxDyn(&vec![n_modes; term_length]), 0.0);
    for (i, v) in tensor.iter_mut().enumerate() {
        *v = ((i % 7) as f64) + 1.0;
    }
    tensor
}

fn bench_encode_symplectic(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_symplectic");

    // One-body term "+-" (rank-2 tensor).
    for n_modes in [8usize, 12, 16] {
        let encoding: MajoranaEncoding = TernaryTree::naive_jordan_wigner(n_modes)
            .build_encoding(n_modes)
            .unwrap();
        let coeffs = varied_coeffs(n_modes, 2);
        let msparse = MajoranaSparse::from_signatures_and_coeffs(
            vec!["+-".to_string()],
            vec![coeffs.view()],
            0.0,
        );
        group.bench_with_input(
            BenchmarkId::new("one_body_+-", n_modes),
            &n_modes,
            |b, _| {
                b.iter(|| {
                    let qham: QubitHamiltonian = encoding.encode(black_box(&msparse));
                    black_box(qham)
                });
            },
        );
    }

    // Two-body term "++--" (rank-4 tensor) — the dominant cost in real
    // Hamiltonians and the case with the most symplectic products.
    for n_modes in [8usize, 12, 16] {
        let encoding: MajoranaEncoding = TernaryTree::naive_jordan_wigner(n_modes)
            .build_encoding(n_modes)
            .unwrap();
        let coeffs = varied_coeffs(n_modes, 4);
        let msparse = MajoranaSparse::from_signatures_and_coeffs(
            vec!["++--".to_string()],
            vec![coeffs.view()],
            0.0,
        );
        group.bench_with_input(
            BenchmarkId::new("two_body_++--", n_modes),
            &n_modes,
            |b, _| {
                b.iter(|| {
                    let qham: QubitHamiltonian = encoding.encode(black_box(&msparse));
                    black_box(qham)
                });
            },
        );
    }

    group.finish();
}

/// Pauli weights of a two-body Hamiltonian over a batch of mode permutations, as
/// in enumeration annealing. Compares materialising each reordered encoding and
/// its string-keyed `QubitHamiltonian` against `batch_pauli_weights`, which reads
/// permuted rows in place and merges terms by their packed X/Z words.
fn bench_batch_pauli_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_pauli_weights");
    group.sample_size(10);
    let n_perms = 64;

    for n_modes in [12usize, 16] {
        let encoding: MajoranaEncoding = TernaryTree::naive_jordan_wigner(n_modes)
            .build_encoding(n_modes)
            .unwrap();
        let coeffs = varied_coeffs(n_modes, 4);
        let msparse = MajoranaSparse::from_signatures_and_coeffs(
            vec!["++--".to_string()],
            vec![coeffs.view()],
            0.0,
        );
        let perms: Vec<Vec<usize>> = (0..n_perms)
            .map(|p| {
                (0..n_modes)
                    .map(|m| (m * (2 * p + 1) + p) % n_modes)
                    .collect()
            })
            .filter(|perm: &Vec<usize>| {
                let mut sorted = perm.clone();
                sorted.sort_unstable();
                sorted.dedup();
                sorted.len() == n_modes
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("apply_and_encode", n_modes),
            &n_modes,
            |b, _| {
                b.iter(|| {
                    let weights: Vec<(f64, f64)> = perms
                        .iter()
                        .map(|perm| {
                            let qham = encoding
                                .apply_mode_enumeration(perm.clone())
                                .encode(black_box(&msparse));
                            (qham.pauli_weight() as f64, qham.coeff_pauli_weight())
                        })
                        .collect();
                    black_box(weights)
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("batch", n_modes), &n_modes, |b, _| {
            b.iter(|| black_box(encoding.batch_pauli_weights(black_box(&msparse), &perms)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_encode_symplectic, bench_batch_pauli_weights);
criterion_main!(benches);
