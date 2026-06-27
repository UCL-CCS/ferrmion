//! Benchmarks the fermion-to-Majorana conversion that powers `MajoranaSparse`
//! construction. The hot path is `MajoranaHashMap::append_term`, exercised here
//! via the public `MajoranaSparse::from_signatures_and_coeffs` entry point with
//! dense one-body (`"+-"`) and two-body (`"++--"`) coefficient tensors.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrmion_core::operators::MajoranaSparse;
use ndarray::{ArrayD, IxDyn};

/// Build a dense coefficient tensor of rank `term_length` over `n_orbitals`
/// modes, filled with a constant so every (allowed) term contributes.
fn dense_coeffs(n_orbitals: usize, term_length: usize) -> ArrayD<f64> {
    let shape = vec![n_orbitals; term_length];
    ArrayD::from_elem(IxDyn(&shape), 1.0)
}

fn bench_majorana_sparse_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("majorana_sparse_construction");

    // One-body term "+-": rank-2 tensor.
    for n_orbitals in [4usize, 8, 16] {
        let coeffs = dense_coeffs(n_orbitals, 2);
        group.bench_with_input(
            BenchmarkId::new("one_body_+-", n_orbitals),
            &n_orbitals,
            |b, _| {
                b.iter(|| {
                    MajoranaSparse::from_signatures_and_coeffs(
                        black_box(vec!["+-".to_string()]),
                        black_box(vec![coeffs.view()]),
                        0.0,
                    )
                });
            },
        );
    }

    // Two-body term "++--": rank-4 tensor, the dominant cost in real Hamiltonians.
    for n_orbitals in [4usize, 6, 8] {
        let coeffs = dense_coeffs(n_orbitals, 4);
        group.bench_with_input(
            BenchmarkId::new("two_body_++--", n_orbitals),
            &n_orbitals,
            |b, _| {
                b.iter(|| {
                    MajoranaSparse::from_signatures_and_coeffs(
                        black_box(vec!["++--".to_string()]),
                        black_box(vec![coeffs.view()]),
                        0.0,
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_majorana_sparse_construction);
criterion_main!(benches);
