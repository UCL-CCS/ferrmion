//! Benchmarks the fermion-to-Majorana conversion that powers `MajoranaSparse`
//! construction. The hot path is the per-term expansion in `append_term_into`,
//! exercised here via the public `MajoranaSparse::from_signatures_and_coeffs`
//! entry point with dense coefficient tensors, and via `MajoranaSparse::from`
//! on a `FermionSparse` whose many independent rows trigger the parallel path of
//! `append_fermion_sparse`.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ferrmion_core::operators::{FermionSparse, LadderOperator, MajoranaSparse};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_complex::Complex64;

/// Build a dense coefficient tensor of rank `term_length` over `n_orbitals`
/// modes, filled with a constant so every (allowed) term contributes.
fn dense_coeffs(n_orbitals: usize, term_length: usize) -> ArrayD<f64> {
    let shape = vec![n_orbitals; term_length];
    ArrayD::from_elem(IxDyn(&shape), 1.0)
}

/// Build the index/coefficient arrays for a two-body (`"++--"`) `FermionSparse`
/// with `n_terms` independent rows over `n_orbitals` modes.
fn two_body_terms(n_terms: usize, n_orbitals: usize) -> (Array2<usize>, Array1<Complex64>) {
    let mut indices = Array2::<usize>::zeros((n_terms, 4));
    for t in 0..n_terms {
        indices[[t, 0]] = t % n_orbitals;
        indices[[t, 1]] = (t / n_orbitals) % n_orbitals;
        indices[[t, 2]] = (t / (n_orbitals * n_orbitals)) % n_orbitals;
        indices[[t, 3]] = (t / (n_orbitals * n_orbitals * n_orbitals)) % n_orbitals;
    }
    let coefficients = Array1::from_elem(n_terms, Complex64::new(1.0, 0.0));
    (indices, coefficients)
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

/// Benchmarks `MajoranaSparse::from(FermionSparse)` across term counts that
/// straddle the serial/parallel threshold, exercising `append_fermion_sparse`.
fn bench_majorana_sparse_from_fermion_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("majorana_sparse_from_fermion_sparse");
    let n_orbitals = 12;
    let action = vec![
        LadderOperator::Creation,
        LadderOperator::Creation,
        LadderOperator::Annihilation,
        LadderOperator::Annihilation,
    ];

    for n_terms in [64usize, 256, 4096, 50_000] {
        let (indices, coefficients) = two_body_terms(n_terms, n_orbitals);
        group.bench_with_input(
            BenchmarkId::new("two_body_++--", n_terms),
            &n_terms,
            |b, _| {
                // `FermionSparse` is consumed by `from`, so build a fresh one per
                // iteration (excluded from the timing via `iter_batched`).
                b.iter_batched(
                    || {
                        FermionSparse::new(action.clone(), indices.clone(), coefficients.clone())
                            .unwrap()
                    },
                    |fsparse| MajoranaSparse::from(black_box(fsparse)),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_majorana_sparse_construction,
    bench_majorana_sparse_from_fermion_sparse
);
criterion_main!(benches);
