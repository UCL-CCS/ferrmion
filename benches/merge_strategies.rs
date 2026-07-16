//! Micro-benchmarks for the merge strategies behind `append_fermion_sparse`.
//!
//! The strategy under test is selected by the `FERRMION_MERGE_STRATEGY`
//! environment variable (see `crates/ferrmion-core/src/operators/merge.rs`),
//! which must be set **before** this process starts — the configuration is
//! read once and cached. `scripts/bench_merge_sweep.sh` drives this bench
//! across every (strategy, `RAYON_NUM_THREADS`) combination, saving criterion
//! baselines that `scripts/bench_merge_report.py` turns into runtime, speedup
//! and parallel-efficiency tables.
//!
//! Two workload regimes stress the merge differently:
//!
//! * **low collision** — every row is a distinct index tuple, so the merged
//!   map is large and the merge cost is dominated by inserts and table growth;
//! * **high collision** — a small set of tuples is repeated many times, so the
//!   merged map is small and the merge cost is dominated by summing duplicate
//!   keys found in every per-chunk partial.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ferrmion_core::operators::{FermionSparse, LadderOperator, MajoranaSparse};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::hint::black_box;

/// Build the index/coefficient arrays for a two-body (`"++--"`) `FermionSparse`.
///
/// Row `t` encodes the base-`n_orbitals` digits of `t`, so rows are distinct
/// while `n_terms <= n_orbitals^4`. Passing `cycle = Some(c)` reuses only the
/// first `c` tuples (`t % c`), producing heavy key collisions across chunks.
fn two_body_terms(
    n_terms: usize,
    n_orbitals: usize,
    cycle: Option<usize>,
) -> (Array2<usize>, Array1<Complex64>) {
    let mut indices = Array2::<usize>::zeros((n_terms, 4));
    for t in 0..n_terms {
        let u = cycle.map_or(t, |c| t % c);
        indices[[t, 0]] = u % n_orbitals;
        indices[[t, 1]] = (u / n_orbitals) % n_orbitals;
        indices[[t, 2]] = (u / (n_orbitals * n_orbitals)) % n_orbitals;
        indices[[t, 3]] = (u / (n_orbitals * n_orbitals * n_orbitals)) % n_orbitals;
    }
    // Small dyadic coefficients: exact in f64, so every strategy produces
    // bit-identical results and the workload is comparison-safe.
    let coefficients =
        Array1::from_shape_fn(n_terms, |t| Complex64::new((t % 4 + 1) as f64 * 0.25, 0.0));
    (indices, coefficients)
}

fn bench_regime(c: &mut Criterion, regime: &str, cycle: Option<usize>) {
    let mut group = c.benchmark_group(format!("merge_strategies/{regime}"));
    // Large inputs take ~seconds per iteration; keep sampling economical so a
    // full strategy x thread sweep stays tractable.
    group.sample_size(10);

    let n_orbitals = 30; // 30^4 = 810_000 distinct tuples >= the largest size below
    let action = vec![
        LadderOperator::Creation,
        LadderOperator::Creation,
        LadderOperator::Annihilation,
        LadderOperator::Annihilation,
    ];

    for n_terms in [4_096usize, 50_000, 500_000] {
        let (indices, coefficients) = two_body_terms(n_terms, n_orbitals, cycle);
        group.bench_with_input(
            BenchmarkId::new("two_body_++--", n_terms),
            &n_terms,
            |b, _| {
                // `FermionSparse` is consumed by `from`, so build a fresh one
                // per iteration (excluded from the timing via `iter_batched`).
                b.iter_batched(
                    || {
                        FermionSparse::new(action.clone(), indices.clone(), coefficients.clone())
                            .unwrap()
                    },
                    |fsparse| MajoranaSparse::from(black_box(fsparse)),
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Mostly-unique keys: merge cost is inserts and hash-table growth.
fn bench_low_collision(c: &mut Criterion) {
    bench_regime(c, "low_collision", None);
}

/// A small key set repeated across all chunks: merge cost is duplicate summing.
fn bench_high_collision(c: &mut Criterion) {
    bench_regime(c, "high_collision", Some(1_024));
}

criterion_group!(benches, bench_low_collision, bench_high_collision);
criterion_main!(benches);
