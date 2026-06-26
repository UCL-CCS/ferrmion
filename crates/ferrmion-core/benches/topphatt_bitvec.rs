//! Compare the index-list ([`ArrayVecTermStore`]) and bit-packed
//! ([`BitTermStore`]) Majorana-term backends for TOPP-HATT.
//!
//! Both backends run the identical orchestration via `topphatt` /
//! `topphatt_impl`; only the per-term weight evaluation and Hamiltonian
//! reduction differ. The benchmark sweeps synthetic Majorana Hamiltonians on
//! JKMN trees over two axes so the performance gap can be read as a function of
//! both:
//! - **number of modes** `m` (problem size), and
//! - **Majorana term degree** `d` (operators per term) — higher `d` does more
//!   per-term work, which is exactly what the bit backends turn into O(1) ops.
//!
//! Four backends are compared where the word width allows:
//! - `index_list`: `Vec<ArrayVec<[u16; 7]>>` (no mode ceiling).
//! - `bit_u64`: one `u64` per term (≤ 31 modes).
//! - `bit_u128`: one `u128` per term (≤ 63 modes).
//! - `bit_u256`: one `bnum` `U256` per term (≤ 127 modes).
//!
//! Note: the bit backends can pick different (but equally valid) encodings on
//! dense inputs because they deduplicate terms by parity-set rather than by
//! multiset, which also trims term counts during reduction. Part of the bit
//! speedup — especially at higher degree — comes from that trimming, not only
//! cheaper per-term ops. The work done is still representative of each approach.

use std::collections::BTreeSet;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ferrmion_core::encode::ternarytree::TernaryTree;
use ferrmion_core::operators::MajoranaSparse;
use ferrmion_core::optimise::{
    topphatt, topphatt_impl, ArrayVecTermStore, BitSlicedTermStore, BitTermStore128,
    BitTermStore256, BitTermStore64, NodeOrderHeuristic,
};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use tinyvec::ArrayVec;

/// Generate a deterministic random Majorana Hamiltonian for `n_modes` modes:
/// `n_terms` distinct terms, each a sorted set of exactly `degree` Majorana
/// indices drawn from `0..2*n_modes`.
fn random_terms(
    n_modes: usize,
    n_terms: usize,
    degree: usize,
    seed: u64,
) -> Vec<ArrayVec<[u16; 7]>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let n_majoranas = 2 * n_modes;
    let mut seen: BTreeSet<ArrayVec<[u16; 7]>> = BTreeSet::new();
    while seen.len() < n_terms {
        let mut chosen: BTreeSet<u16> = BTreeSet::new();
        while chosen.len() < degree {
            chosen.insert(rng.random_range(0..n_majoranas) as u16);
        }
        let term: ArrayVec<[u16; 7]> = chosen.into_iter().collect();
        seen.insert(term);
    }
    seen.into_iter().collect()
}

fn bench_topphatt_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("topphatt_term_store");

    // Term degree (Majorana operators per term); capped by the ArrayVec width.
    for degree in [2usize, 4, 6] {
        for n_modes in [16usize, 31, 63, 80] {
            let n_terms = 12 * n_modes;
            let terms = random_terms(n_modes, n_terms, degree, 0xC0FFEE);
            let coefficients: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); terms.len()];
            let param = format!("m{n_modes}_d{degree}");

            group.bench_with_input(BenchmarkId::new("index_list", &param), &n_modes, |b, &n| {
                b.iter_batched(
                    || {
                        let ham =
                            MajoranaSparse::new(terms.clone(), coefficients.clone(), 0.0).unwrap();
                        (ham, TernaryTree::naive_jkmn(n))
                    },
                    |(ham, tree)| {
                        topphatt(ham, tree, false, NodeOrderHeuristic::MinWeight).unwrap()
                    },
                    BatchSize::SmallInput,
                );
            });

            if n_modes <= BitTermStore64::MAX_MODES {
                group.bench_with_input(BenchmarkId::new("bit_u64", &param), &n_modes, |b, &n| {
                    b.iter_batched(
                        || {
                            let store = BitTermStore64::from_arrayvecs(&terms).unwrap();
                            (store, TernaryTree::naive_jkmn(n))
                        },
                        |(store, tree)| {
                            topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight)
                                .unwrap()
                        },
                        BatchSize::SmallInput,
                    );
                });
            }

            if n_modes <= BitTermStore128::MAX_MODES {
                group.bench_with_input(BenchmarkId::new("bit_u128", &param), &n_modes, |b, &n| {
                    b.iter_batched(
                        || {
                            let store = BitTermStore128::from_arrayvecs(&terms).unwrap();
                            (store, TernaryTree::naive_jkmn(n))
                        },
                        |(store, tree)| {
                            topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight)
                                .unwrap()
                        },
                        BatchSize::SmallInput,
                    );
                });
            }

            if n_modes <= BitTermStore256::MAX_MODES {
                group.bench_with_input(BenchmarkId::new("bit_u256", &param), &n_modes, |b, &n| {
                    b.iter_batched(
                        || {
                            let store = BitTermStore256::from_arrayvecs(&terms).unwrap();
                            (store, TernaryTree::naive_jkmn(n))
                        },
                        |(store, tree)| {
                            topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight)
                                .unwrap()
                        },
                        BatchSize::SmallInput,
                    );
                });
            }

            // The transposed store has no mode ceiling and always runs.
            group.bench_with_input(BenchmarkId::new("bit_sliced", &param), &n_modes, |b, &n| {
                b.iter_batched(
                    || {
                        let store = BitSlicedTermStore::from_arrayvecs(&terms, n);
                        (store, TernaryTree::naive_jkmn(n))
                    },
                    |(store, tree)| {
                        topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight).unwrap()
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

/// Large-mode points beyond every fixed word's ceiling: only the index-list and
/// transposed backends can run here. Kept to a single degree to bound runtime.
fn bench_topphatt_large_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("topphatt_large_modes");
    group.sample_size(10);

    let degree = 4;
    for n_modes in [96usize, 112] {
        let n_terms = 12 * n_modes;
        let terms = random_terms(n_modes, n_terms, degree, 0xC0FFEE);
        let coefficients: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); terms.len()];
        let param = format!("m{n_modes}_d{degree}");

        group.bench_with_input(BenchmarkId::new("index_list", &param), &n_modes, |b, &n| {
            b.iter_batched(
                || {
                    let ham =
                        MajoranaSparse::new(terms.clone(), coefficients.clone(), 0.0).unwrap();
                    (ham, TernaryTree::naive_jkmn(n))
                },
                |(ham, tree)| topphatt(ham, tree, false, NodeOrderHeuristic::MinWeight).unwrap(),
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("bit_sliced", &param), &n_modes, |b, &n| {
            b.iter_batched(
                || {
                    let store = BitSlicedTermStore::from_arrayvecs(&terms, n);
                    (store, TernaryTree::naive_jkmn(n))
                },
                |(store, tree)| {
                    topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight).unwrap()
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Conversion-only cost: time *just* building each backend's store from the
/// canonical ArrayVec terms. This is the per-backend Hamiltonian-preparation the
/// optimization-only groups exclude (it runs in their untimed `iter_batched`
/// setup). Expected to be tiny — `O(T·degree)` bit-sets — next to the optimizer.
fn bench_term_store_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("term_store_build");

    for degree in [2usize, 6] {
        for n_modes in [31usize, 63] {
            let n_terms = 12 * n_modes;
            let terms = random_terms(n_modes, n_terms, degree, 0xC0FFEE);
            let param = format!("m{n_modes}_d{degree}");

            // index_list is already the canonical form; "conversion" is a clone.
            group.bench_with_input(BenchmarkId::new("index_list", &param), &n_modes, |b, _| {
                b.iter(|| black_box(ArrayVecTermStore::new(black_box(&terms).clone())));
            });

            if n_modes <= BitTermStore64::MAX_MODES {
                group.bench_with_input(BenchmarkId::new("bit_u64", &param), &n_modes, |b, _| {
                    b.iter(|| {
                        black_box(BitTermStore64::from_arrayvecs(black_box(&terms)).unwrap())
                    });
                });
            }

            group.bench_with_input(BenchmarkId::new("bit_sliced", &param), &n_modes, |b, &n| {
                b.iter(|| black_box(BitSlicedTermStore::from_arrayvecs(black_box(&terms), n)));
            });
        }
    }

    group.finish();
}

/// End-to-end cost: same optimization as `topphatt_term_store`, but with the
/// store construction (format conversion) folded **into the timed region**. Only
/// the backend-independent tree build stays in the untimed setup, so the delta
/// from the optimization-only group is exactly the conversion overhead.
fn bench_topphatt_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("topphatt_end_to_end");

    for degree in [2usize, 6] {
        for n_modes in [31usize, 63] {
            let n_terms = 12 * n_modes;
            let terms = random_terms(n_modes, n_terms, degree, 0xC0FFEE);
            let param = format!("m{n_modes}_d{degree}");

            group.bench_with_input(BenchmarkId::new("index_list", &param), &n_modes, |b, &n| {
                b.iter_batched(
                    || TernaryTree::naive_jkmn(n),
                    |tree| {
                        let store = ArrayVecTermStore::new(terms.clone());
                        topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight).unwrap()
                    },
                    BatchSize::SmallInput,
                );
            });

            if n_modes <= BitTermStore64::MAX_MODES {
                group.bench_with_input(BenchmarkId::new("bit_u64", &param), &n_modes, |b, &n| {
                    b.iter_batched(
                        || TernaryTree::naive_jkmn(n),
                        |tree| {
                            let store = BitTermStore64::from_arrayvecs(&terms).unwrap();
                            topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight)
                                .unwrap()
                        },
                        BatchSize::SmallInput,
                    );
                });
            }

            if n_modes <= BitTermStore128::MAX_MODES {
                group.bench_with_input(BenchmarkId::new("bit_u128", &param), &n_modes, |b, &n| {
                    b.iter_batched(
                        || TernaryTree::naive_jkmn(n),
                        |tree| {
                            let store = BitTermStore128::from_arrayvecs(&terms).unwrap();
                            topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight)
                                .unwrap()
                        },
                        BatchSize::SmallInput,
                    );
                });
            }

            group.bench_with_input(BenchmarkId::new("bit_sliced", &param), &n_modes, |b, &n| {
                b.iter_batched(
                    || TernaryTree::naive_jkmn(n),
                    |tree| {
                        let store = BitSlicedTermStore::from_arrayvecs(&terms, n);
                        topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight).unwrap()
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_topphatt_backends,
    bench_topphatt_large_modes,
    bench_term_store_build,
    bench_topphatt_end_to_end
);
criterion_main!(benches);
