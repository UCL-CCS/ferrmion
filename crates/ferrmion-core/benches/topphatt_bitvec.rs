//! Compare the index-list ([`ArrayVecTermStore`]) and bit-packed
//! ([`BitTermStore`]) Majorana-term backends for TOPP-HATT.
//!
//! Both backends run the identical orchestration via `topphatt` /
//! `topphatt_impl`; only the per-term weight evaluation and Hamiltonian
//! reduction differ. The benchmark drives them over synthetic Majorana
//! Hamiltonians of increasing size on JKMN trees so the performance gap can be
//! read off as a function of mode count.
//!
//! Note: the two backends can pick different (but equally valid) encodings on
//! dense inputs because the bit backend deduplicates terms by parity-set rather
//! than by multiset. The work done is still representative of each approach.

use std::collections::BTreeSet;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ferrmion_core::encode::ternarytree::TernaryTree;
use ferrmion_core::operators::MajoranaSparse;
use ferrmion_core::optimise::{topphatt, topphatt_impl, BitTermStore, NodeOrderHeuristic};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use tinyvec::ArrayVec;

/// Generate a deterministic random Majorana Hamiltonian for `n_modes` modes:
/// `n_terms` distinct terms of length 2 or 4, each a sorted set of Majorana
/// indices drawn from `0..2*n_modes`.
fn random_terms(n_modes: usize, n_terms: usize, seed: u64) -> Vec<ArrayVec<[u16; 7]>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let n_majoranas = 2 * n_modes;
    let mut seen: BTreeSet<ArrayVec<[u16; 7]>> = BTreeSet::new();
    while seen.len() < n_terms {
        let len = if rng.random_bool(0.5) { 2 } else { 4 };
        let mut chosen: BTreeSet<u16> = BTreeSet::new();
        while chosen.len() < len {
            chosen.insert(rng.random_range(0..n_majoranas) as u16);
        }
        let term: ArrayVec<[u16; 7]> = chosen.into_iter().collect();
        seen.insert(term);
    }
    seen.into_iter().collect()
}

fn bench_topphatt_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("topphatt_term_store");

    // All sizes stay within the u64 bit backend's 31-mode ceiling.
    for n_modes in [6usize, 14, 20] {
        let n_terms = 12 * n_modes;
        let terms = random_terms(n_modes, n_terms, 0xC0FFEE);
        let coefficients: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); terms.len()];

        group.bench_with_input(
            BenchmarkId::new("index_list", n_modes),
            &n_modes,
            |b, &n| {
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
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bit_packed", n_modes),
            &n_modes,
            |b, &n| {
                b.iter_batched(
                    || {
                        let store = BitTermStore::from_arrayvecs(&terms).unwrap();
                        (store, TernaryTree::naive_jkmn(n))
                    },
                    |(store, tree)| {
                        topphatt_impl(store, tree, false, NodeOrderHeuristic::MinWeight).unwrap()
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_topphatt_backends);
criterion_main!(benches);
