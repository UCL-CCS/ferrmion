//! Benchmarks for the simulated-annealing cost evaluation.
//!
//! Compares the legacy string-keyed `encode().pauli_weight()` /
//! `encode().coeff_pauli_weight()` path against the bit-packed
//! `encode_pauli_weight` / `encode_coeff_pauli_weight` fast path used by
//! `OptimalEnumeration::cost`.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrmion_core::encode::encoding::{Encode, MajoranaEncoding};
use ferrmion_core::encode::ternarytree::TernaryTree;
use ferrmion_core::operators::{
    CoefficientPauliWeight, MajoranaSparse, PauliWeight,
};
use num_complex::Complex64;
use tinyvec::ArrayVec;

fn make_msparse(n_modes: usize, n_terms: usize) -> MajoranaSparse {
    let n_majoranas = (2 * n_modes) as u64;
    // Deterministic LCG so benchmark inputs are reproducible.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    let mut indices: Vec<ArrayVec<[u16; 7]>> = Vec::with_capacity(n_terms);
    let mut coefficients: Vec<Complex64> = Vec::with_capacity(n_terms);
    for _ in 0..n_terms {
        let len = (next() % 4 + 1) as usize; // 1..=4 Majorana indices per term
        let mut av: ArrayVec<[u16; 7]> = ArrayVec::new();
        for _ in 0..len {
            av.push((next() % n_majoranas) as u16);
        }
        indices.push(av);
        let re = ((next() % 200) as f64 - 100.0) / 50.0;
        let im = ((next() % 200) as f64 - 100.0) / 50.0;
        coefficients.push(Complex64::new(re, im));
    }
    MajoranaSparse::new(indices, coefficients, 0.0).unwrap()
}

fn build_encoding(n_modes: usize) -> MajoranaEncoding {
    TernaryTree::naive_jordan_wigner(n_modes)
        .build_encoding(n_modes)
        .unwrap()
}

fn bench_anneal_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("anneal_cost");

    for &(n_modes, n_terms) in &[(8usize, 200usize), (16, 800), (24, 1500)] {
        let encoding = build_encoding(n_modes);
        let msparse = make_msparse(n_modes, n_terms);
        let perm: Vec<usize> = (0..n_modes).collect();
        let label = format!("{}modes_{}terms", n_modes, n_terms);

        group.bench_with_input(
            BenchmarkId::new("string_pauli_weight", &label),
            &(),
            |b, _| {
                b.iter(|| {
                    let perm_arr = ndarray::Array1::from(perm.clone());
                    let permuted = encoding.apply_mode_enumeration(perm_arr.to_vec());
                    let qham = permuted.encode(black_box(&msparse));
                    qham.pauli_weight()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bitpacked_pauli_weight", &label),
            &(),
            |b, _| {
                b.iter(|| {
                    encoding.encode_pauli_weight_permuted(black_box(&msparse), &perm)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("string_coeff_weight", &label),
            &(),
            |b, _| {
                b.iter(|| {
                    let perm_arr = ndarray::Array1::from(perm.clone());
                    let permuted = encoding.apply_mode_enumeration(perm_arr.to_vec());
                    let qham = permuted.encode(black_box(&msparse));
                    qham.coeff_pauli_weight()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bitpacked_coeff_weight", &label),
            &(),
            |b, _| {
                b.iter(|| {
                    encoding.encode_coeff_pauli_weight_permuted(black_box(&msparse), &perm)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_anneal_cost);
criterion_main!(benches);
