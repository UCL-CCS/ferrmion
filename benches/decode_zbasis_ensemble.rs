use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrmion_core::encode::majorana::{MajoranaEncoding, TryEncode};
use ferrmion_core::encode::ternarytree::TernaryTree;
use ferrmion_core::states::{FockState, ZBasisEnsemble, ZBasisState};
use ndarray::Array1;
use num_complex::Complex64;

fn make_ensemble(encoding: &MajoranaEncoding, n_states: usize) -> ZBasisEnsemble {
    let n_modes = encoding.n_modes;
    let total_states = 1 << n_modes;
    let states: Vec<ZBasisState> = (0..n_states)
        .map(|i| {
            let bits = i % total_states;
            let occ: Vec<bool> = (0..n_modes).map(|j| (bits >> j) & 1 != 0).collect();
            let fock = FockState::new(Array1::from(occ), Complex64::ONE);
            encoding.try_encode(fock).unwrap()
        })
        .collect();
    ZBasisEnsemble::from(states)
}

fn bench_decode_ensemble(c: &mut Criterion) {
    let n_modes = 4;
    let tree = TernaryTree::naive_jordan_wigner(n_modes);
    let encoding = tree.build_encoding(n_modes).unwrap();

    let mut group = c.benchmark_group("decode_zbasis_ensemble");

    for n_states in [10usize, 100, 1000] {
        let ensemble = make_ensemble(&encoding, n_states);

        group.bench_with_input(
            BenchmarkId::new("batch_parallel", n_states),
            &n_states,
            |b, _| {
                b.iter(|| encoding.decode_zbasis_ensemble(black_box(&ensemble)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sequential", n_states),
            &n_states,
            |b, _| {
                b.iter(|| {
                    ensemble
                        .states
                        .axis_iter(ndarray::Axis(0))
                        .zip(ensemble.coefficients.iter())
                        .map(|(row, &coeff)| {
                            encoding.decode_zbasis_state(ZBasisState::new(row.to_owned(), coeff))
                        })
                        .collect::<Vec<_>>()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_decode_ensemble);
criterion_main!(benches);
