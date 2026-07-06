//! Benchmarks the Clifford-heuristic optimiser's hot path: cloning a
//! [`SymplecticHamiltonian`] (done once per cost-function evaluation inside
//! the simulated-annealing loop) and applying a Clifford chain to it.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrmion_core::encode::majorana::Encode;
use ferrmion_core::encode::ternarytree::TernaryTree;
use ferrmion_core::hamiltonians::SymplecticHamiltonian;
use ferrmion_core::operators::{Clifford, MajoranaSparse};
use ndarray::ArrayD;
use ndarray::IxDyn;

fn varied_coeffs(n_modes: usize, term_length: usize) -> ArrayD<f64> {
    let mut tensor = ArrayD::from_elem(IxDyn(&vec![n_modes; term_length]), 0.0);
    for (i, v) in tensor.iter_mut().enumerate() {
        *v = ((i % 7) as f64) + 1.0;
    }
    tensor
}

fn build_hamiltonian(n_modes: usize) -> SymplecticHamiltonian {
    let encoding = TernaryTree::naive_jordan_wigner(n_modes)
        .build_encoding(n_modes)
        .unwrap();
    let coeffs = varied_coeffs(n_modes, 4);
    let msparse = MajoranaSparse::from_signatures_and_coeffs(
        vec!["++--".to_string()],
        vec![coeffs.view()],
        0.0,
    );
    let qham = encoding.encode(&msparse);
    SymplecticHamiltonian::from_qubit_hamiltonian(&qham, n_modes)
}

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("symplectic_hamiltonian_clone");
    for n_modes in [8usize, 16, 32] {
        let ham = build_hamiltonian(n_modes);
        group.bench_with_input(BenchmarkId::new("clone", n_modes), &n_modes, |b, _| {
            b.iter(|| black_box(ham.clone()));
        });
    }
    group.finish();
}

fn bench_clone_and_apply_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("symplectic_hamiltonian_clone_and_apply_chain");
    for n_modes in [8usize, 16, 32] {
        let ham = build_hamiltonian(n_modes);
        // A representative short Clifford chain, as sampled by CliffordHeuristic::anneal.
        let chain = vec![
            Clifford::H(0),
            Clifford::CNOT {
                control: 0,
                target: 1,
            },
            Clifford::S(2),
            Clifford::CNOT {
                control: 2,
                target: 3,
            },
        ];
        group.bench_with_input(
            BenchmarkId::new("clone_and_apply_chain", n_modes),
            &n_modes,
            |b, _| {
                b.iter(|| {
                    let mut h = ham.clone();
                    h.operators.apply_clifford_chain(black_box(&chain));
                    black_box(h)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_clone, bench_clone_and_apply_chain);
criterion_main!(benches);
