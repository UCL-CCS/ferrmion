//! GPU against CPU on real hardware; built only with the `cuda` feature, via `cargo oxide test`.
#![cfg(feature = "cuda")]

use ferrmion_core::encode::ternarytree::TernaryTree;
use ferrmion_core::operators::MajoranaSparse;
use ferrmion_gpu::{GpuEncoder, PackedProblem};
use num_complex::Complex64;
use tinyvec::array_vec;

#[test]
fn gpu_matches_cpu_and_emulation() {
    for (n_modes, builder) in [(4usize, 0usize), (9, 1), (33, 2), (65, 3)] {
        let tree = match builder {
            0 => TernaryTree::naive_jordan_wigner(n_modes),
            1 => TernaryTree::naive_parity(n_modes),
            2 => TernaryTree::naive_bravyi_kitaev(n_modes),
            _ => TernaryTree::naive_jkmn(n_modes),
        };
        let encoding = tree.build_encoding(n_modes).unwrap();
        let top = (2 * n_modes - 1) as u16;
        let ms = MajoranaSparse::new(
            vec![
                array_vec!([u16; 7] => 0, 1),
                array_vec!([u16; 7] => 0, 3),
                array_vec!([u16; 7] => 0, 3),
                array_vec!([u16; 7] => 1, 2, 3, top),
                array_vec!([u16; 7] => 2, top),
            ],
            vec![
                Complex64::new(1., 0.),
                Complex64::new(2., 0.),
                Complex64::new(-2., 0.),
                Complex64::new(0., 0.5),
                Complex64::new(-3., 1.),
            ],
            0.7,
        )
        .unwrap();
        let perms: Vec<Vec<usize>> = (0..n_modes)
            .map(|shift| (0..n_modes).map(|m| (m + shift) % n_modes).collect())
            .collect();

        let gpu = GpuEncoder::new(&encoding, &ms, 0)
            .unwrap()
            .batch_pauli_weights(&perms)
            .unwrap();
        let cpu = encoding.batch_pauli_weights(&ms, &perms);
        let emulated = PackedProblem::new(&encoding, &ms).emulate(&perms);
        assert_eq!(gpu.0, cpu.0, "plain weights, {n_modes} modes");
        assert_eq!(gpu.0, emulated.0);
        for (g, c) in gpu.1.iter().zip(&cpu.1) {
            assert!((g - c).abs() < 1e-9, "weighted {g} vs {c}");
        }
    }
}
