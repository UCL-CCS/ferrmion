//! Experimental NVIDIA GPU acceleration for ferrmion, built with
//! [cuda-oxide](https://github.com/NVIDIA/cuda-rust).
//!
//! [`GpuEncoder`] evaluates the Pauli weights of one Hamiltonian under many
//! mode permutations in a single kernel launch, the inner loop of enumeration
//! annealing. See `docs/gpu_acceleration.md` in the repository for the design
//! and the other candidate kernels.
//!
//! The `cuda` feature (on by default) needs `cargo oxide`. Without it only
//! [`packed`] is built, which holds the kernel arithmetic, its CPU emulation
//! and the host-side reduction.

pub mod packed;

#[cfg(feature = "cuda")]
mod device;

#[cfg(feature = "cuda")]
pub use device::{GpuEncoder, GpuResult};
pub use packed::PackedProblem;
