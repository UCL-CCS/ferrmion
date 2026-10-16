# ferrmion-gpu (experimental)

Evaluates the Pauli weights of one fermionic Hamiltonian under many mode
permutations on an NVIDIA GPU, using NVIDIA's
[cuda-oxide](https://github.com/NVIDIA/cuda-rust) SIMT kernels written in Rust.
This is the inner loop of enumeration annealing. The result matches
`MajoranaEncoding::batch_pauli_weights`.

```rust
use ferrmion_gpu::GpuEncoder;

let gpu = GpuEncoder::new(&encoding, &msparse, 0)?; // uploads the problem once
let (plain, weighted) = gpu.batch_pauli_weights(&permutations)?;
```

The design, the other candidate kernels and the expected crossover points are
covered in [`docs/source/gpu_acceleration.md`](../../docs/source/gpu_acceleration.md).

## Status

cuda-oxide is alpha software and needs a pinned nightly toolchain, so this
crate sits outside the main workspace and is not part of the Python wheel.

- **Verified without a GPU:** the kernel arithmetic (`packed::lane_weights`),
  which `PackedProblem::emulate` runs on the CPU, matches
  `batch_pauli_weights` on all four standard encodings, including qubit
  counts above 64. The `cuda` feature also type-checks against the pinned
  cuda-rust revision.
- **Not yet run on hardware:** PTX generation and the GPU test and benchmark.

## Requirements

- An NVIDIA GPU with a CUDA 13 driver (R580+) and the CUDA 13 toolkit.
- `clang` and the libclang headers, for `bindgen`.
- `cargo-oxide`:

  ```bash
  cargo +nightly-2026-10-11 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
  ```

The toolchain is pinned in `rust-toolchain.toml`. The cuda-rust revision is
pinned in `Cargo.toml`; `cargo oxide` builds its codegen backend from that
same revision.

## Commands

Run these from this directory:

```bash
cargo oxide doctor                       # check toolkit, driver and backend
cargo oxide test                         # GPU vs CPU and emulation
cargo oxide bench                        # CPU vs GPU sweep
cargo +stable test --no-default-features # emulation tests on stable, no GPU
```
