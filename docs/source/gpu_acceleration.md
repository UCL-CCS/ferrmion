# GPU acceleration

This page covers which parts of ferrmion's encoding and optimisation code
benefit from an NVIDIA GPU, how the experimental `crates/ferrmion-gpu`
prototype is built, and what to do next. It is aimed at contributors.

## Summary

- **Framework:** NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust)
  (SIMT kernels in plain Rust, released September 2026). It fits ferrmion's
  irregular, per-term bit manipulation better than the tile-based cutile-rs.
- **Algorithmic finding:** the Pauli weight of an encoded Hamiltonian under
  a mode enumeration can be computed without merging Pauli strings per
  enumeration (see "Merge once, weigh many" below). Applying this on the CPU
  alone makes `batch_pauli_weights` and enumeration annealing 25–30x faster,
  in the default build, with no GPU.
- **Prototype:** `GpuEncoder::batch_pauli_weights` evaluates a batch of
  permutations in one kernel launch and returns the same result as the CPU.
  The kernel arithmetic is shared with a CPU emulator and tested against the
  CPU path. It has not yet been run on GPU hardware.
- **Biggest remaining blocker:** the simulated annealer evaluates one
  candidate at a time. A GPU only pays off once optimisers propose candidates
  in batches (see "Batched drivers" below).

## Where the time goes

| Feature | Hot path | Data | GPU fit |
|---|---|---|---|
| Enumeration annealing, `batch_pauli_weights` | `MajoranaEncoding::enumerated_pauli_weights` | XOR of ≤7 encoding rows per term, then `popcount(x \| z)` | **High.** P × T independent terms, small read-only rows |
| Clifford heuristic | `CliffordHeuristic::cost` → `apply_clifford_chain` | Gate sequence on the bit-sliced (qubit-major) transpose, then popcount | **High once batched.** Each term word is independent |
| TOPP-HATT | `Backend::evaluate_combination` | For each candidate triple, AND/XOR of three transposed term columns, then popcount | **Medium.** Many combinations, but branch-and-bound pruning is lost |
| HATT | `hatt` | Sequential greedy, small steps | **Low** |
| Decoding | `decode_zbasis_ensemble` | Independent per state | Medium, but outside encoding and optimisation |

All of these reduce to word-wide XOR/AND/OR and popcount on packed `u64`s,
which GPUs do at very high throughput. The encoding rows (`2n` rows of
`⌈n_qubits/64⌉` words) are a few kilobytes, so they stay in L1/L2 while
every thread reads them.

## Merge once, weigh many

`Encode<&MajoranaSparse>` merges terms that map to the same Pauli string and
drops merged coefficients below `COEFFICIENT_TOLERANCE`. The weights depend
on that merge, and it originally cost one `String`-keyed hash insert per term
per enumeration. Each annealing step also rebuilt and re-validated the whole
reordered encoding (pairwise anticommutation and Gaussian elimination).

The rows of a valid `MajoranaEncoding` are linearly independent and pairwise
anticommuting. Linear independence means two Majorana products give the same
Pauli string exactly when their index sets agree up to repeated pairs.
Anticommutation means their relative sign follows from the Majorana algebra
alone. A mode enumeration only relabels which rows are used, so **which terms
merge, and the norm of each merged coefficient, are the same for every
enumeration**. `MajoranaEncoding::pauli_weight_terms` therefore merges once.
After that, each enumeration only needs one `popcount(x | z)` per surviving
term: no phases, strings or hashing.

On the CPU (4 cores, two-body Jordan–Wigner Hamiltonian, 64 permutations,
`benches/encode_symplectic.rs`):

| Modes | Reorder + encode each | Hash-merge per permutation | Merge once (now) |
|---|---|---|---|
| 12 | 70.7 ms | 24.8 ms | 2.9 ms |
| 16 | 330 ms | 93.6 ms | 10.6 ms |

This also turns the GPU kernel into a pure reduction: one number pair per
permutation comes back, instead of every term's Pauli string.

## The prototype kernel

`crates/ferrmion-gpu` sits outside the main workspace because cuda-oxide
builds with its own pinned nightly toolchain.

- **Layout** (`PackedProblem`): the encoding's X and Z blocks as row-major
  `u64` words, plus each merged term's indices padded to 7, its length and
  its coefficient norm. The encoding and Hamiltonian are uploaded once, and
  each call uploads only the permutations (`P × n_modes` `u32`s).
- **Kernel** (`pauli_weights`): one thread per (permutation, lane). Lane `l`
  sums terms `l, l + L, l + 2L, …`, so adjacent threads read adjacent terms
  (coalesced). The lane count `L` depends only on the number of terms, so the
  floating-point summation order, and the result, do not depend on the batch.
  Each thread writes a `u32` plain weight and an `f64` weighted partial.
- **Host:** sums the `L` partials of each permutation in order.
- **Shared code:** the per-thread function `packed::lane_weights` is a plain
  Rust function. cuda-oxide compiles it into the kernel, and
  `PackedProblem::emulate` calls the same function on the CPU. Property tests
  check the emulation against `batch_pauli_weights` on JW, parity, BK and JKMN
  encodings with up to 66 modes (two-word rows).

### Expected performance (to be measured)

Work per call is about `P · T · k · W` word XORs, where `k ≤ 4` is the
indices per term and `W` the words per row. The CPU path measured below
manages about 8.5 × 10⁷ term evaluations per second on 4 cores. For 100 modes,
10⁵ terms and 1024 permutations (10⁸ term evaluations) that is about 1.2 s,
where a data-centre GPU should take milliseconds. The term table (≈40 B per
term) is read once per permutation and stays in L2 for Hamiltonians up to a
few 10⁵ terms. Launch and transfer latency is tens of µs, so the GPU should
start to pay off once `P · T` reaches about 10⁵ (≈1 ms on the CPU).

CPU baseline at the sweep sizes in `benches/gpu_batch_weights.rs`
(4 cores; GPU column to be filled by running `cargo oxide bench` on a GPU
machine):

| Modes | Terms | Permutations | CPU | GPU |
|---|---|---|---|---|
| 20 | 1 000 | 64 | 0.51 ms | – |
| 20 | 1 000 | 1 024 | 3.1 ms | – |
| 50 | 10 000 | 256 | 10.9 ms | – |
| 100 | 10 000 | 256 | 14.3 ms | – |
| 100 | 100 000 | 64 | 74.8 ms | – |

## Batched drivers

argmin's `SimulatedAnnealing` proposes and scores one candidate per
iteration. At realistic sizes (T ≈ 10³–10⁵) one candidate is too little
work for a GPU, so the optimisers need batched proposals:

1. **Independent chains / parallel tempering** for `anneal_enumerations`:
   run K chains at a temperature ladder. Each step scores all K proposals in
   one `GpuEncoder::batch_pauli_weights` call, then does Metropolis
   acceptance and replica swaps on the host. This keeps per-chain
   reproducibility (one seeded RNG per chain) and fits the existing
   `AnnealingParameters`.
2. **Best-of-k neighbours**: each step scores k neighbouring permutations and
   accepts the best by Metropolis. This is a smaller change to the single-chain
   driver.

Both work just as well with the CPU `batch_pauli_weights` (rayon), so the
driver can be written and tested before a GPU path is enabled.

## Other candidates, ranked

1. **Clifford heuristic.** `apply_clifford_chain` already works on the
   bit-sliced transpose (one word holds 64 terms of one qubit), so a gate is a
   few word ops per term-word. A kernel with one thread per (candidate chain,
   term word) would apply its chain and popcount. It is batched in the same
   way as above. CPU first: `cost` clones the Hamiltonian and replays the
   whole growing chain at every step. Applying only the newly appended gates
   to the accepted state (and undoing them on rejection) makes each step cost
   O(new gates) instead of O(chain length).
2. **TOPP-HATT, `DenseTransposeBackend`.** For each active node, all allowed
   (x, y, z) triples are scored by popcounts over `T/64` words. On the GPU:
   one block per triple, a block-level popcount reduction, then a
   deterministic arg-min with the same tie-break as the CPU. Keep the
   transpose resident on the device and apply `reduce` there too, so only the
   chosen triple crosses PCIe each step. Branch-and-bound pruning does not
   carry over. This is worth it only for T ≳ 10⁵, where the steps are
   individually large.
3. **HATT and the tree bookkeeping** are sequential with little work per step.
   Leave them on the CPU.

## Framework assessment

| | cuda-oxide (chosen) | cutile-rs | cudarc + CUDA C++ |
|---|---|---|---|
| Vendor | NVIDIA (NVlabs) | NVIDIA (NVlabs) | Community |
| Kernel language | Rust (SIMT: threads, warps, shared memory) | Rust tile DSL | CUDA C++ |
| Toolchain | Pinned nightly + `cargo oxide` codegen backend | Stable Rust ≥ 1.89, JIT at first launch | Stable Rust + `nvcc` |
| Fit for ferrmion | Good: irregular per-term loops, bit ops, shared code with the CPU | Poor: workloads are not dense tensors | Good, but a second language |
| Maturity | Alpha, API still changing | Further along | Mature |

cuda-oxide is the best fit for kernels that loop over irregular term lists,
and the same plain function can serve the kernel and its CPU emulation. Its
costs:

- **Nightly-only kernel builds.** The host runtime crates (`cuda-core`,
  `cuda-async`) build on stable Rust. For shipping, the kernels can be
  compiled to PTX/LTOIR at release time with `cargo oxide` and loaded by a
  stable host build, so the PyPI wheel never needs nightly.
- **API churn.** The prototype pins one cuda-rust revision, and the codegen
  backend is built from that same revision.
- **Packaging.** GPU support should be an opt-in extra (a separate wheel or a
  `gpu` feature in the maturin build), never a dependency of the default
  wheel.

## Next steps

1. Run `cargo oxide test` and `cargo oxide bench` in `crates/ferrmion-gpu`
   on GPU hardware and fill in the table above.
2. Add a batched annealing driver, CPU first, then switch it to
   `GpuEncoder`.
3. Expose a `gpu=True` option on the Python annealing entry points, behind
   the opt-in build.
4. Make the Clifford heuristic's cost incremental, then port it as a second
   kernel.
