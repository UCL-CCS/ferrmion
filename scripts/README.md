# Merge-strategy benchmark harness

Apparatus for deciding between the alternative hash-map merge algorithms used
by `append_fermion_sparse` when constructing a `MajoranaSparse`
(see `crates/ferrmion-core/src/operators/merge.rs` for the strategies and the
environment variables that select them).

## Quick start

```sh
# Validate the harness end-to-end (~minutes): one dataset, one bench size.
scripts/bench_merge_sweep.sh --smoke

# Full sweep on this machine: every strategy x thread counts up to nproc.
scripts/bench_merge_sweep.sh

# Subset: two strategies, explicit thread counts, phase-timing capture.
scripts/bench_merge_sweep.sh --strategies baseline,shard_phase1 \
    --threads 1,4,16,64 --timing

# Knob sweep: rerun survivors with pre-sized maps and a doubled shard count,
# into the SAME results directory as a default run so the report compares
# them side by side (results are labelled e.g. radix_partition_p1_s22).
scripts/bench_merge_sweep.sh --strategies hash_cache,radix_partition \
    --presize --shards 22 --out bench_results/round2
```

Each run writes a results directory (default `bench_results/<timestamp>/`)
containing:

- `report.md` — runtime tables, speedup/parallel-efficiency tables and a
  strategy ranking, for both measurement layers;
- `pytest_<strategy>_t<threads>.json` — raw pytest-benchmark exports of the
  **end-to-end `encode_topphatt` benchmarks (the decision metric)**;
- `timing_<strategy>_t<threads>.log` — with `--timing`, per-call
  expand/merge phase durations (`[ferrmion-merge-timing] ...` lines), used to
  attribute end-to-end time to the merge phase;
- `meta.json` — host, core count, CPU model, git revision, sweep parameters.

The Rust micro-benchmark medians live in criterion's own tree
(`target/criterion/.../<strategy>_t<threads>/estimates.json`); the report
script reads them from there. To regenerate a report:

```sh
uv run python scripts/bench_merge_report.py \
    --results bench_results/<timestamp> --output report.md
```

## How to read the report

- **Median runtime**: lower is better; the end-to-end pytest tables are the
  primary ranking signal, the criterion tables explain *why* (workload
  regimes: `low_collision` stresses inserts/growth, `high_collision` stresses
  duplicate summing).
- **Speedup (E=…)**: speedup relative to that strategy's smallest measured
  thread count, with parallel efficiency in parentheses (1.00 = perfectly
  linear scaling). ⚠ flags thread counts where runtime *increased*.
- **Ranking**: geometric-mean runtime ratio vs the `baseline` strategy at the
  highest thread count, across workloads. Below 1.0 is faster than baseline.
  Workloads whose baseline runtime is under 1 ms are excluded from the ranking
  (they never reach the parallel merge path — e.g. the `h2_*` pytest datasets —
  so their ratios are noise); their tables are still printed.

## macOS notes

- The sweep script is portable shell: it runs under macOS's stock bash 3.2 and
  detects the core count via `sysctl` when `nproc` is unavailable.
- On Apple Silicon, `hw.ncpu` counts performance **and** efficiency cores
  together, which muddies scaling curves once rayon spills onto E-cores. For a
  clean efficiency measurement, pass an explicit list capped at the P-core
  count, e.g. `--threads 1,2,4,$(sysctl -n hw.perflevel0.logicalcpu)`.

## Notes

- **The production default is `radix_partition` with pre-sizing on**, adopted
  after the benchmark campaign recorded in the repository history (M3 Pro
  11-core and 4-core x86: ~15% faster end-to-end, ~1.6-1.8x faster
  merge-dominated micro-benchmarks at high thread counts than the original
  algorithm). `FERRMION_MERGE_STRATEGY=baseline FERRMION_MERGE_PRESIZE=0`
  reproduces the pre-campaign behaviour exactly. Four dominated candidates
  (`fx_hash`, `tree_reduce`, `sort_scan`, `kway_merge`) were removed; their
  implementations live in the git history.
- Strategy and thread count are read **once per process**, so the sweep runs
  each combination as a fresh process; do not try to switch strategies from
  within Python after `ferrmion` has built its first operator.
- Every strategy is validated against a serial reference by
  `cargo test -p ferrmion-core merge`; run this before benchmarking modified
  strategies.
- The wheel is rebuilt in release mode once per sweep (not per combination).
- All strategies sum every key's contributions in chunk order, so results are
  bit-for-bit deterministic and independent of strategy and thread count.
