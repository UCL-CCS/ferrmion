#!/usr/bin/env bash
# Sweep the merge strategies behind `append_fermion_sparse` across thread
# counts, measuring both the end-to-end Python `encode_topphatt` benchmarks
# (the decision metric) and the Rust criterion micro-benchmarks (the
# diagnostic layer). Results land in a timestamped directory together with a
# machine-readable metadata file and a generated markdown report.
#
# The strategy and thread count are communicated through environment
# variables (`FERRMION_MERGE_STRATEGY`, `RAYON_NUM_THREADS`) that both the
# Rust library and rayon read once at startup, which is why every
# (strategy, threads) combination runs in a fresh process.
#
# Usage:
#   scripts/bench_merge_sweep.sh [options]
#
# Options:
#   --smoke              Quick subset: one dataset/encoding, one bench size,
#                        criterion --quick. Validates the harness end-to-end.
#   --strategies LIST    Comma-separated strategy names (default: all).
#   --threads LIST       Comma-separated thread counts (default: powers of two
#                        up to the core count, plus the core count itself).
#   --out DIR            Results directory (default: bench_results/<timestamp>).
#   --skip-rust          Skip the criterion micro-benchmarks.
#   --skip-python        Skip the pytest end-to-end benchmarks.
#   --timing             Also capture FERRMION_MERGE_TIMING=1 phase-attribution
#                        logs (one single-shot pytest run per combination).
#   -h, --help           Show this help.
set -euo pipefail

cd "$(dirname "$0")/.."

ALL_STRATEGIES="baseline,hash_cache,fx_hash,shard_phase1,tree_reduce,sort_scan,radix_partition,kway_merge"

SMOKE=0
STRATEGIES="$ALL_STRATEGIES"
THREADS=""
OUT=""
SKIP_RUST=0
SKIP_PYTHON=0
TIMING=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --strategies) STRATEGIES="$2"; shift ;;
        --threads) THREADS="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --skip-rust) SKIP_RUST=1 ;;
        --skip-python) SKIP_PYTHON=1 ;;
        --timing) TIMING=1 ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

NPROC="$(nproc)"
if [[ -z "$THREADS" ]]; then
    THREADS="1"
    t=2
    while (( t < NPROC )); do THREADS="$THREADS,$t"; t=$((t * 2)); done
    (( NPROC > 1 )) && THREADS="$THREADS,$NPROC"
fi

OUT="${OUT:-bench_results/merge_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# Pytest selection: the decision metric is test_benchmark_encode_topphatt.
PYTEST_K="test_benchmark_encode_topphatt"
CRITERION_EXTRA=()
CRITERION_FILTER=""
if (( SMOKE )); then
    PYTEST_K="test_benchmark_encode_topphatt and JordanWigner and h2_6"
    CRITERION_EXTRA=(--quick)
    CRITERION_FILTER="50000"
fi

echo "== ferrmion merge-strategy sweep =="
echo "strategies: $STRATEGIES"
echo "threads:    $THREADS"
echo "results:    $OUT"
(( SMOKE )) && echo "mode:       smoke"

# Record enough metadata to interpret the numbers later / on another machine.
{
    echo "{"
    echo "  \"date\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"host\": \"$(hostname)\","
    echo "  \"nproc\": $NPROC,"
    echo "  \"cpu\": \"$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //' || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)\","
    echo "  \"git_rev\": \"$(git rev-parse HEAD 2>/dev/null || echo unknown)\","
    echo "  \"strategies\": \"$STRATEGIES\","
    echo "  \"threads\": \"$THREADS\","
    echo "  \"smoke\": $SMOKE"
    echo "}"
} > "$OUT/meta.json"

# Build everything once, outside the timed loops. Maturin's PEP 517 backend
# builds in release mode by default; a debug wheel would need a deliberate
# MATURIN_PEP517_ARGS="--profile dev" override, so guard against one leaking in.
if [[ -n "${MATURIN_PEP517_ARGS:-}" ]]; then
    echo "refusing to benchmark with MATURIN_PEP517_ARGS=${MATURIN_PEP517_ARGS} set" >&2
    exit 2
fi
if (( ! SKIP_RUST )); then
    echo "-- building criterion benches (release)"
    cargo bench --bench merge_strategies --no-run
fi
if (( ! SKIP_PYTHON )); then
    echo "-- building python wheel (release) + test deps"
    uv sync --group test --reinstall-package ferrmion
fi

IFS=',' read -r -a STRATEGY_ARR <<< "$STRATEGIES"
IFS=',' read -r -a THREAD_ARR <<< "$THREADS"

for s in "${STRATEGY_ARR[@]}"; do
    for t in "${THREAD_ARR[@]}"; do
        echo "== strategy=$s threads=$t =="

        if (( ! SKIP_RUST )); then
            echo "-- criterion micro-bench"
            RAYON_NUM_THREADS="$t" FERRMION_MERGE_STRATEGY="$s" \
                cargo bench --bench merge_strategies -- --noplot \
                --save-baseline "${s}_t${t}" "${CRITERION_EXTRA[@]}" $CRITERION_FILTER
        fi

        if (( ! SKIP_PYTHON )); then
            echo "-- pytest end-to-end (encode_topphatt)"
            RAYON_NUM_THREADS="$t" FERRMION_MERGE_STRATEGY="$s" \
                uv run --group test pytest python/tests/test_ternary_tree.py \
                -k "$PYTEST_K" -q --no-header \
                --benchmark-json="$OUT/pytest_${s}_t${t}.json"
        fi

        if (( TIMING )); then
            echo "-- phase-timing capture"
            RAYON_NUM_THREADS="$t" FERRMION_MERGE_STRATEGY="$s" FERRMION_MERGE_TIMING=1 \
                uv run --group test pytest python/tests/test_ternary_tree.py \
                -k "$PYTEST_K" -q --no-header --benchmark-disable \
                2> "$OUT/timing_${s}_t${t}.log" || {
                    echo "timing capture failed for $s/t$t (see $OUT/timing_${s}_t${t}.log)" >&2
                }
        fi
    done
done

echo "== generating report =="
uv run python scripts/bench_merge_report.py \
    --results "$OUT" --criterion target/criterion --output "$OUT/report.md"
echo "report: $OUT/report.md"
