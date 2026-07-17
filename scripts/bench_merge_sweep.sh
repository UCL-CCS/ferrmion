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
#   --presize            Run every combination with FERRMION_MERGE_PRESIZE=1;
#                        results are labelled <strategy>_p1 so they can be
#                        compared against a default-knob sweep in one report.
#   --shards N           Run every combination with FERRMION_MERGE_SHARDS=N;
#                        results are labelled <strategy>_sN.
#   -h, --help           Show this help.
set -euo pipefail

cd "$(dirname "$0")/.."

# Portable core count: Linux (nproc), macOS/BSD (sysctl), POSIX fallback.
detect_nproc() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif sysctl -n hw.ncpu >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
    fi
}

# Portable CPU-model string for meta.json.
detect_cpu() {
    if [[ -r /proc/cpuinfo ]]; then
        grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'
    elif sysctl -n machdep.cpu.brand_string >/dev/null 2>&1; then
        sysctl -n machdep.cpu.brand_string
    else
        echo unknown
    fi
}

# Print the leading comment block (after the shebang) as --help text.
print_help() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

ALL_STRATEGIES="baseline,hash_cache,shard_phase1,radix_partition"

SMOKE=0
STRATEGIES="$ALL_STRATEGIES"
THREADS=""
OUT=""
SKIP_RUST=0
SKIP_PYTHON=0
TIMING=0
PRESIZE=0
SHARDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --strategies) STRATEGIES="$2"; shift ;;
        --threads) THREADS="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --skip-rust) SKIP_RUST=1 ;;
        --skip-python) SKIP_PYTHON=1 ;;
        --timing) TIMING=1 ;;
        --presize) PRESIZE=1 ;;
        --shards) SHARDS="$2"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

# Knob env vars are constant across the whole sweep (the per-combination env
# only varies strategy and thread count); the result labels carry a suffix so
# knobbed and default runs stay distinguishable in a merged report.
KNOB_LABEL=""
if (( PRESIZE )); then
    export FERRMION_MERGE_PRESIZE=1
    KNOB_LABEL="${KNOB_LABEL}_p1"
fi
if [[ -n "$SHARDS" ]]; then
    export FERRMION_MERGE_SHARDS="$SHARDS"
    KNOB_LABEL="${KNOB_LABEL}_s${SHARDS}"
fi

NPROC="$(detect_nproc)"
if [[ -z "$THREADS" ]]; then
    THREADS="1"
    t=2
    while (( t < NPROC )); do THREADS="$THREADS,$t"; t=$((t * 2)); done
    (( NPROC > 1 )) && THREADS="$THREADS,$NPROC"
fi

OUT="${OUT:-bench_results/merge_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# Pytest selection: the decision metric is test_benchmark_encode_topphatt.
# CRITERION_EXTRA/CRITERION_FILTER hold whitespace-free flags and are expanded
# unquoted; a plain string (not an array) keeps bash 3.2's `set -u` happy on
# macOS, where empty-array expansion counts as an unbound variable.
PYTEST_K="test_benchmark_encode_topphatt"
CRITERION_EXTRA=""
CRITERION_FILTER=""
if (( SMOKE )); then
    # h2o/sto-3g is the smallest dataset that still takes the parallel
    # expand-and-merge path (a few thousand terms); h2_* stay serial.
    PYTEST_K="test_benchmark_encode_topphatt and JordanWigner and h2o_sto"
    CRITERION_EXTRA="--quick"
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
    echo "  \"cpu\": \"$(detect_cpu)\","
    echo "  \"git_rev\": \"$(git rev-parse HEAD 2>/dev/null || echo unknown)\","
    echo "  \"strategies\": \"$STRATEGIES\","
    echo "  \"threads\": \"$THREADS\","
    echo "  \"knobs\": \"${KNOB_LABEL:-default}\","
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
    label="${s}${KNOB_LABEL}"
    for t in "${THREAD_ARR[@]}"; do
        echo "== strategy=$s threads=$t knobs=${KNOB_LABEL:-default} =="

        if (( ! SKIP_RUST )); then
            echo "-- criterion micro-bench"
            RAYON_NUM_THREADS="$t" FERRMION_MERGE_STRATEGY="$s" \
                cargo bench --bench merge_strategies -- --noplot \
                --save-baseline "${label}_t${t}" $CRITERION_EXTRA $CRITERION_FILTER
        fi

        if (( ! SKIP_PYTHON )); then
            echo "-- pytest end-to-end (encode_topphatt)"
            RAYON_NUM_THREADS="$t" FERRMION_MERGE_STRATEGY="$s" \
                uv run --group test pytest python/tests/test_ternary_tree.py \
                -k "$PYTEST_K" -q --no-header \
                --benchmark-json="$OUT/pytest_${label}_t${t}.json"
        fi

        if (( TIMING )); then
            echo "-- phase-timing capture"
            # -s disables pytest's output capture, which would otherwise
            # swallow the timing lines the Rust library prints to stderr.
            RAYON_NUM_THREADS="$t" FERRMION_MERGE_STRATEGY="$s" FERRMION_MERGE_TIMING=1 \
                uv run --group test pytest python/tests/test_ternary_tree.py \
                -k "$PYTEST_K" -q --no-header -s --benchmark-disable \
                2> "$OUT/timing_${label}_t${t}.log" || {
                    echo "timing capture failed for $s/t$t (see $OUT/timing_${label}_t${t}.log)" >&2
                }
        fi
    done
done

echo "== generating report =="
uv run python scripts/bench_merge_report.py \
    --results "$OUT" --criterion target/criterion --output "$OUT/report.md"
echo "report: $OUT/report.md"
