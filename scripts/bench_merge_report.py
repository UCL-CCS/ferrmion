#!/usr/bin/env python3
"""Aggregate merge-strategy sweep results into a markdown report.

Reads the pytest-benchmark JSON files written by ``bench_merge_sweep.sh``
(end-to-end ``encode_topphatt`` runtimes, the decision metric) and the
criterion baselines saved by the Rust micro-benchmarks (the diagnostic
layer), then emits per-workload runtime tables, speedup/parallel-efficiency
tables, and a ranking of strategies at the highest measured thread count.

Only the Python standard library is required, so the script runs anywhere the
sweep does.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

PYTEST_FILE = re.compile(r"pytest_(?P<strategy>.+)_t(?P<threads>\d+)\.json$")
BASELINE_DIR = re.compile(r"^(?P<strategy>[a-z0-9_]+)_t(?P<threads>\d+)$")
CRITERION_RESERVED = {"new", "base", "change", "report"}

# results[source][workload][strategy][threads] = median runtime in seconds
Results = dict[str, dict[str, dict[str, dict[int, float]]]]


def load_pytest_results(results_dir: Path, results: Results) -> None:
    """Collect end-to-end medians from pytest-benchmark JSON exports."""
    for path in sorted(results_dir.glob("pytest_*.json")):
        match = PYTEST_FILE.search(path.name)
        if match is None:
            continue
        strategy = match.group("strategy")
        threads = int(match.group("threads"))
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping unreadable {path}: {exc}")
            continue
        for bench in data.get("benchmarks", []):
            name = bench.get("name", "")
            workload = name[name.find("[") + 1 : name.rfind("]")] if "[" in name else name
            median = bench.get("stats", {}).get("median")
            if median is None:
                continue
            results["end-to-end encode (pytest)"].setdefault(workload, {}).setdefault(
                strategy, {}
            )[threads] = float(median)


def load_criterion_results(criterion_dir: Path, results: Results) -> None:
    """Collect micro-benchmark medians from saved criterion baselines."""
    if not criterion_dir.is_dir():
        return
    for estimates in sorted(criterion_dir.rglob("estimates.json")):
        baseline = estimates.parent.name
        if baseline in CRITERION_RESERVED:
            continue
        match = BASELINE_DIR.match(baseline)
        if match is None:
            continue
        relative = estimates.parent.relative_to(criterion_dir).parts[:-1]
        workload = "/".join(relative)
        try:
            median_ns = json.loads(estimates.read_text())["median"]["point_estimate"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"warning: skipping unreadable {estimates}: {exc}")
            continue
        results["rust micro-bench (criterion)"].setdefault(workload, {}).setdefault(
            match.group("strategy"), {}
        )[int(match.group("threads"))] = float(median_ns) / 1e9


def fmt_time(seconds: float) -> str:
    """Human-readable runtime."""
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.1f} µs"


def strategy_order(strategies: set[str]) -> list[str]:
    """Baseline first, then alphabetical."""
    return sorted(strategies, key=lambda s: (s != "baseline", s))


def runtime_table(per_strategy: dict[str, dict[int, float]], threads: list[int]) -> list[str]:
    lines = [
        "| strategy | " + " | ".join(f"t={t}" for t in threads) + " |",
        "|---" * (len(threads) + 1) + "|",
    ]
    for strategy in strategy_order(set(per_strategy)):
        cells = []
        for t in threads:
            median = per_strategy[strategy].get(t)
            cells.append(fmt_time(median) if median is not None else "—")
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    return lines


def scaling_table(per_strategy: dict[str, dict[int, float]], threads: list[int]) -> list[str]:
    """Speedup S(p) = T(ref)/T(p) and efficiency E(p) = S(p)*ref/p per strategy.

    The reference is the smallest measured thread count for that strategy
    (normally 1). Non-monotone scaling — runtime increasing with threads — is
    flagged with ⚠.
    """
    lines = [
        "| strategy | " + " | ".join(f"t={t}" for t in threads) + " |",
        "|---" * (len(threads) + 1) + "|",
    ]
    for strategy in strategy_order(set(per_strategy)):
        measured = per_strategy[strategy]
        if not measured:
            continue
        ref_threads = min(measured)
        ref = measured[ref_threads]
        cells = []
        previous: float | None = None
        for t in threads:
            median = measured.get(t)
            if median is None:
                cells.append("—")
                continue
            speedup = ref / median
            efficiency = speedup * ref_threads / t
            flag = " ⚠" if previous is not None and median > previous * 1.02 else ""
            cells.append(f"{speedup:.2f}x (E={efficiency:.2f}){flag}")
            previous = median
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    return lines


def ranking(workloads: dict[str, dict[str, dict[int, float]]]) -> list[str]:
    """Rank strategies by geometric-mean runtime ratio vs `baseline` at the
    highest thread count measured for both."""
    ratios: dict[str, list[float]] = defaultdict(list)
    for per_strategy in workloads.values():
        base = per_strategy.get("baseline", {})
        for strategy, measured in per_strategy.items():
            common = set(measured) & set(base)
            if not common:
                continue
            top = max(common)
            if base[top] > 0:
                ratios[strategy].append(measured[top] / base[top])
    if not ratios:
        return ["_no overlapping measurements to rank_"]
    lines = [
        "| rank | strategy | runtime vs baseline (geomean, max threads) |",
        "|---|---|---|",
    ]
    ranked = sorted(
        ratios.items(), key=lambda kv: math.exp(sum(map(math.log, kv[1])) / len(kv[1]))
    )
    for position, (strategy, values) in enumerate(ranked, start=1):
        geomean = math.exp(sum(map(math.log, values)) / len(values))
        lines.append(f"| {position} | {strategy} | {geomean:.3f}x |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="sweep results directory")
    parser.add_argument(
        "--criterion",
        type=Path,
        default=Path("target/criterion"),
        help="criterion output directory (default: target/criterion)",
    )
    parser.add_argument("--output", type=Path, required=True, help="markdown report path")
    args = parser.parse_args()

    results: Results = defaultdict(dict)
    load_pytest_results(args.results, results)
    load_criterion_results(args.criterion, results)

    lines: list[str] = ["# Merge-strategy sweep report", ""]
    meta_path = args.results / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        lines += [
            "| " + " | ".join(meta) + " |",
            "|---" * len(meta) + "|",
            "| " + " | ".join(str(v) for v in meta.values()) + " |",
            "",
        ]

    if not results:
        lines.append("_no benchmark data found_")
    for source, workloads in results.items():
        lines += [f"## {source}", ""]
        threads = sorted({t for w in workloads.values() for m in w.values() for t in m})
        for workload, per_strategy in sorted(workloads.items()):
            lines += [f"### {workload}", "", "**Median runtime**", ""]
            lines += runtime_table(per_strategy, threads)
            lines += ["", "**Speedup (parallel efficiency)** — ⚠ marks non-monotone scaling", ""]
            lines += scaling_table(per_strategy, threads)
            lines.append("")
        lines += [f"### Ranking — {source}", ""]
        lines += ranking(workloads)
        lines.append("")

    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
