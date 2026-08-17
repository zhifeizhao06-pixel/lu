"""Paired significance tests for supplementary experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from summarize_supplementary import collect_runs


METRICS = (
    "psnr",
    "ssim",
    "lpips",
    "num_GS",
    "needle_gt10",
    "opaque_needle_gt10",
)


def bootstrap_ci(values: Sequence[float], repetitions: int, seed: int) -> Tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(repetitions):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lower = means[int(0.025 * (repetitions - 1))]
    upper = means[int(0.975 * (repetitions - 1))]
    return lower, upper


def sign_flip_pvalue(values: Sequence[float], repetitions: int, seed: int) -> float:
    """Two-sided paired randomization test under exchangeability."""
    observed = abs(statistics.fmean(values))
    n = len(values)
    if observed == 0 and all(value == 0 for value in values):
        return 1.0
    extreme = 0
    if n <= 20:
        total = 1 << n
        for mask in range(total):
            permuted = [value if (mask >> i) & 1 else -value for i, value in enumerate(values)]
            if abs(statistics.fmean(permuted)) >= observed - 1e-15:
                extreme += 1
        return extreme / total
    rng = random.Random(seed)
    for _ in range(repetitions):
        permuted = [value if rng.random() < 0.5 else -value for value in values]
        if abs(statistics.fmean(permuted)) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (repetitions + 1)


def compare(
    runs: Sequence[Mapping[str, object]],
    baseline: str,
    repetitions: int,
    seed: int,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in runs:
        grouped[(str(row["suite"]), str(row["method"]))].append(row)
    output: List[Dict[str, object]] = []
    suites = sorted({suite for suite, _ in grouped})
    for suite in suites:
        base = {
            (str(row["scene"]), int(row["seed"])): row
            for row in grouped.get((suite, baseline), [])
        }
        if not base:
            continue
        methods = sorted(method for candidate_suite, method in grouped if candidate_suite == suite and method != baseline)
        for method in methods:
            candidate = {
                (str(row["scene"]), int(row["seed"])): row
                for row in grouped[(suite, method)]
            }
            keys = sorted(set(base) & set(candidate))
            for metric_index, metric in enumerate(METRICS):
                differences = [
                    float(candidate[key][metric]) - float(base[key][metric])
                    for key in keys
                    if metric in base[key] and metric in candidate[key]
                ]
                if not differences:
                    continue
                ci_low, ci_high = bootstrap_ci(
                    differences, repetitions, seed + metric_index
                )
                output.append(
                    {
                        "suite": suite,
                        "baseline": baseline,
                        "method": method,
                        "metric": metric,
                        "n_pairs": len(differences),
                        "mean_delta": statistics.fmean(differences),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "p_value": sign_flip_pvalue(
                            differences, repetitions, seed + 100 + metric_index
                        ),
                    }
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = collect_runs(manifest)
    rows = compare(runs, args.baseline, args.repetitions, args.seed)
    if not rows:
        print("No matched method/baseline pairs were found.")
        return 2
    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "significance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Suite | Baseline | Method | Metric | Pairs | Mean delta | 95% CI | p-value |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['suite']} | {row['baseline']} | {row['method']} | {row['metric']} | "
            f"{row['n_pairs']} | {row['mean_delta']:.6g} | "
            f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] | {row['p_value']:.6g} |"
        )
    (output_dir / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote paired tests for {len(rows)} comparisons to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
