"""Summarize supplementary runs into CSV, Markdown, and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


METRICS = (
    "psnr",
    "ssim",
    "lpips",
    "psnr_dark",
    "psnr_mid",
    "psnr_bright",
    "pixel_fraction_dark",
    "pixel_fraction_mid",
    "pixel_fraction_bright",
    "num_GS",
    "needle_gt10",
    "opaque_needle_gt10",
    "training_time_sec",
    "peak_memory_gb",
    "render_time_sec_per_image",
    "render_fps",
    "checkpoint_size_mb",
)


def step_number(path: Path) -> int:
    match = re.search(r"step(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def latest_json(stats_dir: Path, prefix: str) -> Path | None:
    candidates = list(stats_dir.glob(f"{prefix}_step*.json"))
    return max(candidates, key=step_number) if candidates else None


def load_json(path: Path | None) -> Dict[str, object]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def collect_runs(manifest: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run in manifest.get("runs", []):
        result_dir = Path(str(run["result_dir"]))
        stats_dir = result_dir / "stats"
        val = load_json(latest_json(stats_dir, "val"))
        train = load_json(latest_json(stats_dir, "train"))
        if not val or not train:
            continue
        merged = {**train, **val}
        row: Dict[str, object] = {
            "suite": run["suite"],
            "method": run["method"],
            "scene": run["scene"],
            "seed": run["seed"],
            "settings": json.dumps(run.get("settings", {}), sort_keys=True),
            "result_dir": str(result_dir),
        }
        for metric in METRICS:
            value = finite_number(merged.get(metric))
            if value is not None:
                row[metric] = value
        rows.append(row)
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["suite"]), str(row["method"]), str(row["scene"]))].append(row)

    output: List[Dict[str, object]] = []
    for (suite, method, scene), group in sorted(grouped.items()):
        summary: Dict[str, object] = {
            "suite": suite,
            "method": method,
            "scene": scene,
            "n": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if metric in row]
            if values:
                summary[f"{metric}_mean"] = statistics.fmean(values)
                summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(summary)

    by_method: Dict[tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[(str(row["suite"]), str(row["method"]))].append(row)
    additive_metrics = {"num_GS", "training_time_sec", "checkpoint_size_mb"}
    for (suite, method), group in sorted(by_method.items()):
        # First aggregate scenes within each seed, then report variation across
        # seeds. This avoids mixing scene difficulty into the random-seed std.
        by_seed: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
        for row in group:
            by_seed[int(row["seed"])].append(row)
        summary = {
            "suite": suite,
            "method": method,
            "scene": "MEAN",
            "n": len(by_seed),
        }
        for metric in METRICS:
            seed_values: List[float] = []
            for seed_group in by_seed.values():
                values = [float(row[metric]) for row in seed_group if metric in row]
                if not values:
                    continue
                seed_values.append(
                    sum(values) if metric in additive_metrics else statistics.fmean(values)
                )
            if seed_values:
                summary[f"{metric}_mean"] = statistics.fmean(seed_values)
                summary[f"{metric}_std"] = (
                    statistics.stdev(seed_values) if len(seed_values) > 1 else 0.0
                )
        output.append(summary)
    return output


def fmt(row: Mapping[str, object], metric: str, digits: int) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return "--"
    return f"{float(mean):.{digits}f} +/- {float(std or 0):.{digits}f}"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = ["suite", "method", "scene", "n"]
    fields += [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "| Suite | Method | Scene | n | PSNR | SSIM | LPIPS | GS | Needle10 | OpaqueN10 | Time (s) | Peak mem (GB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {suite} | {method} | {scene} | {n} | {psnr} | {ssim} | {lpips} | {gs} | {needle} | {opaque} | {time} | {mem} |".format(
                **row,
                psnr=fmt(row, "psnr", 3),
                ssim=fmt(row, "ssim", 4),
                lpips=fmt(row, "lpips", 3),
                gs=fmt(row, "num_GS", 0),
                needle=fmt(row, "needle_gt10", 4),
                opaque=fmt(row, "opaque_needle_gt10", 4),
                time=fmt(row, "training_time_sec", 1),
                mem=fmt(row, "peak_memory_gb", 3),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_latex(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Suite & Method & Scene & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & \#GS $\downarrow$ & Needle10 $\downarrow$ & OpaqueN10 $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            latex_escape(row["suite"]),
            latex_escape(row["method"]),
            latex_escape(row["scene"]),
            fmt(row, "psnr", 3).replace("+/-", r"$\pm$"),
            fmt(row, "ssim", 4).replace("+/-", r"$\pm$"),
            fmt(row, "lpips", 3).replace("+/-", r"$\pm$"),
            fmt(row, "num_GS", 0).replace("+/-", r"$\pm$"),
            fmt(row, "needle_gt10", 4).replace("+/-", r"$\pm$"),
            fmt(row, "opaque_needle_gt10", 4).replace("+/-", r"$\pm$"),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_region_latex(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    selected = [row for row in rows if "psnr_dark_mean" in row]
    lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Suite & Method & Scene & Dark PSNR $\uparrow$ & Mid PSNR $\uparrow$ & Bright PSNR $\uparrow$ \\",
        r"\midrule",
    ]
    for row in selected:
        cells = [
            latex_escape(row["suite"]),
            latex_escape(row["method"]),
            latex_escape(row["scene"]),
            fmt(row, "psnr_dark", 3).replace("+/-", r"$\pm$"),
            fmt(row, "psnr_mid", 3).replace("+/-", r"$\pm$"),
            fmt(row, "psnr_bright", 3).replace("+/-", r"$\pm$"),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_efficiency_latex(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    selected = [row for row in rows if row["scene"] == "MEAN"]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Suite & Method & \#GS $\downarrow$ & Train time (s) $\downarrow$ & Peak memory (GB) $\downarrow$ & FPS $\uparrow$ & Checkpoint (MB) $\downarrow$ \\",
        r"\midrule",
    ]
    for row in selected:
        cells = [
            latex_escape(row["suite"]),
            latex_escape(row["method"]),
            fmt(row, "num_GS", 0).replace("+/-", r"$\pm$"),
            fmt(row, "training_time_sec", 1).replace("+/-", r"$\pm$"),
            fmt(row, "peak_memory_gb", 3).replace("+/-", r"$\pm$"),
            fmt(row, "render_fps", 1).replace("+/-", r"$\pm$"),
            fmt(row, "checkpoint_size_mb", 1).replace("+/-", r"$\pm$"),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = collect_runs(manifest)
    if not runs:
        print("No complete runs found (both train and val JSON files are required).")
        return 2

    rows = aggregate(runs)
    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    write_markdown(output_dir / "summary.md", rows)
    write_latex(output_dir / "summary.tex", rows)
    write_region_latex(output_dir / "region_summary.tex", rows)
    write_efficiency_latex(output_dir / "efficiency_summary.tex", rows)
    print(f"Summarized {len(runs)} completed runs into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
