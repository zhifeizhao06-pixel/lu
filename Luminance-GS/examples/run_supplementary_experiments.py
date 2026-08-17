"""Build and optionally execute reproducible supplementary experiment sweeps.

The default mode is a dry run: commands and a machine-readable manifest are
written, but no training is started. Add ``--execute`` on the server to run the
commands sequentially. Results are isolated by suite, method, scene, and seed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_SCENES = ["bike", "buu", "chair", "shrub", "sofa"]
DEFAULT_SEEDS = [0, 1, 2]


@dataclass
class RunSpec:
    suite: str
    method: str
    scene: str
    seed: int
    result_dir: str
    settings: Dict[str, object]
    command: List[str]
    status: str = "planned"
    returncode: int | None = None


def boolean_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def method_args(**overrides: object) -> Dict[str, object]:
    """Return the full proposed method, with explicit flags for reproducibility."""
    settings: Dict[str, object] = {
        "noise-aware": True,
        "noise-model": "linear",
        "confidence-densify": True,
        "needle-regularization": True,
        "confidence-curriculum": True,
        "confidence-schedule": "smoothstep",
    }
    settings.update(overrides)
    return settings


def suite_methods(suite: str) -> Iterable[tuple[str, Dict[str, object]]]:
    if suite == "core":
        yield "baseline", method_args(
            **{
                "noise-aware": False,
                "confidence-densify": False,
                "needle-regularization": False,
                "confidence-curriculum": False,
            }
        )
        yield "noise_nll", method_args(
            **{
                "confidence-densify": False,
                "needle-regularization": False,
                "confidence-curriculum": False,
            }
        )
        yield "confidence", method_args(
            **{
                "needle-regularization": False,
                "confidence-curriculum": False,
            }
        )
        yield "confidence_shape", method_args(
            **{"confidence-curriculum": False}
        )
        yield "full", method_args()
    elif suite == "noise":
        for model in ("fixed", "linear", "poisson_read"):
            yield f"noise_{model}", method_args(**{"noise-model": model})
    elif suite == "schedule":
        for schedule in ("fixed", "step", "linear", "smoothstep", "exponential"):
            yield f"schedule_{schedule}", method_args(
                **{"confidence-schedule": schedule}
            )
    elif suite == "sensitivity":
        for value in (2.0, 3.0, 5.0, 7.0, 10.0):
            yield f"needle_tau_{value:g}", method_args(
                **{"needle-ratio-max": value}
            )
        for value in (1e-5, 1e-4, 5e-4, 1e-3, 5e-3):
            yield f"shape_lambda_{value:g}", method_args(
                **{"needle-reg-lambda": value}
            )
        for value in (0.05, 0.10, 0.15, 0.20, 0.30):
            yield f"confidence_min_{value:g}", method_args(
                **{"densify-confidence-min": value}
            )
    elif suite == "efficiency":
        yield "baseline", method_args(
            **{
                "noise-aware": False,
                "confidence-densify": False,
                "needle-regularization": False,
                "confidence-curriculum": False,
            }
        )
        yield "full", method_args()
    else:
        raise ValueError(f"Unknown suite: {suite}")


def settings_to_cli(settings: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for name, value in settings.items():
        if isinstance(value, bool):
            args.append(boolean_flag(name, value))
        else:
            args.extend([f"--{name}", str(value)])
    return args


def result_complete(result_dir: Path) -> bool:
    stats = result_dir / "stats"
    return stats.is_dir() and any(stats.glob("val_step*.json"))


def build_specs(args: argparse.Namespace) -> List[RunSpec]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    suites = (
        ["core", "noise", "schedule", "sensitivity", "efficiency"]
        if args.suite == "all"
        else [args.suite]
    )
    specs: List[RunSpec] = []
    for suite in suites:
        for method, settings in suite_methods(suite):
            for scene in args.scenes:
                for seed in args.seeds:
                    result_dir = (
                        output_root / suite / method / scene / f"seed_{seed}"
                    )
                    command = [
                        args.python,
                        args.trainer,
                        "--data_dir",
                        str(data_root / scene),
                        "--exp-name",
                        args.exp_name,
                        "--result-dir",
                        str(result_dir),
                        "--seed",
                        str(seed),
                        boolean_flag("save-eval-images", args.save_eval_images),
                        boolean_flag("render-trajectory", args.render_trajectory),
                    ]
                    command.extend(settings_to_cli(settings))
                    specs.append(
                        RunSpec(
                            suite=suite,
                            method=method,
                            scene=scene,
                            seed=seed,
                            result_dir=str(result_dir),
                            settings=settings,
                            command=command,
                        )
                    )
    if args.max_runs is not None:
        specs = specs[: args.max_runs]
    return specs


def write_manifest(path: Path, args: argparse.Namespace, specs: Sequence[RunSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "trainer": args.trainer,
        "runs": [asdict(spec) for spec in specs],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_key(spec: RunSpec) -> tuple[str, str, str, int]:
    return spec.suite, spec.method, spec.scene, spec.seed


def merge_existing_runs(path: Path, current: Sequence[RunSpec]) -> List[RunSpec]:
    """Preserve other suites when several launcher calls share an output root."""
    if not path.exists():
        return list(current)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = [RunSpec(**run) for run in payload.get("runs", [])]
    except (OSError, TypeError, ValueError):
        return list(current)

    current_by_key = {run_key(spec): spec for spec in current}
    merged: List[RunSpec] = []
    for old_spec in existing:
        fresh = current_by_key.pop(run_key(old_spec), None)
        merged.append(fresh if fresh is not None else old_spec)
    merged.extend(current_by_key.values())
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["core", "noise", "schedule", "sensitivity", "efficiency", "all"],
        default="core",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="../results_supplementary")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--trainer", default="simple_trainer_ours.py")
    parser.add_argument("--exp-name", default="low")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-eval-images", action="store_true")
    parser.add_argument("--render-trajectory", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()

    specs = build_specs(args)
    manifest_path = Path(args.output_root).expanduser().resolve() / "run_manifest.json"
    manifest_specs = merge_existing_runs(manifest_path, specs)
    write_manifest(manifest_path, args, manifest_specs)
    print(f"Prepared {len(specs)} runs. Manifest: {manifest_path}")

    for index, spec in enumerate(specs, start=1):
        result_dir = Path(spec.result_dir)
        if args.resume and result_complete(result_dir):
            spec.status = "skipped_complete"
            print(f"[{index}/{len(specs)}] SKIP {spec.method}/{spec.scene}/seed_{spec.seed}")
            write_manifest(manifest_path, args, manifest_specs)
            continue

        printable = shlex.join(spec.command)
        print(f"[{index}/{len(specs)}] {printable}")
        if not args.execute:
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        spec.status = "running"
        write_manifest(manifest_path, args, manifest_specs)
        completed = subprocess.run(spec.command, check=False)
        spec.returncode = completed.returncode
        spec.status = "complete" if completed.returncode == 0 else "failed"
        write_manifest(manifest_path, args, manifest_specs)
        if completed.returncode != 0:
            print("A run failed; the manifest records its return code.", file=sys.stderr)
            return completed.returncode

    write_manifest(manifest_path, args, manifest_specs)
    if not args.execute:
        print("Dry run only. Add --execute to start training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
