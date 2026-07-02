"""Compare Gaussian shape diagnostics from one or more checkpoints."""

import argparse
import json
from pathlib import Path

import torch


def analyze(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    splats = checkpoint["splats"]
    scales = splats["scales"].float().exp().sort(dim=-1).values
    opacity = splats["opacities"].float().sigmoid()

    elongation = scales[:, 2] / scales[:, 0].clamp_min(1e-8)
    needle = scales[:, 2] / scales[:, 1].clamp_min(1e-8)
    flat = scales[:, 1] / scales[:, 0].clamp_min(1e-8)
    opaque = opacity > 0.1
    opaque_count = opaque.sum().clamp_min(1)

    return {
        "step": int(checkpoint.get("step", -1)),
        "num_GS": len(scales),
        "elongation_mean": elongation.mean().item(),
        "elongation_median": elongation.median().item(),
        "needle_mean": needle.mean().item(),
        "needle_median": needle.median().item(),
        "needle_gt5": (needle > 5).float().mean().item(),
        "needle_gt10": (needle > 10).float().mean().item(),
        "flat_mean": flat.mean().item(),
        "flat_median": flat.median().item(),
        "flat_gt10": (flat > 10).float().mean().item(),
        "opaque_fraction": opaque.float().mean().item(),
        "opaque_needle_gt5": (((needle > 5) & opaque).sum() / opaque_count).item(),
        "opaque_needle_gt10": (((needle > 10) & opaque).sum() / opaque_count).item(),
        "opacity_weighted_needle": (
            (needle * opacity).sum() / opacity.sum().clamp_min(1e-8)
        ).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Checkpoint paths, optionally written as label=path.",
    )
    args = parser.parse_args()

    results = {}
    for item in args.checkpoints:
        if "=" in item:
            label, raw_path = item.split("=", 1)
        else:
            raw_path = item
            label = Path(raw_path).parent.parent.name
        results[label] = analyze(Path(raw_path))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
