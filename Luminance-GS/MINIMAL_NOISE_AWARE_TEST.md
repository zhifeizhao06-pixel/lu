# Minimal noise-aware Luminance-GS experiment

The experimental implementation is in `examples/simple_trainer_ours.py` and is
enabled by default. It adds:

1. Per-training-view heteroscedastic noise parameters
   `variance = alpha * intensity + beta`.
2. A Gaussian negative log-likelihood term for the observed low-light branch.
3. Per-Gaussian signal confidence sampled at its projected image position.
4. Confidence-weighted densification, so gradients below the estimated noise
   floor do not create new Gaussians.
5. Complete checkpointing for the curve, adjustment networks, per-view color
   parameters, and noise parameters.

## A/B commands

Run from `examples/`. Keep every option and random seed identical except the
two switches below.

Baseline:

```bash
python simple_trainer_ours.py \
  --data-dir ../data/LOM_full/buu \
  --exp-name low \
  --result-dir ../results/buu_baseline \
  --no-noise-aware \
  --no-confidence-densify
```

Noise-aware experiment:

```bash
python simple_trainer_ours.py \
  --data-dir ../data/LOM_full/buu \
  --exp-name low \
  --result-dir ../results/buu_noise_aware \
  --noise-aware \
  --confidence-densify
```

If Tyro reports a different boolean spelling in the installed version, inspect
the exact flags with:

```bash
python simple_trainer_ours.py --help
```

## Useful ablations

- Noise loss only: `--noise-aware --no-confidence-densify`
- Original loss and densification: `--no-noise-aware --no-confidence-densify`
- Stronger densification filtering:
  `--densify-confidence-min 0.25 --densify-confidence-power 1.5`
- Weaker filtering:
  `--densify-confidence-min 0.05 --densify-confidence-power 0.5`

## What to compare

- PSNR, SSIM, and LPIPS in `stats/val_step*.json`.
- `num_GS` in `stats/train_step*.json`.
- Floaters and needle-like artifacts in trajectory videos.
- `train/noise_alpha`, `train/noise_beta`, and `train/noise_nll` in TensorBoard.

The minimum hypothesis is supported if the noise-aware run reduces floaters or
the number of Gaussians without materially reducing validation quality. Test at
least three scenes before drawing a conclusion.
