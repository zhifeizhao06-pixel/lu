# Minimal noise-aware Luminance-GS experiment

The experimental implementation is in `examples/simple_trainer_ours.py` and is
enabled by default. It adds:

1. Per-training-view heteroscedastic noise parameters
   `variance = alpha * intensity + beta`.
2. A Gaussian negative log-likelihood term for the observed low-light branch.
3. Per-Gaussian signal confidence sampled at its projected image position.
4. Confidence-weighted densification, so gradients below the estimated noise
   floor do not create new Gaussians.
5. Cross-view gradient consensus: projected gradients are approximately lifted
   into world space and inconsistent directions are suppressed during growth.
6. Automatic elongation diagnostics at every checkpoint.
7. A needle-only shape regularizer on `s_max / s_mid` that preserves thin
   surface-aligned discs (`s_mid / s_min`).
8. Complete checkpointing for the curve, adjustment networks, per-view color
   parameters, and noise parameters.
9. Empirical Fisher information for Gaussian position and shape, estimated from
   the heteroscedastic low-light likelihood.
10. Parameter-update decoupling: position Fisher gates center updates and shape
    Fisher gates scale/rotation updates, while appearance remains unconstrained.
    Direct Fisher densification is retained only as a disabled ablation.
11. Noise-normalized structure protection restores densification confidence at
    dark edges only when their gradient exceeds the predicted difference-noise
    floor. This targets texture loss caused by intensity-only confidence.
12. Confidence curriculum keeps early densification permissive and smoothly
    introduces the full noise-aware threshold between steps 500 and 4000.

The validated default method enables noise-aware confidence densification and
needle regularization (`5e-4`). Gradient consensus and both Fisher applications
are disabled by default because they reduced validation quality on `buu`.
Single-view structure protection is also disabled: sRGB ISP noise violated the
simple gradient-SNR assumption and substantially reduced perceptual quality.

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
  --no-confidence-densify \
  --no-gradient-consensus \
  --no-needle-regularization
```

Noise-aware experiment:

```bash
python simple_trainer_ours.py \
  --data-dir ../data/LOM_full/buu \
  --exp-name low \
  --result-dir ../results/buu_noise_aware \
  --noise-aware \
  --confidence-densify \
  --gradient-consensus
```

If Tyro reports a different boolean spelling in the installed version, inspect
the exact flags with:

```bash
python simple_trainer_ours.py --help
```

## Useful ablations

- Noise loss only:
  `--noise-aware --no-confidence-densify --no-gradient-consensus
  --no-needle-regularization`
- Confidence without cross-view consensus:
  `--noise-aware --confidence-densify --no-gradient-consensus
  --no-needle-regularization --no-information-guidance`
- Confidence plus needle regularization (recommended next experiment):
  `--noise-aware --confidence-densify --no-gradient-consensus
  --needle-regularization --no-information-guidance`
- Information-guided version:
  `--noise-aware --confidence-densify --no-gradient-consensus
  --needle-regularization --information-guidance
  --information-gradient-gating --no-information-densify`
- Structure-protected main version:
  `--noise-aware --confidence-densify --structure-protection
  --needle-regularization --no-gradient-consensus --no-information-guidance`
- Curriculum main version:
  `--noise-aware --confidence-densify --confidence-curriculum
  --no-structure-protection --needle-regularization
  --no-gradient-consensus --no-information-guidance`
- Original loss and densification:
  `--no-noise-aware --no-confidence-densify --no-gradient-consensus
  --no-needle-regularization`
- Stronger densification filtering:
  `--densify-confidence-min 0.25 --densify-confidence-power 1.5`
- Weaker filtering:
  `--densify-confidence-min 0.05 --densify-confidence-power 0.5`

## What to compare

- PSNR, SSIM, and LPIPS in `stats/val_step*.json`.
- `num_GS` in `stats/train_step*.json`.
- Floaters and needle-like artifacts in trajectory videos.
- `elongation_mean`, `elongation_gt5`, `elongation_gt10`, and
  `elongation_gt20` in `stats/train_step*.json`.
- Prefer `needle_gt5`, `needle_gt10`, `opaque_needle_gt5`, and
  `opaque_needle_gt10` when diagnosing abnormal stretched Gaussians.
  `flat_*` measures thin surface-like discs and should not be counted as
  floaters by itself. `opacity_weighted_needle` reduces the influence of
  nearly invisible outliers.
- `train/noise_alpha`, `train/noise_beta`, and `train/noise_nll` in TensorBoard.
- `train/signal_confidence`, `train/structure_confidence`, and
  `train/densify_confidence` show how much edge evidence restores confidence.
- `information_supported_fraction`, `information_mean`,
  `information_median`, `information_below_min`, `position_gate_mean`, and
  `shape_gate_mean` in checkpoint statistics.

The minimum hypothesis is supported if the noise-aware run reduces floaters or
the number of Gaussians without materially reducing validation quality. Test at
least three scenes before drawing a conclusion.
