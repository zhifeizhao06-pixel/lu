# Supplementary experiments

This revision adds reproducible controls for the experiments requested by the
paper review: three random seeds, alternative noise models, curriculum forms,
parameter sensitivity, and efficiency reporting.

## What the trainer now records

Each final `train_step*.json` contains the Gaussian count, needle statistics,
training wall-clock time, peak CUDA memory, seed, noise model, and schedule.
Each `val_step*.json` contains PSNR, SSIM, LPIPS, rendering time/FPS, peak CUDA
memory, checkpoint size, seed, and Gaussian count. Existing keys such as
`ellipse_time` and `mem` are retained so older comparison scripts still work.

The `poisson_read` option is a practical heteroscedastic Gaussian approximation
whose variance is evaluated from the detached rendered intensity. It is not an
exact raw-sensor Poisson likelihood, because LOM images are processed sRGB.

## Recommended order

Run commands from `Luminance-GS/examples` on the server. Start with the core
ablation; it is the most important evidence for the paper.

```bash
python run_supplementary_experiments.py \
  --suite core \
  --data-root /home/zhaozhifei/Aleth-NeRF/data/LOM_full \
  --output-root ../results_supplementary \
  --execute --resume
```

This runs five configurations on five scenes with seeds 0, 1, and 2:

1. baseline;
2. noise likelihood only;
3. noise likelihood plus confidence densification;
4. confidence densification plus shape regularization;
5. the full model with confidence curriculum.

Then run the focused supplementary suites:

```bash
python run_supplementary_experiments.py --suite noise \
  --data-root /home/zhaozhifei/Aleth-NeRF/data/LOM_full \
  --output-root ../results_supplementary --execute --resume

python run_supplementary_experiments.py --suite schedule \
  --data-root /home/zhaozhifei/Aleth-NeRF/data/LOM_full \
  --output-root ../results_supplementary --execute --resume

python run_supplementary_experiments.py --suite sensitivity \
  --data-root /home/zhaozhifei/Aleth-NeRF/data/LOM_full \
  --output-root ../results_supplementary --execute --resume

python run_supplementary_experiments.py --suite efficiency \
  --data-root /home/zhaozhifei/Aleth-NeRF/data/LOM_full \
  --output-root ../results_supplementary --execute --resume
```

Without `--execute`, the launcher only prints commands and writes the manifest.
Use `--scenes chair sofa`, `--seeds 0`, or `--max-runs 2` for a smoke test.
Evaluation images and trajectory videos are disabled during sweeps by default;
add `--save-eval-images` and/or `--render-trajectory` for selected qualitative
runs.

## Aggregate the results

```bash
python summarize_supplementary.py \
  --manifest ../results_supplementary/run_manifest.json
```

The command writes `summary.csv`, `summary.md`, and three Overleaf-ready
tables (`summary.tex`, `region_summary.tex`, and `efficiency_summary.tex`)
under `../results_supplementary/summary`. Values are reported as
mean plus/minus sample standard deviation across available seeds. For the
dataset-level `MEAN` row, quality and ratio metrics are averaged over scenes,
whereas Gaussian count, checkpoint size, and total training time are summed
over scenes for each seed before seed statistics are computed.

## Paired significance tests

After the core suite has complete baseline and method runs, compute paired
bootstrap confidence intervals and a two-sided sign-flip randomization test:

```bash
python compare_significance.py \
  --manifest ../results_supplementary/run_manifest.json
```

This creates `significance.csv` and `significance.md`. Pairs are matched by
scene and random seed. A positive delta means the candidate value is larger;
remember that lower LPIPS, Gaussian count, Needle10, and OpaqueN10 are better.
The validation JSON also reports PSNR separately for dark, mid-tone, and bright
ground-truth pixels, using configurable thresholds `--eval-dark-threshold` and
`--eval-bright-threshold`.

## Experiments that still require external data or evaluators

The launcher covers all method-internal ablations. The following reviewer
requests require additional assets and should not be represented as completed
until those assets are available: an external low-light dataset, standard 3DGS
baselines re-run under the same split, and geometry evaluation against ground
truth depth/point clouds (Chamfer distance, F-score, or depth error). The saved
checkpoints and per-run manifest make those additions traceable later.
