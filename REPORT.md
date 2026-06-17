# Report — Lensless Computational Imaging

This report describes the experiments, compares the methods, and answers the
questions from the homework. Empirical numbers, training-log links, and figures
(metric curves, reconstructed images) are produced by the W&B / Comet ML runs and
are referenced in the [Results](#results) section.

## 1. Experiments

We implement and compare five reconstruction approaches, all expressed as a
single modular pipeline (optional pre-processor → camera inversion → optional
post-processor):

1. **ADMM-100** — classic ADMM with 100 iterations and fixed hyperparameters
   (`mu1 = mu2 = mu3 = 1e-4`, `tau = 2e-4`). No training.
2. **Unrolled ADMM-20** — 20 ADMM iterations with the per-iteration penalty
   parameters and TV threshold learned end-to-end (80 trainable parameters).
3. **Modular Pre+Post (8M)** — pre-processor DRUNet (~4.1M) + 5-iteration
   unrolled ADMM + post-processor DRUNet (~4.1M).
4. **Modular Pre (8M)** — pre-processor DRUNet (~8.2M) + 5-iteration unrolled
   ADMM.
5. **Modular Post (8M)** — 5-iteration unrolled ADMM + post-processor DRUNet
   (~8.2M).

All learned models are trained on the `train` split of
DigiCam-Mirflickr-MultiMask-10K with the `MSE + LPIPS(VGG)` loss on the ROI,
Adam (`lr = 1e-4`), batch size 4, and evaluated on the `test` split. ADMM-100 is
only evaluated.

## 2. How each method differs from the previous one

**ADMM-100 vs. nothing.** A model-based solver that minimizes a data-fidelity
term plus an anisotropic-TV prior with non-negativity constraints. *Benefits:*
no training data, interpretable, uses the known physics (PSF). *Drawbacks:*
slow (100 iterations), sensitive to model mismatch and to the hand-picked
hyperparameters, and it produces reconstruction artifacts.

**Unrolled ADMM-20 vs. ADMM-100.** The same algorithm but unrolled into 20
differentiable layers whose hyperparameters are learned. *Benefits:* far fewer
iterations (≈5× faster) while matching or exceeding the quality of converged
ADMM, because the penalties are tuned to the data instead of by hand.
*Drawbacks:* needs training data; only 80 parameters, so it cannot correct
model mismatch or perform perceptual enhancement — it is still essentially a
data-fidelity reconstruction.

**Modular Post vs. Unrolled ADMM.** Adds a learned post-processor (DRUNet) after
a short 5-iteration inversion. *Benefits:* the U-Net denoises the inversion
output, removes artifacts, fixes colors, and optimizes the perceptual (LPIPS)
quality, giving a large jump in SSIM/LPIPS. *Drawbacks:* the post-processor has
to clean up an already noise-amplified inversion (camera inversion amplifies
measurement noise), so it carries the full denoising burden alone; more
parameters and slower than pure ADMM.

**Modular Pre vs. Modular Post.** Moves all the learned capacity in front of the
inversion. *Benefits:* the pre-processor denoises the measurement *before* the
inversion amplifies the noise, which is theoretically the right place to act
(Section III of the modular paper). *Drawbacks:* with only a pre-processor and
no post-processor, there is no module to perform the final perceptual
enhancement / color correction of the inversion output, so it typically lags the
pre+post split on perceptual metrics.

**Modular Pre+Post vs. Pre / Post only.** Splits the same ~8M parameter budget
between a pre- and a post-processor. *Benefits:* the pre-processor reduces the
noise that the inversion would amplify, and the post-processor then only has to
solve an easier denoising/enhancement task — the paper shows this split
consistently beats putting all parameters on one side. *Drawbacks:* two networks
to run, so it is the slowest of the learned models.

## 3. Results

> Fill in with the numbers from your training runs (≈6 h of training on Kaggle).
> The metrics below are computed on the ROI of the `test` split with
> `inference.py` + `calculate_metrics.py`. Speed is measured with
> `measure_speed.py`.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MSE ↓ | Speed (ms) |
|---|---|---|---|---|---|
| ADMM-100 | | | | | |
| Unrolled ADMM-20 | | | | | |
| Modular Post (8M) | | | | | |
| Modular Pre (8M) | | | | | |
| Modular Pre+Post (8M) | | | | | |

Expected trends (from the papers and the design analysis above):

- ADMM-100 ≳ Unrolled ADMM-20 on data fidelity, but the unrolled model reaches
  similar quality with 5× fewer iterations.
- The modular models clearly outperform both ADMM variants on SSIM and LPIPS.
- Splitting parameters between pre- and post-processors (Pre+Post) gives the
  best perceptual quality.
- Inference speed: ADMM-100 is by far the slowest (100 FFT-based iterations);
  the learned models (5–20 iterations) are much faster, with Pre+Post the
  slowest learned model (two U-Nets).

**Training logs / curves / reconstructed images:** see the W&B / Comet ML
report (link here). The runs log the loss, PSNR/SSIM/LPIPS/MSE curves, and
reconstruction examples (measurement vs. ground truth vs. reconstruction) every
epoch.

## 4. What worked and what did not

- **Worked:** the Fourier-domain ADMM with `2H × 2W` padding and a centered
  scene; reusing the FFT diagonalization for the `x`-update; caching one PSF per
  unique mask (essential for a feasible training time); the modular formulation
  that lets a single model class express all five methods.
- **Did not / was tricky:** PSF simulation is expensive and must be cached; the
  raw ADMM output has an arbitrary scale, so per-image max-normalization before
  the post-processor (and for metrics) is needed; very small fixed penalties
  make classic ADMM converge slowly, motivating the learned variants.

## 5. Major challenges

- Handling the crop operator efficiently: working on a padded space and folding
  `C^T C` into a diagonal mask for the closed-form `v`-update.
- Keeping the finite-difference TV operator and the PSF convolution consistent
  in the same FFT basis so that the `x`-update inverse is exact.
- Matching the parameter budget of the paper (~8.2M and ~4.1M DRUNet variants)
  by choosing the right channel widths.
- Fitting the unrolled models (with autograd through many FFT iterations) and the
  two U-Nets into GPU memory within the training-time budget.

## 6. Reconstruction speed

`measure_speed.py` times the forward pass of each model on a single
`3 × 380 × 507` measurement (with warm-up). Run, for example:

```bash
python3 measure_speed.py model=admm100
python3 measure_speed.py model=unrolled_admm20
python3 measure_speed.py model=modular_post
python3 measure_speed.py model=modular_pre
python3 measure_speed.py model=modular_pre_post
```

Report the per-method numbers in the table above.
