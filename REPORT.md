# Report — Lensless Computational Imaging

This report describes the experiments, compares the methods, and answers the
questions from the homework. Empirical numbers, training-log links, and figures
(metric curves, reconstructed images) are produced by the Comet ML runs and
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

**Training logs / curves / reconstructed images:** see the Comet ML
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

## 7. Bonus tasks

### 7.1 A non-ADMM solver: FISTA (1.0 pt)

We implement FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) from scratch
in [`src/model/fista.py`](src/model/fista.py), in both a fixed
([`fista`](src/configs/model/fista.yaml), 100 iters) and a learnable unrolled
([`fista_unrolled`](src/configs/model/fista_unrolled.yaml), 20 iters) variant. It
minimizes the *same* objective as our ADMM — data fidelity
`0.5 ||C H x - b||^2`, an anisotropic-TV prior, and a non-negativity
constraint — and uses the *same* padded Fourier forward model, so the comparison
is apples-to-apples.

**How it differs from ADMM.** ADMM uses variable splitting: it introduces
auxiliary variables (one for the data/crop term, one for the TV term, one for
non-negativity) and dual variables, and each iteration solves these subproblems
in closed form — in particular the `x`-update is a single Fourier-domain division
because every quadratic term diagonalizes. FISTA is an *accelerated proximal
gradient* method: it takes one gradient step on the smooth data term (with the
Lipschitz step size `1/L`, where `L = max |FFT(psf)|^2`), applies the proximal
operator of the regularizer, and adds Nesterov momentum (`y_k`) for `O(1/k^2)`
convergence. There are no dual variables and no penalty parameters. Because the
anisotropic-TV prox has no closed form, we solve it with a few Chambolle dual
iterations, and the non-negativity constraint is applied by clamping after the
TV prox.

**Pros.** Simpler and lighter than ADMM: only two state tensors (`x`, `y`) instead
of three splits plus three duals, and only one real hyperparameter (the TV weight
`tau`) — the step size is set automatically from the PSF, whereas ADMM needs three
penalty parameters `mu1, mu2, mu3`. The acceleration gives a fast `O(1/k^2)`
objective decrease.

**Cons.** FISTA is most natural for *one* smooth term plus *one* prox. With two
non-smooth terms (TV *and* non-negativity) the combination is not exact — we
approximate the joint prox by clamping after the TV prox, whereas ADMM handles
each term exactly through its own split. The TV prox itself needs an inner loop
(extra compute per iteration). Finally, the crop operator sits inside the data
term, so every gradient step needs a full forward+adjoint convolution; ADMM turns
the crop into a closed-form diagonal update. For ill-conditioned PSFs the `1/L`
step is small and convergence slows, where ADMM's penalties act as a
preconditioner.

### 7.2 Pretrained restoration on top of ADMM-100: Real-ESRGAN (0.5 pt)

We run a general-purpose, pretrained restoration GAN — Real-ESRGAN — on top of
the fixed (non-unrolled) ADMM-100 reconstruction
([`admm100_realesrgan`](src/configs/model/admm100_realesrgan.yaml)). The RRDBNet
generator is reimplemented in [`src/model/realesrgan.py`](src/model/realesrgan.py)
with the official layer names, and the published pretrained weights
(`RealESRGAN_x4plus`, 16.7M parameters) are loaded into it. The network upscales
the ADMM output by 4x and we resize back to the original resolution, so it acts
as a same-resolution restorer; it is **frozen** for this task.

**How it differs.** Unlike the modular learned methods, here the inversion is the
classical ADMM-100 (fixed) and the learned component is a *generic* restoration
network that was *not* trained for lensless imaging and is *not* unrolled with the
inversion. It is a pure post-hoc enhancement using a generative image prior.

**Pros.** No training required; it reuses a powerful prior learned from millions
of natural images, so it can sharpen edges and synthesize plausible texture,
typically improving the perceptual metric (LPIPS). It is trivial to attach to any
reconstruction.

**Cons.** Domain mismatch: Real-ESRGAN was trained to undo real-photo
degradations (blur, noise, JPEG), not lensless/ADMM artifacts (color casts,
ringing, low contrast), so frozen it may fail to remove lensless-specific
artifacts and can *hallucinate* detail that is not in the scene. Such
hallucinations hurt fidelity metrics (PSNR, MSE) even when they look sharper. It
is also a large, slow model, and the 4x upscale-then-downscale is wasteful.

### 7.3 Fine-tuning the restoration network (extra 1.0 pt)

We fine-tune the Real-ESRGAN generator on the training split
([`admm100_realesrgan_ft`](src/configs/model/admm100_realesrgan_ft.yaml)),
unfreezing its parameters and training with the same `MSE + LPIPS` ROI loss as the
other learned models, using the ADMM-100 output as input. The inversion stays
fixed and is run without gradients, so only the generator is optimized. Training
uses a small learning rate (`1e-5`) to adapt the prior without destroying it:

```bash
python3 train.py -cn=lensless model=admm100_realesrgan_ft \
    optimizer.lr=1e-5 trainer.n_epochs=10 dataloader.batch_size=2 \
    writer.run_name=realesrgan_ft
```

**How it differs / pros.** Fine-tuning closes the domain gap of Section 7.2: the
generator learns the actual ADMM-100 artifact distribution, so it should improve
*both* fidelity and perceptual quality over the frozen model, while still
benefiting from the pretrained initialization (faster, better than training such
a large network from scratch on limited data).

**Cons.** Training is slow because ADMM-100 is recomputed every step (100
FFT-based iterations per batch, even though no gradient flows through it). The
network is large (16.7M parameters) and can overfit or drift away from the
generative prior if over-trained or the learning rate is too high; depending on
the loss weights it may also regress toward MSE-style smoothing and lose the GAN's
sharpness.

### 7.4 Bonus results

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MSE ↓ | Speed (ms) |
|---|---|---|---|---|---|
| FISTA-100 (fixed) | | | | | |
| Unrolled FISTA-20 | | | | | |
| ADMM-100 + Real-ESRGAN (frozen) | | | | | |
| ADMM-100 + Real-ESRGAN (fine-tuned) | | | | | |

Expected trends (from the analysis above):

- FISTA-100 should be close to ADMM-100 on fidelity (same objective), possibly a
  little lower because of the approximate non-negativity handling; unrolled
  FISTA-20 should track unrolled ADMM-20.
- Frozen Real-ESRGAN on ADMM-100 should improve perceptual sharpness (LPIPS) but
  may not improve — or may even hurt — PSNR/MSE due to hallucination and domain
  mismatch.
- Fine-tuned Real-ESRGAN should beat the frozen version on all metrics and become
  competitive with the modular post-processor model.
