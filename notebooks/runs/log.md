# Run log

One entry per executed notebook in this folder: what the run was for, what changed since
the previous run of that notebook, and the result in a line. Numbers are in the notebooks.

## tc_lora_cifar_run1_f4e6982

First end-to-end run on an A100. Base U-Net 30 epochs, hypernetwork 15 epochs, FID at
10k samples. Purpose: does the pipeline run, and does the hypernetwork change anything.

Result: FID 80.6 (base) vs 79.7 (TC-LoRA). The validation numbers are not comparable. The
base was validated with the training loss, which applies 10% label dropout, and TC-LoRA
without it, so TC-LoRA's "best val 0.0587 at epoch 4" is an artifact of that.

## tc_lora_cifar_run2_e7448aa

Changed since run 1 (`d891abb`, `e7448aa`): both arms are validated fully conditionally on
the same fixed noise and timesteps; the hypernetwork's timestep input is scaled by 1000
before the sinusoidal embedding, since it had been nearly blind to t; the code version is
printed in the first cell and written to the manifest. The adapter-magnitude cell
(`e0bfedf`) was run in the same session afterwards and its output is in the saved file.

Result: val 0.0619 (base) vs 0.0618 (TC-LoRA); FID 76.6 vs 77.4; adapter magnitude
‖BA‖/‖W‖ = 0.11. The adapter moved and it made no difference. A text embedding of the
class name is redundant with the class label the backbone already receives. See
`docs/findings.md`.

## text_xattn_cifar_run1 (code `429ead4`)

First run of the cross-attention arm. Label path off: the backbone gets the null label and
the text is the only route to the class, following the run-2 finding. Base retrained in
this session with the same seed and split as run 2. Wrong-text diagnostic uses fixed label
offsets (1, 37, 73) so the correct and wrong passes see identical noise.

Result: wrong text costs +0.92% loss over correct text; paired samples differ by 19.9%;
FID 78.3 (unconditional base) vs 75.7 (text). Text alone recovers what the label gives
(0.06184 vs 0.06196). Sampled with the DDIM clipping convention from before `4fd05e1`.

## tc_lora_cifar_run3_1aff1df

Changed since run 2 (`4fd05e1`, `1aff1df`): label path off, so the backbone gets the null
label and the hypernetwork's text embedding is the only route to the class; the same four-row
matched validation, wrong-text pass (offsets 1, 37, 73), per-noise-level table and paired
sampling as the cross-attention notebook; DDIM recomputes the noise from the clipped estimate.
Base retrained in this session (same seed and split; val 0.0619 at epoch 30).

Result: val 0.0621 (base, null label) / 0.0619 (base, true label) / 0.0617 (TC-LoRA, correct
text) / 0.0622 (wrong text, all three offsets). Text effect +0.84%; paired samples differ by
12.4%; adapter magnitude 0.11. FID 80.4 (base) vs 81.6 (TC-LoRA). The adapter reads the text
and matches the label on the loss, but does not improve FID. Head-to-head with
`text_xattn_cifar_run1` in `docs/findings.md`.

## tfg_cifar_run1_3c38c51

First run of the TFG arm; sampling only on a base retrained in the session (val 0.0619).
Both classifiers 99.9% on real validation images, so the input conversion is right. ρ = μ = 0
reproduced plain DDIM exactly. Their CIFAR-10 settings, unchanged.

Result, 200 samples, scored by VGG-16-BN: accuracy 1.5% unconditional, 4.5% label-conditioned
base, 3.5% TFG; the wrong-target row scores 5.5% against its target and 1.0% against the
original class, so guidance does follow its target. Sweep over strength peaks at 4.5% at half
the default and falls above it. FID 80.8 (unconditional) vs 151.1 (TFG); saturated pixels 8%
vs 23%. Cost per sample 50 forward + 50 backward denoiser passes and 250 classifier calls,
12× the wall-clock of plain DDIM. At these settings TFG steers weakly and destroys sample
quality. The label-conditioned base itself reaches only 4.5%, so the base model is the
bottleneck. Notebook saved by Colab into `notebooks/run/`; moved here and renamed.

## Planned

- `tfg_cifar` run 2: smaller ρ, μ (the sweep peaked at half the defaults), more samples per
  class for the accuracy rows, FID at the chosen setting.
- Classifier-free guidance sweep on the base model to fix `w*`, then the controller.
