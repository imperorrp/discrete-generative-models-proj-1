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

## Planned

- `tc_lora_cifar` run 3: label path off, same diagnostics as the cross-attention run, so
  the two mechanisms compare head to head. Reuses `base.pt` from the cross-attention run
  if uploaded, otherwise retrains it with the same seed.
- `tfg_cifar` run 1: sampling only, on `base.pt`.
