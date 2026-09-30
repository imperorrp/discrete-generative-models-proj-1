# To do

- **Move shared notebook code into `src/guidance/`.** The U-Net, cosine schedule, DDIM
  sampler, training loop and text encoder are currently defined inline in each notebook.
  Every new arm copies them. They should be imported so a fix to the sampler reaches all
  arms at once, and so a `base.pt` trained once is loaded by every baseline rather than
  retrained.
- **Give the sampler a `cond_fn` hook** so classifier guidance is a parameter of
  `ddim_sample`, not a separate sampler.
- **Null-label switch in every arm**, so each mechanism can be tested as the only route
  by which the model learns the condition. See `docs/findings.md`. Done for TC-LoRA and
  cross-attention (`label_path` in each notebook's config). TFG already treats the
  null-label model as its unconditional base.
- **Modality 2.** Loader done (`src/guidance/data/cells.py`, on the team's `TahoePCs`). Next:
  run `notebooks/tahoe/tc_lora_cells.ipynb` on Colab; then cross-attention and TFG on cells
  (TFG's objective there is the soft k-means cluster from `condition="cluster"`); a control
  metric that needs no trained classifier, such as agreement of the generated mean shift from
  DMSO with the real one per condition.
- **Modality 1 dataset.** The DeepFashion notebook in `notebooks/fashion/` uses a larger U-Net
  with attention and mixed precision. Bring it to the course-style U-Net and the shared
  protocol (fixed-noise validation, manifest, seeds) before the comparison arms move to it.
- **Every manifest** now records hardware and training seconds; the paper's compute table
  reads those.
