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
- **Modality 2.** Loader and all three comparison notebooks exist under `notebooks/tahoe/`;
  none has run on a GPU yet. TFG's objective is a global soft k-means (K = 10) fitted in the
  notebook, scored by a k-nearest-neighbour vote. Still wanted: a control metric that needs
  no classifier at all, such as agreement of the generated mean shift from DMSO with the real
  one per drug (the loader exposes `dmso_reference`).
- **Modality 1 dataset.** The DeepFashion notebook in `notebooks/fashion/` uses a larger U-Net
  with attention and mixed precision. Bring it to the course-style U-Net and the shared
  protocol (fixed-noise validation, manifest, seeds) before the comparison arms move to it.
- **Every manifest** now records hardware and training seconds; the paper's compute table
  reads those.
