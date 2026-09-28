# To do

- **Move shared notebook code into `src/guidance/`.** The U-Net, cosine schedule, DDIM
  sampler, training loop and text encoder are currently defined inline in each notebook.
  Every new arm copies them. They should be imported so a fix to the sampler reaches all
  arms at once, and so a `base.pt` trained once is loaded by every baseline rather than
  retrained.
- **Give the sampler a `cond_fn` hook** so classifier guidance is a parameter of
  `ddim_sample`, not a separate sampler.
- **Null-label switch in every arm**, so each mechanism can be tested as the only route
  by which the model learns the condition. See `docs/findings.md`.
- **Modality 2 loader.** Nothing on the cell-state side runs without it.
