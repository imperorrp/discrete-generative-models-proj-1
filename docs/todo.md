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
- **Modality 2: evaluation.** All four Tahoe notebooks now call `src/guidance/eval/cell_metrics.py`
  (MMD, energy distance, Fréchet distance on the PCs, precision, recall, variance ratio, and the
  k-nearest-neighbour drug scorer with its accuracy on real validation cells as the ceiling),
  and all sample with ancestral DDPM, η = 1, 250 steps, no clamp, as the base's own notebook.
  Still wanted: a Fréchet distance in a frozen single-cell foundation-model embedding (scVI,
  Geneformer or scGPT) as the closer analogue of FID, and a classifier-free control metric,
  agreement of the generated mean shift from DMSO with the real shift per drug (the loader
  exposes `dmso_reference`).
- **Modality 2: base training budget and sharing.** The teammate's base ran 50 epochs at batch 256
  (validation MSE 0.2557, still improving). The comparison notebooks default to 30. Train once
  and share `base.pt` across the four notebooks; `cfg_cells` keeps its own `checkpoints/diffusion_cfg/best.pt`
  format, so a converter or a common format is needed before that is one file.
- **Modality 2: condition text.** Today the text is drug name, mechanism-of-action class and
  targets, plus cell-line name and organ, from the Tahoe metadata. Richer public descriptions
  (DrugBank or ChEMBL mechanism paragraphs, pathway names, the line's driver mutations from
  `panel_review.json`) would give the language model more to work with; write them before any
  result is seen and never describe observed expression effects. The text encoder is CLIP's;
  a biomedical encoder (PubMedBERT, BioLinkBERT) or a general LLM's embeddings is a one-function
  swap in `build_cond_embeddings` and should be an ablation.
- **Modality 1 dataset.** The DeepFashion notebook in `notebooks/fashion/` uses a larger U-Net
  with attention and mixed precision. Bring it to the course-style U-Net and the shared
  protocol (fixed-noise validation, manifest, seeds) before the comparison arms move to it.
- **Every manifest** now records hardware and training seconds; the paper's compute table
  reads those.
