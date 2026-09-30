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
  and all sample with ancestral DDPM, η = 1, 250 steps, as the base's own notebook; the three
  comparison notebooks clamp the predicted clean state to each PC's training range (TFG's
  `clip_sample` translated; `CFG_CLIP_X0=0` removes it) because the first sampling step divides
  the denoiser's residual by 5e-5 under this schedule, and every notebook runs a scale check
  that flags rows whose variance leaves the data range. Evaluation pairs are the largest
  held-out pair of each of the six most populous drugs, one pair per drug, so wrong-condition
  rows are a different drug.
  Still wanted: a Fréchet distance in a frozen single-cell foundation-model embedding (scVI,
  Geneformer or scGPT) as the closer analogue of FID, and a classifier-free control metric,
  agreement of the generated mean shift from DMSO with the real shift per drug (the loader
  exposes `dmso_reference`).
- **Modality 2: one base.** All four cell notebooks now share the recipe of `cfg_cells` (their
  MLP, 50 epochs, batch 256, AdamW 2e-4 constant, fp16 autocast, joint condition dropout 0.1,
  diffusers cosine schedule, seed 42), and the three comparison notebooks load
  `checkpoints/diffusion_cfg/best.pt` from a `cfg_cells` run when present (or `CELL_BASE_CKPT`),
  so they sit on literally the same weights. Run `cfg_cells` first on a machine, then the
  others. The image side follows the same rule: `cfg_fashion` trains the base and saves
  `checkpoints/diffusion_fashion/best.pt` (or `FASHION_BASE_CKPT`), the three DeepFashion
  comparison notebooks load it.
- **Modality 2: condition text.** Today the text is drug name, mechanism-of-action class and
  targets, plus cell-line name and organ, from the Tahoe metadata. Richer public descriptions
  (DrugBank or ChEMBL mechanism paragraphs, pathway names, the line's driver mutations from
  `panel_review.json`) would give the language model more to work with; write them before any
  result is seen and never describe observed expression effects. The text encoder is CLIP's;
  a biomedical encoder (PubMedBERT, BioLinkBERT) or a general LLM's embeddings is a one-function
  swap in `build_cond_embeddings` and should be an ablation.
- **Modality 1: DeepFashion.** The four DeepFashion notebooks share their U-Net (64 base
  channels, bottleneck attention), their schedule, their recipe (AdamW 2e-4 constant, fp16, clip 1,
  label dropout 0.1, 50 epochs with early stopping in `cfg_fashion`; the comparison notebooks
  load its checkpoint), one sampler (ancestral DDPM: the DDIM step with η = 1, 50 steps, clean
  estimate clamped to [-1, 1]; `CFG_ETA=0` gives deterministic DDIM everywhere) and their
  evaluation (2048 class-balanced validation images, fine-tuned ResNet-18 scorer, FID,
  top-1 / top-5, precision / recall). Three things to state when the numbers are reported: the
  base notebook as received used batch 16, `cfg_fashion` uses 64 like its flow-matching twin;
  their sampler was deterministic and `cfg_fashion` adds the η term to it; and the comparison
  notebooks' sampler differs from theirs in two details (rounded vs truncated step grid; noise
  recomputed from the clamped clean estimate) that only matter where the clamp engages. Still to do: the text arms condition on per-image captions and TFG on
  the class, because a classifier gives an objective per class and not per caption; a
  caption-level objective for TFG (CLIP image-text similarity) would make the three arms
  condition on the same thing and is the natural next step.
- **Modality 1: the two base notebooks are not a matched pair.** The flow-matching notebook
  (`CIS6720_Proj1_FlowMatchingCode_Modality_1.ipynb`, as received) and the diffusion one differ
  in U-Net, batch size (64 vs 16), weight decay, learning-rate schedule (cosine vs constant),
  EMA (flow matching only) and sampler steps (100 Euler vs 50 DDIM). A diffusion-vs-flow
  comparison on DeepFashion needs one architecture and one recipe for both; only the diffusion
  base is aligned so far.
- **Every manifest** now records hardware and training seconds; the paper's compute table
  reads those.
