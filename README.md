# discrete-generative-models-proj-1

Hypernetworks and language models for predicting guidance strengths.

Modality 1: images. Method development on CIFAR-100; the dataset for results is DeepFashion
at 64×64 with one caption per image. Modality 2: Tahoe-100M cell states, 25 cell lines × 100
drugs, 50 principal components.

## Setup

```bash
conda env create -f environment.yml
conda activate dgm
pip install -e .
```

Cell-state data: unzip `25lines_100drugs_pca_data_v2.zip` into `data/`; the loader in
`src/guidance/data/cells.py` finds it there, on Colab, or on Drive. Rebuilding that folder from
Tahoe-100M is `notebooks/tahoe/tahoe_colab_downloader_with_pca.ipynb` and needs a Hugging Face
token in the environment (`HF_TOKEN`), never in a file. Image data: unzip the DeepFashion
"Category and Attribute Prediction Benchmark" archive into `data/`, unzip the image archive
(`img/...`) inside that folder, and put `captions.csv` in the folder or next to it;
`src/guidance/data/fashion.py` finds any folder up to two levels under `data/` that holds
`Anno_coarse/` and `Eval/`, or the one `FASHION_ROOT` points to. A fine-tuned scorer from the
flow-matching base notebook (`fashion_resnet18_classifier.pt`) dropped into `checkpoints/` or
`data/` is reused instead of fine-tuning again.

## Layout

```
notebooks/      cifar/, tahoe/, fashion/: one notebook per method and dataset;
                runs/ holds executed copies with outputs, and log.md
src/guidance/
  baselines/    tc_lora.py, cross_attention.py, tfg.py: one file per comparison method,
                provenance and every deviation from the paper in the header
  data/         cifar.py, cells.py, fashion.py: loaders, splits, condition texts and captions
  eval/         cell_metrics.py: distribution metrics and the k-NN scorer for cell states;
                image_metrics.py: the fine-tuned scorer and precision / recall for images
data/           local data, not tracked: the Tahoe PCA folder, DeepFashion
  models/ training/ sampling/ utils/
                planned; shared code is inline in the notebooks for now (docs/todo.md)
scripts/ configs/ tests/
                planned; empty until the notebook code moves into src/
docs/           rubric checklist, findings, to-do
paper/          LaTeX
```

## Status

| | Modality 1 (CIFAR-100, development) | Modality 1 (DeepFashion) | Modality 2 (cell states) |
|---|---|---|---|
| Diffusion baseline | done | notebook ready (`cfg_fashion`), not yet run | 1 run (`cfg_cells`) |
| Flow-matching baseline | done | their notebook as received, not aligned with the diffusion base | — |
| TC-LoRA | 3 runs (runs 1–2 label path on, run 3 off) | notebook ready (captions), not yet run | notebook ready, not yet run |
| Text conditioning, cross-attention | 1 run, label path off | notebook ready (captions), not yet run | notebook ready, not yet run |
| Classifier guidance, TFG | 1 run, their CIFAR-10 settings | notebook ready (fine-tuned classifier), not yet run | notebook ready (per-drug Gaussian classifier), not yet run |
| Proposed method | — | — | — |

## Notebooks

Everything that has run so far ran here. Open a notebook in Colab on a GPU runtime and run it
top to bottom; the first cell clones this repo and prints the commit it is running.

| Notebook | What it does |
|---|---|
| `notebooks/cifar/tc_lora_cifar.ipynb` | train base → freeze → train hypernetwork → wrong-text diagnostics → samples → FID |
| `notebooks/cifar/text_xattn_cifar.ipynb` | train or load base → attach one cross-attention block → train it → same diagnostics |
| `notebooks/cifar/tfg_cifar.ipynb` | load base → guided sampling → accuracy under a second classifier → strength sweep → FID |
| `notebooks/tahoe/cfg_cells.ipynb` | the cell-state base with classifier-free guidance, from `Tahoe_PCA_Diffusion_CFG.ipynb` (kept as received in the same folder); adds held-out-pair evaluation per guidance strength. Run first |
| `notebooks/tahoe/tc_lora_cells.ipynb` | TC-LoRA on cell states: load or train the base → freeze → train hypernetwork → wrong-text diagnostics → per-pair metrics |
| `notebooks/tahoe/text_xattn_cells.ipynb` | cross-attention on cell states: one block after the MLP's input projection, the hidden vector as a single query token; same diagnostics and metrics |
| `notebooks/tahoe/tfg_cells.ipynb` | TFG on cell states: the base told the cell line, a per-drug Gaussian classifier fitted from the training cells supplies the drug; k-nearest-neighbour scorer, strength sweep, per-pair metrics |
| `notebooks/tahoe/tahoe_colab_downloader_with_pca.ipynb`, `Tahoe_dataloader_example.ipynb` | build the Tahoe subset: download, HVGs and PCA on the training split, held-out line-drug pairs; the loader the package reuses |
| `notebooks/fashion/cfg_fashion.ipynb` | the DeepFashion base with classifier-free guidance, from `CIS6720_Proj1_DiffusionCode_Modality_1.ipynb` (kept as received in the same folder); adds the flow-matching notebook's classifier accuracy and precision / recall, and saves the shared base. Run first |
| `notebooks/fashion/tc_lora_fashion.ipynb` | TC-LoRA on DeepFashion, the hypernetwork reading a CLIP embedding of each image's **caption**: load or train the base → freeze → train → wrong-caption diagnostics → accuracy, FID, precision / recall |
| `notebooks/fashion/text_xattn_fashion.ipynb` | cross-attention on DeepFashion: one block after the U-Net's bottleneck attention reading the caption's CLIP tokens; same diagnostics and metrics |
| `notebooks/fashion/tfg_fashion.ipynb` | TFG on DeepFashion: the base with the null label steered toward a class by a fine-tuned ResNet-34; scored by the fine-tuned ResNet-18 every DeepFashion notebook uses; strength sweep, accuracy, FID, precision / recall |

Each opens with the method's equations, a table from each equation to the cell that
implements it, and what the paper did that the notebook does not. Upload `base.pt` from an
earlier session and base training is skipped, after a check that it was trained under the same
split and schedule. On cell states, run `cfg_cells.ipynb` first: it trains the base once, with
the recipe every cell notebook shares (their MLP, 50 epochs, constant learning rate, fp16), and
the three comparison notebooks load its `checkpoints/diffusion_cfg/best.pt`, or the path in
`CELL_BASE_CKPT`, so all four sit on the same weights; if it is absent they train the same recipe
themselves. All four evaluate the same held-out line-drug pairs, the largest pair of each of the
six most populous drugs (one pair per drug: the six largest pairs outright are five lines of one
drug), with 400 samples each and the same metrics. The three comparison notebooks clamp the
predicted clean state to each principal component's training range while sampling, TFG's
`clip_sample` translated to cells: under the base's cosine schedule the first sampling step
divides the denoiser's residual by 5e-5, and without the clamp a slightly imprecise base, or any
guidance through that estimate, leaves the data range. `CFG_CLIP_X0=0` turns it off; every
manifest records the setting, how often it engaged, and a scale check that flags any row whose
variance is more than five times that of the real cells. DeepFashion works the same way: `cfg_fashion.ipynb` trains the base
(their U-Net and recipe) and saves `checkpoints/diffusion_fashion/best.pt`, or the path in
`FASHION_BASE_CKPT`; the three comparison notebooks load it, fine-tune (once, cached) the
ResNet-18 scorer with the flow-matching notebook's recipe, and all four sample with the same
ancestral DDPM sampler (50 steps, `CFG_ETA=0` for deterministic DDIM) and evaluate 2048 generated
images against the same 2048 class-balanced validation images with the same scorer. CIFAR-100
loads from a local copy if one is found, otherwise from the HuggingFace CDN. Every notebook ends by writing a manifest with the commit, the config, the
hardware, training time and every reported number.

Any config field can be set from the environment without editing a cell, for headless runs
or a different GPU: `CFG_BASE_EPOCHS=1 CFG_LIMIT=5000`, `CFG_LABEL_PATH=true`,
`CFG_SWEEP=0,0.5,1`. `N_FID=10000` turns on FID in the CIFAR notebooks; the DeepFashion
notebooks always evaluate `CFG_N_EVAL` images (2048 by default). `CFG_QUICK=1` (or
`%env CFG_QUICK=1` in a cell above the config) runs the DeepFashion notebooks on a class-balanced
10k of the 209k training images with 8 base and 3 adapter epochs and 256 evaluation images: it
shows that everything works in well under an hour, and its numbers are not results. Data and checkpoint locations:
`TAHOE_PCA_DIR`, `FASHION_ROOT`, `CELL_BASE_CKPT`, `FASHION_BASE_CKPT`, `FASHION_CKPT_DIR`.
Headless:

```bash
CFG_BASE_EPOCHS=8 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  notebooks/tahoe/tc_lora_cells.ipynb --output-dir notebooks/runs --output tc_lora_cells_run1_$(git rev-parse --short HEAD).ipynb
```

## Runs

Executed notebooks are saved with their outputs under `notebooks/runs/`, named
`<notebook>_run<N>_<commit>.ipynb`. The commit is the version of the code that produced the
run; it is also printed in the first cell and recorded as `code_version` in the manifest.
`notebooks/runs/log.md` says what each run was for and what changed before it.

To see what changed between two runs:

```bash
git diff e7448aa 1aff1df -- src/ notebooks/tc_lora_cifar.ipynb
```

Results so far. Single seed; 30 base epochs, 15 adapter epochs, 50 DDIM steps. Validation MSE
is the noise-prediction loss on the held-out split with identical fixed noise for every row;
"wrong text" is the same pass with every class description shifted to a different class. FID
is against the 5000-image validation split with 10000 generated images, which biases it
upward. Base models differ between sessions, so compare each arm to the base in its own run.

| | label path | val MSE | wrong text | FID | run |
|---|---|---|---|---|---|
| base, true label | on | 0.0619 | — | 76.6 | `tc_lora_cifar_run2` |
| TC-LoRA | on | 0.0618 | — | 77.4 | `tc_lora_cifar_run2` |
| base, null label | off | 0.0621 | — | 78.3 | `text_xattn_cifar_run1` |
| cross-attention | off | 0.0618 | 0.0624 (+0.92%) | 75.7 | `text_xattn_cifar_run1` |
| base, null label | off | 0.0621 | — | 80.4 | `tc_lora_cifar_run3` |
| TC-LoRA | off | 0.0617 | 0.0622 (+0.84%) | 81.6 | `tc_lora_cifar_run3` |

| | condition accuracy | FID | run |
|---|---|---|---|
| base, null label | 1.5% | 80.8 | `tfg_cifar_run1` |
| base, true label | 4.5% | — | `tfg_cifar_run1` |
| TFG, ρ = 1, μ = 0.25 | 3.5% | 151.1 | `tfg_cifar_run1` |

Accuracy is under a classifier the sampler never used, on 200 samples; chance is 1%. What
these numbers mean is in `docs/findings.md`.

## Credits

Comparison methods implemented in `src/guidance/baselines/`. Where the authors released
code we use it and say so; where they did not, the file is our reimplementation from the
paper and its deviations are listed in the module docstring.

| File | Paper | Source used |
|---|---|---|
| `tc_lora.py` | Cho, Ohana, Jacobsen, Jothi, Chen, Mao, Can. *TC-LoRA: Temporally Modulated Conditional LoRA for Adaptive Diffusion Control.* NeurIPS 2025 SpaVLE workshop. arXiv:2510.09561 | No code released (checked 2026-09-28). Reimplemented from Eq. 1 and Appendices A–B. Seven deviations documented in the file header. |
| `cross_attention.py` | Chen et al. *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.* ICLR 2024. arXiv:2310.00426 | [Official code](https://github.com/PixArt-alpha/PixArt-alpha) read; `MultiHeadCrossAttention` and its zero-init re-expressed in plain PyTorch for a conv U-Net. Five deviations in the file header. |
| `tfg.py` | Ye, Lin, Han, Xu, Liu, Liang, Ma, Zou, Ermon. *TFG: Unified Training-Free Guidance for Diffusion Models.* NeurIPS 2024. arXiv:2409.15761 | [Official code](https://github.com/YWolfeee/Training-Free-Guidance) read; `guide_step` re-expressed against our DDIM sampler with their schedules and their CIFAR-10 hyperparameters as defaults. Six deviations in the file header. |

Pretrained CIFAR-100 classifiers are from
[chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models),
loaded via `torch.hub`, not retrained.

MIT License.
