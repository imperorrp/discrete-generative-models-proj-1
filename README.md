# discrete-generative-models-proj-1

Hypernetworks and language models for predicting guidance strengths.

Modality 1: CIFAR-100 images. Modality 2: single-cell states (PCA-50 of Tahoe-100M).

## Setup

```bash
conda env create -f environment.yml
conda activate dgm
pip install -e .
```

The cell-state data needs a Hugging Face token: `export HF_TOKEN=...`

## Layout

```
notebooks/      one notebook per method; runs/ holds executed copies with outputs, and log.md
src/guidance/
  baselines/    tc_lora.py, cross_attention.py, tfg.py: one file per comparison method,
                provenance and every deviation from the paper in the header
  data/         cifar.py: loader and the fixed split
  models/ training/ sampling/ eval/ utils/
                planned; shared code is inline in the notebooks for now (docs/todo.md)
scripts/ configs/ tests/
                planned; empty until the notebook code moves into src/
docs/           rubric checklist, findings, to-do
paper/          LaTeX
```

## Status

| | Modality 1 (CIFAR-100) | Modality 2 (cell states) |
|---|---|---|
| Diffusion baseline | done | — |
| Flow-matching baseline | done | — |
| TC-LoRA | 3 runs (runs 1–2 label path on, run 3 off) | notebook ready, awaiting data |
| Text conditioning, cross-attention | 1 run, label path off | — |
| Classifier guidance, TFG | 1 run, their CIFAR-10 settings | — |
| Proposed method | — | — |

## Notebooks

Everything that has run so far ran here. Open a notebook in Colab on a GPU runtime and run it
top to bottom; the first cell clones this repo and prints the commit it is running.

| Notebook | What it does |
|---|---|
| `notebooks/tc_lora_cifar.ipynb` | train base → freeze → train hypernetwork → wrong-text diagnostics → samples → FID |
| `notebooks/text_xattn_cifar.ipynb` | train or load base → attach one cross-attention block → train it → same diagnostics |
| `notebooks/tfg_cifar.ipynb` | load base → guided sampling → accuracy under a second classifier → strength sweep → FID |
| `notebooks/tc_lora_cells.ipynb` | TC-LoRA on cell states; synthetic data until the loader lands |

Each opens with the method's equations, a table from each equation to the cell that
implements it, and what the paper did that the notebook does not. Upload `base.pt` from an
earlier session and base training is skipped, after a check that it was trained under the same
split and schedule. CIFAR-100 loads from a local copy if one is found, otherwise from the
HuggingFace CDN. Every notebook ends by writing a manifest with the commit, the config and
every reported number.

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
