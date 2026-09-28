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
src/guidance/   library code
  data/         datasets and splits
  models/       U-Net (images), MLP (cell states)
  training/     flow matching and diffusion objectives
  sampling/     samplers and guidance rules
  eval/         metrics
  utils/        seeding, run manifests
scripts/        command-line entry points
configs/        one YAML per experiment
notebooks/      one notebook per experiment; runs/ holds executed copies with outputs
tests/          correctness checks
results/        run outputs
paper/          LaTeX
docs/           rubric checklist
```

## Status

| | Modality 1 (CIFAR-100) | Modality 2 (cell states) |
|---|---|---|
| Diffusion baseline | done | — |
| Flow-matching baseline | done | — |
| TC-LoRA | run once (`notebooks/runs/`) | notebook ready, awaiting data |
| Classifier guidance | — | — |
| Baseline 3 | — | — |
| Proposed method | — | — |

## Usage

```bash
python scripts/prepare_data.py --modality images
python scripts/train.py        --config configs/m1_diff_base.yaml
python scripts/sample.py       --config configs/m1_diff_base.yaml --guidance 3.0
python scripts/evaluate.py     --config configs/m1_diff_base.yaml
python scripts/make_tables.py  --results results --out paper/tables
```

```bash
python tests/test_guidance_math.py
```

Each run writes `results/<id>/manifest.json`. `make_tables.py` reads only manifests.

## Credits

Comparison methods implemented in `src/guidance/baselines/`. Where the authors released
code we use it and say so; where they did not, the file is our reimplementation from the
paper and its deviations are listed in the module docstring.

| File | Paper | Source used |
|---|---|---|
| `tc_lora.py` | Cho, Ohana, Jacobsen, Jothi, Chen, Mao, Can. *TC-LoRA: Temporally Modulated Conditional LoRA for Adaptive Diffusion Control.* NeurIPS 2025 SpaVLE workshop. arXiv:2510.09561 | No code released (checked 2026-09-28). Reimplemented from Eq. 1 and Appendices A–B. Seven deviations documented in the file header. |

Pretrained CIFAR-100 classifiers are from
[chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models),
loaded via `torch.hub`, not retrained.
