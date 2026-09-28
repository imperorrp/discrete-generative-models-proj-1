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
notebooks/      exploratory work
tests/          correctness checks
results/        run outputs
paper/          LaTeX
docs/           rubric checklist
```

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
