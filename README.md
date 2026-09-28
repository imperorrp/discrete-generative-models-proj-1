# discrete-generative-models-proj-1

Hypernetworks and language models for predicting guidance strengths.

Two continuous modalities: **CIFAR-100 images** (`R^{3×32×32}`) and **single-cell states**
(`R^{50}`, PCA of Tahoe-100M scRNA-seq).

---

## Setup

```bash
conda env create -f environment.yml
conda activate dgm
pip install -e .
```

Or without conda:

```bash
pip install -r requirements.txt && pip install -e .
```

## Repository structure

```
src/guidance/          Library code. Everything reusable lives here.
  data/                Dataset loading and splits
  models/              Backbones (U-Net for images, MLP for cell states)
  training/            Flow-matching and diffusion objectives
  sampling/            Samplers and every guidance rule, including ours
  eval/                Metrics
  utils/               Seeding and run manifests
scripts/               Command-line entry points. Thin wrappers over src/.
configs/               One YAML per experiment. Referenced by name in the paper.
notebooks/             Exploratory notebooks. Not the source of any paper number.
tests/                 Fast correctness checks
results/               Run outputs. Every run writes manifest.json
paper/                 LaTeX. tables/ is generated, never hand-edited
docs/                  Rubric checklist and the baseline/prior-art notes
```

## Data preparation

```bash
python scripts/prepare_data.py --modality images    # downloads CIFAR-100
python scripts/prepare_data.py --modality cells     # builds PCA-50 from Tahoe-100M
```

CIFAR-100 downloads automatically. The cell-state data requires a Hugging Face token; set it as
an environment variable, never in a file:

```bash
export HF_TOKEN=...
```

## Running an experiment

Every experiment is one config. Train, sample, evaluate:

```bash
python scripts/train.py     --config configs/m1_diffusion_base.yaml
python scripts/sample.py    --config configs/m1_diffusion_base.yaml --guidance 3.0
python scripts/evaluate.py  --config configs/m1_diffusion_base.yaml
```

Regenerate every paper table from the run manifests:

```bash
python scripts/make_tables.py --results results --out paper/tables
```

Check the guidance math without a GPU or any data:

```bash
python tests/test_guidance_math.py
```

## Which config produced which table

See [EXPERIMENTS.md](EXPERIMENTS.md). Every table and figure in the paper maps to a config file
and a results directory there.

## Two rules

**Every number in the paper comes from a manifest.** `scripts/make_tables.py` is the only path
from results to LaTeX. If a number is in the paper and not in a `results/*/manifest.json`, it does
not go in the paper.

**Notebooks are for exploring, not for results.** Anything that produces a number in the paper
runs from `scripts/` with a config, so it can be rerun.

## Documentation

| File | Contents |
|---|---|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Every experiment: config, output, which paper table it feeds |
| [docs/rubric.md](docs/rubric.md) | The graded items, with point values |
| [docs/baselines.md](docs/baselines.md) | The three comparison methods and the closest prior work, with verified citations |
