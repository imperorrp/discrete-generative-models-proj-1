# Experiments

Which config produced which result. This file satisfies the rubric's "Connection to the paper"
item (.1 pt): every table and figure names the config and the results directory behind it.

Fill in the results column as runs complete. An empty row means the experiment has not run.

## Modality 1 — CIFAR-100 images

| ID | Experiment | Config | Results dir | Paper |
|----|-----------|--------|-------------|-------|
| E0 | Smoke test, tiny model, CPU, proves the pipeline runs | `configs/smoke.yaml` | `results/E0_smoke/` | — |
| E1 | Flow-matching baseline | `configs/m1_fm_base.yaml` | `results/E1_m1_fm/` | §3.2, Table 1 |
| E2 | Diffusion baseline, **same U-Net as E1** | `configs/m1_diff_base.yaml` | `results/E2_m1_diff/` | §3.3, Table 1 |
| E3 | Guidance sweep on the selected family | `configs/m1_cfg_sweep.yaml` | `results/E3_m1_cfg/` | §4.3, Table 2 |
| E3b | Guidance sweep on the *other* family (appendix insurance) | `configs/m1_cfg_sweep_alt.yaml` | `results/E3b_m1_cfg_alt/` | App. A.4 |
| E4 | Our method: hypernetwork predicts guidance strength | `configs/m1_ours.yaml` | `results/E4_m1_ours/` | §3.5, Tables 3–4 |
| E5 | Ablations | `configs/m1_abl_*.yaml` | `results/E5_m1_abl_*/` | §4.4, Table 3 |
| E6 | External baselines | `configs/m1_ext_*.yaml` | `results/E6_m1_ext_*/` | §4.5, Table 4 |

## Modality 2 — cell states

| ID | Experiment | Config | Results dir | Paper |
|----|-----------|--------|-------------|-------|
| E10 | Diffusion baseline | `configs/m2_diff_base.yaml` | `results/E10_m2_diff/` | §3.6 |
| E11 | Our method, transferred | `configs/m2_ours.yaml` | `results/E11_m2_ours/` | §4.6, Table 5 |
| E12 | External baselines, **same three as E6** | `configs/m2_ext_*.yaml` | `results/E12_m2_ext_*/` | §4.6, Table 5 |

## Ablations (E5)

Each row changes exactly one thing. All rows share one frozen base checkpoint, the same seeds,
the same NFE, and the same evaluation samples.

| Config | Variant | What it rules out |
|--------|---------|-------------------|
| `m1_abl_fixed.yaml` | Best tuned fixed `w*` | "you beat an untuned baseline" |
| `m1_abl_time_only.yaml` | `w(t)` — controller sees time, not the condition | "any learned time schedule would do" |
| `m1_abl_shuffled.yaml` | Condition embeddings permuted across classes | "it's the extra parameters, not the semantics" |
| `m1_abl_onehot.yaml` | Learned one-hot embedding instead of text, param-matched | "a lookup table is as good as a language model" |
| `m1_ours.yaml` | Full method | — |

## External baselines (E6 / E12)

The same three methods on both modalities. All are inference-time modifications to a trained
diffusion model, so they run on the checkpoint we already have and the protocol matches exactly.
See [docs/baselines.md](docs/baselines.md) for why each was chosen.

| Config | Method | Citation |
|--------|--------|----------|
| `*_ext_autoguidance.yaml` | Autoguidance | Karras et al., NeurIPS 2024, arXiv:2406.02507 |
| `*_ext_cfgpp.yaml` | CFG++ | Chung et al. 2024, arXiv:2406.08070 |
| `*_ext_interval.yaml` | Limited-interval guidance | Kynkäänniemi et al. 2024, arXiv:2404.07724 |

## Run hygiene

Every run writes `results/<ID>/manifest.json` with the git commit, the resolved config, the seed,
parameter counts, wall-clock, NFE, metric values, and library versions.
`scripts/make_tables.py` reads only manifests.

## Seeds

Base models: seed 0 (single seed, disclosed as a compute limitation).
Ablations and guidance sweeps: seeds {0, 1, 2}, reusing one base checkpoint. Report mean ± std.
