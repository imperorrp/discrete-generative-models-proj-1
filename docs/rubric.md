# Rubric checklist

15 points total: **7.5 paper + code**, **7.5 defense** (4.5 presentation + 3.0 Q&A).

Every row is a graded item. Tick one only when the artifact exists and contains real content.

---

## What the study must contain

**Modality 1** — one continuous modality:

1. Train one continuous **flow-matching** model and one **diffusion** model on the same task
2. Compare them under the same evaluation protocol
3. Demonstrate that at least one **guidance mechanism** works
4. Choose the stronger family and add a **methodological innovation**
5. **Ablations** showing the innovation causes the improvement
6. Compare against **≥3 methods published within the last five years**

**Modality 2** — a genuinely different continuous state space:

- Transfer the **same core innovation**; compare against **≥3 recent methods**
- Does *not* need to repeat: flow-matching-vs-diffusion, guided-vs-unguided, or the ablations

**A modality is defined by the state space the generative dynamics are modelled on.** Two datasets
sharing a state space are not two modalities. Probability-simplex representations of discrete data
do count as continuous.

**What does not count as the innovation.** Hyperparameter tuning, more epochs, more compute, a
different seed, a larger sweep, changing batch size or learning rate, swapping in a bigger standard
backbone, or applying a published guidance method unchanged.

**Format.** NeurIPS workshop format, 5-page main-text limit; *or* ICLR main format, 9 pages, if
aiming for an ICLR submission. References and appendix are excluded from the limit.

---

## Paper (6.975 pts)

| ✓ | Pts | Item | What must be shown | Artifact |
|---|-----|------|--------------------|----------|
| ☐ | .225 | Abstract | Problem, both modalities, chosen family, guidance, innovation, strongest result, transfer, conclusion. 200–250 words, no citations or equations. | `paper/sections/00_abstract.tex` |
| ☐ | .100 | 1 Intro ¶1 | Motivate generation; why flow matching and diffusion are relevant | `paper/sections/01_intro.tex` |
| ☐ | .225 | 1 Intro ¶2 | The limitation motivating the innovation, with the closest prior work cited | `paper/sections/01_intro.tex` |
| ☐ | .100 | 1 Intro ¶3 | Two-modality design and the main hypothesis | `paper/sections/01_intro.tex` |
| ☐ | .100 | 1 Intro bullets | 4–5 contributions, each matching something actually shown | `paper/sections/01_intro.tex` |
| ☐ | .100 | 2 Related ¶1 | Flow matching and diffusion work relevant to this setup | `paper/sections/02_related.tex` |
| ☐ | .125 | 2 Related ¶2 | The papers closest to the innovation | `paper/sections/02_related.tex` |
| ☐ | .100 | 2 Related ¶3–4 | Modality-specific literature and the 3+3 benchmark methods | `paper/sections/02_related.tex` |
| ☐ | .100 | 2 Related | Explicit "what is known vs. what is different here" | `paper/sections/02_related.tex` |
| ☐ | .300 | 3.1 Data | Both modalities: sources, splits, sizes, preprocessing, conditioning, and **why the two state spaces are genuinely different** | `paper/sections/03_methods.tex` |
| ☐ | .475 | 3.2 Flow-matching baseline | Path, target velocity, loss, architecture, training settings, ODE sampler. **Equations must match the code.** | `paper/sections/03_methods.tex` |
| ☐ | .475 | 3.3 Diffusion baseline | Forward process, schedule, prediction target, loss, reverse sampler | `paper/sections/03_methods.tex` |
| ☐ | .300 | 3.4 Guidance | Objective, where the signal enters, strength and schedule, what is trained and what is frozen | `paper/sections/03_methods.tex` |
| ☐ | .625 | **3.5 Innovation** | Hypothesis, mechanism, equations, difference from the base model and from the closest prior work, which ablations isolate it | `paper/sections/03_methods.tex` |
| ☐ | .200 | 3.6 Transfer | What stays the same and what must change for Modality 2 | `paper/sections/03_methods.tex` |
| ☐ | .200 | 3.7 Overview figure | One publishable two-panel figure; trainable, pretrained, and frozen components labelled | `paper/figures/` |
| ☐ | .400 | 4.1 Eval protocol | Metrics, what each measures, and its limitations; sampling and compute budget; seeds; uncertainty | `paper/sections/04_results.tex` |
| ☐ | .450 | 4.2 FM vs diffusion | Matched comparison: quality, diversity, NFE, parameters, training cost; justify the family carried forward | `paper/tables/` |
| ☐ | .325 | 4.3 Guidance | Guided vs unguided; the target metric **and** ≥1 quality or diversity metric; multiple guidance strengths | `paper/tables/` |
| ☐ | .675 | **4.4 Ablations** | Base vs full, one-at-a-time removals, trade-offs visible | `paper/tables/` |
| ☐ | .450 | 4.5 Recent methods (M1) | ≥3 external methods under 5 years old, plus the internal base; where better, similar, and worse | `paper/tables/` |
| ☐ | .400 | 4.6 Transfer (M2) | ≥3 recent methods; what had to be adapted; whether it still helps | `paper/tables/` |
| ☐ | .200 | 4.7 Cross-modality | Compare across modalities; **≥2 real limitations or failure modes** | `paper/sections/04_results.tex` |
| ☐ | .125 | 5 Discussion ¶1 | Findings without overstatement; causal (ablation) vs correlational (benchmark) | `paper/sections/05_discussion.tex` |
| ☐ | .100 | 5 Discussion ¶2 | Mechanistic explanation tied to geometry, objective, guidance, or data | `paper/sections/05_discussion.tex` |
| ☐ | .100 | 5 Discussion ¶3 | Limitations and next steps; the narrowest claim the evidence supports | `paper/sections/05_discussion.tex` |

## Code (0.525 pts)

| ✓ | Pts | Item | What must be shown | Artifact |
|---|-----|------|--------------------|----------|
| ☐ | .200 | README and setup | Structure, environment, data preparation, how to run training, sampling, evaluation, and figures | `README.md` |
| ☐ | .225 | Organization | Organized configs; non-obvious code commented, **especially the innovation** | `configs/`, `src/guidance/` |
| ☐ | .100 | Paper linkage | Which script and config produced each experiment; external code credited | `configs/`, `scripts/` |

## Defense (7.5 pts)

Slides must be **labelled 1a–3e in the top-right corner**, in order. Numbered citations on each
slide, small font, bottom-right. Bibliography slide at the end. **No notes.** Even speaking time.
Hard 15-minute limit, and slides cannot be changed after submission.

| ✓ | Pts | Slide | Content |
|---|-----|-------|---------|
| ☐ | .100 | 1a | Problem and central hypothesis; both state spaces |
| ☐ | .275 | 1b | Datasets, splits and sizes, **how leakage is prevented**, representations, conditioning, normalization |
| ☐ | .275 | 1c | Flow matching: path, target velocity, loss, symbols, citations; architecture and optimizer; model selection; ODE solver and steps |
| ☐ | .275 | 1d | Diffusion: forward process, schedule, target, loss, reverse sampler, citations; architecture; selection; steps |
| ☐ | .275 | 1e | Metrics — implementation, sample counts, limitations — and how fairness is matched |
| ☐ | .100 | 1f | Overview figure |
| ☐ | .275 | 2a | The §4.2 table, and which family is carried forward and why |
| ☐ | .300 | 2b | Guidance rule, frozen and trained parts, schedule, clipping, and the §4.3 table |
| ☐ | .475 | 2c | Motivate the innovation (hypothesis, weakness addressed, difference from prior work) and give the full specification: equations, algorithm, new hyperparameters |
| ☐ | .450 | 2d | The §4.4 table, ≥2 ablations, and what each one isolates |
| ☐ | .750 | 2e | The §4.5 benchmark table; reproduced vs quoted numbers; fair discussion including unfavourable results |
| ☐ | .300 | 3a | Transfer: what stays, what changes, Modality 2 training and sampling |
| ☐ | .275 | 3b | The §4.6 table, fairness, and fair discussion |
| ☐ | .100 | 3c | Does it transfer? Direction and magnitude |
| ☐ | .100 | 3d | ≥2 limitations or failure modes, with samples or trajectories |
| ☐ | .175 | 3e | Back to the hypothesis: mechanism, trade-offs, specific next experiments |
| ☐ | 3.000 | Q&A | Every member must know the datasets, preprocessing, both formulations, guidance, ablations, baselines, metrics, and Modality 2 |

## Highest-density items

If time collapses, these are worth the most per hour:

1. Slide 2e + §4.5, the benchmark comparison — **1.2 pts**
2. Slide 2d + §4.4, the ablations — **1.125 pts**
3. Slide 2c + §3.5, the innovation specification — **1.1 pts**
4. §3.2 and §3.3, the two base-model descriptions — **0.95 pts**

Items 3 and 4 are writing, not experiments. They can be finished even if a training run fails.
