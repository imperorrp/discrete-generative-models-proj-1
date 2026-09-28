# Baselines and closest prior work

Every citation below was checked against the arXiv listing page on 2026-09-28. Title, authors,
and ID are as shown there. Anything added later gets the same treatment before it enters
`paper/refs.bib`.

---

## The three comparison methods (§4.5 and §4.6)

The rubric asks for at least three methods published within the last five years, per modality.
We use **the same three on both modalities**. This is not required — the rubric allows
modality-specific baselines, and its own examples are modality-specific — but all three of these
are *inference-time modifications to a trained diffusion model* with no image-specific
assumptions, so they run unchanged on a U-Net over images and on an MLP over cell states. That
makes §4.7's cross-modality comparison a like-for-like one, which is worth more than picking the
most famous method in each field.

| Method | Citation | What it does | Why it is the right comparison |
|---|---|---|---|
| **Autoguidance** | Karras, Aittala, Lehtinen, Hellsten, Aila, Laine. *Guiding a Diffusion Model with a Bad Version of Itself.* NeurIPS 2024. arXiv:2406.02507 | Replaces the unconditional branch with a smaller, under-trained copy of the same model | A fundamentally different answer to "what should we guide away from." Strongest recent guidance result. |
| **CFG++** | Chung, Kim, Park, Nam, Ye. *CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models.* arXiv:2406.08070 | Reframes text guidance as an inverse problem; fixes the off-manifold drift that causes mode collapse at high `w` | Directly targets the failure mode our method claims to avoid by adapting `w`. |
| **Limited-interval guidance** | Kynkäänniemi, Aittala, Karras, Laine, Aila, Lehtinen. *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models.* arXiv:2404.07724 | Applies guidance only in a middle band of noise levels; none early, none late | **The key comparison.** This is a hand-designed schedule for guidance strength over time. Our method learns one. If we cannot beat a hand-designed schedule, we do not have a contribution. |

All three need no retraining, so each costs one evaluation pass per modality.

**For Modality 2, cite additionally in §2 ¶4** (not necessarily as a §4.6 table row):

- **BranchSBM** — Tang, Zhang, Tong, Chatterjee. *Branched Schrödinger Bridge Matching.* ICLR
  2026. arXiv:2506.09007. Same dataset (Tahoe-100M), same cell line (A-549, CVCL_0023), same
  drugs (Trametinib, Clonidine), same representation (top-50 PCs). From the course
  instructor's lab. **Must be cited.** Their Trametinib Table 5: single-branch SBM
  RBF-MMD 0.246 ± 0.013, BranchSBM 0.053 ± 0.001.

---

## Closest prior work (§1 ¶2, worth .225 pt, and §2 ¶2, worth .125 pt)

Our innovation predicts guidance strength from a language-model embedding of the condition.
Several groups have converged on nearby ideas. Cite these rather than hoping nobody notices.

Ordered by how close they are.

### 1. Prompt-aware CFG — the closest

Zhang & Li. *Prompt-aware classifier free guidance for diffusion models.* arXiv:2509.22728
(Sep 2025).

Predicts scale-dependent quality from **semantic embeddings and linguistic complexity of the
prompt**, then picks the best scale per prompt via a utility function. Images and audio.

**How we differ:** theirs is training-free — it builds a synthetic dataset by sampling at many
scales, scores it, and fits a predictor. It selects one scalar per prompt. Ours trains a
hypernetwork end-to-end and emits a schedule that varies over the trajectory. Also they never
test conditions absent from training, and never transfer across modalities.

### 2. Learn to Guide Your Diffusion Model

Galashov, Pokle, Doucet, Gretton, Delbracio, De Bortoli. arXiv:2510.00815 (Oct 2025).

Learns guidance weights `ω_{c,(s,t)}` as continuous functions of the conditioning `c` and both
times, by minimising distributional mismatch against the true conditional. Tested on
low-dimensional toy problems **and** images.

**How we differ:** their `c` enters as the model's own conditioning vector. Ours is a frozen,
model-external text embedding, which is what makes it defined for conditions never trained on.
Note their low-dimensional experiments make this the most directly comparable prior work for
our cell-state modality.

### 3. Adversarial Learning of Classifier-Free Guidance Schedules

Pokle, Galashov, Doucet, Delbracio, De Bortoli. arXiv:2608.14038 (Aug 2026).

Learns the schedule as a function of time, conditioning, **and the current noisy sample**, via
density-ratio estimation with a discriminator. A strict superset of what our controller reads.

**How we differ:** reading the noisy state ties the scheduler to one backbone's internal
geometry. Reading less is the point, not a concession — a model-external embedding is defined
for any condition, including unseen ones. Check whether they evaluate on held-out conditions
before claiming this.

### 4. LeSAMP — an LLM emitting guidance schedules

Lim & Gandelsman. *Learning Sampling Parameters for Diffusion Models.* arXiv:2607.23488
(Jul 2026).

An LLM trained with RL emits prompt-conditioned, timestep-varying sampling parameters including
the CFG scale. Rewards from human preference models and VLM-as-a-judge.

**How we differ:** this is the closest thing to "a language model predicts guidance strength,"
so it must be cited. Theirs is an RL policy over sampling hyperparameters for large pretrained
text-to-image models. Ours is a small hypernetwork over a frozen semantic embedding, trained
directly against a distributional objective, on models we train ourselves.

### 5. Dynamic CFG via Online Feedback

Papalampidi, Wiles, Ktena, Shtedritski, Bugliarello, Kajic, Albuquerque, Nematzadeh
(Google DeepMind). arXiv:2509.16131 (Sep 2025).

Per-prompt, per-timestep scale chosen by greedy search using online feedback from CLIP, a
discriminator, and a preference model.

**How we differ:** theirs is search at inference time and needs evaluators in the loop. Ours is
a single forward pass through a small network.

### 6. Classifier-free Guidance with Adaptive Scaling (β-CFG)

arXiv:2502.10574 (Feb 2025). Analytic time-dependent β-curve schedule. A hand-designed
schedule, in the same family as limited-interval guidance.

### Also relevant

- **TC-LoRA** (arXiv:2510.09561) — hypernetwork generates LoRA adapters conditioned on time and
  condition for a frozen diffusion backbone. The closest prior art for the *hypernetwork* half
  of our method, as opposed to the guidance half. Verify before citing.
- **Classifier guidance** — Dhariwal & Nichol, arXiv:2105.05233 (May 2021). **Outside the
  five-year window** as of Sept 2026, so it cannot count toward the three required methods. Still
  cite it as background in §2 ¶2.

---

## Phrasing that is defensible

> Learned and adaptive guidance schedules are an active area: schedules have been learned as
> functions of time and conditioning [2510.00815], of time, conditioning and the noisy state
> [2608.14038], selected per prompt from semantic embeddings [2509.22728], searched online
> against reward models [2509.16131], and emitted by an RL-trained language model [2607.23488].
> Hand-designed schedules restricted to an interval of noise levels are also effective
> [2404.07724]. We ask a question this line of work does not: whether a guidance policy
> conditioned on a frozen, model-external description of the condition remains useful for
> conditions absent from training, and whether the same mechanism transfers between two
> unrelated continuous state spaces.

**Do not write** "first", "novel adaptive guidance", or "state of the art". Adaptive guidance is
not new. The question we ask about it may be.

---

## Still to verify

- [ ] Does arXiv:2608.14038 evaluate on conditions held out from training? (If it does, our
      framing has to change.)
- [ ] Does arXiv:2509.22728 report per-timestep schedules or only a single scale per prompt?
- [ ] TC-LoRA arXiv ID and author list
- [ ] Re-search the day before submission
