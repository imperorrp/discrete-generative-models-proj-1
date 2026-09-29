# Findings

Results that changed how we think about the experiments. Numbers are in the executed
notebooks under `notebooks/runs/`.

## TC-LoRA gives no gain on CIFAR-100 when the backbone already has the label

Run 2 (`notebooks/runs/tc_lora_cifar_run2_e7448aa.ipynb`, single seed):

| | base | + TC-LoRA |
|---|---|---|
| validation MSE, matched noise | 0.0619 | 0.0618 |
| FID (5000 real, 10000 generated, 50 DDIM steps) | 76.6 | 77.4 |
| adapter magnitude, mean ‖BA‖/‖W‖ | — | 0.11 (range 0.03–0.21 over 12 layers) |

The adapter moved substantially and it made no difference. The hypernetwork read a CLIP
embedding of the class name; the backbone read the class label. For a fixed set of 100
classes those carry the same information, so there was nothing for the adapter to add.

**What this means for every text-conditioned mechanism in this project, including ours:**
a text embedding of a class name is redundant with the class label. Any method that adds
text on top of a label-conditioned backbone should be expected to show nothing on
CIFAR-100. That is not a failure of the method; it is the wrong test.

## Open question: how to test semantic conditioning so it can actually show something

Options, roughly in order of how much they change:

1. **Remove the label path.** Run the backbone with the null label always, so the tested
   mechanism (adapter, cross-attention, classifier gradient, guidance controller) is the
   *only* route by which the model learns the class. Then the comparison between
   mechanisms is meaningful, and text has to carry the information or nothing does.
   Cheapest change; the base checkpoint already supports it because it was trained with
   label dropout.
2. **Richer text than the class name.** WordNet glosses or short descriptions carry
   attribute information the label index does not ("large spotted wild cat, tawny
   fur"). Only helps if the model can use attributes the label alone would not give it.
3. **Held-out classes.** Train on 80 classes, condition on text for the other 20 at test
   time. There is no label for an unseen class, so the text is all the model has. This is
   the experiment that separates "reads a description" from "reads a lookup table", and
   the one a label-conditioned baseline cannot run at all.
4. **A modality where the text is not a class name.** Cell states: the text is a drug's
   mechanism of action, which is information the training data never encodes as a label.

Option 1 is the immediate fix for the comparison table. Option 3 is the experiment that
would make the paper's claim. Options 1 and 3 combine.

## Cross-attention with the label path off: the text is read

Run 1 (`notebooks/runs/text_xattn_cifar_run1.ipynb`, code `429ead4`, single seed). Backbone
given the null label throughout; a 394k-parameter cross-attention block at the bottleneck,
reading CLIP token embeddings of the class name, is the only route to the class.

| | val MSE, matched noise |
|---|---|
| base, null label | 0.06213 |
| base, true label (reference) | 0.06196 |
| cross-attention, correct text | 0.06184 |
| cross-attention, wrong text (three label offsets, all 0.0624) | 0.06241 |

Wrong text costs 0.92% more loss than correct text, stable across offsets; samples from the
same noise differ by 20%. FID 78.3 (unconditional) → 75.7 (text). This answers the open
question above, option 1: remove the label path and a text mechanism has something to
learn. Text alone recovers what the label gives (0.06184 vs 0.06196), and wrong text is
worse than no text (0.06241 vs 0.06213), so the block steers rather than merely adds
capacity.

Caveats: one seed; base still improving at epoch 30; FID against 5000 validation images;
sampled with the pre-`4fd05e1` DDIM convention.

## TC-LoRA with the label path off: the text is read here too

Run 3 (`notebooks/runs/tc_lora_cifar_run3_1aff1df.ipynb`, single seed), same protocol as the
cross-attention run. The two text mechanisms side by side, each against its own session's base:

| | TC-LoRA (weights) | cross-attention (activations) |
|---|---|---|
| trainable parameters | 1.66M | 0.39M |
| base, null label | 0.0621 | 0.0621 |
| base, true label | 0.0619 | 0.0620 |
| correct text | 0.0617 | 0.0618 |
| wrong text | 0.0622 | 0.0624 |
| text effect | +0.84% | +0.92% |
| paired samples changed by wrong text | 12.4% | 19.9% |
| FID, base → adapted | 80.4 → 81.6 | 78.3 → 75.7 |

Both read the text: wrong text costs each about 0.9% on the loss, and both recover the loss
the true label would give. They differ on the samples. Cross-attention lowers FID and moves
the samples more; TC-LoRA, with four times the parameters, lowers the loss by a hair more and
raises FID. A weight-space adapter that fits the denoising objective slightly better does not
give better images here. Base FIDs differ between sessions (80.4 vs 78.3) with the same seed
and code path, so compare each arm to its own base, not across columns.

## TFG at its CIFAR-10 settings: it steers, weakly, and wrecks quality

Run 1 (`notebooks/runs/tfg_cifar_run1_3c38c51.ipynb`, 200 samples, two per class, scored by a
classifier the sampler never saw):

| | accuracy | FID (5000 real, 10000 generated) | saturated pixels |
|---|---|---|---|
| unconditional base | 1.5% | 80.8 | 8% |
| label-conditioned base | 4.5% | — | 8% |
| TFG, ρ = 1, μ = 0.25 | 3.5% | 151.1 | 23% |
| TFG, wrong target, scored against that target | 5.5% | | 23% |

Guidance follows its target: the wrong-target row scores 5.5% against the class it was pushed
toward and 1.0% against the original. But the effect is small and the price is large: FID
nearly doubles and a quarter of the pixels sit at the clamp, where the classifier gradient is
zero. Over a strength sweep accuracy peaks at 4.5% at half the default and falls above it.

The bottleneck is the base model. Even given the true label it produces images a CIFAR-100
classifier recognises 4.5% of the time, at FID 80. TFG's settings were tuned for a strong
CIFAR-10 DDPM; on a 1.9M-parameter model trained for 30 epochs there is little signal for the
classifier to amplify. Any guidance comparison on this base, including the proposed method,
inherits this ceiling, and the paper has to say so. Next: smaller ρ, μ; more samples per class
for the accuracy rows; and a better base if time allows.
