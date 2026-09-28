"""Backbones.

unet.py  Modality 1. ONE class, used by BOTH the flow-matching and the diffusion model.
         Only the objective and the sampler differ between them -- that is what makes
         the section 4.2 comparison matched, and it is easier to defend than a claim.
mlp.py   Modality 2. Plain MLP over a 50-dim vector; no convolutions, nothing spatial.

Both take (x, t, c) and return a prediction with the same shape as x.
"""
