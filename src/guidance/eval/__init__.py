"""Metrics.

Modality 1 (images): FID (state the feature network and sample count), class accuracy
via a frozen CLIP zero-shot classifier, and a diversity measure.

Modality 2 (cells): RBF-MMD over all 50 PCs, Wasserstein W1/W2 over the top 2 PCs, and
the variance ratio against the real population. The first two match BranchSBM's Table 5
so our numbers sit in the same table as theirs.

Never report a guidance target metric alone. High class accuracy with collapsed
diversity is reward hacking, not success -- that is what the variance ratio catches.
"""
