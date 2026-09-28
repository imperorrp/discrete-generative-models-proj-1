"""Datasets and splits.

cifar.py  Modality 1: CIFAR-100, class-conditional. R^{3x32x32}.
cells.py  Modality 2: Tahoe-100M A-549 cell states, PCA-50. R^50.

Both expose the same interface so scripts/ does not branch on modality:

    make_loaders(cfg) -> (train_loader, val_loader, test_loader, meta)

`meta` carries num_classes, the condition strings (class names / drug names), and
anything the metrics need. Splits are seeded and reported in the manifest.
"""
