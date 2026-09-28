"""CIFAR-100 loading and splits."""

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 10
batch_size = 64
num_classes = 100
null_class = num_classes

# Places a pre-downloaded copy might already be sitting.
SEARCH_PATHS = ("./cifar", "./data", "../cifar", "../../cifar")


def find_root(root=None):
    """Return (root, download). Reuse a local copy if there is one, else download."""
    if root is not None:
        return root, not os.path.isdir(os.path.join(root, "cifar-100-python"))
    for path in SEARCH_PATHS:
        if os.path.isdir(os.path.join(path, "cifar-100-python")):
            print(f"using pre-downloaded CIFAR-100 at {path}")
            return path, False
    return "./data", True


def make_loaders(root=None, batch_size=batch_size, seed=seed):
    torch.manual_seed(seed)
    root, download = find_root(root)

    transform = transforms.Compose([transforms.ToTensor(),  # Pixels in [0, 1]
                                    transforms.Normalize((0.5,) * 3, (0.5,) * 3)])  # Pixels in [-1, 1]
    full_train = datasets.CIFAR100(root=root, train=True, download=download, transform=transform)

    # Reserve 50 images per class for validation, using a reproducible split.
    rng = torch.Generator().manual_seed(seed)
    targets = torch.tensor(full_train.targets)
    train_indices, val_indices = [], []

    for class_id in range(num_classes):
        indices = torch.where(targets == class_id)[0]
        indices = indices[torch.randperm(len(indices), generator=rng)]
        val_indices.extend(indices[:50].tolist())
        train_indices.extend(indices[50:].tolist())

    train_loader = DataLoader(Subset(full_train, train_indices),
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=2,
                              pin_memory=(device.type == "cuda"))

    val_loader = DataLoader(Subset(full_train, val_indices),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=(device.type == "cuda"))

    meta = {"n_conditions": num_classes,
            "shape": (3, 32, 32),
            "cond_text": [f"a photo of a {c.replace('_', ' ')}" for c in full_train.classes]}

    return train_loader, val_loader, meta
