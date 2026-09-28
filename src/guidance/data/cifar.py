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

# Places a pre-downloaded copy might already be sitting. torchvision fetches from
# www.cs.toronto.edu/~kriz/, which is rate-limited to roughly 50 kB/s, so a 169MB
# download takes the better part of an hour. Always prefer a local copy.
SEARCH_PATHS = (
    "./cifar", "./data", "../cifar", "../../cifar",       # local checkout
    "/content/cifar", "/content/data",                     # Colab session storage
    "/content/drive/MyDrive/cifar",                        # Colab with Drive mounted
)


def find_root(root=None):
    """Return the folder holding cifar-100-python, or None if there isn't one."""
    for path in ([root] if root is not None else SEARCH_PATHS):
        if path and os.path.isdir(os.path.join(path, "cifar-100-python")):
            return path
    return None


class HFCifar100(torch.utils.data.Dataset):
    """CIFAR-100 from the HuggingFace CDN.

    Same images, same labels, but downloaded from a CDN instead of cs.toronto.edu,
    so it takes seconds rather than most of an hour. Exposes .targets and .classes
    so the split code below does not care which source was used.
    """

    def __init__(self, transform):
        from datasets import load_dataset
        ds = load_dataset("uoft-cs/cifar100", split="train")
        self.ds = ds
        self.transform = transform
        self.targets = list(ds["fine_label"])
        self.classes = ds.features["fine_label"].names

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return self.transform(row["img"].convert("RGB")), row["fine_label"]


def make_loaders(root=None, batch_size=batch_size, seed=seed, limit=0):
    """limit > 0 keeps only that many training images, for quick smoke tests.

    `seed` controls only the split, through its own generator below. This function
    deliberately leaves the global torch RNG alone: a caller that passes a fixed split
    seed and a varying training seed must get different model initialisations.
    """
    transform = transforms.Compose([transforms.ToTensor(),  # Pixels in [0, 1]
                                    transforms.Normalize((0.5,) * 3, (0.5,) * 3)])  # Pixels in [-1, 1]

    found = find_root(root)
    if found is not None:
        print(f"using pre-downloaded CIFAR-100 at {found}")
        full_train = datasets.CIFAR100(root=found, train=True, download=False,
                                       transform=transform)
    else:
        print("no local copy found; fetching from the HuggingFace CDN")
        full_train = HFCifar100(transform)

    # Reserve 50 images per class for validation, using a reproducible split.
    rng = torch.Generator().manual_seed(seed)
    targets = torch.tensor(full_train.targets)
    train_indices, val_indices = [], []

    for class_id in range(num_classes):
        indices = torch.where(targets == class_id)[0]
        indices = indices[torch.randperm(len(indices), generator=rng)]
        val_indices.extend(indices[:50].tolist())
        train_indices.extend(indices[50:].tolist())

    if limit:
        train_indices = train_indices[:limit]
        val_indices = val_indices[:max(limit // 9, batch_size)]

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
