# Copyright (c) 2026 Fatemeh Salmani. Licensed under the MIT License (see LICENSE).
#
# BloodMNIST loaders. The dataset is part of MedMNIST v2 (Yang et al., Scientific
# Data 2023), CC BY 4.0, derived from Acevedo et al., Data in Brief 2020.
# Images are downloaded at runtime by the `medmnist` package (Apache-2.0) and are
# never stored in this repository.
"""Data loading for BloodMNIST."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

# Inputs are mapped to [-1, 1]; attacks clamp perturbed images to this range.
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
X_MIN, X_MAX = -1.0, 1.0
NUM_CLASSES = 8
IN_CHANNELS = 3


def _squeeze_label(y):
    # medmnist returns labels with shape (1,)
    return torch.as_tensor(y).long().view(-1)[0]


def get_datasets(size: int = 64, root: str | None = None, augment: bool = False):
    """Return (train, val, test) BloodMNIST datasets.

    `size` is one of 28, 64, 128, 224. The default VGG-19 needs at least 32 px,
    so 64 is the default here.
    """
    # imported lazily so the core modules and tests do not need them
    import torchvision.transforms as T
    from medmnist import BloodMNIST

    base = [T.ToTensor(), T.Normalize(MEAN, STD)]
    train_tf = T.Compose(
        ([T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if augment else []) + base
    )
    eval_tf = T.Compose(base)
    kw = dict(download=True, size=size, target_transform=_squeeze_label)
    if root is not None:
        kw["root"] = root
    train = BloodMNIST(split="train", transform=train_tf, **kw)
    val = BloodMNIST(split="val", transform=eval_tf, **kw)
    test = BloodMNIST(split="test", transform=eval_tf, **kw)
    return train, val, test


def make_loader(ds, batch_size: int = 128, shuffle: bool = False, num_workers: int = 2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def subset(ds, n: int, seed: int = 0):
    """Random subset of `n` samples (used for AS analysis and attack training sets)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)
