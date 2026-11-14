# MedMNIST v2 datasets (CC BY 4.0), downloaded by the medmnist package at runtime.
# BloodMNIST is the default; Phase 3 also uses DermaMNIST and PathMNIST.

import torch
from torch.utils.data import DataLoader, Subset

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
X_MIN, X_MAX = -1.0, 1.0   # range of normalized images
NUM_CLASSES = 8          # BloodMNIST
CLASSES = {"bloodmnist": 8, "dermamnist": 7, "pathmnist": 9}


def _label(y):
    return torch.as_tensor(y).long().view(-1)[0]


def get_datasets(size=64, root=None, augment=False, name="bloodmnist"):
    import medmnist
    import torchvision.transforms as T

    cls = getattr(medmnist, medmnist.INFO[name]["python_class"])

    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    flips = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if augment else []
    kw = dict(download=True, size=size, target_transform=_label)
    if root:
        kw["root"] = root
    train = cls(split="train", transform=T.Compose(flips + norm), **kw)
    val = cls(split="val", transform=T.Compose(norm), **kw)
    test = cls(split="test", transform=T.Compose(norm), **kw)
    return train, val, test


def make_loader(ds, batch_size=128, shuffle=False, num_workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def subset(ds, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)
