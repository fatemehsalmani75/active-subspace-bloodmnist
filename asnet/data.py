# BloodMNIST (MedMNIST v2, CC BY 4.0). Downloaded by the medmnist package at runtime.

import torch
from torch.utils.data import DataLoader, Subset

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
X_MIN, X_MAX = -1.0, 1.0   # range of normalized images
NUM_CLASSES = 8


def _label(y):
    return torch.as_tensor(y).long().view(-1)[0]


def get_datasets(size=64, root=None, augment=False):
    import torchvision.transforms as T
    from medmnist import BloodMNIST

    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    flips = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if augment else []
    kw = dict(download=True, size=size, target_transform=_label)
    if root:
        kw["root"] = root
    train = BloodMNIST(split="train", transform=T.Compose(flips + norm), **kw)
    val = BloodMNIST(split="val", transform=T.Compose(norm), **kw)
    test = BloodMNIST(split="test", transform=T.Compose(norm), **kw)
    return train, val, test


def make_loader(ds, batch_size=128, shuffle=False, num_workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def subset(ds, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)
