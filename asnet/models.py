# Copyright (c) 2026 Fatemeh Salmani. Licensed under the MIT License (see LICENSE).
"""Networks written as an ordered list of "units" so they can be cut at any
layer index l into a pre-model (units[:l]) and a post-model (units[l:] + head).
"""

from __future__ import annotations

import torch
import torch.nn as nn

VGG19_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
             512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]


class UnitNet(nn.Module):
    """A network = sequence of units followed by a head."""

    def __init__(self, units: list[nn.Module], head: nn.Module):
        super().__init__()
        self.units = nn.ModuleList(units)
        self.head = head

    def forward(self, x):
        for u in self.units:
            x = u(x)
        return self.head(x)

    @property
    def num_units(self) -> int:
        return len(self.units)

    def split(self, cut: int):
        """Return (pre_model, post_model). The pre-model output is flattened, so
        the post-model takes vectors of size n and reshapes them back."""
        if not 1 <= cut <= self.num_units:
            raise ValueError(f"cut must be in [1, {self.num_units}]")
        pre = nn.Sequential(*self.units[:cut], nn.Flatten())
        post = _PostModel(list(self.units[cut:]), self.head)
        return pre, post


class _PostModel(nn.Module):
    def __init__(self, units, head):
        super().__init__()
        self.units = nn.ModuleList(units)
        self.head = head
        self.feature_shape = None  # set on first call of the pre-model output

    def forward(self, z):
        x = z.view(z.shape[0], *self.feature_shape)
        for u in self.units:
            x = u(x)
        return self.head(x)


def split_model(model: UnitNet, cut: int, example_input: torch.Tensor):
    """Split and record the intermediate feature shape. Returns (pre, post, n)."""
    pre, post = model.split(cut)
    with torch.no_grad():
        x = example_input[:1]
        for u in model.units[:cut]:
            x = u(x)
    post.feature_shape = tuple(x.shape[1:])
    return pre, post, int(x[0].numel())


def vgg19(num_classes: int = 8, in_channels: int = 3) -> UnitNet:
    """VGG-19 with batch norm (CIFAR-style head). Works for inputs >= 32 px."""
    units, c = [], in_channels
    for v in VGG19_CFG:
        if v == "M":
            units.append(nn.MaxPool2d(2))
        else:
            units.append(nn.Sequential(nn.Conv2d(c, v, 3, padding=1),
                                       nn.BatchNorm2d(v), nn.ReLU(inplace=True)))
            c = v
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
    return UnitNet(units, head)


def small_cnn(num_classes: int = 8, in_channels: int = 3, image_size: int = 64) -> UnitNet:
    """The 4-layer CNN used for the Fashion-MNIST attack experiments in the
    ASNet paper (conv5x5-20, conv5x5-50, FC-500), adapted to RGB inputs."""
    s = ((image_size - 4) // 2 - 4) // 2
    units = [
        nn.Sequential(nn.Conv2d(in_channels, 20, 5), nn.ReLU(inplace=True)),
        nn.MaxPool2d(2),
        nn.Sequential(nn.Conv2d(20, 50, 5), nn.ReLU(inplace=True)),
        nn.MaxPool2d(2),
        nn.Sequential(nn.Flatten(), nn.Linear(50 * s * s, 500), nn.ReLU(inplace=True)),
    ]
    return UnitNet(units, nn.Linear(500, num_classes))


def build_model(name: str, num_classes: int = 8, image_size: int = 64) -> UnitNet:
    if name == "vgg19":
        return vgg19(num_classes)
    if name == "smallcnn":
        return small_cnn(num_classes, image_size=image_size)
    raise ValueError(f"unknown model {name!r}")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
