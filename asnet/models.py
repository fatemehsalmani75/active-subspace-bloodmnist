# Networks are stored as a list of "units" so they can be cut after any unit:
# pre-model = units[:cut], post-model = units[cut:] + head.

import torch
import torch.nn as nn

VGG19_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
             512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]


class UnitNet(nn.Module):
    def __init__(self, units, head):
        super().__init__()
        self.units = nn.ModuleList(units)
        self.head = head

    def forward(self, x):
        for u in self.units:
            x = u(x)
        return self.head(x)

    @property
    def num_units(self):
        return len(self.units)


class PostModel(nn.Module):
    """units[cut:] + head, takes the flattened output of the pre-model."""

    def __init__(self, units, head, feature_shape):
        super().__init__()
        self.units = nn.ModuleList(units)
        self.head = head
        self.feature_shape = feature_shape

    def forward(self, z):
        x = z.view(z.shape[0], *self.feature_shape)
        for u in self.units:
            x = u(x)
        return self.head(x)


def split_model(model, cut, x_example):
    """Returns (pre, post, n) where n is the number of neurons at the cut."""
    assert 1 <= cut <= model.num_units
    with torch.no_grad():
        h = x_example[:1]
        for u in model.units[:cut]:
            h = u(h)
    pre = nn.Sequential(*model.units[:cut], nn.Flatten())
    post = PostModel(list(model.units[cut:]), model.head, tuple(h.shape[1:]))
    return pre, post, h[0].numel()


def vgg19(num_classes=8, in_ch=3):
    units, c = [], in_ch
    for v in VGG19_CFG:
        if v == "M":
            units.append(nn.MaxPool2d(2))
        else:
            units.append(nn.Sequential(nn.Conv2d(c, v, 3, padding=1),
                                       nn.BatchNorm2d(v), nn.ReLU(inplace=True)))
            c = v
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
    return UnitNet(units, head)


def small_cnn(num_classes=8, in_ch=3, image_size=64):
    # same layout as the Fashion-MNIST CNN in Cui et al. (20, 50 channels, FC 500)
    s = ((image_size - 4) // 2 - 4) // 2
    units = [
        nn.Sequential(nn.Conv2d(in_ch, 20, 5), nn.ReLU(inplace=True)),
        nn.MaxPool2d(2),
        nn.Sequential(nn.Conv2d(20, 50, 5), nn.ReLU(inplace=True)),
        nn.MaxPool2d(2),
        nn.Sequential(nn.Flatten(), nn.Linear(50 * s * s, 500), nn.ReLU(inplace=True)),
    ]
    return UnitNet(units, nn.Linear(500, num_classes))


def build_model(name, num_classes=8, image_size=64):
    if name == "vgg19":
        return vgg19(num_classes)
    if name == "smallcnn":
        return small_cnn(num_classes, image_size=image_size)
    raise ValueError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
