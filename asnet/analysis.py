# Why do universal perturbations send most images to one class?
# A few cheap checks on a trained model and a perturbation v.

import torch
import torch.nn.functional as F

from .attack import predict
from .data import X_MAX, X_MIN


@torch.no_grad()
def target_histogram(model, X, v, num_classes):
    """Where do the fooled images go?"""
    clean = predict(model, X)
    adv = predict(model, (X + v).clamp(X_MIN, X_MAX))
    fooled = adv != clean
    return torch.bincount(adv[fooled].cpu(), minlength=num_classes)


@torch.no_grad()
def runner_up_histogram(model, X, num_classes, bs=256):
    """Second most likely class on clean images: the closest decision region."""
    second = []
    for i in range(0, len(X), bs):
        second.append(model(X[i:i + bs]).topk(2, dim=1).indices[:, 1].cpu())
    return torch.bincount(torch.cat(second), minlength=num_classes)


@torch.no_grad()
def margin_to_class(model, X, c, bs=256):
    """Mean of (max logit - logit_c) over images not predicted as c: how far class c is."""
    out = []
    for i in range(0, len(X), bs):
        z = model(X[i:i + bs])
        m = z.max(1).values - z[:, c]
        out.append(m[z.argmax(1) != c].cpu())
    return torch.cat(out).mean().item()


@torch.no_grad()
def class_direction_similarity(X, y, v, num_classes):
    """cos(v, mean_c - mean_all): does v look like 'the average image of class c'?"""
    mean_all = X.mean(0).flatten()
    vf = v.flatten()
    sims = []
    for c in range(num_classes):
        d = X[y == c].mean(0).flatten() - mean_all
        sims.append(F.cosine_similarity(vf, d, dim=0).item())
    return torch.tensor(sims)


def gradient_toward(model, X, c, bs=128):
    """Mean input gradient of log p_c: the direction that increases class c the most on average."""
    gs = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].clone().requires_grad_(True)
        lp = F.log_softmax(model(x), 1)[:, c].sum()
        gs.append(torch.autograd.grad(lp, x)[0].sum(0))
    return torch.stack(gs).sum(0, keepdim=True) / len(X)
