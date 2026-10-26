# Copyright (c) 2026 Fatemeh Salmani. Licensed under the MIT License (see LICENSE).
#
# Independent re-implementation of ASNet (pre-model + active-subspace layer +
# polynomial-chaos-expansion layer, Algorithm 3.1) and of the l1 proximal step used
# by the sparse variant ASNet-s, as described in: C. Cui, K. Zhang, T. Daulbaev,
# J. Gusak, I. Oseledets, Z. Zhang, "Active Subspace of Neural Networks: Structural
# Analysis and Universal Attacks", SIAM J. Math. Data Sci. 2(4):1096-1122, 2020.
# doi:10.1137/19M1296070. Written from the paper; no code from
# https://github.com/chunfengc/ASNet was copied.
"""Build and fine-tune ASNet from a trained UnitNet."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .active_subspace import ActiveSubspace, compute_active_subspace
from .models import UnitNet, split_model
from .pce import PCELayer


class ASNet(nn.Module):
    def __init__(self, pre: nn.Module, V: torch.Tensor, pce: PCELayer):
        super().__init__()
        self.pre = pre
        self.as_layer = nn.Linear(V.shape[0], V.shape[1], bias=False)
        with torch.no_grad():
            self.as_layer.weight.copy_(V.T)
        self.pce = pce

    def forward(self, x):
        return self.pce(self.as_layer(self.pre(x)))


@torch.no_grad()
def _collect(pre, as_layer, teacher, loader, max_samples, device):
    Zs, Ys, seen = [], [], 0
    for x, _ in loader:
        x = x.to(device)
        Zs.append(as_layer(pre(x)).cpu())
        Ys.append(teacher(x).cpu())          # fit the original network's logits
        seen += x.shape[0]
        if max_samples is not None and seen >= max_samples:
            break
    return torch.cat(Zs)[:max_samples], torch.cat(Ys)[:max_samples]


def build_asnet(model: UnitNet, cut: int, as_loader, fit_loader, *, r: int | None = 50,
                p: int = 2, eps: float = 0.05, sketch_size: int | None = 200,
                as_samples: int | None = None, fit_samples: int | None = None,
                device="cpu") -> tuple[ASNet, ActiveSubspace]:
    """Algorithm 3.1 (construction step): cut `model` at layer `cut`, compute the
    active subspace V, and initialise the PCE layer by least squares so that
    PCE(V^T z) approximates the original logits."""
    teacher = model.eval().to(device)
    example = next(iter(as_loader))[0].to(device)
    pre, post, _ = split_model(copy.deepcopy(model), cut, example)
    pre, post = pre.to(device), post.to(device)
    sub = compute_active_subspace(pre, post, as_loader, r=r, eps=eps,
                                  sketch_size=sketch_size, max_samples=as_samples,
                                  device=device)
    as_layer = nn.Linear(sub.V.shape[0], sub.r, bias=False).to(device)
    with torch.no_grad():
        as_layer.weight.copy_(sub.V.T)
    Z, Y = _collect(pre, as_layer, teacher, fit_loader, fit_samples, device)
    num_classes = Y.shape[1]
    pce = PCELayer(sub.r, p, num_classes)
    pce.fit_least_squares(Z, Y)
    return ASNet(pre, sub.V, pce).to(device), sub


def distillation_loss(student_logits, teacher_logits, y, beta: float = 0.1, T: float = 1.0):
    """(1 - beta) * CE(student, y) + beta * T^2 * KL(teacher || student).

    The ASNet paper re-trains with a knowledge-distillation loss weighted by
    beta = 0.1; this is a standard (Hinton et al., 2015) form of that loss.
    """
    ce = F.cross_entropy(student_logits, y)
    kd = F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits / T, 1),
                  reduction="batchmean") * T * T
    return (1 - beta) * ce + beta * kd


@torch.no_grad()
def prox_l1_(params, threshold: float):
    """Soft-thresholding: the proximal step of an l1 penalty (ASNet-s)."""
    for w in params:
        w.copy_(torch.sign(w) * torch.clamp(w.abs() - threshold, min=0))


def finetune_asnet(net: ASNet, teacher: nn.Module, loader, *, epochs: int = 50,
                   beta: float = 0.1, lr_pre: float = 1e-4, lr_head: float = 1e-5,
                   l1: float = 0.0, device="cpu", log=print):
    """Fine-tune all ASNet layers with Adam (pre-model and AS/PCE layers get
    separate learning rates, as in the paper). l1 > 0 gives the sparse ASNet-s."""
    teacher.eval()
    opt = torch.optim.Adam([
        {"params": net.pre.parameters(), "lr": lr_pre},
        {"params": list(net.as_layer.parameters()) + list(net.pce.parameters()), "lr": lr_head},
    ])
    for ep in range(epochs):
        net.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t = teacher(x)
            loss = distillation_loss(net(x), t, y, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            if l1 > 0:
                for g in opt.param_groups:
                    prox_l1_([p for p in g["params"] if p.dim() > 1], g["lr"] * l1)
            total += loss.item() * x.shape[0]; n += x.shape[0]
        log(f"[finetune] epoch {ep + 1}/{epochs} loss {total / n:.4f}")
    return net
