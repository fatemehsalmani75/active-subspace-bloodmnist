# Copyright (c) 2026 Fatemeh Salmani. Licensed under the MIT License (see LICENSE).
#
# Independent re-implementation of the recursive active-subspace universal attack
# (Algorithm 4.1) described in: C. Cui, K. Zhang, T. Daulbaev, J. Gusak, I. Oseledets,
# Z. Zhang, "Active Subspace of Neural Networks: Structural Analysis and Universal
# Attacks", SIAM J. Math. Data Sci. 2(4):1096-1122, 2020. doi:10.1137/19M1296070.
# Written from the paper; no code from https://github.com/chunfengc/ASNet was copied.
# The step-size search below is this repository's own implementation choice.
"""Active-subspace universal adversarial perturbation (input space)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import X_MAX, X_MIN


@torch.no_grad()
def predict(model, X, batch_size=256):
    return torch.cat([model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def fooling_rate(model, X, v, clean_pred=None, batch_size=256) -> float:
    """Fraction of samples whose predicted label changes when v is added."""
    if clean_pred is None:
        clean_pred = predict(model, X, batch_size)
    adv = predict(model, (X + v).clamp(X_MIN, X_MAX), batch_size)
    return (adv != clean_pred).float().mean().item()


def project_l2(v: torch.Tensor, delta: float) -> torch.Tensor:
    n = v.norm()
    return v if n <= delta else v * (delta / n)


def dominant_direction(model, X, y, v, batch_size=128) -> torch.Tensor:
    """Leading right singular vector of the matrix of input gradients
    d loss(f(x_i + v), y_i) / d x  (the 1-D active subspace of the attack loss)."""
    grads = []
    for i in range(0, len(X), batch_size):
        x = (X[i:i + batch_size] + v).clamp(X_MIN, X_MAX).requires_grad_(True)
        loss = F.cross_entropy(model(x), y[i:i + batch_size], reduction="sum")
        (g,) = torch.autograd.grad(loss, x)
        grads.append(g.flatten(1))
    G = torch.cat(grads)
    _, _, vh = torch.linalg.svd(G, full_matrices=False)
    return vh[0].view_as(v)


@dataclass
class AttackResult:
    v: torch.Tensor
    history: list = field(default_factory=list)   # fooling rate after each accepted step


def as_universal_attack(model: nn.Module, X: torch.Tensor, y: torch.Tensor, *,
                        delta: float, alpha: int = 10, max_iter: int = 100,
                        n_steps: int = 6, seed: int = 0, log=print) -> AttackResult:
    """Recursive active-subspace universal attack.

    v_0 = 0. At every iteration, on the samples still classified as originally
    predicted, take the dominant active direction u of the loss gradients and try
    v + s * t * u for s in {+1, -1} and step lengths t = delta * 2^-j (j < n_steps),
    each projected onto the l2 ball of radius delta. The candidate with the highest
    fooling rate on (X, y) is accepted if it improves it; otherwise a failure is
    counted (the next direction is then estimated on a random half of the remaining
    samples). Stops after `alpha` consecutive failures or when every sample is fooled.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    v = torch.zeros_like(X[:1])
    clean = predict(model, X)
    best = 0.0
    hist, fails = [], 0
    for it in range(max_iter):
        adv = predict(model, (X + v).clamp(X_MIN, X_MAX))
        alive = adv == clean
        if not alive.any():
            break
        idx = alive.nonzero().squeeze(1)
        if fails > 0 and len(idx) > 1:
            # after a failure, re-estimate the direction on a random half of the
            # remaining samples so the next try is not identical
            perm = torch.randperm(len(idx), generator=gen)[: max(1, len(idx) // 2)]
            idx = idx[perm.to(idx.device)]
        u = dominant_direction(model, X[idx], y[idx], v)
        cand_best, cand_v = best, None
        for j in range(n_steps):
            t = delta * 2.0 ** (-j)
            for s in (1.0, -1.0):
                cand = project_l2(v + s * t * u, delta)
                fr = fooling_rate(model, X, cand, clean)
                if fr > cand_best:
                    cand_best, cand_v = fr, cand
        if cand_v is None:
            fails += 1
            if fails >= alpha:
                break
            continue
        fails = 0
        v, best = cand_v, cand_best
        hist.append(best)
        log(f"[AS attack] iter {it + 1}: fooling rate {best:.4f}, ||v||_2 = {v.norm():.3f}")
    return AttackResult(v=v.detach(), history=hist)
