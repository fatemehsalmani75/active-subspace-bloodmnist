# Recursive active-subspace universal attack (Cui et al., SIAM J. Math. Data Sci.
# 2020, Alg. 4.1). v is updated along the dominant direction of the input gradients
# of the samples that are not fooled yet, and projected on the l2 ball of radius delta.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .data import X_MAX, X_MIN


@torch.no_grad()
def predict(model, X, bs=256):
    return torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])


@torch.no_grad()
def fooling_rate(model, X, v, clean=None):
    """Fraction of samples whose label changes after adding v."""
    if clean is None:
        clean = predict(model, X)
    return (predict(model, (X + v).clamp(X_MIN, X_MAX)) != clean).float().mean().item()


def project_l2(v, delta):
    n = v.norm()
    return v * (delta / n) if n > delta else v


def top_direction(model, X, y, v, bs=128):
    grads = []
    for i in range(0, len(X), bs):
        x = (X[i:i + bs] + v).clamp(X_MIN, X_MAX).requires_grad_(True)
        loss = F.cross_entropy(model(x), y[i:i + bs], reduction="sum")
        grads.append(torch.autograd.grad(loss, x)[0].flatten(1))
    _, _, vt = torch.linalg.svd(torch.cat(grads), full_matrices=False)
    return vt[0].view_as(v)


@dataclass
class AttackResult:
    v: torch.Tensor
    history: list = field(default_factory=list)


def as_universal_attack(model, X, y, delta, alpha=10, max_iter=100, n_steps=6, seed=0, log=print):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    v = torch.zeros_like(X[:1])
    clean = predict(model, X)
    best, fails, hist = 0.0, 0, []

    for it in range(max_iter):
        alive = (predict(model, (X + v).clamp(X_MIN, X_MAX)) == clean).nonzero().squeeze(1)
        if len(alive) == 0:
            break
        if fails > 0 and len(alive) > 1:
            # last step failed: use a random half of the samples for a new direction
            keep = torch.randperm(len(alive), generator=g)[:max(1, len(alive) // 2)]
            alive = alive[keep.to(alive.device)]
        u = top_direction(model, X[alive], y[alive], v)

        # try both signs and step lengths delta, delta/2, ...
        new_v, new_best = None, best
        for j in range(n_steps):
            for s in (1.0, -1.0):
                cand = project_l2(v + s * delta * 2.0 ** -j * u, delta)
                fr = fooling_rate(model, X, cand, clean)
                if fr > new_best:
                    new_v, new_best = cand, fr

        if new_v is None:
            fails += 1
            if fails >= alpha:
                break
            continue
        fails = 0
        v, best = new_v, new_best
        hist.append(best)
        log(f"iter {it + 1}: fooling rate {best:.3f}, |v| = {v.norm():.2f}")

    return AttackResult(v.detach(), hist)
