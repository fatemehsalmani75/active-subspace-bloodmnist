# Defense side of the active subspace.
#
# Universal perturbations live mostly in the span of the leading input-space
# active directions U = top-k eigenvectors of E[grad_x loss grad_x loss^T].
# Two simple uses of U:
#   * purification: remove the component of x along U before classifying
#   * detection: score(x) = ||U^T (x - mu)||^2, large for perturbed inputs
# Both are evaluated against the plain attacks and an adaptive attack that
# knows the defense (the attack is run on the purified model directly).

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .active_subspace import FrequentDirections


@dataclass
class InputSubspace:
    U: torch.Tensor        # D x k, orthonormal columns (flattened input space)
    eigvals: torch.Tensor
    mu: torch.Tensor       # mean clean image, flattened


def input_active_subspace(model, loader, k=20, sketch_size=200, max_samples=None, device="cpu"):
    model.eval()
    fd, N, s = None, 0, None
    for x, y in loader:
        if max_samples is not None and N >= max_samples:
            break
        x, y = x.to(device), y.to(device)
        if max_samples is not None:
            x, y = x[:max_samples - N], y[:max_samples - N]
        x = x.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x), y, reduction="sum")
        g = torch.autograd.grad(loss, x)[0].flatten(1)
        if fd is None:
            fd = FrequentDirections(g.shape[1], sketch_size, g.device)
        fd.update(g)
        s = x.detach().sum(0) if s is None else s + x.detach().sum(0)
        N += len(x)
    _, sv, vt = torch.linalg.svd(fd.sketch(), full_matrices=False)
    k = min(k, len(vt))
    return InputSubspace(U=vt[:k].T.float(), eigvals=(sv ** 2 / N).float().cpu(),
                         mu=(s / N).flatten().float())


class Purified(nn.Module):
    """model(x - U U^T (x - mu)): the input loses its component along U."""

    def __init__(self, model, sub):
        super().__init__()
        self.model = model
        self.register_buffer("U", sub.U.clone())
        self.register_buffer("mu", sub.mu.clone())

    def purify(self, x):
        d = x.flatten(1) - self.mu
        return x - (d @ self.U @ self.U.T).view_as(x)

    def forward(self, x):
        return self.model(self.purify(x))


@torch.no_grad()
def detection_scores(sub, X, bs=256):
    U, mu = sub.U.to(X.device), sub.mu.to(X.device)
    out = []
    for i in range(0, len(X), bs):
        out.append(((X[i:i + bs].flatten(1) - mu) @ U).pow(2).sum(1))
    return torch.cat(out).cpu()


@torch.no_grad()
def projection_variance(sub, X, bs=256, eps=1e-8):
    """Variance of clean images along each column of U (for the Mahalanobis score)."""
    U, mu = sub.U.to(X.device), sub.mu.to(X.device)
    P = torch.cat([(X[i:i + bs].flatten(1) - mu) @ U for i in range(0, len(X), bs)])
    return (P.var(0) + eps).cpu()


@torch.no_grad()
def mahalanobis_scores(sub, X, var, bs=256):
    """sum_j (u_j^T (x - mu))^2 / var_j : each direction measured in units of its
    natural spread, so a small shift along a quiet direction counts."""
    U, mu, var = sub.U.to(X.device), sub.mu.to(X.device), var.to(X.device)
    out = []
    for i in range(0, len(X), bs):
        out.append((((X[i:i + bs].flatten(1) - mu) @ U).pow(2) / var).sum(1))
    return torch.cat(out).cpu()


def roc_auc(neg, pos):
    """AUC of a score that should be larger for `pos`; Mann-Whitney form, ties count 1/2."""
    neg, pos = torch.as_tensor(neg).double(), torch.as_tensor(pos).double()
    gt = (pos[:, None] > neg[None, :]).double().mean()
    eq = (pos[:, None] == neg[None, :]).double().mean()
    return (gt + 0.5 * eq).item()


def overlap(v, sub):
    """Fraction of ||v||^2 that lies in span(U)."""
    v = v.flatten().to(sub.U.device)
    return ((sub.U.T @ v).pow(2).sum() / v.pow(2).sum()).item()
