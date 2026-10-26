# Copyright (c) 2026 Fatemeh Salmani. Licensed under the MIT License (see LICENSE).
#
# Independent re-implementation of the active-subspace analysis of neural networks
# described in: C. Cui, K. Zhang, T. Daulbaev, J. Gusak, I. Oseledets, Z. Zhang,
# "Active Subspace of Neural Networks: Structural Analysis and Universal Attacks",
# SIAM J. Math. Data Sci. 2(4):1096-1122, 2020. doi:10.1137/19M1296070
# (Definition 3.1 and the frequent-directions variant, Algorithm 3.2).
# The active-subspace method itself is due to P. G. Constantine et al.
# The frequent-directions sketch is due to Liberty (2013) / Ghashami et al. (2016).
# Written from the papers; no code from https://github.com/chunfengc/ASNet was copied.
"""Active subspace of a cut network: C = E[g g^T], g = d loss / d z_l."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequentDirections:
    """Streaming sketch B (ell x n) with B^T B ~= G^T G for the rows fed in.

    Uses a 2*ell row buffer; when it is full, the buffer is compressed by an SVD
    and every squared singular value is shrunk by the ell-th one.
    """

    def __init__(self, n: int, ell: int, device=None, dtype=torch.float64):
        self.n, self.ell = n, ell
        self.buf = torch.zeros(2 * ell, n, device=device, dtype=dtype)
        self.next_row = 0
        self.rows_seen = 0

    def update(self, rows: torch.Tensor):
        rows = rows.to(self.buf)
        self.rows_seen += rows.shape[0]
        for start in range(0, rows.shape[0], self.ell):
            chunk = rows[start:start + self.ell]
            if self.next_row + chunk.shape[0] > self.buf.shape[0]:
                self._compress()
            self.buf[self.next_row:self.next_row + chunk.shape[0]] = chunk
            self.next_row += chunk.shape[0]

    def _compress(self):
        _, s, vh = torch.linalg.svd(self.buf[: self.next_row], full_matrices=False)
        k = min(self.ell, s.numel())
        delta = s[k - 1] ** 2 if s.numel() >= self.ell else s.new_zeros(())
        s_new = torch.sqrt(torch.clamp(s[:k] ** 2 - delta, min=0))
        self.buf.zero_()
        self.buf[:k] = s_new[:, None] * vh[:k]
        self.next_row = k

    def sketch(self) -> torch.Tensor:
        return self.buf[: self.next_row]


@dataclass
class ActiveSubspace:
    V: torch.Tensor          # (n, r) active directions (columns)
    eigvals: torch.Tensor    # (k,) leading eigenvalues of C, descending
    r: int                   # number of active neurons

    @property
    def energy(self) -> torch.Tensor:
        """Cumulative fraction of eigenvalue mass captured by the leading i directions."""
        return torch.cumsum(self.eigvals, 0) / self.eigvals.sum()


def per_sample_grads(post: nn.Module, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Rows are d loss(post(z_i), y_i) / d z_i. `post` must be in eval mode so
    that samples do not interact (e.g. through batch norm)."""
    z = z.detach().requires_grad_(True)
    loss = F.cross_entropy(post(z), y, reduction="sum")
    (g,) = torch.autograd.grad(loss, z)
    return g


def choose_rank(eigvals: torch.Tensor, eps: float) -> int:
    """Smallest r with sum_{i<=r} lambda_i >= (1 - eps) * sum lambda (Definition 3.1)."""
    frac = torch.cumsum(eigvals, 0) / eigvals.sum()
    return int(torch.searchsorted(frac, torch.tensor(1 - eps, dtype=frac.dtype)).item()) + 1


@torch.no_grad()
def _forward_pre(pre, x):
    return pre(x)


def compute_active_subspace(pre: nn.Module, post: nn.Module, loader, *, r: int | None = None,
                            eps: float = 0.05, sketch_size: int | None = 200,
                            max_samples: int | None = None, device="cpu") -> ActiveSubspace:
    """Estimate the active subspace of the loss w.r.t. the layer-l neurons.

    If `sketch_size` is None the full gradient matrix is kept in memory (exact SVD);
    otherwise frequent directions with `sketch_size` rows is used.
    If `r` is None it is chosen from `eps`.
    """
    pre.eval(); post.eval()
    fd, rows, seen = None, [], 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if max_samples is not None:
            take = max_samples - seen
            if take <= 0:
                break
            x, y = x[:take], y[:take]
        g = per_sample_grads(post, _forward_pre(pre, x), y)
        seen += g.shape[0]
        if sketch_size is None:
            rows.append(g.double().cpu())
        else:
            if fd is None:
                fd = FrequentDirections(g.shape[1], sketch_size, device=g.device)
            fd.update(g)
    B = torch.cat(rows) if sketch_size is None else fd.sketch()
    _, s, vh = torch.linalg.svd(B, full_matrices=False)
    eigvals = s ** 2 / seen                     # eigenvalues of C = G^T G / N
    if r is None:
        r = choose_rank(eigvals, eps)
    r = min(r, vh.shape[0])
    return ActiveSubspace(V=vh[:r].T.float().cpu(), eigvals=eigvals.float().cpu(), r=r)
