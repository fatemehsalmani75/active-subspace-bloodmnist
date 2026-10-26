# Copyright (c) 2026 Fatemeh Salmani. Licensed under the MIT License (see LICENSE).
#
# Independent re-implementation of the polynomial-chaos-expansion (PCE) layer used
# by ASNet in: C. Cui et al., "Active Subspace of Neural Networks: Structural
# Analysis and Universal Attacks", SIAM J. Math. Data Sci. 2(4):1096-1122, 2020.
# doi:10.1137/19M1296070. Written from the paper; no code from
# https://github.com/chunfengc/ASNet was copied.
"""Total-degree Hermite polynomial chaos layer with least-squares initialisation."""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn


def multi_indices(r: int, p: int) -> list[list[tuple[int, int]]]:
    """All multi-indices with total degree <= p in r variables, each stored sparsely
    as a list of (variable, degree) pairs. Count = C(r + p, p)."""
    out = [[]]
    for deg in range(1, p + 1):
        for parts in _partitions(deg):
            for vars_ in itertools.combinations(range(r), len(parts)):
                for perm in set(itertools.permutations(parts)):
                    out.append(list(zip(vars_, perm)))
    return out


def _partitions(n, max_part=None):
    max_part = n if max_part is None else max_part
    if n == 0:
        yield ()
        return
    for k in range(min(n, max_part), 0, -1):
        for rest in _partitions(n - k, k):
            yield (k,) + rest


def hermite_table(z: torch.Tensor, p: int) -> torch.Tensor:
    """Orthonormal probabilists' Hermite polynomials He_k(z)/sqrt(k!), k=0..p.
    z: (B, r) -> (B, r, p+1)."""
    h = [torch.ones_like(z), z]
    for k in range(1, p):
        h.append(z * h[k] - k * h[k - 1])
    h = torch.stack(h[: p + 1], dim=-1)
    norm = torch.tensor([math.sqrt(math.factorial(k)) for k in range(p + 1)],
                        device=z.device, dtype=z.dtype)
    return h / norm


class PCELayer(nn.Module):
    """y = W phi(z_std) + b, where phi are the PCE basis functions of the
    standardised active variables z_std = (z - mu) / sigma."""

    def __init__(self, r: int, p: int, out_features: int):
        super().__init__()
        self.r, self.p = r, p
        idx = multi_indices(r, p)
        self.num_terms = len(idx)
        # pad every multi-index to p factors with (var 0, degree 0) == 1
        var = torch.zeros(self.num_terms, max(p, 1), dtype=torch.long)
        deg = torch.zeros_like(var)
        for j, mi in enumerate(idx):
            for k, (v, d) in enumerate(mi):
                var[j, k], deg[j, k] = v, d
        self.register_buffer("var", var)
        self.register_buffer("deg", deg)
        self.register_buffer("mu", torch.zeros(r))
        self.register_buffer("sigma", torch.ones(r))
        self.linear = nn.Linear(self.num_terms, out_features)

    def basis(self, z: torch.Tensor) -> torch.Tensor:
        z = (z - self.mu) / self.sigma
        H = hermite_table(z, self.p)                       # (B, r, p+1)
        vals = H[:, self.var, self.deg]                    # (B, M, p)
        return vals.prod(dim=-1)                           # (B, M)

    def forward(self, z):
        return self.linear(self.basis(z))

    @torch.no_grad()
    def fit_least_squares(self, Z: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-6):
        """Set mu/sigma from Z and solve min ||Phi W^T + b - Y||^2 (+ ridge)."""
        self.mu.copy_(Z.mean(0))
        self.sigma.copy_(Z.std(0).clamp_min(1e-8))
        Phi = self.basis(Z).double()
        Phi1 = torch.cat([Phi, torch.ones(Phi.shape[0], 1, dtype=Phi.dtype)], 1)
        A = Phi1.T @ Phi1 + ridge * torch.eye(Phi1.shape[1], dtype=Phi.dtype)
        coef = torch.linalg.solve(A, Phi1.T @ Y.double())   # (M+1, K)
        self.linear.weight.copy_(coef[:-1].T.float())
        self.linear.bias.copy_(coef[-1].float())
