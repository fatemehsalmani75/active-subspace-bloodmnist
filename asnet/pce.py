# Polynomial chaos expansion (Hermite, total degree <= p) used as the output layer
# of ASNet (Cui et al., SIAM J. Math. Data Sci. 2020, Sec. 3.3).

import itertools
import math

import torch
import torch.nn as nn


def _partitions(n, largest=None):
    largest = n if largest is None else largest
    if n == 0:
        yield ()
        return
    for k in range(min(n, largest), 0, -1):
        for rest in _partitions(n - k, k):
            yield (k,) + rest


def multi_indices(r, p):
    """Multi-indices with |alpha| <= p as lists of (variable, degree); C(r+p, p) of them."""
    out = [[]]
    for total in range(1, p + 1):
        for parts in _partitions(total):
            for vs in itertools.combinations(range(r), len(parts)):
                for degs in set(itertools.permutations(parts)):
                    out.append(list(zip(vs, degs)))
    return out


def hermite(z, p):
    """Normalized probabilists' Hermite polynomials, shape (B, r, p+1)."""
    H = [torch.ones_like(z), z]
    for k in range(1, p):
        H.append(z * H[k] - k * H[k - 1])
    H = torch.stack(H[:p + 1], -1)
    c = torch.tensor([math.sqrt(math.factorial(k)) for k in range(p + 1)],
                     dtype=z.dtype, device=z.device)
    return H / c


class PCELayer(nn.Module):
    def __init__(self, r, p, out_features):
        super().__init__()
        self.r, self.p = r, p
        idx = multi_indices(r, p)
        self.num_terms = len(idx)
        var = torch.zeros(len(idx), max(p, 1), dtype=torch.long)
        deg = torch.zeros_like(var)   # unused slots point to He_0 = 1
        for j, a in enumerate(idx):
            for k, (v, d) in enumerate(a):
                var[j, k], deg[j, k] = v, d
        self.register_buffer("var", var)
        self.register_buffer("deg", deg)
        self.register_buffer("mu", torch.zeros(r))
        self.register_buffer("sigma", torch.ones(r))
        self.linear = nn.Linear(len(idx), out_features)

    def basis(self, z):
        H = hermite((z - self.mu) / self.sigma, self.p)
        return H[:, self.var, self.deg].prod(-1)

    def forward(self, z):
        return self.linear(self.basis(z))

    @torch.no_grad()
    def fit_least_squares(self, Z, Y, ridge=1e-6):
        self.mu.copy_(Z.mean(0))
        self.sigma.copy_(Z.std(0).clamp_min(1e-8))
        A = self.basis(Z).double()
        A = torch.cat([A, torch.ones(len(A), 1, dtype=A.dtype)], 1)
        coef = torch.linalg.solve(A.T @ A + ridge * torch.eye(A.shape[1], dtype=A.dtype),
                                  A.T @ Y.double())
        self.linear.weight.copy_(coef[:-1].T.float())
        self.linear.bias.copy_(coef[-1].float())
