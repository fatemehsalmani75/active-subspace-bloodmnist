# Active subspace of the loss w.r.t. the neurons of layer l:
#   C = E[g g^T],  g = d loss(post(z), y) / dz,  z = pre(x)
# Method: Cui et al., "Active Subspace of Neural Networks: Structural Analysis and
# Universal Attacks", SIAM J. Math. Data Sci. 2020 (Def. 3.1, Alg. 3.2).
# Frequent directions: Liberty 2013, Ghashami et al. 2016.

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class FrequentDirections:
    """Sketch B (ell x n) with B^T B ~ G^T G, fed one block of rows at a time."""

    def __init__(self, n, ell, device=None):
        self.ell = ell
        self.B = torch.zeros(2 * ell, n, device=device, dtype=torch.float64)
        self.k = 0  # filled rows

    def update(self, rows):
        rows = rows.to(self.B)
        for i in range(0, len(rows), self.ell):
            r = rows[i:i + self.ell]
            if self.k + len(r) > len(self.B):
                self._shrink()
            self.B[self.k:self.k + len(r)] = r
            self.k += len(r)

    def _shrink(self):
        _, s, vt = torch.linalg.svd(self.B[:self.k], full_matrices=False)
        m = min(self.ell, len(s))
        d = s[self.ell - 1] ** 2 if len(s) >= self.ell else 0.0
        s = torch.sqrt(torch.clamp(s[:m] ** 2 - d, min=0))
        self.B.zero_()
        self.B[:m] = s[:, None] * vt[:m]
        self.k = m

    def sketch(self):
        return self.B[:self.k]


@dataclass
class ActiveSubspace:
    V: torch.Tensor        # n x r
    eigvals: torch.Tensor  # leading eigenvalues of C
    r: int

    @property
    def energy(self):
        return torch.cumsum(self.eigvals, 0) / self.eigvals.sum()


def input_grads(post, z, y):
    # post must be in eval mode, otherwise batchnorm mixes the samples
    z = z.detach().requires_grad_(True)
    loss = F.cross_entropy(post(z), y, reduction="sum")
    return torch.autograd.grad(loss, z)[0]


def choose_rank(eigvals, eps):
    # smallest r with sum_{i<=r} lambda_i >= (1 - eps) * sum lambda_i
    frac = torch.cumsum(eigvals, 0) / eigvals.sum()
    return int((frac < 1 - eps).sum().item()) + 1


def compute_active_subspace(pre, post, loader, r=None, eps=0.05, sketch_size=200,
                            max_samples=None, device="cpu"):
    """sketch_size=None keeps all gradients and uses an exact SVD."""
    pre.eval()
    post.eval()
    fd, rows, N = None, [], 0
    for x, y in loader:
        if max_samples is not None and N >= max_samples:
            break
        if max_samples is not None:
            x, y = x[:max_samples - N], y[:max_samples - N]
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            z = pre(x)
        g = input_grads(post, z, y)
        N += len(g)
        if sketch_size is None:
            rows.append(g.double().cpu())
        else:
            if fd is None:
                fd = FrequentDirections(g.shape[1], sketch_size, g.device)
            fd.update(g)

    G = torch.cat(rows) if sketch_size is None else fd.sketch()
    _, s, vt = torch.linalg.svd(G, full_matrices=False)
    lam = s ** 2 / N
    if r is None:
        r = choose_rank(lam, eps)
    r = min(r, len(vt))
    return ActiveSubspace(V=vt[:r].T.float().cpu(), eigvals=lam.float().cpu(), r=r)
