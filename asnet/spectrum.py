# Spectrum of the input-gradient covariance C = E[g g^T], g = grad_x loss.
#
# To first order a perturbation v changes the loss by g^T v, so over the data
#   E[(g^T v)^2] = v^T C v <= ||v||^2 * lambda_1,  with equality at v = ||v|| u_1,
# while a random direction gets only ||v||^2 * tr(C) / D on average.
# lambda_1 (and its share of tr C) should therefore measure how much a single
# universal perturbation can hurt a model. This module computes those numbers.

import torch
import torch.nn.functional as F

from .active_subspace import FrequentDirections


def input_spectrum(model, loader, sketch_size=None, max_samples=None, device="cpu", top=200):
    """Returns dict with the leading eigenvalues of C, the exact trace of C (mean squared
    gradient norm) and derived measures. sketch_size=None keeps every gradient and
    takes an exact SVD (2000 images of 64x64x3 fit easily on a GPU); a number uses
    the frequent-directions sketch, which can underestimate lambda_1 when the spectrum is flat."""
    model.eval()
    fd, rows, N, trace = None, [], 0, 0.0
    for x, y in loader:
        if max_samples is not None and N >= max_samples:
            break
        if max_samples is not None:
            x, y = x[:max_samples - N], y[:max_samples - N]
        x, y = x.to(device), y.to(device)
        x = x.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x), y, reduction="sum")
        g = torch.autograd.grad(loss, x)[0].flatten(1)
        if sketch_size is None:
            rows.append(g.detach().float())
        else:
            if fd is None:
                fd = FrequentDirections(g.shape[1], sketch_size, g.device)
            fd.update(g)
        trace += g.double().pow(2).sum().item()
        N += len(g)
    G = torch.cat(rows) if sketch_size is None else fd.sketch()
    D = G.shape[1]
    s = torch.linalg.svdvals(G.double() if G.shape[0] * G.shape[1] < 3e7 else G)[:top]
    lam = (s.double() ** 2 / N).cpu()
    tr = trace / N
    return {
        "eigvals": lam,
        "lambda1": lam[0].item(),
        "trace": tr,
        "lambda1_share": lam[0].item() / tr,
        "top5_share": lam[:5].sum().item() / tr,
        "gain_over_random": lam[0].item() * D / tr,   # best universal direction vs a random one
    }
