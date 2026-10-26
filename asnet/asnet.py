# ASNet = pre-model + active-subspace layer + PCE layer (Cui et al., SIAM J. Math.
# Data Sci. 2020, Alg. 3.1). l1 > 0 in finetune gives the sparse version (ASNet-s).

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .active_subspace import compute_active_subspace
from .models import split_model
from .pce import PCELayer


class ASNet(nn.Module):
    def __init__(self, pre, V, pce):
        super().__init__()
        self.pre = pre
        self.as_layer = nn.Linear(V.shape[0], V.shape[1], bias=False)
        with torch.no_grad():
            self.as_layer.weight.copy_(V.T)
        self.pce = pce

    def forward(self, x):
        return self.pce(self.as_layer(self.pre(x)))


@torch.no_grad()
def _features_and_logits(pre, proj, model, loader, n, device):
    Z, Y, seen = [], [], 0
    for x, _ in loader:
        x = x.to(device)
        Z.append(proj(pre(x)).cpu())
        Y.append(model(x).cpu())
        seen += len(x)
        if n is not None and seen >= n:
            break
    return torch.cat(Z)[:n], torch.cat(Y)[:n]


def build_asnet(model, cut, as_loader, fit_loader, r=50, p=2, eps=0.05, sketch_size=200,
                as_samples=None, fit_samples=None, device="cpu"):
    model = model.eval().to(device)
    x0 = next(iter(as_loader))[0].to(device)
    pre, post, _ = split_model(copy.deepcopy(model), cut, x0)
    pre, post = pre.to(device), post.to(device)

    sub = compute_active_subspace(pre, post, as_loader, r=r, eps=eps, sketch_size=sketch_size,
                                  max_samples=as_samples, device=device)
    proj = nn.Linear(sub.V.shape[0], sub.r, bias=False).to(device)
    with torch.no_grad():
        proj.weight.copy_(sub.V.T)

    # PCE is fitted to the logits of the original network
    Z, Y = _features_and_logits(pre, proj, model, fit_loader, fit_samples, device)
    pce = PCELayer(sub.r, p, Y.shape[1])
    pce.fit_least_squares(Z, Y)
    return ASNet(pre, sub.V, pce).to(device), sub


def kd_loss(out, teacher_out, y, beta=0.1, T=1.0):
    ce = F.cross_entropy(out, y)
    kl = F.kl_div(F.log_softmax(out / T, 1), F.softmax(teacher_out / T, 1),
                  reduction="batchmean")
    return (1 - beta) * ce + beta * T * T * kl


@torch.no_grad()
def soft_threshold_(params, t):
    for w in params:
        w.copy_(w.sign() * (w.abs() - t).clamp(min=0))


def finetune_asnet(net, teacher, loader, epochs=50, beta=0.1, lr_pre=1e-4, lr_head=1e-5,
                   l1=0.0, device="cpu", log=print):
    teacher.eval()
    head_params = list(net.as_layer.parameters()) + list(net.pce.parameters())
    opt = torch.optim.Adam([{"params": net.pre.parameters(), "lr": lr_pre},
                            {"params": head_params, "lr": lr_head}])
    for ep in range(epochs):
        net.train()
        running, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t = teacher(x)
            loss = kd_loss(net(x), t, y, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if l1 > 0:  # proximal step for the l1 penalty
                for g in opt.param_groups:
                    soft_threshold_([w for w in g["params"] if w.dim() > 1], g["lr"] * l1)
            running += loss.item() * len(x)
            n += len(x)
        log(f"epoch {ep + 1}/{epochs}  loss {running / n:.4f}")
    return net
