# Baselines for universal perturbations:
#   uap_baseline: UAP (Moosavi-Dezfooli et al., CVPR 2017) using the implementation in
#                 IBM's Adversarial Robustness Toolbox (MIT license);
#   sgd_uap:      stochastic gradient ascent on the clamped cross-entropy over the
#                 attack set, projected on the l2 ball (Shafahi et al., AAAI 2020, Alg. 1).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import X_MAX, X_MIN


def uap_baseline(model, X, num_classes, delta, max_iter=10, overshoot=0.02, device="cpu"):
    from art.attacks.evasion import UniversalPerturbation
    from art.estimators.classification import PyTorchClassifier

    model.eval()
    clf = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                            input_shape=tuple(X.shape[1:]), nb_classes=num_classes,
                            clip_values=(X_MIN, X_MAX),
                            device_type="gpu" if str(device).startswith("cuda") else "cpu")
    attack = UniversalPerturbation(clf, attacker="deepfool",
                                   attacker_params={"epsilon": overshoot, "verbose": False},
                                   delta=0.0, max_iter=max_iter, eps=delta, norm=2,
                                   verbose=False)
    attack.generate(X.detach().cpu().numpy().astype(np.float32))
    v = torch.from_numpy(np.asarray(attack.noise, dtype=np.float32))
    return v.reshape(1, *X.shape[1:]).to(X.device)


def sgd_uap(model, X, y, delta, epochs=10, bs=32, lr=None, beta=9.0, seed=0):
    """Universal l2 perturbation by minibatch gradient ascent on min(CE, beta)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    v = torch.zeros_like(X[:1])
    lr = lr or delta / 10
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(bs):
            idx = idx.to(X.device)
            v.requires_grad_(True)
            ce = F.cross_entropy(model((X[idx] + v).clamp(X_MIN, X_MAX)), y[idx], reduction="none")
            gv = torch.autograd.grad(ce.clamp(max=beta).mean(), v)[0]
            with torch.no_grad():
                v = v + lr * gv / gv.norm().clamp_min(1e-12)
                n = v.norm()
                v = v * (delta / n) if n > delta else v
    return v.detach()


# ---------- the same UAP + DeepFool, written directly in PyTorch ----------
#
# A line-by-line port of ART's UniversalPerturbation(attacker="deepfool", norm=2) and
# DeepFool(nb_grads >= num_classes), so it gives the same perturbation as uap_baseline
# (tests/test_uap_torch.py checks this), but it stays on the GPU and gets all the class
# gradients of an image from one batched backward pass, where ART does one backward
# pass per class and copies to numpy at every step. Details kept from ART:
#   - DeepFool steps are scaled by (clip_max - clip_min) and clipped to the image range;
#   - the overshoot is applied once at the end: x + (1 + overshoot) * (x_df - x);
#   - x + v is not clipped before DeepFool, only the final x + v for the fooling rate;
#   - v is replaced (not summed) by x_adv - x and rescaled onto the l2 ball.

def _logits_and_jacobian(model, x, nc):
    xb = x.expand(nc, *x.shape[1:]).clone().requires_grad_(True)
    out = model(xb)
    jac = torch.autograd.grad(out.diagonal().sum(), xb)[0]
    return out[0].detach(), jac.flatten(1)


def _deepfool_one(model, x, nc, overshoot, max_iter=100, tol=10e-8):
    rng_scale = X_MAX - X_MIN
    batch = x.clone()
    f, grd = _logits_and_jacobian(model, batch, nc)
    k0 = int(f.argmax())
    for _ in range(max_iter):
        grad_diff = grd - grd[k0]
        f_diff = f - f[k0]
        value = f_diff.abs() / (grad_diff.norm(dim=1) + tol)
        value[k0] = float("inf")
        l = int(value.argmin())
        r = f_diff[l].abs() / (grad_diff[l].norm() ** 2 + tol) * grad_diff[l]
        batch = (batch + r.view_as(batch) * rng_scale).clamp(X_MIN, X_MAX)
        f, grd = _logits_and_jacobian(model, batch, nc)
        if int(f.argmax()) != k0:
            break
    return (x + (1 + overshoot) * (batch - x)).clamp(X_MIN, X_MAX)


def uap_baseline_torch(model, X, num_classes, delta, max_iter=10, overshoot=0.02,
                       df_max_iter=100, seed=None, device=None):
    """Same arguments and result as uap_baseline. seed fixes the order in which the images
    are visited (ART uses Python's global random module; seed=None does the same)."""
    import random

    model.eval()
    rng = random if seed is None else random.Random(seed)
    with torch.no_grad():
        y0 = torch.cat([model(X[i:i + 256]).argmax(1) for i in range(0, len(X), 256)])
    noise = torch.zeros_like(X[:1])
    n, fooling, it = len(X), 0.0, 0
    while fooling < 1.0 and it < max_iter:
        for i in rng.sample(range(n), n):
            x_i = X[i:i + 1]
            with torch.no_grad():
                cur = int(model(x_i + noise).argmax(1))
            if cur != int(y0[i]):
                continue
            adv = _deepfool_one(model, x_i + noise, num_classes, overshoot, df_max_iter)
            with torch.no_grad():
                new = int(model(adv).argmax(1))
            if new != cur:
                noise = adv - x_i
                nrm = noise.norm()
                noise = noise * (delta / nrm).clamp(max=1.0) if nrm > 0 else noise * 0
        it += 1
        with torch.no_grad():
            pred = torch.cat([model((X[i:i + 256] + noise).clamp(X_MIN, X_MAX)).argmax(1)
                              for i in range(0, n, 256)])
        fooling = (pred != y0).float().mean().item()
    return noise.detach()
