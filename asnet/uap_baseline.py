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
