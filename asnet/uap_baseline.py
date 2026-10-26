# Copyright (c) 2026 Fatemeh Salmani. Licensed under the MIT License (see LICENSE).
#
# Baseline: Universal Adversarial Perturbations (UAP) of S.-M. Moosavi-Dezfooli,
# A. Fawzi, O. Fawzi, P. Frossard, CVPR 2017 (doi:10.1109/CVPR.2017.17), with DeepFool
# (Moosavi-Dezfooli et al., CVPR 2016) as the inner attack.
# This file does not re-implement UAP: it calls the MIT-licensed implementation in
# IBM's Adversarial Robustness Toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox),
# installed as a dependency. No code from https://github.com/LTS4/universal was copied.
"""UAP baseline through the Adversarial Robustness Toolbox (ART)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import X_MAX, X_MIN


def uap_baseline(model: nn.Module, X: torch.Tensor, num_classes: int, *, delta: float,
                 max_iter: int = 10, deepfool_overshoot: float = 0.02,
                 device: str = "cpu") -> torch.Tensor:
    """Return a universal perturbation v (shape of one input) with ||v||_2 <= delta."""
    from art.attacks.evasion import UniversalPerturbation
    from art.estimators.classification import PyTorchClassifier

    model.eval()
    clf = PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(), input_shape=tuple(X.shape[1:]),
        nb_classes=num_classes, clip_values=(X_MIN, X_MAX),
        device_type="gpu" if str(device).startswith("cuda") else "cpu",
    )
    attack = UniversalPerturbation(
        classifier=clf, attacker="deepfool",
        attacker_params={"epsilon": deepfool_overshoot, "verbose": False}, delta=0.0,
        max_iter=max_iter, eps=delta, norm=2, verbose=False,
    )
    x_np = X.detach().cpu().numpy().astype(np.float32)
    attack.generate(x_np)
    v = torch.from_numpy(np.asarray(attack.noise, dtype=np.float32)).reshape(1, *X.shape[1:])
    return v.to(X.device)
