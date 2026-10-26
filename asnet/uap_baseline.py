# UAP baseline (Moosavi-Dezfooli et al., CVPR 2017) using the implementation in
# IBM's Adversarial Robustness Toolbox (MIT license).

import numpy as np
import torch
import torch.nn as nn

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
