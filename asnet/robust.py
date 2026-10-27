# Fine-tuning against universal perturbations.
#
# Every `refresh` steps a new universal perturbation v is built on the current
# model (by `make_v`), and each batch is trained half clean, half x + v.
# With make_v = AS attack this is universal adversarial training in the spirit of
# Shafahi et al. (AAAI 2020), but with the active-subspace attack generating v.
# make_v = random noise of the same norm gives the control experiment.

import torch
import torch.nn.functional as F

from .attack import as_universal_attack
from .data import X_MAX, X_MIN


def as_perturbation(Xa, ya, delta, max_iter=30, seed=0):
    def make_v(model, step):
        return as_universal_attack(model, Xa, ya, delta=delta, max_iter=max_iter,
                                   seed=seed + step, log=lambda *a: None).v
    return make_v


def noise_perturbation(shape, delta, seed=0):
    g = torch.Generator().manual_seed(seed)

    def make_v(model, step):
        v = torch.randn(shape, generator=g)
        return (v / v.norm() * delta).to(next(model.parameters()).device)
    return make_v


def adv_finetune(model, loader, make_v, epochs=5, lr=0.005, refresh=60, frac=0.5,
                 device="cpu", log=print):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    step, v = 0, None
    for ep in range(epochs):
        running, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if v is None or step % refresh == 0:
                model.eval()
                v = make_v(model, step).detach()
            model.train()
            m = int(len(x) * frac)
            x = torch.cat([(x[:m] + v).clamp(X_MIN, X_MAX), x[m:]])
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(x)
            n += len(x)
            step += 1
        log(f"epoch {ep + 1}/{epochs}  loss {running / n:.4f}  |v| {v.norm():.2f}")
    model.eval()
    return model
