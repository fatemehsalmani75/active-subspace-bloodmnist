# Fine-tuning against universal perturbations, and the baselines it is compared with.
#
# adv_finetune: every `refresh` steps a new universal perturbation v is built on the
#   current model by `make_v`; each batch is trained half clean, half x + v.
#   make_v = AS attack   -> our method
#   make_v = UAP (ART)   -> the same loop with DeepFool-based UAP
#   make_v = noise       -> control
# uat_finetune: Universal Adversarial Training (Shafahi et al., AAAI 2020), the
#   "simultaneous" version: one backward pass updates the weights (descent) and the
#   universal perturbation (ascent). Written here for the l2 ball.
# pgd_finetune: per-image PGD adversarial training (Madry et al., ICLR 2018), l2.

import time

import torch
import torch.nn.functional as F

from .attack import as_universal_attack
from .data import X_MAX, X_MIN


# ---------- perturbation generators for adv_finetune ----------

def as_perturbation(Xa, ya, delta, max_iter=30, seed=0):
    def make_v(model, step):
        return as_universal_attack(model, Xa, ya, delta=delta, max_iter=max_iter,
                                   seed=seed + step, log=lambda *a: None).v
    return make_v


def uap_perturbation(Xa, num_classes, delta, max_iter=10):
    from .uap_baseline import uap_baseline

    def make_v(model, step):
        return uap_baseline(model, Xa, num_classes, delta=delta, max_iter=max_iter,
                            device=str(Xa.device))
    return make_v


def noise_perturbation(shape, delta, seed=0):
    g = torch.Generator().manual_seed(seed)

    def make_v(model, step):
        v = torch.randn(shape, generator=g)
        return (v / v.norm() * delta).to(next(model.parameters()).device)
    return make_v


# ---------- helpers ----------

def _optimizer(model, lr, steps):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def project_l2_batch(d, eps):
    n = d.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *[1] * (d.dim() - 1))
    return d * (eps / n).clamp(max=1.0)


# ---------- training loops (each returns the model and the wall-clock seconds) ----------

def adv_finetune(model, loader, make_v, epochs=5, lr=0.005, refresh=60, frac=0.5,
                 device="cpu", log=print):
    model.to(device)
    opt, sched = _optimizer(model, lr, epochs * len(loader))
    step, v = 0, None
    _sync(device); t0 = time.time()
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
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += loss.item() * len(x); n += len(x); step += 1
        log(f"epoch {ep + 1}/{epochs}  loss {running / n:.4f}  |v| {v.norm():.2f}")
    _sync(device)
    model.eval()
    return model, time.time() - t0


def uat_finetune(model, loader, delta, epochs=5, lr=0.005, v_step=None, frac=0.5,
                 device="cpu", log=print):
    model.to(device)
    opt, sched = _optimizer(model, lr, epochs * len(loader))
    x0 = next(iter(loader))[0]
    v = torch.zeros_like(x0[:1]).to(device)
    v_step = v_step or delta / 4
    _sync(device); t0 = time.time()
    for ep in range(epochs):
        running, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.train()
            m = int(len(x) * frac)
            v.requires_grad_(True)
            xa = (x[:m] + v).clamp(X_MIN, X_MAX)
            loss = F.cross_entropy(model(torch.cat([xa, x[m:]])), y)
            opt.zero_grad()
            loss.backward()
            g = v.grad.detach()
            opt.step(); sched.step()
            with torch.no_grad():   # ascent step on v, normalized, then back into the ball
                v = project_l2_batch(v.detach() + v_step * g / g.norm().clamp_min(1e-12), delta)
            running += loss.item() * len(x); n += len(x)
        log(f"epoch {ep + 1}/{epochs}  loss {running / n:.4f}  |v| {v.norm():.2f}")
    _sync(device)
    model.eval()
    return model, time.time() - t0


def pgd_l2(model, x, y, eps, steps=5, alpha=None):
    """Per-image l2 PGD (random start), keeps images in range."""
    alpha = alpha or 2.5 * eps / steps
    was_training = model.training
    model.eval()
    d = project_l2_batch(torch.randn_like(x), eps * torch.rand(1).item())
    for _ in range(steps):
        d.requires_grad_(True)
        loss = F.cross_entropy(model((x + d).clamp(X_MIN, X_MAX)), y)
        g = torch.autograd.grad(loss, d)[0]
        g = g / g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *[1] * (x.dim() - 1))
        d = project_l2_batch(d.detach() + alpha * g, eps)
    model.train(was_training)
    return (x + d).clamp(X_MIN, X_MAX).detach()


def pgd_finetune(model, loader, eps, epochs=5, lr=0.005, steps=5, frac=0.5, device="cpu", log=print):
    model.to(device)
    opt, sched = _optimizer(model, lr, epochs * len(loader))
    _sync(device); t0 = time.time()
    for ep in range(epochs):
        running, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            m = int(len(x) * frac)
            xa = pgd_l2(model, x[:m], y[:m], eps, steps)
            model.train()
            loss = F.cross_entropy(model(torch.cat([xa, x[m:]])), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += loss.item() * len(x); n += len(x)
        log(f"epoch {ep + 1}/{epochs}  loss {running / n:.4f}")
    _sync(device)
    model.eval()
    return model, time.time() - t0


@torch.no_grad()
def _acc(model, X, y, bs=256):
    return torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]).eq(y).float().mean().item()


def pgd_accuracy(model, X, y, eps, steps=10, bs=128):
    """Accuracy under per-image l2 PGD."""
    correct = 0
    for i in range(0, len(X), bs):
        xa = pgd_l2(model, X[i:i + bs], y[i:i + bs], eps, steps)
        with torch.no_grad():
            correct += model(xa).argmax(1).eq(y[i:i + bs]).sum().item()
    return correct / len(X)
