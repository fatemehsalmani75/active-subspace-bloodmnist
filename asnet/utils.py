# Copyright (c) 2026 Fatemeh Salmani. Licensed under the MIT License (see LICENSE).
"""Training / evaluation helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> str:
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model: nn.Module, loader, device="cpu") -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_classifier(model, train_loader, val_loader, *, epochs=30, lr=0.05,
                     weight_decay=5e-4, device="cpu", log=print):
    """SGD + cosine schedule; keeps the weights with the best validation accuracy."""
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, val_loader, device)
        log(f"[train] epoch {ep + 1}/{epochs} val acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


def stack_dataset(ds, n: int | None = None, device="cpu"):
    """Materialise (X, y) tensors from a dataset (used by the attacks)."""
    n = len(ds) if n is None else min(n, len(ds))
    xs, ys = zip(*(ds[i] for i in range(n)))
    return torch.stack(xs).to(device), torch.as_tensor(ys).long().to(device)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
