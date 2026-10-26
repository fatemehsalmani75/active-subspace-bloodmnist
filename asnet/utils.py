import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name=None):
    return name or ("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def train_classifier(model, train_loader, val_loader, epochs=30, lr=0.05, wd=5e-4,
                     device="cpu", log=print):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        acc = evaluate(model, val_loader, device)
        log(f"epoch {ep + 1}/{epochs}  val acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


def stack_dataset(ds, n=None, device="cpu"):
    n = len(ds) if n is None else min(n, len(ds))
    xs, ys = zip(*(ds[i] for i in range(n)))
    return torch.stack(xs).to(device), torch.as_tensor(ys).long().to(device)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
