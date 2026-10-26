"""Train the baseline network on BloodMNIST.

Example:
    python scripts/train.py --model vgg19 --size 64 --epochs 30
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asnet.data import NUM_CLASSES, get_datasets, make_loader
from asnet.models import build_model, count_params
from asnet.utils import evaluate, get_device, save_json, set_seed, train_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vgg19", choices=["vgg19", "smallcnn"])
    ap.add_argument("--size", type=int, default=64, choices=[28, 64, 128, 224])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = get_device(a.device)
    train, val, test = get_datasets(a.size, augment=True)
    tl = make_loader(train, a.batch_size, shuffle=True)
    vl, te = make_loader(val, 256), make_loader(test, 256)

    model = build_model(a.model, NUM_CLASSES, a.size)
    print(f"{a.model}: {count_params(model):,} parameters")
    model, best_val = train_classifier(model, tl, vl, epochs=a.epochs, lr=a.lr, device=dev)
    test_acc = evaluate(model, te, dev)
    print(f"best val acc {best_val:.4f} | test acc {test_acc:.4f}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{a.model}_s{a.size}.pt"
    torch.save({"model": a.model, "size": a.size, "state_dict": model.state_dict()}, ckpt)
    save_json({"model": a.model, "size": a.size, "val_acc": best_val, "test_acc": test_acc,
               "params": count_params(model)}, out / f"{a.model}_s{a.size}_train.json")
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
