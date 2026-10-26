"""Active-subspace universal attack vs. the UAP baseline.

Example:
    python scripts/attack.py --ckpt checkpoints/vgg19_s64.pt --delta 5 10 --n-train 200
    python scripts/attack.py --ckpt ... --target-class 3   # one-class attack (100 samples)
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asnet.attack import as_universal_attack, fooling_rate  # noqa: E402
from asnet.data import NUM_CLASSES, get_datasets, subset  # noqa: E402
from asnet.models import build_model  # noqa: E402
from asnet.utils import get_device, save_json, set_seed, stack_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--delta", type=float, nargs="+", default=[5.0, 10.0],
                    help="l2 radius of v in normalised input space; rescale for your image size")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--target-class", type=int, default=None,
                    help="attack only images of this class (one-class attack)")
    ap.add_argument("--alpha", type=int, default=10)
    ap.add_argument("--no-uap", action="store_true")
    ap.add_argument("--uap-iter", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = get_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu")
    model = build_model(ck["model"], NUM_CLASSES, ck["size"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    train, _, test = get_datasets(ck["size"])
    if a.target_class is not None:
        tr_idx = (train.labels.reshape(-1) == a.target_class).nonzero()[0].tolist()
        te_idx = (test.labels.reshape(-1) == a.target_class).nonzero()[0].tolist()
        g = torch.Generator().manual_seed(a.seed)
        tr_idx = [tr_idx[i] for i in torch.randperm(len(tr_idx), generator=g)[: a.n_train]]
        train_sub, test_sub = Subset(train, tr_idx), Subset(test, te_idx)
    else:
        train_sub, test_sub = subset(train, a.n_train, a.seed), test
    Xtr, ytr = stack_dataset(train_sub, device=dev)
    Xte, yte = stack_dataset(test_sub, device=dev)

    results = []
    for delta in a.delta:
        res = as_universal_attack(model, Xtr, ytr, delta=delta, alpha=a.alpha, seed=a.seed)
        row = {"delta": delta, "method": "active-subspace",
               "train_fooling": fooling_rate(model, Xtr, res.v),
               "test_fooling": fooling_rate(model, Xte, res.v),
               "norm": res.v.norm().item()}
        results.append(row); print(row)
        if not a.no_uap:
            from asnet.uap_baseline import uap_baseline
            v = uap_baseline(model, Xtr, NUM_CLASSES, delta=delta, max_iter=a.uap_iter, device=dev)
            row = {"delta": delta, "method": "UAP (ART)",
                   "train_fooling": fooling_rate(model, Xtr, v),
                   "test_fooling": fooling_rate(model, Xte, v), "norm": v.norm().item()}
            results.append(row); print(row)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        torch.save(res.v.cpu(), out / f"as_uap_delta{delta:g}.pt")

    tag = f"{ck['model']}_s{ck['size']}" + (f"_class{a.target_class}" if a.target_class is not None else "")
    save_json(results, Path(a.out) / f"attack_{tag}.json")


if __name__ == "__main__":
    main()
