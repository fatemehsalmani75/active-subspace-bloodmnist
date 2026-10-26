"""Active-subspace analysis + ASNet compression of a trained network.

Example:
    python scripts/compress.py --ckpt checkpoints/vgg19_s64.pt --cut 16 --r 50 --p 2
"""

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asnet.asnet import build_asnet, finetune_asnet
from asnet.data import NUM_CLASSES, get_datasets, make_loader, subset
from asnet.models import build_model, count_params
from asnet.utils import evaluate, get_device, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cut", type=int, required=True, help="number of units kept in the pre-model")
    ap.add_argument("--r", type=int, default=50, help="active neurons (use --r 0 to pick by --eps)")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--p", type=int, default=2, help="PCE total degree")
    ap.add_argument("--sketch", type=int, default=200, help="frequent-directions rows (0 = exact)")
    ap.add_argument("--as-samples", type=int, default=5000)
    ap.add_argument("--fit-samples", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr-pre", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-5)
    ap.add_argument("--l1", type=float, default=0.0, help="> 0 trains the sparse ASNet-s")
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

    train, val, test = get_datasets(ck["size"])
    as_loader = make_loader(subset(train, a.as_samples, a.seed), 128)
    fit_loader = make_loader(subset(train, a.fit_samples, a.seed + 1), 256)
    test_loader = make_loader(test, 256)

    base_acc = evaluate(model, test_loader, dev)
    net, sub = build_asnet(model, a.cut, as_loader, fit_loader, r=a.r or None, p=a.p,
                           eps=a.eps, sketch_size=a.sketch or None, as_samples=a.as_samples,
                           fit_samples=a.fit_samples, device=dev)
    init_acc = evaluate(net, test_loader, dev)
    print(f"r = {sub.r}, energy captured = {sub.energy[sub.r - 1]:.4f}")
    print(f"original acc {base_acc:.4f} | ASNet after LS init {init_acc:.4f}")

    if a.epochs > 0:
        teacher = copy.deepcopy(model)
        finetune_asnet(net, teacher, make_loader(train, 128, shuffle=True), epochs=a.epochs,
                       beta=a.beta, lr_pre=a.lr_pre, lr_head=a.lr_head, l1=a.l1, device=dev)
    final_acc = evaluate(net, test_loader, dev)
    p_orig, p_new = count_params(model), count_params(net)
    nnz = sum((p != 0).sum().item() for p in net.parameters())
    print(f"ASNet acc {final_acc:.4f} | params {p_orig:,} -> {p_new:,} "
          f"({p_orig / p_new:.2f}x) | non-zeros {nnz:,}")

    tag = f"{ck['model']}_s{ck['size']}_cut{a.cut}_r{sub.r}_p{a.p}" + ("_sparse" if a.l1 else "")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / f"asnet_{tag}.pt")
    torch.save({"eigvals": sub.eigvals, "V": sub.V}, out / f"active_subspace_{tag}.pt")
    save_json({"cut": a.cut, "r": sub.r, "p": a.p, "orig_acc": base_acc, "init_acc": init_acc,
               "asnet_acc": final_acc, "params_orig": p_orig, "params_asnet": p_new,
               "nonzeros_asnet": nnz, "eigvals": sub.eigvals[:200].tolist()},
              out / f"compress_{tag}.json")


if __name__ == "__main__":
    main()
