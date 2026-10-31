# quick checks on random data, no download needed: pytest -q

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asnet.active_subspace import FrequentDirections, choose_rank, compute_active_subspace
from asnet.asnet import build_asnet, finetune_asnet
from asnet.attack import as_universal_attack, fooling_rate
from asnet.models import small_cnn, split_model, vgg19
from asnet.pce import PCELayer, multi_indices

torch.manual_seed(0)


def _loader(n=64, size=32, bs=16):
    X = torch.rand(n, 3, size, size) * 2 - 1
    y = torch.randint(0, 8, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=bs), X, y


def test_multi_index_count():
    for r, p in [(3, 2), (5, 3), (50, 2)]:
        assert len(multi_indices(r, p)) == math.comb(r + p, p)


def test_pce_least_squares_fits_quadratic():
    Z = torch.randn(2000, 3)
    Y = torch.stack([Z[:, 0] ** 2 + Z[:, 1] * Z[:, 2], 2 * Z[:, 0] - 1], 1)
    layer = PCELayer(3, 2, 2)
    layer.fit_least_squares(Z, Y)
    assert torch.allclose(layer(Z), Y, atol=1e-3)


def test_frequent_directions_matches_exact():
    G = torch.randn(500, 40, dtype=torch.float64) @ torch.diag(torch.logspace(0, -3, 40, dtype=torch.float64))
    fd = FrequentDirections(40, 20)
    fd.update(G)
    B = fd.sketch()
    err = torch.linalg.matrix_norm(G.T @ G - B.T @ B, ord=2)
    # FD guarantee: ||G^T G - B^T B||_2 <= ||G||_F^2 / ell
    assert err <= torch.linalg.matrix_norm(G) ** 2 / 20 + 1e-9


def test_choose_rank():
    assert choose_rank(torch.tensor([90.0, 6.0, 3.0, 1.0]), 0.05) == 2


def test_split_is_exact():
    model = vgg19().eval()
    loader, X, _ = _loader()
    pre, post, n = split_model(model, 8, X)
    with torch.no_grad():
        assert torch.allclose(post(pre(X[:4])), model(X[:4]), atol=1e-5)


def test_active_subspace_and_asnet_pipeline():
    model = small_cnn(image_size=32).eval()
    loader, X, y = _loader()
    sub = compute_active_subspace(*split_model(model, 3, X)[:2], loader, r=None, eps=0.05)
    assert sub.V.shape[1] == sub.r and torch.allclose(sub.V.T @ sub.V, torch.eye(sub.r), atol=1e-4)
    net, _ = build_asnet(model, 3, loader, loader, r=4, p=2)
    assert net(X[:2]).shape == (2, 8)
    finetune_asnet(net, model, loader, epochs=1, log=lambda *_: None)


def test_attack_respects_radius():
    model = small_cnn(image_size=32).eval()
    _, X, y = _loader(32)
    res = as_universal_attack(model, X, y, delta=3.0, alpha=2, max_iter=5, log=lambda *_: None)
    assert res.v.norm() <= 3.0 + 1e-5
    assert 0.0 <= fooling_rate(model, X, res.v) <= 1.0


def test_purification_removes_subspace():
    from asnet.defense import Purified, input_active_subspace, overlap, roc_auc
    model = small_cnn(image_size=32).eval()
    loader, X, _ = _loader()
    sub = input_active_subspace(model, loader, k=5, sketch_size=20)
    assert torch.allclose(sub.U.T @ sub.U, torch.eye(5), atol=1e-4)
    net = Purified(model, sub)
    d = (net.purify(X[:4]).flatten(1) - sub.mu) @ sub.U
    assert d.abs().max() < 1e-4
    assert abs(overlap(sub.U[:, 0].view(1, 3, 32, 32), sub) - 1) < 1e-4
    assert roc_auc([0, 1], [2, 3]) == 1.0 and roc_auc([1], [1]) == 0.5


def test_mahalanobis_and_finetune():
    from asnet.defense import input_active_subspace, mahalanobis_scores, projection_variance
    from asnet.robust import adv_finetune, as_perturbation, noise_perturbation
    model = small_cnn(image_size=32)
    loader, X, y = _loader()
    sub = input_active_subspace(model.eval(), loader, k=4, sketch_size=20)
    var = projection_variance(sub, X)
    s = mahalanobis_scores(sub, X, var)
    assert s.shape == (len(X),) and abs(s.mean().item() / 4 - 1) < 0.2   # about k on clean data
    _, t = adv_finetune(model, loader, noise_perturbation(X[:1].shape, 2.0), epochs=1, refresh=2, log=lambda *a: None)
    adv_finetune(model, loader, as_perturbation(X[:16], y[:16], 2.0, max_iter=2), epochs=1, refresh=3,
                 log=lambda *a: None)
    assert not model.training and t > 0


def test_uat_and_pgd():
    from asnet.robust import pgd_accuracy, pgd_finetune, pgd_l2, uat_finetune
    model = small_cnn(image_size=32)
    loader, X, y = _loader()
    xa = pgd_l2(model.eval(), X[:8], y[:8], eps=1.5, steps=3)
    assert ((xa - X[:8]).flatten(1).norm(dim=1) <= 1.5 + 1e-4).all()
    uat_finetune(model, loader, delta=2.0, epochs=1, log=lambda *a: None)
    pgd_finetune(model, loader, eps=1.0, epochs=1, steps=2, log=lambda *a: None)
    assert 0 <= pgd_accuracy(model, X[:16], y[:16], eps=1.0, steps=2) <= 1


def test_input_spectrum():
    from asnet.spectrum import input_spectrum
    model = small_cnn(image_size=32).eval()
    loader, X, y = _loader()
    s = input_spectrum(model, loader)
    assert 0 < s["lambda1"] <= s["trace"] + 1e-6
    assert 0 < s["lambda1_share"] <= s["top5_share"] <= 1 + 1e-6
    # exact check of lambda1 against a full SVD of the gradient matrix
    G = []
    for x, yy in loader:
        x = x.clone().requires_grad_(True)
        l = torch.nn.functional.cross_entropy(model(x), yy, reduction="sum")
        G.append(torch.autograd.grad(l, x)[0].flatten(1))
    G = torch.cat(G).double()
    lam1 = (torch.linalg.svdvals(G)[0] ** 2 / len(G)).item()
    assert abs(s["lambda1"] - lam1) / lam1 < 1e-4
