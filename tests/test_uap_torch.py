# uap_baseline_torch must give the same perturbation as ART's UAP (uap_baseline)
# when both visit the images in the same order.

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asnet.models import small_cnn, vgg19
from asnet.uap_baseline import uap_baseline, uap_baseline_torch

pytest.importorskip("art")


@pytest.mark.parametrize("arch,delta", [("small", 2.0), ("small", 6.0)])
def test_torch_uap_matches_art(arch, delta):
    torch.manual_seed(0)
    model = (small_cnn(8, image_size=32) if arch == "small" else vgg19(8)).eval()
    X = torch.rand(12, 3, 32, 32) * 2 - 1
    random.seed(123)
    v_art = uap_baseline(model, X, 8, delta=delta, max_iter=2)
    random.seed(123)
    v_torch = uap_baseline_torch(model, X, 8, delta=delta, max_iter=2)
    assert v_art.norm() > 0
    assert torch.allclose(v_art, v_torch, atol=1e-4), (v_art - v_torch).abs().max()
