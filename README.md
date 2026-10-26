# Active Subspace of Neural Networks on BloodMNIST

Structural analysis, compression and universal adversarial attacks for a deep network trained on
**BloodMNIST** (8 blood-cell classes), using **active subspaces**.

This is an independent, from-scratch **PyTorch re-implementation** written by Fatemeh Salmani,
based on the methods of [Cui et al. (2020)](#acknowledgements-and-credits) and developed for her MSc thesis
*"Structural Analysis of Deep Neural Networks"* (Shahid Beheshti University, supervisor Dr. Masoud Hajarian).

> **What is new here:** the original paper evaluated CIFAR-10/100 and Fashion-MNIST. As far as I know,
> active-subspace compression and active-subspace universal attacks have not been published on
> BloodMNIST / MedMNIST; this repository applies them to a medical-imaging benchmark.
> It is **not** a reproduction of the paper's numbers.

## What the code does

| Step | Idea | Code |
|---|---|---|
| 1. Train | VGG-19 (BN) or the paper's 4-layer CNN on BloodMNIST | `scripts/train.py` |
| 2. Active subspace | Cut the network after layer *l*. The covariance of the loss gradients w.r.t. the layer-*l* neurons, **C = E[∇ℓ ∇ℓᵀ]**, has few large eigenvalues: those directions are the *active neurons*. Estimated with a streaming **frequent-directions** sketch. | `asnet/active_subspace.py` |
| 3. ASNet | Replace everything after layer *l* by an **AS layer** (projection on the *r* active directions) and a **polynomial chaos expansion (PCE)** layer, initialised by least squares and fine-tuned with distillation. `--l1 > 0` gives the sparse **ASNet-s**. | `asnet/asnet.py`, `asnet/pce.py`, `scripts/compress.py` |
| 4. Universal attack | One input-space perturbation **v** (‖v‖₂ ≤ δ) built recursively from the dominant active direction of the input gradients, compared with the classic **UAP** baseline. | `asnet/attack.py`, `asnet/uap_baseline.py`, `scripts/attack.py` |

## Quick start

```bash
pip install -r requirements.txt

# 1) train the baseline (BloodMNIST is downloaded automatically)
python scripts/train.py --model vgg19 --size 64 --epochs 30

# 2) active subspace + ASNet (keep the first 12 units, r = 50 active neurons, PCE degree 2)
python scripts/compress.py --ckpt checkpoints/vgg19_s64.pt --cut 12 --r 50 --p 2
python scripts/compress.py --ckpt checkpoints/vgg19_s64.pt --cut 12 --r 50 --l1 1e-4   # ASNet-s

# 3) universal attacks: active subspace vs UAP
python scripts/attack.py --ckpt checkpoints/vgg19_s64.pt --delta 5 10 --n-train 200
python scripts/attack.py --ckpt checkpoints/vgg19_s64.pt --target-class 3 --n-train 100

# tests (random data, no download)
pytest -q
```

Outputs go to `checkpoints/` and `results/` (JSON summaries, eigenvalues, saved perturbations).

### Notes on choices

- **Image size.** VGG-19 has five 2×2 pooling stages, so 28×28 inputs collapse to zero size. The default is the
  64×64 version of BloodMNIST (`--size 64`).
- **Defaults follow the paper:** PCE degree p = 2, r = 50 active neurons (or choose r with ε = 0.05),
  distillation weight β = 0.1, 50 epochs of Adam with learning rates 1e-4 (pre-model) / 1e-5 (AS + PCE layers),
  attack failure limit α = 10, 100 samples for one-class and 200 for all-class attacks, UAP capped at 10 iterations.
- **Attack radius.** Inputs are normalised to [-1, 1] and δ is measured in that space. The paper's δ = 5–10 was set
  for CIFAR images; rescale it for 64×64 inputs and report the value you use.
- **Fooling rate** = fraction of samples whose predicted label changes after adding v.
- The attack's step-size search (trying ±t·u with t = δ, δ/2, …) and the distillation loss form are my own
  implementation choices; they are documented in the code.

## Repository layout

```
asnet/
  data.py             BloodMNIST loaders (medmnist package)
  models.py           VGG-19 / small CNN written as cuttable "units"
  active_subspace.py  gradient covariance, frequent directions, rank selection
  pce.py              Hermite polynomial chaos layer + least-squares fit
  asnet.py            ASNet construction, distillation fine-tuning, l1 proximal step
  attack.py           recursive active-subspace universal attack
  uap_baseline.py     UAP baseline via the Adversarial Robustness Toolbox
  utils.py            training / evaluation helpers
scripts/              train.py, compress.py, attack.py
tests/                smoke tests on random data
```

## Acknowledgements and Credits

This repository contains **no source code copied from the original authors' repositories**.

**Methods re-implemented (please cite the original authors):**

- *Active subspace analysis, ASNet (active-subspace + polynomial-chaos-expansion layers) and the
  active-subspace universal adversarial attack* — Chunfeng Cui, Kaiqi Zhang, Talgat Daulbaev,
  Julia Gusak, Ivan Oseledets and Zheng Zhang, "Active Subspace of Neural Networks: Structural
  Analysis and Universal Attacks", *SIAM J. Math. Data Sci.* 2(4):1096–1122, 2020,
  [doi:10.1137/19M1296070](https://doi.org/10.1137/19M1296070) ([arXiv:1910.13025](https://arxiv.org/abs/1910.13025)).
  Original code: <https://github.com/chunfengc/ASNet> (no license; not reused).
- *Active subspaces* — P. G. Constantine, *Active Subspaces*, SIAM, 2015; P. G. Constantine, E. Dow, Q. Wang,
  *SIAM J. Sci. Comput.* 36(4):A1500–A1524, 2014.
- *Frequent directions* — E. Liberty, KDD 2013; M. Ghashami, E. Liberty, J. M. Phillips, D. P. Woodruff,
  *SIAM J. Comput.* 45(5), 2016.
- *Universal Adversarial Perturbations (baseline)* — S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, P. Frossard,
  CVPR 2017 ([doi:10.1109/CVPR.2017.17](https://doi.org/10.1109/CVPR.2017.17)), with DeepFool (CVPR 2016).
  Original code: <https://github.com/LTS4/universal> (no license; not reused). The baseline here calls the
  MIT-licensed [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).
- *Knowledge distillation* — G. Hinton, O. Vinyals, J. Dean, 2015.

**Data:** BloodMNIST from MedMNIST v2 (Yang et al., *Scientific Data* 10:41, 2023; Yang et al., ISBI 2021),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), derived from Acevedo et al.,
*Data in Brief* 30:105474 (2020). Images are downloaded at runtime from Zenodo
([doi:10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652)) by the `medmnist` package (Apache-2.0)
and are not redistributed here; images are normalised to [-1, 1]. **Not for clinical use.**

The results reported in the thesis were produced with an earlier codebase. Any errors in this re-implementation are
my own. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`CITATION.bib`](CITATION.bib).

## License

[MIT](LICENSE) © 2026 Fatemeh Salmani — for the code in this repository only. Third-party packages and the dataset
keep their own licenses.
