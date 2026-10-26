# Third-Party Notices

## Runtime dependencies (installed with pip, not vendored)
- **medmnist** — Apache License 2.0 — MedMNIST Team — https://github.com/MedMNIST/MedMNIST
- **adversarial-robustness-toolbox** — MIT License — Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018 — https://github.com/Trusted-AI/adversarial-robustness-toolbox
- **PyTorch / torchvision** — BSD 3-Clause License — https://github.com/pytorch/pytorch, https://github.com/pytorch/vision
- **NumPy** — BSD 3-Clause License — https://github.com/numpy/numpy

## Data (not redistributed)
- **BloodMNIST** (MedMNIST v2) — CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
  Source images: Acevedo et al., *Data in Brief* 30:105474 (2020).
  Changes: resized by the MedMNIST authors; normalised to [-1, 1] in this repository.

## Methods re-implemented without code reuse
- ASNet, active-subspace analysis and the active-subspace universal attack — Cui et al.,
  *SIAM J. Math. Data Sci.* 2(4):1096–1122, 2020. Original repository https://github.com/chunfengc/ASNet has
  no license; it was not copied.
- Universal Adversarial Perturbations / DeepFool — Moosavi-Dezfooli et al., CVPR 2017 / CVPR 2016. Original
  repositories https://github.com/LTS4/universal and https://github.com/LTS4/DeepFool have no license; they were
  not copied. The UAP baseline uses ART (above).

## Vendored code
- None.
