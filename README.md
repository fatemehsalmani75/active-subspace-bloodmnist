# Active Subspace of Neural Networks on BloodMNIST

<a href="https://colab.research.google.com/github/fatemehsalmani75/active-subspace-bloodmnist/blob/main/notebooks/bloodmnist_active_subspace.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

PyTorch code for my MSc thesis *Structural Analysis of Deep Neural Networks* (Shahid Beheshti University,
supervisor Dr. Masoud Hajarian), rewritten from scratch for BloodMNIST.

The idea comes from Cui et al. (2020): cut a trained network after some layer and look at the gradients of the
loss with respect to the neurons of that layer. Their covariance matrix usually has only a few large eigenvalues,
so a small number of directions (the *active subspace*) explains most of what the rest of the network does.
This is used for two things:

- **compression (ASNet):** keep the first layers, project onto the active subspace and replace the rest of the
  network with a polynomial chaos expansion;
- **universal attacks:** build one perturbation for all images from the dominant active direction of the input
  gradients, and compare it with the UAP attack of Moosavi-Dezfooli et al. (2017).

The paper used CIFAR-10/100 and Fashion-MNIST. Here the same methods are applied to BloodMNIST
(8 classes of blood cell microscopy images).

## Run it

The easiest way is the Colab notebook (button above). It trains the model, plots the eigenvalue spectrum,
builds ASNet and compares the two attacks.

Locally:

```bash
pip install -r requirements.txt

python scripts/train.py --model vgg19 --size 64 --epochs 30
python scripts/compress.py --ckpt checkpoints/vgg19_s64.pt --cut 16 --r 50 --p 2
python scripts/compress.py --ckpt checkpoints/vgg19_s64.pt --cut 16 --r 50 --l1 1e-4   # sparse ASNet-s
python scripts/attack.py --ckpt checkpoints/vgg19_s64.pt --delta 2 4 6 8 10 --n-train 200
python scripts/attack.py --ckpt checkpoints/vgg19_s64.pt --target-class 3 --n-train 100

pytest -q   # small tests on random data
```

BloodMNIST is downloaded automatically by the `medmnist` package.

## Code

```
asnet/
  data.py             BloodMNIST loaders
  models.py           VGG-19 and a small CNN, written so they can be cut after any layer
  active_subspace.py  gradient covariance, frequent-directions sketch, choice of r
  pce.py              Hermite polynomial chaos layer, least-squares fit
  asnet.py            ASNet, distillation fine-tuning, l1 proximal step
  attack.py           active-subspace universal attack
  uap_baseline.py     UAP baseline (through the Adversarial Robustness Toolbox)
scripts/              train / compress / attack from the command line
notebooks/            Colab notebook with all plots
tests/
```

Some details:

- 64x64 images are used because VGG-19 has five pooling layers and 28x28 images become too small.
- Defaults follow the paper where possible: p = 2, r = 50 (or chosen with eps = 0.05), distillation weight 0.1,
  Adam with learning rate 1e-4 for the first layers and 1e-5 for the new layers, alpha = 10 failed steps,
  200 images for the attacks and 10 iterations for UAP.
- Images are normalized to [-1, 1] and delta is the l2 norm of the perturbation in that space. The values in the
  paper were for CIFAR images, so a range of deltas is tried here.
- Fooling rate = fraction of images whose predicted label changes after adding the perturbation.
- The step-size search in the attack (signs ±1, steps delta, delta/2, ...) and the exact form of the
  distillation loss are my choices; the paper does not fix them completely.

## References and credits

All code in this repository was written for this project; nothing was copied from the authors' original
repositories (they have no license).

- C. Cui, K. Zhang, T. Daulbaev, J. Gusak, I. Oseledets, Z. Zhang. Active Subspace of Neural Networks: Structural
  Analysis and Universal Attacks. *SIAM J. Math. Data Sci.* 2(4):1096–1122, 2020.
  [doi:10.1137/19M1296070](https://doi.org/10.1137/19M1296070), [arXiv:1910.13025](https://arxiv.org/abs/1910.13025).
  Original code: https://github.com/chunfengc/ASNet
- P. G. Constantine. *Active Subspaces*. SIAM, 2015.
- E. Liberty (KDD 2013); M. Ghashami, E. Liberty, J. M. Phillips, D. P. Woodruff (SIAM J. Comput. 2016).
  Frequent directions.
- S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, P. Frossard. Universal Adversarial Perturbations. CVPR 2017.
  Original code: https://github.com/LTS4/universal. The baseline here uses the implementation in
  [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (MIT).
- G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network, 2015.
- **Data:** BloodMNIST from MedMNIST v2 (J. Yang et al., *Scientific Data* 10:41, 2023; ISBI 2021), CC BY 4.0,
  based on A. Acevedo et al., *Data in Brief* 30:105474, 2020. The images are not included in this repository.
  Not for clinical use.

The numbers in the thesis were obtained with an earlier version of the code. BibTeX entries are in
[`CITATION.bib`](CITATION.bib) and licenses of the dependencies in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

MIT, see [LICENSE](LICENSE).
