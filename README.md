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

## Results

VGG-19 trained for 20 epochs on 64x64 BloodMNIST (Colab, T4 GPU). Full run: [the notebook](notebooks/bloodmnist_active_subspace.ipynb).

### Active subspace

![spectrum](figures/spectrum.png)

The gradient covariance is strongly low-rank. After layer 16 (4th pooling) a handful of directions already capture
95% of the energy, out of 8192 neurons. Deeper in the network (layer 20) the energy is more spread out.

### ASNet

| model | test accuracy | parameters | compression |
|---|---|---|---|
| VGG-19 | 96.52% | 20.04 M | 1.00x |
| ASNet (cut 16, r = 50, p = 2), least-squares init only | 96.23% | 11.01 M | 1.82x |
| ASNet (cut 16, r = 50, p = 2), after 10 epochs fine-tuning | 96.29% | 11.01 M | 1.82x |

![asnet](figures/asnet.png)

Replacing the last four 512-channel conv layers and the classifier (about 9.4 M parameters) with the AS + PCE
layers (about 0.4 M) costs only 0.2% accuracy, and the
least-squares initialization alone is almost as good as fine-tuning. Most remaining parameters are in the first 16
layers, so cutting earlier (e.g. layer 11) should give a larger compression; that is the next experiment.

### Universal attacks

Fooling rate on the **test set**, perturbation built from 200 training images:

| ‖v‖₂ | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| active subspace | **24.1%** | **73.2%** | **88.1%** | **89.9%** | 90.3% |
| UAP | 12.2% | 23.3% | 71.9% | 82.1% | **91.2%** |

![attack](figures/attack.png)

For small and medium perturbations the active-subspace attack is much stronger than UAP (73% vs 23% at
‖v‖₂ = 4). At ‖v‖₂ = 10 both methods fool about 90% of the images. The results on training and test images are
close, so the perturbation generalizes to unseen images.

![examples](figures/examples.png)

The AS perturbation is concentrated in the center of the image, where the cell is, and pushes most images towards
the *erythroblast* class.

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
