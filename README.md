# 🧠 MRI Contrast Synthesis — T1 ↔ T2

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Domain-Medical%20Imaging-6A5ACD?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Research-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Institution-ÉTS%20Montréal-004B8D?style=flat-square"/>
</p>

<p align="center">
  <b>Unsupervised and semi-supervised MRI synthesis using CycleGAN and VAE-CycleGAN</b><br/>
  <i>Translating brain MRI scans between T1 and T2 contrast — without requiring paired training data</i>
</p>

---

## Why This Problem Matters

Acquiring both T1 and T2 brain MRI sequences is time-consuming, expensive, and clinically burdensome for patients. If one contrast can be reliably synthesised from the other, clinicians gain a complete diagnostic picture from a single acquisition — reducing scan time, cost, and patient discomfort, while enabling downstream tasks like segmentation and anomaly detection on modalities that weren't directly imaged.

This project investigates how much paired supervision is actually necessary for high-quality synthesis — and demonstrates that a well-designed unsupervised architecture can outperform semi-supervised baselines trained on up to 75% paired data.

---

## Key Results

| Model | Paired Data Used | SSIM ↑ (T1) | SSIM ↑ (T2) | PSNR ↑ T1 (dB) | PSNR ↑ T2 (dB) |
|---|:---:|:---:|:---:|:---:|:---:|
| **VAE-CycleGAN** | **0%** | **0.75** | **0.66** | **25.46** | **19.66** |
| Paired CycleGAN | 0% | 0.52 | 0.39 | 9.40 | 4.24 |
| Paired CycleGAN | 25% | 0.67 | 0.60 | 11.33 | 5.36 |
| Paired CycleGAN | 50% | 0.72 | 0.63 | 12.97 | 6.55 |
| Paired CycleGAN | 75% | 0.72 | 0.65 | 13.81 | 6.77 |
| Paired CycleGAN | 100% | 0.72 | 0.66 | 13.08 | 7.13 |

**The headline finding:** VAE-CycleGAN with zero paired training data achieves better T1 synthesis (SSIM 0.75, PSNR 25.46 dB) than a Paired CycleGAN trained on 100% paired data (SSIM 0.72, PSNR 13.08 dB). The attention mechanism and VAE bottleneck together eliminate the need for expensive paired annotations.

---

## Cycle-Consistency Principle

The core constraint that makes unpaired translation tractable: translating an image A→B→A should faithfully recover the original input.

```
  Real T1 ──► G_A ──► Fake T2 ──► G_B ──► Reconstructed T1
     │                                           │
     └──────────── Cycle-Consistency Loss ───────┘

  Real T2 ──► G_B ──► Fake T1 ──► G_A ──► Reconstructed T2
     │                                           │
     └──────────── Cycle-Consistency Loss ───────┘
```

---

## Architectures

### Standard CycleGAN (Baseline)

- Two **ResNet-9 generators** (9 residual blocks)
- Two **PatchGAN discriminators** operating on 70×70 patches
- Loss composition: adversarial (LSGAN), cycle-consistency (L1, λ=10), identity (L1, λ=0.5)

### Paired CycleGAN (Semi-Supervised)

Extends the baseline with two **conditional discriminators** that evaluate concatenated real/generated pairs when paired samples are available:

| Additional Loss | Role |
|---|---|
| Conditional Adversarial Loss | Closes the pixel/content gap between generated image and paired ground truth |
| Conditional Cycle-Consistency Loss | Closes the pixel/content gap between reconstructed image and source ground truth |

### VAE-CycleGAN (Unsupervised — Best Performer)

Two architectural additions over the baseline, with no requirement for any paired data:

- **Attention discriminators** — direct gradient signal toward anatomically meaningful regions rather than texture or background artefacts
- **VAE bottleneck** — regularises the latent representation, stabilising adversarial training and reducing mode collapse

---

## Dataset

7 Tesla brain MRI scans from 10 healthy subjects (ages 25–41), yielding approximately **1,000 2D slices per contrast**. An 80/20 train/test split is applied. The dataset is not distributed with this repository.

### Preprocessing Pipeline

| Step | Training | Testing |
|---|:---:|:---:|
| Resize to 286×286 | ✅ | ✅ |
| Random crop to 256×256 | ✅ | ❌ |
| Random horizontal flip | ✅ | ❌ |
| Normalise to [−1, 1] | ✅ | ✅ |

### Expected Directory Structure

```
datasets/
└── t12t2_brain/
    ├── 0/                    ← 0% paired (fully unpaired)
    │   ├── trainA/           ← T1 training slices (.png / .jpg)
    │   ├── trainB/           ← T2 training slices
    │   ├── testA/
    │   └── testB/
    ├── 0.25/                 ← 25% paired
    │   ├── trainA/
    │   ├── trainB/
    │   ├── trainA_paired/    ← paired subset for supervised losses
    │   ├── trainB_paired/
    │   ├── testA/
    │   └── testB/
    ├── 0.5/
    ├── 0.75/
    └── 1/                    ← 100% paired
```

---

## Installation

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA-capable GPU recommended (8 GB VRAM+)

```bash
# Clone
git clone https://github.com/eya-ml/Mri-cyclegan-.git
cd Mri-cyclegan-

# Environment
python -m venv venv && source venv/bin/activate   # or: conda create -n mri-cyclegan python=3.10

# Dependencies
pip install --upgrade pip && pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> CPU-only mode is supported via `--gpu_ids -1` on all commands, but expect 10–50× slower training.

---

## Usage

### Training

```bash
# Fully unpaired (VAE-CycleGAN / standard CycleGAN at 0%)
python train.py \
  --dataroot ./datasets/t12t2_brain/0 \
  --model cycle_gan \
  --dataset_mode unaligned \
  --netG resnet_9blocks \
  --direction AtoB \
  --n_epochs 100 --n_epochs_decay 100 \
  --super_start 0 \
  --name unpaired_gan --no_dropout

# 25% paired data
python train.py \
  --dataroot ./datasets/t12t2_brain/0.25 \
  --model cycle_gan --dataset_mode unaligned \
  --netG resnet_9blocks --direction AtoB \
  --super_epochs 100 --super_mode aligned --super_start 1 \
  --name paired_gan_25 --no_dropout

# 100% paired data
python train.py \
  --dataroot ./datasets/t12t2_brain/1 \
  --model cycle_gan --dataset_mode unaligned \
  --netG resnet_9blocks --direction AtoB \
  --super_epochs 50 --super_mode aligned --super_start 1 \
  --name paired_gan_100 --no_dropout --n_epochs 50
```

### Inference

```bash
python test.py \
  --dataroot ./datasets/t12t2_brain/0.5 \
  --results_dir results/t12t2_brain/0.5 \
  --model cycle_gan --dataset_mode single \
  --netG resnet_9blocks --direction AtoB \
  --name paired_gan_50 --num_test 500 --no_dropout
```

Output structure:

```
results/t12t2_brain/0.5/
├── realA/      ← Input T1
├── fakeB/      ← Synthesised T2  ✦ primary output
├── recA/       ← Reconstructed T1 (cycle check)
├── realB/      ← Ground-truth T2
├── fakeA/      ← Synthesised T1 (reverse direction)
└── recB/       ← Reconstructed T2 (cycle check)
```

### Evaluation

```bash
# FID score using DenseNet-121 features (more appropriate than Inception-v3 for medical imaging)
python evaluation/FID_densenet121.py --base_path results/t12t2_brain/0.5
```

Full end-to-end pipeline (all regimes, inference, metrics) is available in [`notebooks/experiments.ipynb`](notebooks/experiments.ipynb).

---

## Project Structure

```
mri-cyclegan/
├── models/
│   ├── cycle_gan_model.py       # Generators, discriminators, losses, optimisers
│   └── networks.py              # ResNet generator, PatchGAN, GANLoss
├── data/
│   ├── base_dataset.py          # Abstract base + image transform pipeline
│   ├── cyclegan_dataset.py      # Unaligned / aligned dataset implementations
│   └── cyclegan_dataloader.py   # DataLoader wrapper (paired/unpaired switching)
├── options/
│   ├── base_options.py          # Shared options
│   ├── train_options.py         # Training hyperparameters
│   └── test_options.py          # Inference options
├── util/
│   ├── util.py                  # tensor2im, save_image, mkdirs
│   └── image_pool.py            # Replay buffer for discriminator training
├── evaluation/
│   └── FID_densenet121.py       # FID with DenseNet-121 feature extractor
├── notebooks/
│   ├── experiments.ipynb        # Full training / evaluation pipeline
│   └── VAE_CycleGAN.ipynb       # Architecture ablation
├── train.py                     # Training entry point
├── test.py                      # Inference entry point
└── requirements.txt
```

---

## Evaluation Metrics

**PSNR** (Peak Signal-to-Noise Ratio, dB — higher is better): measures pixel-level fidelity. Values above ~25 dB generally indicate good perceptual quality for medical images.

**SSIM** (Structural Similarity Index, 0–1 — higher is better): jointly evaluates luminance, contrast, and structural similarity, making it more clinically meaningful than pure pixel error.

**FID** (Fréchet Inception Distance — lower is better): measures distributional distance between real and generated images in feature space. This implementation uses DenseNet-121 features rather than standard Inception-v3, which are better calibrated to medical imaging data.

---

## Discussion & Limitations

**Why VAE-CycleGAN wins without paired data:**
The attention discriminator focuses gradient signal on anatomically relevant regions, preventing the network from gaming cycle-consistency through imperceptible perturbations. The VAE bottleneck regularises the latent space, reducing mode collapse and improving generalisation across subjects.

**Paired CycleGAN saturation:**
Performance plateaus at 75% paired data, suggesting the conditional losses become overly constraining before the network capacity is saturated — a useful signal for data collection strategy in resource-limited clinical settings.

**Known limitations:**
- Attention calibration is imperfect and may under-weight clinically subtle structures
- VAE + attention requires significantly more training iterations than the baseline
- SSIM/PSNR/FID do not directly measure clinical utility — downstream validation on segmentation or detection tasks is future work
- Results are specific to 7T brain MRI; generalisability to other field strengths or anatomy is untested

---

## References

1. Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2020). Unpaired image-to-image translation using cycle-consistent adversarial networks. *arXiv*. https://doi.org/10.48550/arXiv.1703.10593
2. Tripathy, S., Kannala, J., & Rahtu, E. (2019). Learning image-to-image translation using paired and unpaired training samples. *ACCV 2018*, pp. 51–66. Springer.
3. Yang, Q., et al. (2020). MRI cross-modality image-to-image translation. *Scientific Reports*, 10(1), 3753.
4. Goodfellow, I., et al. (2014). Generative adversarial nets. *NeurIPS*.
5. Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2016). Image-to-image translation with conditional adversarial networks. *arXiv*. https://arxiv.org/abs/1611.07004
6. Heusel, M., et al. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *NeurIPS*.
7. Wang, Z., et al. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE Trans. Image Processing*, 13(4), 600–612.
