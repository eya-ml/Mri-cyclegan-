# 🧠 MRI Contrast Translation — T1 ↔ T2

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Research-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/École-ÉTS%20Montréal-004B8D?style=flat-square"/>
  <img src="https://img.shields.io/badge/Course-MTI805-grey?style=flat-square"/>
</p>

<p align="center">
  <b>Unsupervised and Semi-Supervised MRI Synthesis using CycleGAN and VAE-CycleGAN</b><br/>
  <i>MTI805 – Image Understanding &nbsp;|&nbsp; École de Technologie Supérieure, Montréal &nbsp;|&nbsp; April 2024</i>
</p>

---

## 📽️ Visual Overview

The diagram below illustrates the cycle-consistency principle. A real T1 scan is translated to a synthetic T2, then translated back — the reconstruction should match the original input.

```
  Real T1 ──► G_A ──► Fake T2 ──► G_B ──► Reconstructed T1
     │                                           │
     └──────────── Cycle-Consistency Loss ───────┘

  Real T2 ──► G_B ──► Fake T1 ──► G_A ──► Reconstructed T2
     │                                           │
     └──────────── Cycle-Consistency Loss ───────┘
```

### Sample Results — T1 → T2 Translation

| Model | Paired (%) | Input T1 | Generated T2 | Ground Truth T2 |
|---|:---:|:---:|:---:|:---:|
| VAE-CycleGAN | 0 | `docs/assets/vae_input.png` | `docs/assets/vae_fake.png` | `docs/assets/gt.png` |
| Paired CycleGAN | 25 | `docs/assets/p25_input.png` | `docs/assets/p25_fake.png` | `docs/assets/gt.png` |
| Paired CycleGAN | 75 | `docs/assets/p75_input.png` | `docs/assets/p75_fake.png` | `docs/assets/gt.png` |

> 💡 Replace the cells above with actual image paths from `results/` once training is complete.
> Recommended: `![desc](docs/assets/filename.png)`

### Quantitative Summary

| Model | Paired (%) | SSIM ↑ (T1) | SSIM ↑ (T2) | PSNR ↑ T1 (dB) | PSNR ↑ T2 (dB) |
|---|:---:|:---:|:---:|:---:|:---:|
| **VAE-CycleGAN** | **0** | **0.75** | **0.66** | **25.46** | **19.66** |
| Paired CycleGAN | 0 | 0.52 | 0.39 | 9.40 | 4.24 |
| Paired CycleGAN | 25 | 0.67 | 0.60 | 11.33 | 5.36 |
| Paired CycleGAN | 50 | 0.72 | 0.63 | 12.97 | 6.55 |
| Paired CycleGAN | 75 | 0.72 | 0.65 | 13.81 | 6.77 |
| Paired CycleGAN | 100 | 0.72 | 0.66 | 13.08 | 7.13 |

> **Key finding:** VAE-CycleGAN achieves superior SSIM and PSNR with *zero* paired training data, outperforming the Paired CycleGAN at every level of supervision on T1 synthesis.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Background & Motivation](#-background--motivation)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Evaluation Metrics](#-evaluation-metrics)
- [Discussion & Limitations](#-discussion--limitations)
- [Authors](#-authors)
- [References](#-references)

---

## 🔍 Overview

This project investigates the synthesis of brain MRI scans across T1 and T2 contrasts using two Generative Adversarial Network architectures:

| Architecture | Supervision | Key Innovation |
|---|---|---|
| **VAE-CycleGAN** | Unsupervised (0 % paired) | Attention discriminator + VAE information bottleneck |
| **Paired CycleGAN** | Semi-supervised (0–100 % paired) | Conditional adversarial losses (Tripathy et al., 2019) |

The core motivation is to reduce the clinical burden of acquiring multiple MRI sequences by synthesising one contrast from another — enabling downstream tasks such as segmentation and anomaly detection without additional scan time or cost.

---

## 📚 Background & Motivation

Magnetic Resonance Imaging (MRI) is central to neurological diagnosis. T1-weighted images provide excellent anatomical contrast, while T2-weighted images are more sensitive to fluid and pathological tissues. Acquiring both sequences is time-consuming, expensive, and can be uncomfortable for patients.

This project builds upon two seminal works:

- **Zhu et al. (2020)** — *CycleGAN* — enables cross-domain image translation without paired training data via a cycle-consistency constraint: translating an image A→B→A should recover the original.
- **Tripathy et al. (2019)** — extends CycleGAN with conditional adversarial losses to exploit available paired samples, reducing the under-determination inherent to the unpaired setting.

---

## 🏗️ Architecture

### Standard CycleGAN (Baseline)

```
Domain A (T1) ──► G_A (ResNet-9) ──► Fake B ──► D_A
Domain B (T2) ──► G_B (ResNet-9) ──► Fake A ──► D_B
```

- Two **ResNet generators** with 9 residual blocks
- Two **PatchGAN discriminators** on 70×70 patches
- **Adversarial loss** (LSGAN / MSE), **Cycle loss** (L1, λ=10), **Identity loss** (L1, λ=0.5)

### Paired CycleGAN (Semi-Supervised)

Adds two **conditional discriminators** (`D_C`, `D_D`) evaluating concatenated image pairs:

| Loss | Role |
|---|---|
| Conditional Adversarial Loss | Minimises pixel/content gap between translated image and target ground truth (paired data) |
| Conditional Adversarial Cycle Consistency Loss | Minimises pixel/content gap between reconstructed image and source ground truth |

### VAE-CycleGAN (Unsupervised)

- **Attention mechanism** in discriminators → focuses on anatomically relevant regions
- **Variational Autoencoder (VAE)** bottleneck → stabilises training and improves generalisation

---

## 🛠️ Installation

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10 |
| GPU VRAM | — (CPU supported) | 8 GB (NVIDIA CUDA) |
| RAM | 8 GB | 16 GB |
| Disk space | 5 GB | 20 GB |
| OS | Linux / macOS / Windows | Ubuntu 22.04 LTS |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/mri-cyclegan.git
cd mri-cyclegan
```

### Step 2 — Create a Virtual Environment

**Using `venv` (recommended):**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows PowerShell
```

**Using `conda`:**

```bash
conda create -n mri-cyclegan python=3.10 -y
conda activate mri-cyclegan
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Verify PyTorch & GPU

```python
import torch
print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
```

> **CPU-only mode:** All training and testing commands accept `--gpu_ids -1`. Expect significantly longer training times (~10–50× slower than GPU).

### Step 5 — Verify the Install

```bash
python train.py --help
```

You should see the full list of available options without any import errors.

---

## 🗄️ Dataset Setup

The dataset consists of paired T1/T2 brain MRI scans acquired at **7 Tesla** field strength from 10 healthy subjects (ages 25–41), yielding approximately **1,000 2D slices** per contrast. An 80/20 train/test split is applied.

### Expected Directory Structure

```
datasets/
└── t12t2_brain/
    ├── 0/                    ← 0 % paired (fully unpaired)
    │   ├── trainA/           ← T1 training slices  (.png / .jpg)
    │   ├── trainB/           ← T2 training slices
    │   ├── testA/
    │   └── testB/
    ├── 0.25/                 ← 25 % paired
    │   ├── trainA/
    │   ├── trainB/
    │   ├── trainA_paired/    ← paired subset for supervised losses
    │   ├── trainB_paired/
    │   ├── testA/
    │   └── testB/
    ├── 0.5/
    ├── 0.75/
    └── 1/                    ← 100 % paired
```

### Preprocessing Pipeline (Applied Automatically)

| Step | Training | Testing |
|---|:---:|:---:|
| Resize to 286×286 | ✅ | ✅ |
| Random crop to 256×256 | ✅ | ❌ |
| Random horizontal flip | ✅ | ❌ |
| Normalise to [−1, 1] | ✅ | ✅ |

> **Note:** The dataset is not distributed with this repository. See `docs/Rapport_MTI805.pdf` for the acquisition protocol and data source.

---

## 🚀 Usage

### Training

#### 0 % Paired — Fully Unpaired (Standard CycleGAN)

```bash
python train.py \
  --dataroot ./datasets/t12t2_brain/0 \
  --model cycle_gan \
  --dataset_mode unaligned \
  --netG resnet_9blocks \
  --direction AtoB \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --super_start 0 \
  --name unpaired_gan \
  --no_dropout
```

#### 25 % Paired Data

```bash
python train.py \
  --dataroot ./datasets/t12t2_brain/0.25 \
  --model cycle_gan \
  --dataset_mode unaligned \
  --netG resnet_9blocks \
  --direction AtoB \
  --super_epochs 100 \
  --super_mode aligned \
  --super_start 1 \
  --name paired_gan_25 \
  --no_dropout
```

#### 50 % Paired Data

```bash
python train.py \
  --dataroot ./datasets/t12t2_brain/0.5 \
  --model cycle_gan \
  --dataset_mode unaligned \
  --netG resnet_9blocks \
  --direction AtoB \
  --super_epochs 50 \
  --super_mode aligned \
  --super_start 1 \
  --name paired_gan_50 \
  --no_dropout
```

#### 100 % Paired Data

```bash
python train.py \
  --dataroot ./datasets/t12t2_brain/1 \
  --model cycle_gan \
  --dataset_mode unaligned \
  --netG resnet_9blocks \
  --direction AtoB \
  --super_epochs 50 \
  --super_mode aligned \
  --super_start 1 \
  --name paired_gan_100 \
  --no_dropout \
  --n_epochs 50
```

> Add `--gpu_ids -1` to any command above for CPU-only training.

### Testing / Inference

```bash
python test.py \
  --dataroot ./datasets/t12t2_brain/0.5 \
  --results_dir results/t12t2_brain/0.5 \
  --model cycle_gan \
  --dataset_mode single \
  --netG resnet_9blocks \
  --direction AtoB \
  --name paired_gan_50 \
  --num_test 500 \
  --no_dropout
```

Generated images are organised as follows:

```
results/t12t2_brain/0.5/
├── realA/      ← Input T1 slices
├── fakeB/      ← Synthesised T2 slices  ✦ primary output
├── recA/       ← Reconstructed T1 (forward cycle)
├── realB/      ← Ground-truth T2
├── fakeA/      ← Synthesised T1 (backward direction)
└── recB/       ← Reconstructed T2 (backward cycle)
```

### Evaluation — FID Score (DenseNet-121)

```bash
python evaluation/FID_densenet121.py \
  --base_path results/t12t2_brain/0.5
```

Results are printed to the console and saved to `results/t12t2_brain/0.5/fid_score.txt`.

### Full Experiment Pipeline

Open [`notebooks/experiments.ipynb`](notebooks/experiments.ipynb) for the complete end-to-end pipeline — training all five regimes, running inference, and computing evaluation metrics.

---

## 📁 Project Structure

```
mri-cyclegan/
│
├── models/                          # Network definitions & model logic
│   ├── __init__.py
│   ├── cycle_gan_model.py           # Generators, discriminators, losses, optimisers
│   └── networks.py                  # ResNet generator, PatchGAN, GANLoss
│
├── data/                            # Dataset & data loading
│   ├── __init__.py
│   ├── base_dataset.py              # Abstract base + image transform pipeline
│   ├── cyclegan_dataset.py          # Unaligned / aligned dataset implementations
│   └── cyclegan_dataloader.py       # DataLoader wrapper (paired/unpaired switching)
│
├── options/                         # CLI argument parsers
│   ├── __init__.py
│   ├── base_options.py              # Shared options (dataset, model, hardware)
│   ├── train_options.py             # Training: lr, epochs, loss weights
│   └── test_options.py              # Testing: num_test, results_dir
│
├── util/                            # Utility functions
│   ├── __init__.py
│   ├── util.py                      # tensor2im, save_image, mkdirs
│   └── image_pool.py                # Replay buffer for discriminator training
│
├── evaluation/                      # Evaluation scripts
│   ├── __init__.py
│   └── FID_densenet121.py           # FID with DenseNet-121 feature extractor
│
├── notebooks/                       # Jupyter notebooks
│   ├── experiments.ipynb            # Full training / testing / evaluation pipeline
│   └── VAE_CycleGAN.ipynb           # VAE-CycleGAN architecture & ablation
│
├── docs/                            # Documentation
│   └── Rapport_MTI805.pdf           # Full project report (French)
│
├── datasets/                        # ⚠ Not tracked by Git
├── checkpoints/                     # ⚠ Not tracked by Git
├── results/                         # ⚠ Not tracked by Git
│
├── train.py                         # Training entry point
├── test.py                          # Inference entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📐 Evaluation Metrics

### PSNR — Peak Signal-to-Noise Ratio

$$\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$

Expressed in decibels (dB). Higher is better. Values above ~25 dB typically indicate good perceptual fidelity for medical images.

### SSIM — Structural Similarity Index

$$\text{SSIM}(x, y) = \bigl[l(x,y)\bigr]^\alpha \cdot \bigl[c(x,y)\bigr]^\beta \cdot \bigl[s(x,y)\bigr]^\gamma$$

Jointly evaluates luminance *l*, contrast *c*, and structural similarity *s*. Ranges from −1 to 1 (1 = perfect similarity).

### FID — Fréchet Inception Distance (DenseNet-121 variant)

$$\text{FID} = \|\mu_r - \mu_f\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_f - 2\sqrt{\Sigma_r \Sigma_f}\right)$$

Measures distributional distance between real and generated images in feature space. Lower is better. This implementation uses DenseNet-121 features, which are more appropriate for medical imaging than standard Inception-v3 features.

---

## 💬 Discussion & Limitations

### Why VAE-CycleGAN Outperforms

The attention mechanism directs the discriminator toward anatomically relevant regions, enabling finer distinction between real and synthetic scans. The VAE bottleneck regularises the latent space, stabilising adversarial training and reducing mode collapse. VAE-CycleGAN was also trained for substantially more epochs (~10²), contributing to its performance advantage.

### Saturation of the Paired CycleGAN

Performance plateaus at 75 % paired data, suggesting either a capacity limit or the onset of over-fitting, where the conditional losses become overly constraining.

### Known Limitations

- Imperfect attention calibration may neglect clinically relevant but visually subtle structures
- The VAE + attention architecture requires significantly more training iterations than the baseline
- Conditional losses in the Paired CycleGAN increase risk of poor generalisation to unseen images
- SSIM/PSNR/FID do not directly measure clinical utility — downstream validation (segmentation, detection) remains as future work

---

## 👥 Authors

| Name | Student ID |
|---|---|
| Eya Bdah | BDAE0432990 |
| Eya Mlika | MLIE02379900 |
| Bianca Popa | POPB27589607 |
| Nilufar Estiry | ESTN85310101 |

Supervised by **Luc Duong** — MTI805, Image Understanding, ÉTS Montréal

---

## 📖 References

1. **Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A.** (2020). Unpaired image-to-image translation using cycle-consistent adversarial networks. *arXiv*. https://doi.org/10.48550/arXiv.1703.10593

2. **Tripathy, S., Kannala, J., & Rahtu, E.** (2019). Learning image-to-image translation using paired and unpaired training samples. *ACCV 2018*, pp. 51–66. Springer.

3. **Yang, Q., Li, N., Zhao, Z., Fan, X., Chang, E. I. C., & Xu, Y.** (2020). MRI cross-modality image-to-image translation. *Scientific Reports*, 10(1), 3753.

4. **Goodfellow, I., et al.** (2014). Generative adversarial nets. *NeurIPS*.

5. **Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A.** (2016). Image-to-image translation with conditional adversarial networks. *arXiv*. https://arxiv.org/abs/1611.07004

6. **Heusel, M., et al.** (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *NeurIPS* (FID metric).

7. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). Image quality assessment: from error visibility to structural similarity. *IEEE Trans. Image Processing*, 13(4), 600–612.

---

> *École de Technologie Supérieure (ÉTS), Montréal — April 2024. All rights reserved by the authors.*
