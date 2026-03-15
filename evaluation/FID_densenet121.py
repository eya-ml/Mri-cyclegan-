"""
FID (Fréchet Inception Distance) computation using DenseNet-121 features.

This script computes an FID-like score between generated MRI images and real
ground-truth images, using features extracted from a pre-trained DenseNet-121
rather than the standard Inception-v3 network. DenseNet-121 (pre-trained on
ImageNet via torchvision) has been shown to provide discriminative features
well-suited to medical imaging domains.

Usage
-----
    python evaluation/FID_densenet121.py --base_path results/t12t2_brain/0.5

References
----------
Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S.
(2017). GANs trained by a two time-scale update rule converge to a local Nash
equilibrium. *Advances in NeurIPS*.
"""

import argparse
import os

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """Minimal dataset that loads all PNG images from a directory."""

    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}

    def __init__(self, folder: str, transform=None):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    """Return a DenseNet-121 with the classifier head removed."""
    densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    # Remove the fully-connected classifier to expose pooled feature vectors
    feature_extractor = torch.nn.Sequential(
        densenet.features,
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
    )
    feature_extractor.eval().to(device)
    return feature_extractor


@torch.no_grad()
def extract_features(
    folder: str,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract DenseNet-121 features for all images in *folder*.

    Returns
    -------
    np.ndarray
        Array of shape (N, 1024) containing the feature vectors.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

    all_features = []
    for batch in loader:
        feats = model(batch.to(device)).cpu().numpy()
        all_features.append(feats)

    return np.concatenate(all_features, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# FID computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_fid(mu1: np.ndarray, sigma1: np.ndarray,
                mu2: np.ndarray, sigma2: np.ndarray,
                eps: float = 1e-6) -> float:
    """Compute FID between two multivariate Gaussians.

    FID = ||μ₁ − μ₂||² + Tr(Σ₁ + Σ₂ − 2√(Σ₁Σ₂))

    Parameters
    ----------
    mu1, mu2 : np.ndarray  — mean vectors of the two distributions.
    sigma1, sigma2 : np.ndarray — covariance matrices.
    eps : float — small value added to the diagonal for numerical stability.
    """
    diff = mu1 - mu2

    # Compute the matrix square-root of (sigma1 @ sigma2)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical correction for imaginary parts
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component too large: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


def gaussian_stats(features: np.ndarray):
    """Return (mean, covariance) for a feature matrix of shape (N, D)."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute FID between generated and real MRI images "
                    "using DenseNet-121 features."
    )
    parser.add_argument('--base_path', required=True,
                        help='Base results directory; should contain fakeB/ and realB/ sub-folders.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU index; use -1 for CPU.')
    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")

    real_dir = os.path.join(args.base_path, 'realB')
    fake_dir = os.path.join(args.base_path, 'fakeB')

    for d in (real_dir, fake_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    print("Loading DenseNet-121 feature extractor …")
    model = build_feature_extractor(device)

    print(f"Extracting features from real images ({real_dir}) …")
    real_feats = extract_features(real_dir, model, device, args.batch_size)

    print(f"Extracting features from generated images ({fake_dir}) …")
    fake_feats = extract_features(fake_dir, model, device, args.batch_size)

    mu_r, sigma_r = gaussian_stats(real_feats)
    mu_f, sigma_f = gaussian_stats(fake_feats)

    fid_score = compute_fid(mu_r, sigma_r, mu_f, sigma_f)
    print(f"\nFID (DenseNet-121): {fid_score:.4f}")
    print(f"  Real images  : {len(real_feats)}")
    print(f"  Fake images  : {len(fake_feats)}")

    # Save result
    out_file = os.path.join(args.base_path, 'fid_score.txt')
    with open(out_file, 'w') as f:
        f.write(f"FID (DenseNet-121): {fid_score:.4f}\n")
        f.write(f"Real: {len(real_feats)} images\n")
        f.write(f"Fake: {len(fake_feats)} images\n")
    print(f"Result saved to {out_file}")


if __name__ == '__main__':
    main()
