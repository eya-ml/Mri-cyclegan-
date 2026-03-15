"""
base_dataset.py
===============
Abstract base class for all dataset implementations and shared
preprocessing utilities (image transforms, file discovery).

All concrete dataset classes (e.g. ``CycleGANDataset``) must inherit
from :class:`BaseDataset` and implement ``__len__`` and ``__getitem__``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# Supported image extensions
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(directory: str | Path) -> List[str]:
    """Recursively collect all image file paths under *directory*.

    Files are sorted for reproducibility and filtered by extension
    (case-insensitive).

    Args:
        directory: Root folder to search.

    Returns:
        Sorted list of absolute image file paths.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    paths: List[str] = []
    for dirpath, _, filenames in sorted(os.walk(root)):
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in _IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, fname))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Transform factory
# ─────────────────────────────────────────────────────────────────────────────

def get_transform(
    opt,
    grayscale: bool = False,
    aligned: bool = False,
    method: int = Image.BICUBIC,
) -> Callable:
    """Build a torchvision transform pipeline from training options.

    Pipeline (in order, applied only when relevant flags are set):

    1. Grayscale conversion  (if ``grayscale=True``)
    2. Resize                (if ``'resize'`` in ``opt.preprocess``)
    3. Scale-width resize    (if ``'scale_width'`` in ``opt.preprocess``)
    4. Random crop           (if ``'crop'`` in ``opt.preprocess`` and
                              not ``aligned``)
    5. Random horizontal flip (if not ``opt.no_flip`` and not ``aligned``)
    6. ToTensor
    7. Normalise to [-1, 1]

    Args:
        opt:        Options namespace; must expose ``preprocess``,
                    ``load_size``, ``crop_size``, ``no_flip``.
        grayscale:  Convert to single-channel grayscale.
        aligned:    If ``True``, skip all random spatial transforms
                    (used for paired validation/test images).
        method:     Interpolation filter for resizing
                    (default: ``Image.BICUBIC``).

    Returns:
        A composed ``torchvision.transforms`` pipeline.
    """
    steps: list = []

    # ── Channel ──────────────────────────────────────────────────────
    if grayscale:
        steps.append(T.Grayscale(num_output_channels=1))

    # ── Spatial transforms (skipped for aligned / test data) ─────────
    if not aligned:
        preprocess = getattr(opt, "preprocess", "resize_and_crop")

        if "resize" in preprocess:
            steps.append(T.Resize((opt.load_size, opt.load_size), method))

        if "scale_width" in preprocess:
            target_h = int(opt.load_size * opt.height / opt.width)
            steps.append(T.Resize((target_h, opt.load_size), method))

        if "crop" in preprocess:
            steps.append(T.RandomCrop(opt.crop_size))

        if not getattr(opt, "no_flip", False):
            steps.append(T.RandomHorizontalFlip())

    # ── Tensor conversion & normalisation ────────────────────────────
    steps.append(T.ToTensor())

    if grayscale:
        steps.append(T.Normalize(mean=(0.5,), std=(0.5,)))
    else:
        steps.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    return T.Compose(steps)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseDataset(data.Dataset, ABC):
    """Abstract base class for CycleGAN dataset implementations.

    All concrete subclasses must implement :meth:`__len__` and
    :meth:`__getitem__`.

    Args:
        opt: Options namespace; must expose ``dataroot``.
    """

    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.root: Path = Path(opt.dataroot)

    # ------------------------------------------------------------------
    @staticmethod
    def modify_commandline_options(parser, is_train: bool):
        """Hook for subclasses to add dataset-specific CLI options."""
        return parser

    def name(self) -> str:
        """Human-readable dataset class name (used for logging)."""
        return type(self).__name__

    # ------------------------------------------------------------------
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Return a single sample as a dictionary of tensors and metadata."""
