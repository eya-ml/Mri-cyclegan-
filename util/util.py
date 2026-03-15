"""
util.py
=======
General-purpose utility functions for image I/O and filesystem management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ─────────────────────────────────────────────────────────────────────────────

def mkdirs(paths: Union[str, Path, list]) -> None:
    """Create one or more directories, including any missing parents.

    Args:
        paths: A single path (``str`` or ``Path``) or a list of paths.
               Existing directories are silently ignored.
    """
    if isinstance(paths, (list, tuple)):
        for p in paths:
            Path(p).mkdir(parents=True, exist_ok=True)
    else:
        Path(paths).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor ↔ image conversion
# ─────────────────────────────────────────────────────────────────────────────

def tensor2im(
    tensor: Union[torch.Tensor, np.ndarray],
    dtype: type = np.uint8,
) -> np.ndarray:
    """Convert a normalised PyTorch tensor to a displayable NumPy image.

    The tensor is expected to be normalised to [-1, 1] (standard CycleGAN
    convention). It is rescaled to [0, 255] and transposed from CHW to HWC.

    Handles both batched tensors (shape ``(1, C, H, W)``) and single
    tensors (shape ``(C, H, W)``).

    Args:
        tensor: Image tensor or numpy array in [-1, 1].
        dtype:  Output numpy dtype (default: ``np.uint8``).

    Returns:
        NumPy array of shape ``(H, W, C)`` in [0, 255].
    """
    if isinstance(tensor, np.ndarray):
        return tensor.astype(dtype)

    arr: np.ndarray = tensor.detach().cpu().float().numpy()
    if arr.ndim == 4:          # remove batch dimension if present
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0))   # CHW → HWC
    arr = (arr + 1.0) / 2.0 * 255.0     # [-1,1] → [0,255]
    return arr.clip(0, 255).astype(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """Save a NumPy image array to disk as a PNG file.

    Parent directories are created automatically.

    Args:
        image: Array of shape ``(H, W, C)`` or ``(H, W)`` in [0, 255].
        path:  Destination file path (must end with a supported PIL format,
               e.g. ``.png``, ``.jpg``).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(dest)


def display_image(image: np.ndarray) -> None:
    """Display a NumPy image array using the system default image viewer.

    Primarily useful inside Jupyter notebooks.

    Args:
        image: Array of shape ``(H, W, C)`` or ``(H, W)`` in [0, 255].
    """
    Image.fromarray(image).show()
