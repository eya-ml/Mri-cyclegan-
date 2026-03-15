"""
image_pool.py
=============
Replay buffer for stabilising discriminator training in CycleGAN.

Keeping a history of previously generated images and randomly returning
stored images instead of the most recent ones reduces temporal correlation
in the discriminator updates, which in turn reduces oscillation during
adversarial training.

Reference
---------
Shrivastava, A., et al. (2017). Learning from Simulated and Unsupervised
    Images through Adversarial Training. CVPR 2017.
"""

from __future__ import annotations

import random
from typing import List

import torch


class ImagePool:
    """Fixed-size FIFO replay buffer with probabilistic sampling.

    When the pool is full, each new image is either:

    * Returned directly to the caller (with probability 0.5), **or**
    * Used to replace a random entry in the pool, while the displaced
      entry is returned to the caller (with probability 0.5).

    Setting ``pool_size=0`` disables buffering entirely — the input
    images are always returned unchanged (useful for debugging).

    Args:
        pool_size: Maximum number of images to hold in the buffer.
                   Use 0 to disable.
    """

    def __init__(self, pool_size: int) -> None:
        if pool_size < 0:
            raise ValueError(f"pool_size must be >= 0, got {pool_size}")
        self.pool_size = pool_size
        self._buffer: List[torch.Tensor] = []

    # ------------------------------------------------------------------

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Sample images from the pool, updating it with the incoming batch.

        Args:
            images: Batch tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, C, H, W)`` drawn from the pool.
            When the pool is disabled (``pool_size == 0``), returns
            *images* unchanged.
        """
        if self.pool_size == 0:
            return images

        output: List[torch.Tensor] = []

        for image in images:
            image_unsqueezed = image.data.unsqueeze(0)  # (1, C, H, W)

            if len(self._buffer) < self.pool_size:
                # Pool not yet full: always accept and return the new image
                self._buffer.append(image_unsqueezed)
                output.append(image_unsqueezed)
            elif random.random() > 0.5:
                # Replace a random pool entry; return the displaced image
                idx = random.randrange(len(self._buffer))
                displaced = self._buffer[idx].clone()
                self._buffer[idx] = image_unsqueezed
                output.append(displaced)
            else:
                # Return the new image without modifying the pool
                output.append(image_unsqueezed)

        return torch.cat(output, dim=0)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Current number of images stored in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"pool_size={self.pool_size}, "
            f"current_size={len(self._buffer)})"
        )
