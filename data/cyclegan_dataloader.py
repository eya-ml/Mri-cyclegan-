"""
cyclegan_dataloader.py
======================
DataLoader wrapper for the CycleGAN training and testing pipeline.

Selects between *unaligned* (unpaired) and *aligned* (paired) dataset
modes based on the training configuration, then wraps the dataset in a
``torch.utils.data.DataLoader`` with the appropriate shuffle, batch, and
worker settings.
"""

from __future__ import annotations

import logging
from typing import Iterator

from torch.utils.data import DataLoader

from data.cyclegan_dataset import CycleGANDataset

logger = logging.getLogger(__name__)


class CycleGANDataLoader:
    """Iterable DataLoader wrapper for CycleGAN experiments.

    Determines the dataset mode at construction time:

    * If ``opt.super_start == 1`` (paired training), the dataset is loaded
      in ``opt.super_mode`` (usually ``'aligned'``).
    * Otherwise, the standard ``opt.dataset_mode`` (usually
      ``'unaligned'``) is used.

    Args:
        opt: Options namespace.  Required fields:
             ``batch_size``, ``serial_batches``, ``num_threads``,
             ``max_dataset_size``, ``dataset_mode``, ``super_start``,
             ``super_mode``.
    """

    def __init__(self, opt) -> None:
        self.opt = opt
        self.dataset: CycleGANDataset = self._create_dataset(opt)
        self.dataloader: DataLoader = DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            pin_memory=True,
            drop_last=False,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dataset_mode(opt) -> str:
        """Return the dataset mode string for the current training regime."""
        use_paired = hasattr(opt, "super_start") and opt.super_start == 1
        return opt.super_mode if use_paired else opt.dataset_mode

    def _create_dataset(self, opt) -> CycleGANDataset:
        mode = self._resolve_dataset_mode(opt)
        dataset = CycleGANDataset(opt, dataset_mode=mode)
        logger.info("Dataset [%s | mode=%s] created — %d samples",
                    dataset.name(), mode, len(dataset))
        return dataset

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_data(self) -> "CycleGANDataLoader":
        """Return self (enables the ``dataset = loader.load_data()`` idiom)."""
        return self

    def __len__(self) -> int:
        """Effective dataset size, capped at ``opt.max_dataset_size``."""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self) -> Iterator[dict]:
        """Iterate over batches, stopping at ``opt.max_dataset_size``."""
        for i, batch in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield batch
