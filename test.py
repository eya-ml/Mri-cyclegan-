"""
test.py
=======
Inference entry point for the CycleGAN MRI contrast-translation pipeline.

For each input image the script saves six output images:

  real_A  — original domain-A input
  fake_B  — synthesised domain-B image          ← primary output
  rec_A   — cycle-reconstructed domain-A image
  real_B  — original domain-B reference
  fake_A  — synthesised domain-A image
  rec_B   — cycle-reconstructed domain-B image

Usage
-----
    python test.py --dataroot  ./datasets/t12t2_brain/0.5 \\
                   --results_dir results/t12t2_brain/0.5 \\
                   --model      cycle_gan \\
                   --dataset_mode single \\
                   --netG       resnet_9blocks \\
                   --direction  AtoB \\
                   --name       paired_gan_50 \\
                   --num_test   500 \\
                   --no_dropout
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from data.cyclegan_dataloader import CycleGANDataLoader
from models.cycle_gan_model import CycleGANModel
from options.test_options import TestOptions
from util.util import mkdirs, save_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Maps visual label → which source path provides the filename stem
_LABEL_TO_PATH_KEY: dict[str, str] = {
    "real_A": "A_path",
    "fake_B": "A_path",
    "rec_A":  "A_path",
    "real_B": "B_path",
    "fake_A": "B_path",
    "rec_B":  "B_path",
}


# ─────────────────────────────────────────────────────────────────────────────

def _prepare_output_dirs(results_dir: Path) -> None:
    """Create one sub-folder per visual type inside *results_dir*."""
    for label in _LABEL_TO_PATH_KEY:
        mkdirs(results_dir / label)


def _stem(path_list: list, key: str, batch: dict) -> str:
    """Extract the filename stem (without extension) for a given path key."""
    return Path(batch[key][0]).stem


# ─────────────────────────────────────────────────────────────────────────────

def test() -> None:
    """Run inference and save all visual outputs to disk."""

    # ── Configuration ────────────────────────────────────────────────
    opt = TestOptions().parse()
    opt.num_threads   = 1     # single-threaded inference
    opt.batch_size    = 1     # one image at a time
    opt.serial_batches = True  # preserve image order
    opt.no_flip        = True  # no augmentation at test time

    results_dir = Path(opt.results_dir)
    _prepare_output_dirs(results_dir)

    # ── Data & Model ─────────────────────────────────────────────────
    loader = CycleGANDataLoader(opt).load_data()
    model  = CycleGANModel(opt)
    logger.info("Running inference on %d samples (limit: %d)",
                len(loader), opt.num_test)

    # ── Inference loop ───────────────────────────────────────────────
    for idx, batch in enumerate(loader):
        if idx >= opt.num_test:
            break

        model.set_input(batch)
        model.test()
        visuals = model.get_current_visuals()

        # ── Save each visual ─────────────────────────────────────────
        for label, image in visuals.items():
            path_key = _LABEL_TO_PATH_KEY[label]
            stem     = _stem([], path_key, batch)
            filename = f"{stem}_{label}.png"
            dest     = results_dir / label / filename
            save_image(image, dest)

        if (idx + 1) % 50 == 0:
            logger.info("  Processed %d / %d", idx + 1, opt.num_test)

    logger.info("Inference complete. Results saved to %s", results_dir)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test()
