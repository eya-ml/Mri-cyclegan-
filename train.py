"""
train.py
========
Training entry point for the CycleGAN MRI contrast-translation pipeline.

Usage
-----
    python train.py --dataroot ./datasets/t12t2_brain/0.5 \\
                    --model cycle_gan \\
                    --dataset_mode unaligned \\
                    --netG resnet_9blocks \\
                    --direction AtoB \\
                    --super_start 1 \\
                    --super_mode aligned \\
                    --name paired_gan_50 \\
                    --no_dropout

    # CPU-only training:
    python train.py ... --gpu_ids -1
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from data.cyclegan_dataloader import CycleGANDataLoader
from models.cycle_gan_model import CycleGANModel
from options.train_options import TrainOptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def _format_losses(losses: dict) -> str:
    """Format a loss dictionary into a compact, human-readable string."""
    return "  ".join(f"{k}: {v:.4f}" for k, v in losses.items())


def _save_losses(log_path: Path, line: str) -> None:
    """Append a single line to the training loss log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    """Run the full CycleGAN training loop."""

    # ── Configuration ────────────────────────────────────────────────
    opt = TrainOptions().parse()
    log_path = Path("checkpoints") / opt.name / "losses.txt"

    # ── Data ─────────────────────────────────────────────────────────
    loader = CycleGANDataLoader(opt).load_data()
    logger.info("Dataset size : %d samples", len(loader))

    # ── Model ────────────────────────────────────────────────────────
    model = CycleGANModel(opt)

    # ── Training loop ────────────────────────────────────────────────
    total_steps = 0
    n_total_epochs = opt.n_epochs + opt.n_epochs_decay

    for epoch in range(opt.epoch_count, n_total_epochs + 1):
        epoch_start = time.time()
        logger.info("━━━ Epoch %d / %d ━━━", epoch, n_total_epochs)

        # ── Inner batch loop ─────────────────────────────────────────
        for batch in loader:
            total_steps += opt.batch_size
            model.set_input(batch)
            model.optimize_parameters()

        # ── End-of-epoch actions ─────────────────────────────────────
        elapsed = time.time() - epoch_start

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks("latest")
            model.save_networks(epoch)

            losses = model.get_current_losses()
            loss_str = _format_losses(losses)
            log_line = f"Epoch {epoch:04d} | step {total_steps:08d} | {loss_str}"

            logger.info(log_line)
            _save_losses(log_path, log_line)

        logger.info("Epoch %d finished — %.1f s", epoch, elapsed)

        # ── LR schedule ──────────────────────────────────────────────
        model.update_learning_rate()

    _save_losses(log_path, "Training complete.")
    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
