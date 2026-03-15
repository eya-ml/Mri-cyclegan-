"""
cycle_gan_model.py
==================
Core CycleGAN model supporting both fully unpaired training and the
semi-supervised paired extension from Tripathy et al. (2019).

Architecture
------------
Generators
    G_A : domain A → domain B  (e.g. T1 → T2)
    G_B : domain B → domain A  (e.g. T2 → T1)

Discriminators
    D_A, D_B : standard single-image PatchGAN discriminators
    D_C, D_D : conditional PatchGAN discriminators that evaluate
               concatenated image pairs (used when paired data
               is available, i.e. ``opt.super_start == 1``)

Loss terms
    • Adversarial loss           (LSGAN)
    • Cycle-consistency loss     (L1)
    • Identity loss              (L1, optional)
    • Conditional adversarial loss        (paired mode only)
    • Conditional cycle-consistency loss  (paired mode only)

References
----------
Zhu, J.-Y. et al. (2020). Unpaired Image-to-Image Translation using
    Cycle-Consistent Adversarial Networks. arXiv:1703.10593.
Tripathy, S. et al. (2019). Learning Image-to-Image Translation using
    Paired and Unpaired Training Samples. ACCV 2018.
"""

from __future__ import annotations

import itertools
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from util.image_pool import ImagePool
import util.util as util
from . import networks

logger = logging.getLogger(__name__)


class CycleGANModel:
    """Full CycleGAN model with optional semi-supervised paired-data support.

    Args:
        opt: Parsed options namespace (see ``options/``).  Required fields:
             ``gpu_ids``, ``isTrain``, ``checkpoints_dir``, ``name``,
             ``input_nc``, ``output_nc``, ``ngf``, ``ndf``, ``netG``,
             ``no_dropout``, ``pool_size``, ``lr``, ``beta1``,
             ``lambda_A``, ``lambda_B``, ``lambda_identity``,
             ``super_start``, ``continue_train``, ``epoch``.
    """

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(self, opt) -> None:
        self.opt = self._set_defaults(opt)
        self.is_train: bool = opt.isTrain
        self.device: torch.device = self._resolve_device(opt.gpu_ids)
        self.save_dir: Path = Path(opt.checkpoints_dir) / opt.name

        logger.info("Device: %s", self.device)

        self._build_generators()
        if self.is_train:
            self._build_discriminators()
            self._build_losses()
            self._build_optimisers()
            self._build_schedulers()

        if not self.is_train or self.opt.continue_train:
            self._load_generators()

    # ──────────────────────────────────────────────────────────────────
    # Private helpers — initialisation
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _set_defaults(opt):
        """Back-fill any options that may be absent (e.g. during testing)."""
        defaults = {"lambda_identity": 0.5, "lambda_A": 10.0, "lambda_B": 10.0}
        for key, value in defaults.items():
            if not hasattr(opt, key):
                setattr(opt, key, value)
        return opt

    @staticmethod
    def _resolve_device(gpu_ids: List[int]) -> torch.device:
        """Pick CUDA or CPU based on the requested GPU IDs."""
        if gpu_ids and gpu_ids[0] != -1 and torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_ids[0]}")
        return torch.device("cpu")

    def _make_generator(self, in_ch: int, out_ch: int) -> nn.Module:
        n_blocks = 9 if self.opt.netG == "resnet_9blocks" else 6
        return networks.ResnetGenerator(
            in_ch, out_ch,
            ngf=self.opt.ngf,
            use_dropout=not self.opt.no_dropout,
            n_blocks=n_blocks,
        ).to(self.device)

    def _make_discriminator(self, in_ch: int) -> nn.Module:
        return networks.NLayerDiscriminator(
            in_ch, ndf=self.opt.ndf, n_layers=3
        ).to(self.device)

    def _build_generators(self) -> None:
        self.netG_A = self._make_generator(self.opt.input_nc, self.opt.output_nc)
        self.netG_B = self._make_generator(self.opt.output_nc, self.opt.input_nc)

    def _build_discriminators(self) -> None:
        # Standard single-image discriminators
        self.netD_A = self._make_discriminator(self.opt.output_nc)
        self.netD_B = self._make_discriminator(self.opt.input_nc)
        # Conditional pair discriminators (input = 2 × 3 = 6 channels)
        self.netD_C = self._make_discriminator(6)
        self.netD_D = self._make_discriminator(6)

        # Replay buffers for stabilised discriminator updates
        self.fake_A_pool = ImagePool(self.opt.pool_size)
        self.fake_B_pool = ImagePool(self.opt.pool_size)

    def _build_losses(self) -> None:
        self.criterion_gan   = networks.GANLoss().to(self.device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_idt   = nn.L1Loss()

    def _build_optimisers(self) -> None:
        lr, betas = self.opt.lr, (self.opt.beta1, 0.999)

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=lr, betas=betas,
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(
                self.netD_A.parameters(), self.netD_B.parameters(),
                self.netD_C.parameters(), self.netD_D.parameters(),
            ),
            lr=lr, betas=betas,
        )
        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def _build_schedulers(self) -> None:
        self.schedulers = [
            networks.get_scheduler(opt, self.opt) for opt in self.optimizers
        ]

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────────

    def _load_generators(self) -> None:
        """Restore generator weights from disk."""
        for name in ("G_A", "G_B"):
            net = getattr(self, f"net{name}")
            path = self.save_dir / f"{self.opt.epoch}_net_{name}.pth"
            net.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            logger.info("Loaded %s from %s", name, path)

    def save_networks(self, epoch: int | str) -> None:
        """Persist all network weights to ``save_dir``.

        Args:
            epoch: Epoch identifier used as a filename prefix
                   (e.g. ``50`` or ``'latest'``).
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for name in ("G_A", "G_B", "D_A", "D_B", "D_C", "D_D"):
            net = getattr(self, f"net{name}")
            path = self.save_dir / f"{epoch}_net_{name}.pth"
            torch.save(net.state_dict(), path)
        logger.debug("Saved all networks for epoch %s", epoch)

    # ──────────────────────────────────────────────────────────────────
    # Data flow
    # ──────────────────────────────────────────────────────────────────

    def set_input(self, batch: Dict[str, torch.Tensor]) -> None:
        """Unpack a data batch from the DataLoader.

        Args:
            batch: Dictionary with keys ``'A'``, ``'B'``, ``'A_path'``,
                   ``'B_path'``.
        """
        self.real_A: torch.Tensor = batch["A"].to(self.device)
        self.real_B: torch.Tensor = batch["B"].to(self.device)
        key = "A_path" if self.opt.direction == "AtoB" else "B_path"
        self.image_paths: list = batch[key]

    def forward(self) -> None:
        """Run the full forward pass (generation + cycle reconstruction)."""
        # Forward translation
        self.fake_B: torch.Tensor = self.netG_A(self.real_A)  # A → B
        self.fake_A: torch.Tensor = self.netG_B(self.real_B)  # B → A

        # Cycle reconstruction
        self.rec_A: torch.Tensor = self.netG_B(self.fake_B)   # A → B → A
        self.rec_B: torch.Tensor = self.netG_A(self.fake_A)   # B → A → B

    # ──────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self) -> None:
        """Run inference (no gradient computation, eval mode)."""
        self.netG_A.eval()
        self.netG_B.eval()
        self.forward()
        self.netG_A.train()
        self.netG_B.train()

    # ──────────────────────────────────────────────────────────────────
    # Loss computation — discriminators
    # ──────────────────────────────────────────────────────────────────

    def _discriminator_loss(
        self,
        net_D: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute and back-propagate the standard GAN discriminator loss.

        Loss = 0.5 × [L(D(real), 1) + L(D(fake.detach()), 0)]

        Args:
            net_D: Discriminator network.
            real:  Real image tensor.
            fake:  Generated image tensor (detached to stop generator gradients).

        Returns:
            Scalar discriminator loss (already back-propagated).
        """
        loss_real = self.criterion_gan(net_D(real), target_is_real=True)
        loss_fake = self.criterion_gan(net_D(fake.detach()), target_is_real=False)
        loss = (loss_real + loss_fake) * 0.5
        loss.backward()
        return loss

    def _update_D_A(self) -> None:
        """Update discriminator D_A (judges whether B images are real)."""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self._discriminator_loss(self.netD_A, self.real_B, fake_B)

    def _update_D_B(self) -> None:
        """Update discriminator D_B (judges whether A images are real)."""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self._discriminator_loss(self.netD_B, self.real_A, fake_A)

    def _update_D_C(self) -> None:
        """Update conditional discriminator D_C.

        In paired mode:  real pair  = (A, B),  fake pair = (fake_A, B)
        In unpaired mode: real pair = (A, fake_B), fake pair = (rec_A, fake_B)
        """
        if self.opt.super_start == 1:
            real = torch.cat([self.real_A, self.real_B], dim=1)
            fake = torch.cat([self.fake_A.detach(), self.real_B], dim=1)
        else:
            real = torch.cat([self.real_A, self.fake_B.detach()], dim=1)
            fake = torch.cat([self.rec_A.detach(), self.fake_B.detach()], dim=1)

        self.loss_D_C = self._discriminator_loss(self.netD_C, real, fake)

    def _update_D_D(self) -> None:
        """Update conditional discriminator D_D.

        In paired mode:  real pair  = (B, A),  fake pair = (fake_B, A)
        In unpaired mode: real pair = (B, fake_A), fake pair = (rec_B, fake_A)
        """
        if self.opt.super_start == 1:
            real = torch.cat([self.real_B, self.real_A], dim=1)
            fake = torch.cat([self.fake_B.detach(), self.real_A], dim=1)
        else:
            real = torch.cat([self.real_B, self.fake_A.detach()], dim=1)
            fake = torch.cat([self.rec_B.detach(), self.fake_A.detach()], dim=1)

        self.loss_D_D = self._discriminator_loss(self.netD_D, real, fake)

    # ──────────────────────────────────────────────────────────────────
    # Loss computation — generators
    # ──────────────────────────────────────────────────────────────────

    def _adversarial_cycle_loss(
        self,
        translated: torch.Tensor,
        source: torch.Tensor,
        net_D: nn.Module,
    ) -> torch.Tensor:
        """Conditional adversarial cycle-consistency loss.

        Encourages the pair (translated, source) to look like a real pair
        to the conditional discriminator.

        Args:
            translated: Translated image (e.g. fake_B).
            source:     Original source image (e.g. real_A).
            net_D:      Conditional discriminator to fool.

        Returns:
            Scalar adversarial loss.
        """
        if self.opt.super_start == 1:
            pair = torch.cat([translated, source], dim=1)
        else:
            pair = torch.cat([source, translated], dim=1)
        return self.criterion_gan(net_D(pair), target_is_real=True)

    def _compute_generator_loss(self) -> None:
        """Compute all generator losses and trigger back-propagation."""
        λ_A   = self.opt.lambda_A
        λ_B   = self.opt.lambda_B
        λ_idt = self.opt.lambda_identity

        # ── Identity loss ──────────────────────────────────────────
        if λ_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_A = self.criterion_idt(self.idt_A, self.real_B) * λ_B * λ_idt
            self.loss_idt_B = self.criterion_idt(self.idt_B, self.real_A) * λ_A * λ_idt
        else:
            self.loss_idt_A = torch.tensor(0.0, device=self.device)
            self.loss_idt_B = torch.tensor(0.0, device=self.device)

        # ── Standard adversarial losses ────────────────────────────
        self.loss_G_A = self.criterion_gan(self.netD_A(self.fake_B), target_is_real=True)
        self.loss_G_B = self.criterion_gan(self.netD_B(self.fake_A), target_is_real=True)

        # ── Cycle-consistency losses (L1) ───────────────────────────
        self.loss_cycle_A = self.criterion_cycle(self.rec_A, self.real_A) * λ_A
        self.loss_cycle_B = self.criterion_cycle(self.rec_B, self.real_B) * λ_B

        # ── Conditional adversarial cycle losses ────────────────────
        # Discriminator routing differs by training mode
        disc_for_A = self.netD_D if self.opt.super_start == 1 else self.netD_C
        disc_for_B = self.netD_C if self.opt.super_start == 1 else self.netD_D

        self.loss_cycle_A_adv = self._adversarial_cycle_loss(
            self.fake_B, self.real_A, disc_for_A
        )
        self.loss_cycle_B_adv = self._adversarial_cycle_loss(
            self.fake_A, self.real_B, disc_for_B
        )

        # ── Total generator loss ────────────────────────────────────
        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A
            + self.loss_cycle_B
            + self.loss_idt_A
            + self.loss_idt_B
            + self.loss_cycle_A_adv
            + self.loss_cycle_B_adv
        )
        self.loss_G.backward()

    # ──────────────────────────────────────────────────────────────────
    # Optimisation step
    # ──────────────────────────────────────────────────────────────────

    def optimize_parameters(self) -> None:
        """Perform one full generator + discriminator optimisation step.

        Execution order
        ~~~~~~~~~~~~~~~
        1. Forward pass (generate fake and reconstructed images).
        2. Freeze discriminators → update generators.
        3. Unfreeze discriminators → update all four discriminators.
        """
        all_discriminators = [self.netD_A, self.netD_B, self.netD_C, self.netD_D]

        # ── Step 1: forward ─────────────────────────────────────────
        self.forward()

        # ── Step 2: update generators ────────────────────────────────
        self._set_requires_grad(all_discriminators, requires_grad=False)
        self.optimizer_G.zero_grad()
        self._compute_generator_loss()
        self.optimizer_G.step()

        # ── Step 3: update discriminators ────────────────────────────
        self._set_requires_grad(all_discriminators, requires_grad=True)
        self.optimizer_D.zero_grad()
        self._update_D_A()
        self._update_D_B()
        self._update_D_C()
        self._update_D_D()
        self.optimizer_D.step()

    # ──────────────────────────────────────────────────────────────────
    # Learning-rate scheduling
    # ──────────────────────────────────────────────────────────────────

    def update_learning_rate(self) -> None:
        """Step all LR schedulers and log the current learning rate."""
        for scheduler in self.schedulers:
            scheduler.step()
        current_lr = self.optimizers[0].param_groups[0]["lr"]
        logger.info("Learning rate updated → %.7f", current_lr)

    # ──────────────────────────────────────────────────────────────────
    # Logging helpers
    # ──────────────────────────────────────────────────────────────────

    def get_current_losses(self) -> OrderedDict:
        """Return all current loss values as a labelled ordered dict.

        Returns:
            OrderedDict mapping loss name → scalar float.
        """
        paired = self.opt.super_start == 1
        return OrderedDict([
            ("loss_G",         self.loss_G.item()),
            ("loss_G_A",       self.loss_G_A.item()),
            ("loss_G_B",       self.loss_G_B.item()),
            ("loss_cycle_A",   self.loss_cycle_A.item()),
            ("loss_cycle_B",   self.loss_cycle_B.item()),
            ("loss_idt_A",     self.loss_idt_A.item()),
            ("loss_idt_B",     self.loss_idt_B.item()),
            ("loss_D_A",       self.loss_D_A.item()),
            ("loss_D_B",       self.loss_D_B.item()),
            ("loss_D_C",       self.loss_D_C.item() if paired else 0.0),
            ("loss_D_D",       self.loss_D_D.item() if paired else 0.0),
        ])

    # Backward-compat alias
    get_current_errors = get_current_losses

    def get_current_visuals(self) -> OrderedDict:
        """Return a dict of numpy images for logging / visualisation.

        Returns:
            OrderedDict mapping label → uint8 numpy array (H, W, C).
        """
        return OrderedDict([
            ("real_A", util.tensor2im(self.real_A)),
            ("fake_B", util.tensor2im(self.fake_B)),
            ("rec_A",  util.tensor2im(self.rec_A)),
            ("real_B", util.tensor2im(self.real_B)),
            ("fake_A", util.tensor2im(self.fake_A)),
            ("rec_B",  util.tensor2im(self.rec_B)),
        ])

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _set_requires_grad(
        nets: List[nn.Module], requires_grad: bool
    ) -> None:
        """Toggle gradient computation for a list of networks.

        Args:
            nets:           Networks to modify.
            requires_grad:  ``True`` to enable gradients, ``False`` to disable.
        """
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
