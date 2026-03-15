"""
networks.py
===========
All neural network building blocks for the CycleGAN pipeline:

  - ResnetGenerator   : ResNet-based encoder-decoder generator
  - ResnetBlock       : Single residual block used inside the generator
  - NLayerDiscriminator : PatchGAN discriminator (70×70 receptive field)
  - GANLoss           : Least-squares adversarial loss (LSGAN)
  - init_weights      : Gaussian weight initialisation for conv layers
  - get_scheduler     : Linear learning-rate decay scheduler

References
----------
Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2020).
    Unpaired Image-to-Image Translation using Cycle-Consistent
    Adversarial Networks. arXiv:1703.10593.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────
NormLayer = type[nn.InstanceNorm2d]   # only InstanceNorm2d is used here


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_weights(net: nn.Module, gain: float = 0.02) -> None:
    """Initialise convolutional weights with a zero-mean Gaussian.

    Biases are set to zero.  InstanceNorm2d layers are skipped because
    they are constructed with ``affine=False`` and have no learnable
    parameters.

    Args:
        net:  The network whose parameters are to be initialised.
        gain: Standard deviation of the normal distribution (default 0.02).
    """
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, mean=0.0, std=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler
# ─────────────────────────────────────────────────────────────────────────────

def get_scheduler(optimizer: Optimizer, opt) -> LambdaLR:
    """Return a linear learning-rate decay scheduler.

    The learning rate stays constant for the first ``n_epochs`` epochs, then
    decays linearly to zero over the following ``n_epochs_decay`` epochs.

    Args:
        optimizer: The optimiser whose LR will be scheduled.
        opt:       Training options namespace; must expose ``epoch_count``,
                   ``n_epochs``, and ``n_epochs_decay``.

    Returns:
        A ``LambdaLR`` scheduler instance.
    """
    def _lr_lambda(epoch: int) -> float:
        decay_steps = max(0, epoch + opt.epoch_count - opt.n_epochs)
        return 1.0 - decay_steps / float(opt.n_epochs_decay + 1)

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class ResnetBlock(nn.Module):
    """A single residual block used inside :class:`ResnetGenerator`.

    The block applies two 3×3 convolutions with normalisation and ReLU,
    then adds the input skip connection.

    Args:
        channels:     Number of input (and output) feature channels.
        padding_type: ``'reflect'`` (default) or ``'zero'``.
        norm_layer:   Normalisation layer class (default: ``InstanceNorm2d``).
        use_dropout:  If ``True``, insert a 50 % dropout layer between the two
                      convolutions.
    """

    def __init__(
        self,
        channels: int,
        padding_type: str = "reflect",
        norm_layer: NormLayer = nn.InstanceNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.conv_block = self._build_conv_block(
            channels, padding_type, norm_layer, use_dropout
        )

    # ------------------------------------------------------------------
    def _build_conv_block(
        self,
        channels: int,
        padding_type: str,
        norm_layer: NormLayer,
        use_dropout: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []

        # ── First conv ───────────────────────────────────────────────
        pad = nn.ReflectionPad2d(1) if padding_type == "reflect" else nn.ZeroPad2d(1)
        layers += [pad, nn.Conv2d(channels, channels, kernel_size=3, padding=0),
                   norm_layer(channels), nn.ReLU(inplace=True)]

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        # ── Second conv ──────────────────────────────────────────────
        layers += [nn.ReflectionPad2d(1),
                   nn.Conv2d(channels, channels, kernel_size=3, padding=0),
                   norm_layer(channels)]

        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual (skip) connection."""
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet-based encoder–bottleneck–decoder generator.

    Architecture:
      1. Initial 7×7 convolution with reflection padding.
      2. Two strided 3×3 downsampling convolutions (stride 2).
      3. ``n_blocks`` residual blocks at the bottleneck.
      4. Two transposed convolutions for upsampling.
      5. Final 7×7 convolution + Tanh to map to [-1, 1].

    Args:
        input_nc:     Number of input image channels.
        output_nc:    Number of output image channels.
        ngf:          Base number of filters (multiplied along the encoder).
        use_dropout:  Enable dropout inside residual blocks.
        n_blocks:     Number of residual blocks (6 or 9).
        padding_type: Padding strategy for residual blocks
                      (``'reflect'`` or ``'zero'``).
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
    ) -> None:
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be non-negative, got {n_blocks}")
        super().__init__()

        norm = nn.InstanceNorm2d
        layers: list[nn.Module] = []

        # ── Encoder head ─────────────────────────────────────────────
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm(ngf),
            nn.ReLU(inplace=True),
        ]

        # ── Downsampling ─────────────────────────────────────────────
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2,
                          kernel_size=3, stride=2, padding=1),
                norm(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # ── Bottleneck: residual blocks ───────────────────────────────
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            layers.append(
                ResnetBlock(ngf * mult, padding_type=padding_type,
                            norm_layer=norm, use_dropout=use_dropout)
            )

        # ── Upsampling ────────────────────────────────────────────────
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                norm(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]

        # ── Output head ──────────────────────────────────────────────
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        init_weights(self)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator that classifies overlapping 70×70 patches.

    Composed of a sequence of strided convolutions that progressively
    reduce spatial resolution, ending with a single-channel prediction map
    where each value corresponds to a patch of the input image.

    Args:
        input_nc:  Number of input channels (3 for standard images, 6 for
                   concatenated pairs used in conditional discriminators).
        ndf:       Number of filters in the first convolutional layer.
        n_layers:  Number of intermediate strided conv layers (default: 3).
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        norm = nn.InstanceNorm2d
        kw, padw = 4, 1

        # ── Input layer (no normalisation) ───────────────────────────
        layers: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # ── Intermediate strided conv layers ─────────────────────────
        nf_prev = 1
        for n in range(1, n_layers):
            nf = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_prev, ndf * nf,
                          kernel_size=kw, stride=2, padding=padw, bias=True),
                norm(ndf * nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            nf_prev = nf

        # ── Penultimate layer (stride 1) ──────────────────────────────
        nf = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_prev, ndf * nf,
                      kernel_size=kw, stride=1, padding=padw, bias=True),
            norm(ndf * nf),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # ── Output: single-channel patch prediction ──────────────────
        layers.append(
            nn.Conv2d(ndf * nf, 1, kernel_size=kw, stride=1, padding=padw)
        )

        self.model = nn.Sequential(*layers)
        init_weights(self)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """Least-squares GAN loss (LSGAN).

    Instead of binary cross-entropy, this implementation uses mean squared
    error between the discriminator predictions and soft targets (1 for real,
    0 for fake), following Mao et al. (2017).

    Target tensors are created once and broadcast to the prediction shape,
    avoiding repeated allocation on the GPU.

    Args:
        real_label: Target value for real images (default: 1.0).
        fake_label: Target value for fake images (default: 0.0).
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self._criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    def _get_target(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        label = self.real_label if target_is_real else self.fake_label  # type: ignore[attr-defined]
        return label.expand_as(prediction)

    # ------------------------------------------------------------------
    def forward(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """Compute LSGAN loss.

        Args:
            prediction:     Raw discriminator output (patch map).
            target_is_real: ``True`` → compare against real-label targets;
                            ``False`` → compare against fake-label targets.

        Returns:
            Scalar loss tensor.
        """
        target = self._get_target(prediction, target_is_real)
        return self._criterion(prediction, target)
