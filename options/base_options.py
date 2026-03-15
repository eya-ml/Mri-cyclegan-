import argparse
import os
import torch


class BaseOptions:
    """Base class for training and testing options."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="MRI T1↔T2 Translation with CycleGAN",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):
        # ── Dataset ──────────────────────────────────────────────────────────
        self.parser.add_argument('--dataroot', required=True,
                                 help='Path to dataset root directory')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned',
                                 choices=['unaligned', 'aligned', 'single'],
                                 help='Dataset loading mode')
        self.parser.add_argument('--direction', type=str, default='AtoB',
                                 choices=['AtoB', 'BtoA'],
                                 help='Translation direction')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='Disable random shuffling of data')
        self.parser.add_argument('--num_threads', type=int, default=4,
                                 help='Number of DataLoader worker threads')
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='Input batch size')
        self.parser.add_argument('--max_dataset_size', type=int, default=float('inf'),
                                 help='Maximum number of samples per epoch')

        # ── Preprocessing ────────────────────────────────────────────────────
        self.parser.add_argument('--load_size', type=int, default=286,
                                 help='Scale images to this size before cropping')
        self.parser.add_argument('--crop_size', type=int, default=256,
                                 help='Crop images to this size')
        self.parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                                 help='Preprocessing pipeline (resize, crop, scale_width, etc.)')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='Disable random horizontal flip augmentation')
        self.parser.add_argument('--height', type=int, default=256,
                                 help='Image height (used with scale_width preprocessing)')
        self.parser.add_argument('--width', type=int, default=256,
                                 help='Image width (used with scale_width preprocessing)')

        # ── Model ────────────────────────────────────────────────────────────
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='Model architecture to use')
        self.parser.add_argument('--input_nc', type=int, default=3,
                                 help='Number of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3,
                                 help='Number of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64,
                                 help='Number of generator filters in the first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64,
                                 help='Number of discriminator filters in the first conv layer')
        self.parser.add_argument('--netG', type=str, default='resnet_9blocks',
                                 choices=['resnet_9blocks', 'resnet_6blocks'],
                                 help='Generator architecture')
        self.parser.add_argument('--no_dropout', action='store_true',
                                 help='Disable dropout in generators')

        # ── Hardware ─────────────────────────────────────────────────────────
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='GPU IDs to use, e.g. "0,1". Use "-1" for CPU.')

        # ── Checkpoints ──────────────────────────────────────────────────────
        self.parser.add_argument('--name', type=str, default='experiment',
                                 help='Experiment name (used for checkpoint directory)')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                 help='Root directory for saving checkpoints')
        self.parser.add_argument('--epoch', type=str, default='latest',
                                 help='Epoch to load (e.g. "latest" or an integer)')

        # ── Semi-supervised / Paired training ────────────────────────────────
        self.parser.add_argument('--super_start', type=int, default=0,
                                 help='Enable paired (supervised) training mode (1=yes, 0=no)')
        self.parser.add_argument('--super_mode', type=str, default='aligned',
                                 help='Dataset mode for paired samples')
        self.parser.add_argument('--super_epochs', type=int, default=50,
                                 help='Number of epochs to use paired data')
        self.parser.add_argument('--super_epoch_start', type=int, default=0,
                                 help='Epoch at which to start using paired data')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        # Parse GPU IDs
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = [int(i) for i in str_ids if int(i) >= -1]

        # Validate GPU availability
        if self.opt.gpu_ids[0] != -1 and not torch.cuda.is_available():
            print("Warning: CUDA not available. Falling back to CPU.")
            self.opt.gpu_ids = [-1]

        self._print_options()
        return self.opt

    def _print_options(self):
        """Print and save options to a text file."""
        message = '\n' + '─' * 60 + '\n'
        message += f"{'Options':^60}\n"
        message += '─' * 60 + '\n'
        for k, v in sorted(vars(self.opt).items()):
            message += f'  {k:<30}: {v}\n'
        message += '─' * 60 + '\n'
        print(message)

        os.makedirs(os.path.join(self.opt.checkpoints_dir, self.opt.name), exist_ok=True)
        file_name = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'opt.txt')
        with open(file_name, 'wt') as f:
            f.write(message)
