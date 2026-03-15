from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """Command-line options for training."""

    def __init__(self):
        super().__init__()
        self.isTrain = True

    def initialize(self):
        super().initialize()

        # ── Training schedule ────────────────────────────────────────────────
        self.parser.add_argument('--n_epochs', type=int, default=100,
                                 help='Number of epochs with constant learning rate')
        self.parser.add_argument('--n_epochs_decay', type=int, default=100,
                                 help='Number of epochs for linear LR decay to zero')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='Starting epoch counter')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='Resume training from the latest checkpoint')

        # ── Optimiser ────────────────────────────────────────────────────────
        self.parser.add_argument('--lr', type=float, default=0.0002,
                                 help='Initial learning rate for Adam optimiser')
        self.parser.add_argument('--beta1', type=float, default=0.5,
                                 help='Adam β₁ momentum term')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='Size of the image replay buffer for discriminator training')

        # ── Loss weights ─────────────────────────────────────────────────────
        self.parser.add_argument('--lambda_A', type=float, default=10.0,
                                 help='Weight for forward cycle-consistency loss (A→B→A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0,
                                 help='Weight for backward cycle-consistency loss (B→A→B)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5,
                                 help='Weight for identity loss; 0 disables it')

        # ── Logging & checkpointing ──────────────────────────────────────────
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='Frequency (in epochs) at which to save checkpoints')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='Frequency (in iterations) at which to print losses')
