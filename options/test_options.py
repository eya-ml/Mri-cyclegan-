from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """Command-line options for testing / inference."""

    def __init__(self):
        super().__init__()
        self.isTrain = False

    def initialize(self):
        super().initialize()

        self.parser.add_argument('--results_dir', type=str, default='./results',
                                 help='Directory where translated images will be saved')
        self.parser.add_argument('--num_test', type=int, default=50,
                                 help='Number of test images to process')
        self.parser.add_argument('--eval', action='store_true',
                                 help='Run generators in eval mode during testing')
