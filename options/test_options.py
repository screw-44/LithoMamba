"""Arguments used by the paired inference entry point test.py."""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument('--results_dir', default='./results/', help='PNG prediction directory')
        self.parser.add_argument('--num_test', type=int, default=50, help='maximum images to generate')
        self.parser.add_argument('--which_epoch', default='latest', help='generator checkpoint epoch to load')
        self.parser.set_defaults(is_train=False,
                                 layout_image_dir='./datasets/test/layout',
                                 sem_image_dir='./datasets/test/sem')
