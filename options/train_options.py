"""Arguments used by train_mamba.py."""

import argparse
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument('--display_freq', type=int, default=5,
                                 help='save checkpoints and preview images every this many epochs')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='warm start from saved generator/discriminator weights')
        self.parser.add_argument('--which_epoch', default='latest', help='checkpoint epoch to load')
        self.parser.add_argument('--pretrained_path', default='',
                                 help='optional VMamba initialization checkpoint for fresh training')
        self.parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True,
                                 help='train with AMP; --no-fp16 disables mixed precision')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam first-moment coefficient')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='Adam learning rate')
        self.parser.add_argument('--epochs', type=int, default=500, help='total training epochs')
        self.parser.add_argument('--lambda_l1', type=float, default=1.0,
                                 help='reconstruction weight; legacy default 1.0, paper reports 0.1')
        self.parser.add_argument('--lambda_edge', type=float, default=1.0,
                                 help='legacy edge-region loss weight; use 0 for the paper objective')
        self.parser.add_argument('--edge_threshold', type=float, default=240 / 255,
                                 help='legacy edge threshold in the normalized tensor domain')
