"""Shared arguments for the LithoMamba training and inference entry points."""

import argparse
from pathlib import Path

import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.set_defaults(is_train=True)
        self.parser.add_argument('--name', default='DefaultExperimentName',
                                 help='experiment name used for checkpoint storage')
        self.parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu', 'mps'],
                                 help='device selection; full LithoMamba execution requires CUDA')
        self.parser.add_argument('--gpu_ids', type=int, default=0,
                                 help='single CUDA device index; -1 selects CPU')
        self.parser.add_argument('--checkpoints_dir', default='./checkpoints',
                                 help='checkpoint directory')
        self.parser.add_argument('--layout_image_dir', default='./datasets/train/layout',
                                 help='directory of grayscale layout images')
        self.parser.add_argument('--sem_image_dir', default='./datasets/train/sem',
                                 help='directory of paired grayscale SEM images')
        self.parser.add_argument('--shuffle', action='store_true', help='shuffle image pairs')
        self.parser.add_argument('--num_workers', type=int, default=16, help='data loader workers')
        self.parser.add_argument('--batch_size', type=int, default=8, help='images per batch')
        self.parser.add_argument('--load_size', type=int, default=256, help='square image resize size')
        self.parser.add_argument('--crop_size', type=int, default=256, help='crop size after resizing')
        self.parser.add_argument('--flip_ratio', type=float, default=0.5,
                                 help='probability of each horizontal/vertical training flip')
        self.parser.add_argument('--resize_or_crop', default='scale_short_and_crop',
                                 choices=['resize_and_crop', 'crop', 'scale_short', 'scale_short_and_crop'],
                                 help='all modes resize to load_size; crop modes also crop to crop_size')
        self.parser.add_argument('--max_dataset_size', type=int, default=float('inf'),
                                 help='maximum number of image pairs to load')
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        if self.opt.gpu_ids < 0:
            self.opt.device = 'cpu'
        elif self.opt.device == 'auto':
            self.opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lines = ['------------ Options -------------']
        lines.extend('{}: {}'.format(key, value) for key, value in sorted(vars(self.opt).items()))
        lines.append('-------------- End ----------------')
        output = '\n'.join(lines) + '\n'
        print(output, end='')
        if save:
            experiment_dir = Path(self.opt.checkpoints_dir) / self.opt.name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            (experiment_dir / 'opt.txt').write_text(output)
        return self.opt
