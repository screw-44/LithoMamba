"""Portable generator/discriminator checkpoint helpers."""

from pathlib import Path

import torch


def save_network(network, save_dir, network_label, epoch_label):
    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = 'epoch_{}_{}.pth'.format(epoch_label, network_label)
    state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
    torch.save(state, directory / filename)


def load_network(network, save_dir, network_label, epoch_label):
    filename = 'epoch_{}_{}.pth'.format(epoch_label, network_label)
    path = Path(save_dir) / filename
    if not path.is_file():
        raise FileNotFoundError('{} does not exist'.format(path))
    network.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    print('INFO: Loaded network {} at epoch {}'.format(network_label, epoch_label))
