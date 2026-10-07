"""LithoMamba generator wrapper and optional VMamba weight initialization."""

import copy

import torch
import torch.nn as nn

from .mamba_sys import VSSM


class MambaUnet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.mamba_unet = VSSM(in_chans=1, num_classes=num_classes)

    def forward(self, inputs):
        return self.mamba_unet(inputs)

    def load_from(self, pretrained_path):
        print('Loading VMamba initialization: {}'.format(pretrained_path))
        pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if 'model' not in pretrained_dict:
            pretrained_dict = {key[17:]: value for key, value in pretrained_dict.items()}
            pretrained_dict = {key: value for key, value in pretrained_dict.items() if 'output' not in key}
            self.mamba_unet.load_state_dict(pretrained_dict, strict=False)
            return
        pretrained_dict = pretrained_dict['model']
        model_dict = self.mamba_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for key, value in pretrained_dict.items():
            if 'layers.' in key:
                decoder_layer = 3 - int(key[7:8])
                decoder_key = 'layers_up.' + str(decoder_layer) + key[8:]
                full_dict[decoder_key] = value
        for key in list(full_dict):
            if key in model_dict and full_dict[key].shape != model_dict[key].shape:
                print('Skipping incompatible initialization tensor: {}'.format(key))
                del full_dict[key]
        self.mamba_unet.load_state_dict(full_dict, strict=False)
