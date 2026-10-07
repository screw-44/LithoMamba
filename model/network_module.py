"""Local convolutional discriminator and least-squares adversarial loss."""

import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        kernel_size = 4
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(input_nc, ndf, kernel_size, stride=2, padding=padding)]
        nf = ndf
        for index in range(n_layers):
            previous_nf, nf = nf, min(nf * 2, 512)
            stride = 2 if index != n_layers - 1 else 1
            layers.extend([
                nn.Conv2d(previous_nf, nf, kernel_size, stride=stride, padding=padding),
                norm_layer(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        layers.append(nn.Conv2d(nf, 1, kernel_size, stride=1, padding=padding))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, predictions, target_is_real):
        """Compute the released LSGAN objective over every sample and spatial score."""
        target = torch.full_like(predictions, 1.0 if target_is_real else 0.0, dtype=torch.float32)
        return self.loss(predictions, target)
