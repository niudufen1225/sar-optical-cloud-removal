"""SN-PatchGAN Discriminator — mask-conditioned, spectral-normalized.

Exact replica of the original DeepFill v2 TensorFlow implementation from:
  ICCV 2019 (Oral) "Free-Form Image Inpainting with Gated Convolution"
  Yu et al.  https://github.com/JiahuiYu/generative_inpainting
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm


class SNPatchGANDiscriminator(nn.Module):
    """Spectral-Normalized PatchGAN discriminator.

    When ``cond_mask=True`` the input is 4-channel (image + binary mask).
    Output: dense 3D feature map (256 × H' × W') — every element is an
    independent GAN score for hinge loss.
    """

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 6, cond_mask: bool = False) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.cond_mask = cond_mask
        in_channels = input_nc + (1 if cond_mask else 0)

        kw = 5
        padw = int(np.ceil((kw - 1.0) / 2))

        channels = [ndf]
        for i in range(1, n_layers):
            channels.append(channels[-1] * 2 if i < 3 else channels[-1])

        sequence: list[nn.Sequential] = []
        prev_ch = in_channels
        for ch in channels:
            sequence.append(nn.Sequential(
                spectral_norm(nn.Conv2d(prev_ch, ch, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            prev_ch = ch

        for n, seq in enumerate(sequence):
            setattr(self, 'model' + str(n), seq)

    def get_all_activations(self, x: Tensor) -> list[Tensor]:
        res = [x]
        for n in range(self.n_layers):
            res.append(getattr(self, 'model' + str(n))(res[-1]))
        return res[1:]

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, list[Tensor]]:
        if self.cond_mask and mask is not None:
            if mask.shape[-2:] != x.shape[-2:]:
                mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
            x = torch.cat([x, mask], dim=1)
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


def make_sn_patchgan_discriminator(input_nc: int = 3, ndf: int = 64, n_layers: int = 6, cond_mask: bool = True) -> SNPatchGANDiscriminator:
    return SNPatchGANDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers, cond_mask=cond_mask)
