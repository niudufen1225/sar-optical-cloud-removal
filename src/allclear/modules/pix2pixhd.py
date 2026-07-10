"""pix2pixHD NLayer discriminator used by the official LaMa configs.

This is a local, minimal port of LaMa's
``saicinpainting.training.modules.pix2pixhd.NLayerDiscriminator`` so the
ALLClear trainer does not depend on importing from an external checkout.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from torch import Tensor


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator from pix2pixHD/LaMa.

    Official Big-LaMa uses ``input_nc=3``, ``ndf=64`` and ``n_layers=4``.
    The forward method returns the final score map and intermediate features,
    matching LaMa's feature matching interface.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 4,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence: list[list[nn.Module]] = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]]

        nf = ndf
        for _ in range(1, self.n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence.append([
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ])

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence.append([
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True),
        ])
        sequence.append([nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)])

        for idx, layers in enumerate(sequence):
            setattr(self, f"model{idx}", nn.Sequential(*layers))

    def get_all_activations(self, x: Tensor) -> list[Tensor]:
        activations = [x]
        for idx in range(self.n_layers + 2):
            model = getattr(self, f"model{idx}")
            activations.append(model(activations[-1]))
        return activations[1:]

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        activations = self.get_all_activations(x.float())
        return activations[-1], activations[:-1]


def make_pix2pixhd_nlayer_discriminator(
    input_nc: int = 3,
    ndf: int = 64,
    n_layers: int = 4,
) -> NLayerDiscriminator:
    return NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers)
