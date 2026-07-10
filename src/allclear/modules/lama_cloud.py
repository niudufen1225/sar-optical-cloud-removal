"""UFFC/LaMa-based cloud inpainting branch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.allclear.modules.uffc import UFFCResNetGenerator


class _BaseCloudBranch(nn.Module):
    """Shared forward logic for LaMa-style cloud inpainting."""

    s2_channels: int
    sar_channels: int
    use_sar: bool
    mask_cloud_input: bool
    generator: nn.Module

    def forward(self, s2: Tensor, sar: Tensor | None, cloud_mask: Tensor) -> dict[str, Tensor]:
        m = cloud_mask.float().clamp(0, 1)
        s2_in = s2.float()
        if self.mask_cloud_input:
            s2_in = s2_in * (1.0 - m)
        parts = [s2_in]
        if self.use_sar:
            if sar is None:
                raise ValueError("LaMa cloud branch was configured with use_sar=True, but no SAR tensor was provided.")
            parts.append(sar.float())
        parts.append(m)
        inp = torch.cat(parts, dim=1)
        fill = self.generator(inp)
        image = (1.0 - m) * s2.float() + m * fill
        return {"I_cloud": image, "I_cloud_raw": fill, "F_cloud": fill}


class UFFCCloudBranch(_BaseCloudBranch):
    """UFFC-based LaMa cloud inpainting branch.

    By default this follows LaMa's data contract: masked optical image plus the
    binary mask.  SAR can be enabled for ablations, but it is not part of the
    original LaMa inpainting formulation.
    """

    def __init__(
        self, s2_channels: int = 13, sar_channels: int = 2,
        ngf: int = 64, n_downsampling: int = 3, n_blocks: int = 18,
        ratio_gin: float = 0.75, ratio_gout: float = 0.75,
        enable_lfu: bool = True,
        pretrained_checkpoint: str | None = None,
        use_sar: bool = False,
        mask_cloud_input: bool = True,
        rgb_indices: tuple[int, int, int] = (3, 2, 1),
        output_prior: float = 0.10,
    ) -> None:
        super().__init__()
        self.s2_channels = int(s2_channels)
        self.sar_channels = int(sar_channels)
        self.use_sar = bool(use_sar)
        self.mask_cloud_input = bool(mask_cloud_input)
        self.rgb_indices = tuple(int(i) for i in rgb_indices)
        self.output_prior = float(output_prior)
        input_nc = self.s2_channels + (self.sar_channels if self.use_sar else 0) + 1
        self.generator = UFFCResNetGenerator(
            input_nc=input_nc, output_nc=self.s2_channels,
            ngf=ngf, n_downsampling=n_downsampling, n_blocks=n_blocks,
            init_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": enable_lfu},
            downsample_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": enable_lfu},
            resnet_conv_kwargs={"ratio_gin": ratio_gin, "ratio_gout": ratio_gout, "enable_lfu": enable_lfu},
            add_out_act="sigmoid",
        )
        if pretrained_checkpoint:
            self._load_pretrained_biglama(pretrained_checkpoint)

    def _adapt_biglama_tensor(self, own_key: str, own_tensor: Tensor, pretrained: Tensor) -> Tensor | None:
        if own_tensor.shape == pretrained.shape:
            return pretrained
        if own_tensor.ndim == 4 and pretrained.ndim == 4:
            # big-lama input stem: RGB + mask -> ALLClear: 13-band S2 + mask.
            if (
                pretrained.shape[1] == 4
                and own_tensor.shape[1] == self.s2_channels + (self.sar_channels if self.use_sar else 0) + 1
                and own_tensor.shape[0] == pretrained.shape[0]
                and own_tensor.shape[2:] == pretrained.shape[2:]
            ):
                adapted = torch.zeros_like(own_tensor)
                for src_ch, dst_ch in enumerate(self.rgb_indices):
                    adapted[:, dst_ch] = pretrained[:, src_ch]
                adapted[:, self.s2_channels + (self.sar_channels if self.use_sar else 0)] = pretrained[:, 3]
                return adapted
            # UFFC adds one frequency-location channel; copy the pretrained Fourier
            # weights and initialize the added channel to zero.
            if (
                own_tensor.shape[0] == pretrained.shape[0]
                and own_tensor.shape[1] == pretrained.shape[1] + 1
                and own_tensor.shape[2:] == pretrained.shape[2:]
            ):
                adapted = torch.zeros_like(own_tensor)
                adapted[:, : pretrained.shape[1]] = pretrained
                return adapted
            # The RGB output head is domain-specific. Reinitialize the
            # multispectral head to a low-reflectance prior instead of copying
            # natural-image RGB logits into Sentinel-2 bands.
            if (
                own_tensor.shape[0] == self.s2_channels
                and pretrained.shape[0] == 3
                and own_tensor.shape[1:] == pretrained.shape[1:]
            ):
                return torch.zeros_like(own_tensor)
        if own_tensor.ndim == 1 and pretrained.ndim == 1:
            if own_tensor.shape[0] == self.s2_channels and pretrained.shape[0] == 3:
                prior = min(max(self.output_prior, 1.0e-4), 1.0 - 1.0e-4)
                return torch.full_like(own_tensor, float(torch.logit(torch.tensor(prior))))
        return None

    def _load_pretrained_biglama(self, ckpt_path: str) -> None:
        """Load compatible weights from a big-lama FFCResNetGenerator checkpoint.

        Maps big-lama's ``generator.model.X.*`` keys to our ``model.X.*`` keys
        and ``.ffc.`` → ``.uffc.`` for FFC_BN_ACT / FFCResnetBlock internals.
        RGB+mask input/output layers are adapted to Sentinel-2 band order.
        """
        import torch

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        lama_sd: dict[str, torch.Tensor] = raw.get("state_dict", raw)
        # Strip Lightning prefix: generator.model.X → model.X
        lama_sd = {k.replace("generator.", ""): v for k, v in lama_sd.items()}

        own = self.generator.state_dict()
        loaded = 0
        adapted_count = 0
        skipped = 0
        mismatched = 0

        for own_key, own_tensor in own.items():
            # Map our key → big-lama key
            lama_key = own_key
            # uffc → ffc
            lama_key = lama_key.replace(".uffc.", ".ffc.")
            # UFFCFourierUnit → FourierUnit: conv_1x1 → conv_layer
            lama_key = lama_key.replace("conv_1x1.weight", "conv_layer.weight")
            lama_key = lama_key.replace("conv_1x1.bias", "conv_layer.bias")

            if lama_key in lama_sd:
                pretrained = lama_sd[lama_key]
                adapted = self._adapt_biglama_tensor(own_key, own_tensor, pretrained)
                if adapted is not None:
                    own[own_key].copy_(adapted)
                    loaded += 1
                    adapted_count += int(adapted.shape != pretrained.shape)
                else:
                    mismatched += 1
            else:
                skipped += 1

        self.generator.load_state_dict(own)
        print(f"[UFFCCloudBranch] Loaded {loaded} params from big-lama, "
              f"adapted {adapted_count}, skipped {skipped} (UFFC-specific), mismatched {mismatched}")
