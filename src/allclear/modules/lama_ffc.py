"""LaMa WACV 2022 Fast Fourier Convolution blocks.

The classes in this file are a compact local port of LaMa's public
``saicinpainting.training.modules.ffc`` implementation.  They keep the same
local/global channel split, FourierUnit, SpectralTransform, FFC, FFC_BN_ACT,
and FFCResnetBlock mechanics, with one ALLClear-specific wrapper:
``LaMaFFCFeatureContext`` and ``LaMaFFCResidualContext``.  The first wrapper is
a direct feature-context stack that mirrors LaMa's FFC residual-block middle
path; the second keeps the earlier residual-adapter ablation path.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from kornia.geometry.transform import rotate
from torch import Tensor, nn


class LearnableSpatialTransformWrapper(nn.Module):
    """LaMa learnable spatial transform wrapper.

    This is a local port of LaMa's
    ``saicinpainting.training.modules.spatial_transform`` implementation: pad
    by reflection, rotate by a learnable angle, apply the wrapped module, rotate
    back, and crop to the original feature size.
    """

    def __init__(self, impl: nn.Module, pad_coef: float = 0.5, angle_init_range: float = 80, train_angle: bool = True) -> None:
        super().__init__()
        self.impl = impl
        angle = torch.rand(1) * float(angle_init_range)
        if train_angle:
            self.angle = nn.Parameter(angle, requires_grad=True)
        else:
            self.register_buffer("angle", angle, persistent=True)
        self.pad_coef = float(pad_coef)

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> tuple[Tensor, Tensor] | Tensor:
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        if isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) if torch.is_tensor(elem) else elem for elem in x)
            y_trans = self.impl(x_trans)
            if not isinstance(y_trans, tuple):
                raise TypeError("Wrapped LaMa spatial transform module returned a non-tuple for tuple input.")
            return tuple(
                self.inverse_transform(elem, orig_x) if torch.is_tensor(elem) and torch.is_tensor(orig_x) else elem
                for elem, orig_x in zip(y_trans, x)
            )
        raise ValueError(f"Unexpected input type {type(x)}")

    def transform(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        context = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else nullcontext()
        with context:
            x_padded = F.pad(x.float(), [pad_w, pad_w, pad_h, pad_h], mode="reflect")
            y = rotate(x_padded, angle=self.angle.to(device=x.device, dtype=torch.float32))
        return y.to(dtype=orig_dtype)

    def inverse_transform(self, y_padded_rotated: Tensor, orig_x: Tensor) -> Tensor:
        orig_dtype = orig_x.dtype
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        context = torch.amp.autocast("cuda", enabled=False) if y_padded_rotated.is_cuda else nullcontext()
        with context:
            y_padded = rotate(
                y_padded_rotated.float(),
                angle=-self.angle.to(device=y_padded_rotated.device, dtype=torch.float32),
            )
            y_height, y_width = y_padded.shape[2:]
            y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y.to(dtype=orig_dtype)


class ConcatTupleLayer(nn.Module):
    """Concatenate (local, global) tuple back into a single tensor.

    From LaMa's saicinpainting.training.modules.ffc
    """

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> Tensor:
        if isinstance(x, tuple):
            x_l, x_g = x
            if not torch.is_tensor(x_g):
                return x_l
            return torch.cat(x, dim=1)
        return x


class FourierUnit(nn.Module):
    """LaMa Fourier unit: real FFT -> 1x1 conv over real/imag -> inverse FFT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        spatial_scale_factor: float | None = None,
        spatial_scale_mode: str = "bilinear",
        spectral_pos_encoding: bool = False,
        use_se: bool = False,
        se_kwargs: dict | None = None,
        ffc3d: bool = False,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        if use_se:
            raise NotImplementedError("LaMa FourierUnit SE path is not used by ALLClear configs.")
        self.groups = int(groups)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = str(spatial_scale_mode)
        self.spectral_pos_encoding = bool(spectral_pos_encoding)
        self.ffc3d = bool(ffc3d)
        self.fft_norm = str(fft_norm)
        extra_channels = 2 if self.spectral_pos_encoding else 0
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2 + extra_channels,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _autocast_off(x: Tensor):
        if x.is_cuda:
            return torch.amp.autocast("cuda", enabled=False)
        return nullcontext()

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        orig_dtype = x.dtype
        if self.spatial_scale_factor is not None:
            x = F.interpolate(
                x,
                scale_factor=self.spatial_scale_factor,
                mode=self.spatial_scale_mode,
                align_corners=False if self.spatial_scale_mode in {"bilinear", "bicubic"} else None,
            )
        orig_size = x.shape[-3:] if self.ffc3d else x.shape[-2:]

        # torch.fft is more reliable in fp32; the result is cast back so AMP
        # still saves memory in surrounding convolutions.
        with self._autocast_off(x):
            x_f = x.float()
            fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
            ffted = torch.fft.rfftn(x_f, dim=fft_dim, norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view(batch, -1, *ffted.size()[3:])

            if self.spectral_pos_encoding:
                height, width = ffted.shape[-2:]
                coords_vert = torch.linspace(0, 1, height, device=ffted.device, dtype=ffted.dtype)[None, None, :, None]
                coords_vert = coords_vert.expand(batch, 1, height, width)
                coords_hor = torch.linspace(0, 1, width, device=ffted.device, dtype=ffted.dtype)[None, None, None, :]
                coords_hor = coords_hor.expand(batch, 1, height, width)
                ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

            ffted = self.conv_layer(ffted)
            ffted = self.relu(self.bn(ffted))
            ffted = ffted.view(batch, -1, 2, *ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])
            output = torch.fft.irfftn(ffted, s=orig_size, dim=fft_dim, norm=self.fft_norm)

        return output.to(dtype=orig_dtype)


class SpectralTransform(nn.Module):
    """LaMa spectral transform used inside the global branch of FFC."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        enable_lfu: bool = True,
        **fu_kwargs,
    ) -> None:
        super().__init__()
        self.enable_lfu = bool(enable_lfu)
        if stride == 2:
            self.downsample: nn.Module = nn.AvgPool2d(kernel_size=2, stride=2)
        elif stride == 1:
            self.downsample = nn.Identity()
        else:
            raise ValueError("SpectralTransform stride must be 1 or 2.")
        hidden_channels = max(1, out_channels // 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(hidden_channels, hidden_channels, groups=groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(hidden_channels, hidden_channels, groups=groups, **fu_kwargs)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        xs: Tensor | float = 0
        if self.enable_lfu:
            batch, channels, height, width = x.shape
            lfu_channels = channels // 4
            if height >= 2 and width >= 2 and height % 2 == 0 and width % 2 == 0 and lfu_channels > 0:
                split_no = 2
                split_s = height // split_no
                xs = torch.cat(torch.split(x[:, :lfu_channels], split_s, dim=-2), dim=1)
                xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1)
                xs = self.lfu(xs)
                xs = xs.repeat(1, 1, split_no, split_no)
        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):
    """Fast Fourier Convolution with local/global channel branches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        enable_lfu: bool = True,
        padding_type: str = "reflect",
        gated: bool = False,
        **spectral_kwargs,
    ) -> None:
        super().__init__()
        if padding_type != "reflect":
            raise ValueError("This local LaMa FFC port currently supports padding_type='reflect'.")
        self.stride = int(stride)
        self.ratio_gin = float(ratio_gin)
        self.ratio_gout = float(ratio_gout)
        self.global_in_num = int(in_channels * self.ratio_gin)
        self.global_out_num = int(out_channels * self.ratio_gout)
        self.local_in_num = int(in_channels) - self.global_in_num
        self.local_out_num = int(out_channels) - self.global_out_num
        self.gated = bool(gated)

        module = nn.Identity if self.local_in_num == 0 or self.local_out_num == 0 else nn.Conv2d
        self.convl2l = module(
            self.local_in_num,
            self.local_out_num,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if self.local_in_num == 0 or self.global_out_num == 0 else nn.Conv2d
        self.convl2g = module(
            self.local_in_num,
            self.global_out_num,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if self.global_in_num == 0 or self.local_out_num == 0 else nn.Conv2d
        self.convg2l = module(
            self.global_in_num,
            self.local_out_num,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if self.global_in_num == 0 or self.global_out_num == 0 else SpectralTransform
        self.convg2g = module(
            self.global_in_num,
            self.global_out_num,
            stride,
            1 if groups == 1 else groups // 2,
            enable_lfu,
            **spectral_kwargs,
        )
        if self.gated:
            self.gate = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> tuple[Tensor, Tensor]:
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl: Tensor | float = 0
        out_xg: Tensor | float = 0

        if self.gated:
            total_input = x_l if not torch.is_tensor(x_g) else torch.cat((x_l, x_g), dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.local_out_num != 0:
            out_xl = self.convl2l(x_l)
            if torch.is_tensor(x_g):
                out_xl = out_xl + self.convg2l(x_g) * g2l_gate
        if self.global_out_num != 0:
            if torch.is_tensor(x_g):
                out_xg = self.convg2g(x_g)
            out_xg = out_xg + self.convl2g(x_l) * l2g_gate
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    """LaMa FFC followed by independent BN/activation for local/global parts."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.ReLU,
        padding_type: str = "reflect",
        enable_lfu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            stride,
            padding,
            dilation,
            groups,
            bias,
            enable_lfu,
            padding_type=padding_type,
            **kwargs,
        )
        local_channels = int(out_channels * (1.0 - ratio_gout))
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = nn.Identity() if local_channels == 0 else norm_layer(local_channels)
        self.bn_g = nn.Identity() if global_channels == 0 else norm_layer(global_channels)
        self.act_l = nn.Identity() if local_channels == 0 else activation_layer(inplace=True)
        self.act_g = nn.Identity() if global_channels == 0 else activation_layer(inplace=True)

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> tuple[Tensor, Tensor]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l)) if torch.is_tensor(x_l) else x_l
        x_g = self.act_g(self.bn_g(x_g)) if torch.is_tensor(x_g) else x_g
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    """LaMa residual block with two FFC_BN_ACT layers."""

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.ReLU,
        dilation: int = 1,
        spatial_transform_kwargs: dict | None = None,
        inline: bool = False,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.inline = bool(inline)
        self.conv1 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        self.conv2 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> tuple[Tensor, Tensor] | Tensor:
        if self.inline:
            if not torch.is_tensor(x):
                raise TypeError("inline FFCResnetBlock expects a tensor input.")
            global_channels = int(self.conv1.ffc.global_in_num)
            if global_channels > 0:
                x = (x[:, :-global_channels], x[:, -global_channels:])
            else:
                x = (x, 0)
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x_l, x_g
        out_l, out_g = self.conv1((x_l, x_g))
        out_l, out_g = self.conv2((out_l, out_g))
        out_l = id_l + out_l if torch.is_tensor(out_l) else id_l
        out_g = id_g + out_g if torch.is_tensor(id_g) and torch.is_tensor(out_g) else out_g
        return torch.cat((out_l, out_g), dim=1) if self.inline else (out_l, out_g)


class LaMaFFCResidualContext(nn.Module):
    """Residual LaMa-FFC context adapter for an existing feature map.

    LaMa's generator applies many FFC residual blocks after convolutional
    downsampling.  DADIGAN keeps a full-resolution feature stream, so this
    adapter optionally downsamples the fused PDAFM feature, runs a small stack
    of LaMa FFC blocks, upsamples the correction, and adds it with a small
    residual scale.  This keeps DaDiGAN's structure path intact while adding
    the image-wide Fourier context that LaMa uses for large holes.
    """

    def __init__(
        self,
        channels: int,
        blocks: int = 6,
        ratio_g: float = 0.75,
        enable_lfu: bool = False,
        downsample: int = 4,
        residual_scale: float = 0.1,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.downsample = max(1, int(downsample))
        self.entry = FFC_BN_ACT(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            ratio_gin=0.0,
            ratio_gout=float(ratio_g),
            enable_lfu=bool(enable_lfu),
            fft_norm=fft_norm,
        )
        self.blocks = nn.ModuleList(
            [
                FFCResnetBlock(
                    self.channels,
                    ratio_gin=float(ratio_g),
                    ratio_gout=float(ratio_g),
                    enable_lfu=bool(enable_lfu),
                    fft_norm=fft_norm,
                )
                for _ in range(max(1, int(blocks)))
            ]
        )
        self.concat = ConcatTupleLayer()
        self.proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: Tensor) -> Tensor:
        ref_size = x.shape[-2:]
        y = x
        if self.downsample > 1:
            y = F.interpolate(
                y,
                size=(max(1, ref_size[0] // self.downsample), max(1, ref_size[1] // self.downsample)),
                mode="bilinear",
                align_corners=False,
            )
        feat: tuple[Tensor, Tensor] | Tensor = self.entry(y)
        for block in self.blocks:
            feat = block(feat)
        y = self.proj(self.concat(feat))
        if y.shape[-2:] != ref_size:
            y = F.interpolate(y, size=ref_size, mode="bilinear", align_corners=False)
        scale = self.residual_scale.to(device=x.device, dtype=x.dtype)
        return x + scale * y.to(dtype=x.dtype)


class LaMaFFCFeatureContext(nn.Module):
    """Direct LaMa-style FFC feature context without an outer residual scale."""

    def __init__(
        self,
        channels: int,
        blocks: int = 9,
        ratio_g: float = 0.75,
        enable_lfu: bool = False,
        fft_norm: str = "ortho",
        spatial_transform_layers: tuple[int, ...] | None = None,
        spatial_transform_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.entry = FFC_BN_ACT(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            ratio_gin=0.0,
            ratio_gout=float(ratio_g),
            enable_lfu=bool(enable_lfu),
            fft_norm=fft_norm,
        )
        selected_layers = set(int(i) for i in spatial_transform_layers) if spatial_transform_layers is not None else set()
        block_list: list[nn.Module] = []
        for i in range(max(1, int(blocks))):
            block: nn.Module = FFCResnetBlock(
                self.channels,
                ratio_gin=float(ratio_g),
                ratio_gout=float(ratio_g),
                enable_lfu=bool(enable_lfu),
                fft_norm=fft_norm,
            )
            if i in selected_layers:
                block = LearnableSpatialTransformWrapper(block, **(spatial_transform_kwargs or {}))
            block_list.append(block)
        self.blocks = nn.ModuleList(block_list)
        self.concat = ConcatTupleLayer()

    def forward(self, x: Tensor) -> Tensor:
        feat: tuple[Tensor, Tensor] | Tensor = self.entry(x)
        for block in self.blocks:
            feat = block(feat)
        return self.concat(feat).to(dtype=x.dtype)
