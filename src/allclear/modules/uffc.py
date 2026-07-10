"""Unbiased Fast Fourier Convolution (UFFC) for Image Inpainting.

Exact replica of the original implementation from:
  ICCV 2023 "Rethinking Fast Fourier Convolution in Image Inpainting"
  https://github.com/1911cty/Unbiased-Fast-Fourier-Convolution
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.allclear.modules.lama_ffc import ConcatTupleLayer


# ---------------------------------------------------------------------------
#  mygate — learnable per-channel gating (from original UFFC codebase)
# ---------------------------------------------------------------------------


class mygate(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x: Tensor, clip: float | None = None) -> Tensor:
        w = self.weight
        if w.dim() == 1:
            w = w.view(1, -1, 1, 1)
        elif w.dim() == 2:
            w = w.view(1, 1, w.shape[0], w.shape[1])
        if clip is not None:
            w = torch.clamp(w, -clip, clip)
        return x * w


# ---------------------------------------------------------------------------
#  UFFCFourierUnit
# ---------------------------------------------------------------------------


class UFFCFourierUnit(nn.Module):
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
        freq_resolution: int = 32,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.freq_resolution = freq_resolution

        self.locMap = nn.Parameter(torch.rand(freq_resolution, freq_resolution // 2 + 1))
        self.lambda_base = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        in_ch = in_channels * 2 + 1
        out_ch = out_channels * 2
        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=groups, bias=False)
        self.conv_dilated = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=2, dilation=2, groups=groups, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
        self.distill: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        # FFT and complex-valued spectral filtering are numerically fragile under
        # autocast. Keep this unit in fp32 while the surrounding convolutional
        # trunk can still benefit from AMP.
        autocast_off = torch.amp.autocast(device_type=x.device.type, enabled=False) if x.is_cuda else nullcontext()
        with autocast_off:
            x = x.float()
            batch = x.shape[0]

            orig_size = None
            if self.spatial_scale_factor is not None:
                orig_size = x.shape[-2:]
                x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

            fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            locMap = self.locMap.float()[None, None, :, :]
            if locMap.shape[-2:] != ffted.shape[-2:]:
                locMap = F.interpolate(locMap, size=ffted.shape[-2:], mode="bilinear", align_corners=False)
            locMap = locMap.expand(batch, 1, -1, -1)
            ffted_orig = ffted.clone()

            cat1 = torch.cat([ffted[:, :self.in_channels, :, :], ffted[:, self.in_channels:, :, :], locMap], dim=1)
            ffted = self.conv_1x1(cat1)
            ffted = torch.fft.fftshift(ffted, dim=-2)
            ffted = self.relu(ffted)

            locMap_shifted = torch.fft.fftshift(locMap, dim=-2)
            cat2 = torch.cat([ffted[:, :self.in_channels, :, :], ffted[:, self.in_channels:, :, :], locMap_shifted], dim=1)
            ffted = self.conv_dilated(cat2)
            ffted = torch.fft.ifftshift(ffted, dim=-2)

            lam = torch.sigmoid(self.lambda_base.float())
            ffted = ffted_orig * lam + ffted * (1.0 - lam)

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])

            ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
            output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

            if orig_size is not None:
                output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

            epsilon = 0.5
            output = output - output.mean() + x.mean()
            output = torch.clamp(output, float(x.min() - epsilon), float(x.max() + epsilon))

            self.distill = output
            return output


# ---------------------------------------------------------------------------
#  UFFCSpectralTransform
# ---------------------------------------------------------------------------


class UFFCSpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        enable_lfu: bool = True,
        freq_resolution: int = 32,
        **fu_kwargs,
    ) -> None:
        super().__init__()
        self.enable_lfu = enable_lfu

        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        mid = out_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.fu = UFFCFourierUnit(mid, mid, groups, freq_resolution=freq_resolution, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = UFFCFourierUnit(mid, mid, groups, freq_resolution=freq_resolution // 2, **fu_kwargs)
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = x[:, :c // 4]
            xs = torch.cat(torch.split(xs, split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        return self.conv2(x + output + xs)


# ---------------------------------------------------------------------------
#  UFFCBlock
# ---------------------------------------------------------------------------


class UFFCBlock(nn.Module):
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
        freq_resolution: int = 32,
        **spectral_kwargs,
    ) -> None:
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        m = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = m(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        m = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = m(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        m = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = m(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        m = nn.Identity if in_cg == 0 or out_cg == 0 else UFFCSpectralTransform
        self.convg2g = m(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu=enable_lfu, freq_resolution=freq_resolution, **spectral_kwargs)

        self.gated = gated
        m = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = m(in_channels, 2, 1)

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl, out_xg = 0, 0
        if self.gated:
            parts = [x_l]
            if torch.is_tensor(x_g):
                parts.append(x_g)
            gates = torch.sigmoid(self.gate(torch.cat(parts, dim=1)))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg


# ---------------------------------------------------------------------------
#  UFFC_BN_ACT
# ---------------------------------------------------------------------------


class UFFC_BN_ACT(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        ratio_gin: float, ratio_gout: float, stride: int = 1, padding: int = 0,
        dilation: int = 1, groups: int = 1, bias: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.Identity,
        padding_type: str = "reflect",
        enable_lfu: bool = True, freq_resolution: int = 32, **kwargs,
    ) -> None:
        super().__init__()
        self.uffc = UFFCBlock(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
            stride, padding, dilation, groups, bias, enable_lfu=enable_lfu,
            padding_type=padding_type, freq_resolution=freq_resolution, **kwargs,
        )
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        gc = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - gc)
        self.bn_g = gnorm(gc)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x_l, x_g = self.uffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))


# ---------------------------------------------------------------------------
#  UFFCResnetBlock
# ---------------------------------------------------------------------------


class UFFCResnetBlock(nn.Module):
    def __init__(
        self, dim: int, padding_type: str = "reflect", norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation_layer: type[nn.Module] = nn.ReLU, dilation: int = 1,
        freq_resolution: int = 32, **conv_kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = UFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                 norm_layer=norm_layer, activation_layer=activation_layer,
                                 padding_type=padding_type, freq_resolution=freq_resolution, **conv_kwargs)
        self.conv2 = UFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                 norm_layer=norm_layer, activation_layer=activation_layer,
                                 padding_type=padding_type, freq_resolution=freq_resolution, **conv_kwargs)

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv2(self.conv1((x_l, x_g)))
        return id_l + x_l, id_g + x_g


# ---------------------------------------------------------------------------
#  UFFCResNetGenerator
# ---------------------------------------------------------------------------


class UFFCResNetGenerator(nn.Module):
    def __init__(
        self, input_nc: int, output_nc: int, ngf: int = 64, n_downsampling: int = 3,
        n_blocks: int = 9, norm_layer: type[nn.Module] = nn.BatchNorm2d,
        padding_type: str = "reflect", activation_layer: type[nn.Module] = nn.ReLU,
        up_norm_layer: type[nn.Module] = nn.BatchNorm2d,
        up_activation: type[nn.Module] = nn.ReLU(True),
        init_conv_kwargs: dict | None = None,
        downsample_conv_kwargs: dict | None = None,
        resnet_conv_kwargs: dict | None = None,
        add_out_act: bool | str = True, max_features: int = 1024,
    ) -> None:
        assert n_blocks >= 0
        super().__init__()
        init_conv_kwargs = init_conv_kwargs or {}
        downsample_conv_kwargs = downsample_conv_kwargs or {}
        resnet_conv_kwargs = resnet_conv_kwargs or {}

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            UFFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0,
                        norm_layer=norm_layer, activation_layer=activation_layer, **init_conv_kwargs),
        ]
        for i in range(n_downsampling):
            mult = 2 ** i
            cur_kwargs = dict(downsample_conv_kwargs)
            if i == n_downsampling - 1:
                cur_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            model += [UFFC_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer, activation_layer=activation_layer, **cur_kwargs)]

        bottleneck_ch = min(max_features, ngf * (2 ** n_downsampling))
        for _ in range(n_blocks):
            model += [UFFCResnetBlock(bottleneck_ch, padding_type=padding_type,
                                     activation_layer=activation_layer, norm_layer=norm_layer, **resnet_conv_kwargs)]
        model += [ConcatTupleLayer()]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))), up_activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            act = "tanh" if add_out_act is True else str(add_out_act)
            model.append(nn.Sigmoid() if act == "sigmoid" else nn.Tanh())
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
