"""Common layers shared by TG-ECNet, SoftShadow, and DADIGAN modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """NCHW layer norm used by restoration blocks."""

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class RestorationBlock(nn.Module):
    """TG-ECNet resblock-style local restoration block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        expand: int = 2,
        scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        del expand, scale_init
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


def make_blocks(channels: int, count: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(*[RestorationBlock(channels, kernel_size=kernel_size) for _ in range(max(1, int(count)))])


def resize_like(x: Tensor, ref: Tensor) -> Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def downsample_mask(mask: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(mask.float(), size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def sobel_xy(x: Tensor) -> tuple[Tensor, Tensor]:
    channels = x.shape[1]
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    kx = kx.repeat(channels, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1)
    return (
        F.conv2d(x, kx, padding=1, groups=channels),
        F.conv2d(x, ky, padding=1, groups=channels),
    )


def gradient_magnitude(x: Tensor) -> Tensor:
    gx, gy = sobel_xy(x)
    return torch.sqrt(gx.pow(2) + gy.pow(2) + 1.0e-6)


def y_luma_from_s2(s2: Tensor, rgb_indices: tuple[int, int, int] = (3, 2, 1)) -> Tensor:
    """Return a luma channel from S2 RGB bands in normalized reflectance space."""

    if s2.shape[1] <= max(rgb_indices):
        raise ValueError(f"s2 has {s2.shape[1]} channels, cannot index {rgb_indices}")
    rgb = s2[:, list(rgb_indices)].float()
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def low_pass(x: Tensor, kernel_size: int = 5) -> Tensor:
    if kernel_size <= 1:
        return x
    kernel = x.new_ones((1, 1, kernel_size, kernel_size)) / float(kernel_size * kernel_size)
    return F.conv2d(x, kernel, padding=kernel_size // 2)


def bbox_from_mask(mask: Tensor, pad: int = 4) -> Tensor:
    """Create [B, 4] xyxy boxes from binary masks for optional SAM-style prompts."""

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [B,1,H,W], got {tuple(mask.shape)}")
    boxes = []
    _, _, h, w = mask.shape
    for m in mask.detach():
        ys, xs = torch.where(m[0] > 0.5)
        if ys.numel() == 0:
            boxes.append(m.new_tensor([0.0, 0.0, float(w - 1), float(h - 1)]))
            continue
        x1 = max(0, int(xs.min().item()) - pad)
        y1 = max(0, int(ys.min().item()) - pad)
        x2 = min(w - 1, int(xs.max().item()) + pad)
        y2 = min(h - 1, int(ys.max().item()) + pad)
        boxes.append(m.new_tensor([float(x1), float(y1), float(x2), float(y2)]))
    return torch.stack(boxes, dim=0)


@dataclass(frozen=True)
class RegionMasks:
    clear: Tensor
    shadow: Tensor
    cloud: Tensor


def masks_from_cld_shdw(cld_shdw: Tensor, shadow_index: int = 3, cloud_index: int = 1) -> RegionMasks:
    """Convert ALLClear cld_shdw to mutually exclusive clear/shadow/cloud masks.

    Supports either a categorical [B,1,H,W] map or a one-hot/probability
    [B,K,H,W] tensor.  Cloud takes precedence over shadow because the shadow
    branch is only for non-cloud shadow regions in this framework.
    """

    if cld_shdw.ndim != 4:
        raise ValueError(f"cld_shdw must be [B,C,H,W], got {tuple(cld_shdw.shape)}")
    x = cld_shdw.float()
    if x.shape[1] == 1:
        labels = x.round().long()
        shadow = (labels == int(shadow_index)).float()
        cloud = (labels == int(cloud_index)).float()
    else:
        if x.shape[1] <= max(shadow_index, cloud_index):
            raise ValueError(
                f"cld_shdw has {x.shape[1]} channels, cannot read shadow={shadow_index}, cloud={cloud_index}"
            )
        shadow = x[:, shadow_index : shadow_index + 1].clamp(0.0, 1.0)
        cloud = x[:, cloud_index : cloud_index + 1].clamp(0.0, 1.0)
    shadow = (shadow * (1.0 - cloud)).clamp(0.0, 1.0)
    clear = (1.0 - shadow - cloud).clamp(0.0, 1.0)
    return RegionMasks(clear=clear, shadow=shadow, cloud=cloud)


class PatchDiscriminator(nn.Module):
    """DADIGAN/Pix2Pix-style conditional discriminator.

    DADIGAN specifies five repeated blocks of 4x4 stride-2 convolution,
    normalization, and LeakyReLU, followed by a 3x3 stride-1 convolution.  The
    paper does not identify the normalization type or whether the final result
    is a spatial PatchGAN map or a pooled scalar, so both are configurable.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_layers: int = 4,
        norm_type: str = "group",
        output_mode: str = "patch",
    ) -> None:
        super().__init__()
        norm_type = str(norm_type).lower()
        output_mode = str(output_mode).lower()
        if norm_type not in {"group", "batch", "instance", "none"}:
            raise ValueError("norm_type must be one of: group, batch, instance, none")
        if output_mode not in {"patch", "scalar"}:
            raise ValueError("output_mode must be one of: patch, scalar")
        self.output_mode = output_mode

        def make_norm(channels: int) -> nn.Module:
            if norm_type == "batch":
                return nn.BatchNorm2d(channels)
            if norm_type == "instance":
                return nn.InstanceNorm2d(channels, affine=True)
            if norm_type == "group":
                return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
            return nn.Identity()

        layers: list[nn.Module] = []
        ch = base_channels
        layers.extend([nn.Conv2d(in_channels, ch, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)])
        for i in range(1, num_layers):
            prev = ch
            ch = min(base_channels * (2**i), 512)
            layers.extend(
                [
                    nn.Conv2d(prev, ch, 4, stride=2, padding=1, bias=norm_type == "none"),
                    make_norm(ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.append(nn.Conv2d(ch, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        scores = self.net(x.float())
        if self.output_mode == "scalar":
            return F.adaptive_avg_pool2d(scores, output_size=1).flatten(1)
        return scores


def hinge_d_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: Tensor) -> Tensor:
    return -fake_logits.mean()


def masked_l1(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    weight: float = 1.0,
    reduction: str = "mask_mean",
) -> Tensor:
    mask = mask.float()
    while mask.shape[1] != pred.shape[1]:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
        break
    masked_error = (mask * (pred.float() - target.float()).abs()).mean()
    mask_fraction = mask.mean().clamp_min(1.0e-6)
    mode = str(reduction).lower()
    if mode in {"mask_mean", "mask_normalized", "region_mean"}:
        loss = masked_error / mask_fraction
    elif mode in {"image_mean", "full_mean"}:
        loss = masked_error
    elif mode in {"sqrt_area", "sqrt_mask"}:
        loss = masked_error / mask_fraction.sqrt()
    elif mode == "hybrid":
        loss = 0.5 * (masked_error / mask_fraction + masked_error)
    else:
        raise ValueError("masked_l1 reduction must be one of: mask_mean, image_mean, sqrt_area, hybrid")
    return float(weight) * loss


def gradient_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    diff = (gradient_magnitude(pred.float()) - gradient_magnitude(target.float())).abs()
    if mask is None:
        return diff.mean()
    if mask.shape[1] == 1 and diff.shape[1] != 1:
        mask = mask.expand(-1, diff.shape[1], -1, -1)
    return (diff * mask.float()).mean() / mask.float().mean().clamp_min(1.0e-6)


# ---------------------------------------------------------------------------
#  NAFBlock  —  from NAFNet (Chen et al., ECCV 2022)
#  https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py
# ---------------------------------------------------------------------------


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Nonlinear Activation Free Block for image restoration.

    Exact reproduction of the official NAFNet implementation:
    https://github.com/megvii-research/NAFNet

    Args:
        c:          number of input / output channels.
        DW_Expand:  channel expansion factor for the first 1×1 conv (default 2).
        FFN_Expand: channel expansion factor for the FFN 1×1 conv (default 2).
        drop_out_rate: dropout probability (0. = no dropout).
    """

    def __init__(
        self,
        c: int,
        DW_Expand: float = 2,
        FFN_Expand: float = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channel = int(c * DW_Expand)
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True,
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention (SCA) —  not a separate class in the
        # official code;  built inline inside NAFBlock.__init__.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True,
            ),
        )

        self.sg = SimpleGate()

        ffn_channel = int(FFN_Expand * c)
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: Tensor) -> Tensor:
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma
