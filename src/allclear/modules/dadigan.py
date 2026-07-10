"""DADIGAN-constrained SAR-optical cloudy branch.

This module implements the generator structure described in the DADIGAN paper:

- DDIN: iterative shared/private disentanglement for SAR and cloudy optical.
- PDAFM: progressive dual-attention fusion.
- CAB: the two DADIGAN CAB positions are implemented with PVT-source
  spatial-reduction cross-attention (SRA-CAB), optionally using the stable
  normalized complement attention that preserves DADIGAN's discrepancy-aware
  ``1 - Attention`` semantics without full-resolution quadratic memory.
- MSAB: DADIGAN's MSAB position, supporting paper attention, the previous
  Efficient Attention implementation, and official Restormer MDTA replacement.
- GAN discriminator support for cloud generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.allclear.modules.common import LayerNorm2d, PatchDiscriminator
from src.allclear.modules.lama_ffc import LaMaFFCFeatureContext, LaMaFFCResidualContext
from src.allclear.modules.pvt_sra_cab import SRACAB
from src.allclear.modules.restormer_official import TransformerBlock as OfficialRestormerTransformerBlock


def _inv_softplus(value: float) -> float:
    x = torch.tensor(float(value))
    return float(torch.log(torch.expm1(x)).item())


def _spatial_transform_kwargs(pad_coef: float, angle_init_range: float, train_angle: bool) -> dict[str, float | bool]:
    return {
        "pad_coef": float(pad_coef),
        "angle_init_range": float(angle_init_range),
        "train_angle": bool(train_angle),
    }


class DADIGANResidualBlock(nn.Module):
    """DADIGAN RB unit: three 3x3 convolutions with ReLU on the first two."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class ProxNet(nn.Module):
    """ResNet proximal operator used by DADIGAN's unfolded PGDA step."""

    def __init__(self, channels: int, blocks: int = 2) -> None:
        super().__init__()
        blocks = max(1, int(blocks))
        self.body = nn.Sequential(
            *[DADIGANResidualBlock(channels) for _ in range(blocks)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class DDINStep(nn.Module):
    """One DADIGAN PGDA-unfolded shared/private update step."""

    def __init__(self, channels: int, prox_blocks: int = 2) -> None:
        super().__init__()
        self.x_s = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.x_p = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.y_s = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.y_v = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.l_s = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False)
        self.x_p_t = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.y_v_t = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.l_s_t = nn.ConvTranspose2d(channels * 2, channels, kernel_size=3, padding=1, bias=False)
        self.prox_p = ProxNet(channels, blocks=prox_blocks)
        self.prox_v = ProxNet(channels, blocks=prox_blocks)
        self.prox_s = ProxNet(channels, blocks=prox_blocks)
        eta_init = _inv_softplus(0.1)
        self.eta_p = nn.Parameter(torch.tensor(eta_init))
        self.eta_v = nn.Parameter(torch.tensor(eta_init))
        self.eta_s = nn.Parameter(torch.tensor(eta_init))

    def forward(self, opt: Tensor, sar: Tensor, shared: Tensor, opt_private: Tensor, sar_private: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eta_p = F.softplus(self.eta_p)
        eta_v = F.softplus(self.eta_v)
        eta_s = F.softplus(self.eta_s)

        grad_p = self.x_p_t(self.x_s(shared) + self.x_p(opt_private) - opt.float())
        opt_private = self.prox_p(opt_private - eta_p * grad_p)

        grad_v = self.y_v_t(self.y_s(shared) + self.y_v(sar_private) - sar.float())
        sar_private = self.prox_v(sar_private - eta_v * grad_v)

        opt_residual = opt.float() - self.x_p(opt_private)
        sar_residual = sar.float() - self.y_v(sar_private)
        joint_observation = torch.cat([opt_residual, sar_residual], dim=1)
        grad_s = self.l_s_t(self.l_s(shared) - joint_observation)
        shared = self.prox_s(shared - eta_s * grad_s)
        return shared, opt_private, sar_private


class DDIN(nn.Module):
    """Deep disentangled iterative network."""

    def __init__(self, channels: int, steps: int = 3, prox_blocks: int = 2) -> None:
        super().__init__()
        self.init_p = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.init_v = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.init_s = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.steps = nn.ModuleList([DDINStep(channels, prox_blocks=prox_blocks) for _ in range(steps)])

    def forward(self, opt: Tensor, sar: Tensor) -> dict[str, Tensor]:
        opt = opt.float()
        sar = sar.float()
        opt_private = self.init_p(opt)
        sar_private = self.init_v(sar)
        shared = self.init_s(torch.cat([opt, sar], dim=1))
        for step in self.steps:
            shared, opt_private, sar_private = step(opt, sar, shared, opt_private, sar_private)
        return {"shared": shared, "opt_private": opt_private, "sar_private": sar_private}


class MSAB(nn.Module):
    """DADIGAN MSAB position.

    ``mode="paper"`` follows DADIGAN Eq. (27)-(31) using scaled dot-product
    multi-head self-attention.  ``mode="efficient"`` keeps the previous linear
    Efficient Attention implementation for ablation.  ``mode="restormer_mdta"``
    uses the official Restormer TransformerBlock (MDTA + GDFN) as the
    high-resolution restoration substitute at the same PDAFM output position.
    """

    def __init__(self, channels: int, heads: int = 8, mode: str = "efficient") -> None:
        super().__init__()
        self.heads = int(heads)
        if channels % self.heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={self.heads}")
        self.mode = str(mode).lower()
        if self.mode not in {"efficient", "paper", "restormer_mdta"}:
            raise ValueError("MSAB mode must be one of: efficient, paper, restormer_mdta")
        self.channels = int(channels)
        self.head_channels = self.channels // self.heads
        if self.mode == "restormer_mdta":
            self.restormer = OfficialRestormerTransformerBlock(
                dim=channels,
                num_heads=self.heads,
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type="WithBias",
            )
            return
        self.norm = LayerNorm2d(channels)
        self.keys = nn.Conv2d(channels, channels, kernel_size=1)
        self.queries = nn.Conv2d(channels, channels, kernel_size=1)
        self.values = nn.Conv2d(channels, channels, kernel_size=1)
        self.reprojection = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "restormer_mdta":
            return self.restormer(x)
        b, c, h, w = x.shape
        x_attn = self.norm(x)
        if self.mode == "paper":
            d = self.head_channels
            q = self.queries(x_attn).flatten(2).transpose(1, 2).view(b, h * w, self.heads, d).transpose(1, 2).contiguous()
            k = self.keys(x_attn).flatten(2).transpose(1, 2).view(b, h * w, self.heads, d).transpose(1, 2).contiguous()
            v = self.values(x_attn).flatten(2).transpose(1, 2).view(b, h * w, self.heads, d).transpose(1, 2).contiguous()
            y = F.scaled_dot_product_attention(q, k, v)
            y = y.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
            y = self.reprojection(y)
            return y * self.gate(x) + x

        keys = self.keys(x_attn).reshape(b, c, h * w)
        queries = self.queries(x_attn).reshape(b, c, h * w)
        values = self.values(x_attn).reshape(b, c, h * w)
        attended_values: list[Tensor] = []
        for i in range(self.heads):
            key = F.softmax(
                keys[:, i * self.head_channels : (i + 1) * self.head_channels, :],
                dim=2,
            )
            query = F.softmax(
                queries[:, i * self.head_channels : (i + 1) * self.head_channels, :],
                dim=1,
            )
            value = values[:, i * self.head_channels : (i + 1) * self.head_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(b, self.head_channels, h, w)
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1)
        y = self.reprojection(aggregated_values)
        return y * self.gate(x) + x


class PDAFMScale(nn.Module):
    """Progressive Dual Attention Fusion Module from DADIGAN Sec. 3.2.2.

    The paper's detailed equations define the two CABs by their Q/K/V sources:

    - Eq. (21)-(23): Qp comes from optical-private P, while K/V come from
      shared S, producing the intermediate feature FM.
    - Eq. (24)-(26): Qv comes from SAR-private V, while K/V come from FM,
      producing F.

    Therefore the original data flow is ``CAB(P, S) -> CAB(V, FM) -> MSAB(F)``.
    The shorter prose/diagram notation can look like ``CAB(CAB(P,S), V)``, but
    the Q/K/V equations make the second CAB order unambiguous.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        cab_sr_ratio: int = 8,
        cab_attention_mode: str = "standard",
        msab_mode: str = "efficient",
    ) -> None:
        super().__init__()
        self.cab_ps = SRACAB(
            channels,
            heads=heads,
            sr_ratio=cab_sr_ratio,
            attention_mode=cab_attention_mode,
        )   # CAB(P, S)
        self.cab_fv = SRACAB(
            channels,
            heads=heads,
            sr_ratio=cab_sr_ratio,
            attention_mode=cab_attention_mode,
        )   # CAB(V, FM)
        self.msab = MSAB(channels, heads=heads, mode=msab_mode)

    def forward(self, shared: Tensor, opt_private: Tensor, sar_private: Tensor) -> Tensor:
        # Eq.(21-23): CAB(P, S) — opt_private→Q, shared→K,V
        fm = self.cab_ps(opt_private, shared)
        # Eq.(24-26): CAB(V, FM) — sar_private→Q, fm→K,V
        fm = self.cab_fv(sar_private, fm)
        # MSAB on fused features
        return self.msab(fm)


class DDINOutputContext(nn.Module):
    """Optional LaMa-FFC context before DADIGAN's PDAFM fusion.

    The adapter is deliberately branch-preserving: shared, optical-private,
    and SAR-private DDIN outputs get separate residual FFC context modules.
    This keeps the DADIGAN PDAFM equations readable while moving image-wide
    context earlier than the post-PDAFM bottleneck adapter.
    """

    def __init__(
        self,
        channels: int,
        context: str = "none",
        blocks: int = 3,
        ratio_g: float = 0.75,
        enable_lfu: bool = False,
        downsample: int = 4,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.context = str(context).lower()
        if self.context in {"none", "identity", ""}:
            self.adapters = nn.ModuleDict()
            return
        if self.context not in {"lama_ffc", "ffc"}:
            raise ValueError("pre_pda_context must be one of: none, identity, lama_ffc")
        self.adapters = nn.ModuleDict(
            {
                name: LaMaFFCResidualContext(
                    channels,
                    blocks=max(1, int(blocks)),
                    ratio_g=float(ratio_g),
                    enable_lfu=bool(enable_lfu),
                    downsample=max(1, int(downsample)),
                    residual_scale=float(residual_scale),
                )
                for name in ("shared", "opt_private", "sar_private")
            }
        )

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.adapters:
            return features
        out = dict(features)
        for name, adapter in self.adapters.items():
            out[name] = adapter(out[name])
        return out


def _glfcr_df_conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=True,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


class GLFCRDFResBlock(nn.Module):
    """GLF-CR dynamic-filter residual block from the public ``submodules.py``."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x) + x


class GLFCRDynamicFilterGenerator(nn.Module):
    """GLF-CR DFG: concat OPT/SAR features and predict per-channel 2D kernels."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.body = nn.Sequential(
            _glfcr_df_conv(channels * 2, channels, kernel_size=3),
            GLFCRDFResBlock(channels, kernel_size=3),
            GLFCRDFResBlock(channels, kernel_size=3),
            _glfcr_df_conv(channels, channels * self.kernel_size * self.kernel_size, kernel_size=1),
        )

    def forward(self, opt_f: Tensor, sar_f: Tensor) -> Tensor:
        return self.body(torch.cat([opt_f, sar_f], dim=1))


def _glfcr_kernel2d_conv(feat_in: Tensor, kernel: Tensor, kernel_size: int) -> Tensor:
    """PyTorch port of GLF-CR's Python fallback for FAC ``KernelConv2D``."""

    channels = feat_in.size(1)
    n, kernels, h, w = kernel.size()
    expected = channels * kernel_size * kernel_size
    if kernels != expected:
        raise ValueError(f"GLF-CR dynamic kernel has {kernels} channels, expected {expected}")
    pad = (kernel_size - 1) // 2
    feat = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat = feat.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    feat = feat.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat = feat.reshape(n, h, w, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(n, h, w, channels, kernel_size, kernel_size)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(n, h, w, channels, -1)
    out = torch.sum(feat * kernel, dim=-1)
    return out.permute(0, 3, 1, 2).contiguous()


class GLFCRFusionStep(nn.Module):
    """GLF-CR fusion step: dynamic SAR filtering plus difference-gated updates."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 != 1:
            raise ValueError("GLF-CR dynamic kernel size must be odd")
        self.dynamic_filter = GLFCRDynamicFilterGenerator(channels, kernel_size=self.kernel_size)
        self.sar_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.opt_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, opt: Tensor, sar: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        opt_m = opt
        sar_m = sar
        kernel_sar = self.dynamic_filter(opt_m, sar_m)
        sar_m = _glfcr_kernel2d_conv(sar_m, kernel_sar, self.kernel_size)

        sar_fuse_gate = torch.sigmoid(self.sar_fuse_1x1conv(sar_m - opt_m))
        opt = opt + (sar_m - opt_m) * sar_fuse_gate

        opt_distribute_gate = torch.sigmoid(self.opt_distribute_1x1conv(opt - sar_m))
        sar = sar + (opt - sar_m) * opt_distribute_gate
        return opt, sar, {
            "M_glfcr_sar_fuse_gate": sar_fuse_gate.mean(dim=1, keepdim=True),
            "M_glfcr_opt_distribute_gate": opt_distribute_gate.mean(dim=1, keepdim=True),
        }


class GLFCRCoupledDDINStep(nn.Module):
    """DaDiGAN DDIN step with GLF-CR coupled P/V update before shared update."""

    def __init__(self, channels: int, prox_blocks: int = 2, kernel_size: int = 5) -> None:
        super().__init__()
        self.x_s = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.x_p = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.y_s = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.y_v = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.l_s = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False)
        self.x_p_t = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.y_v_t = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.l_s_t = nn.ConvTranspose2d(channels * 2, channels, kernel_size=3, padding=1, bias=False)
        self.prox_p = ProxNet(channels, blocks=prox_blocks)
        self.prox_v = ProxNet(channels, blocks=prox_blocks)
        self.prox_s = ProxNet(channels, blocks=prox_blocks)
        self.glfcr = GLFCRFusionStep(channels, kernel_size=kernel_size)
        eta_init = _inv_softplus(0.1)
        self.eta_p = nn.Parameter(torch.tensor(eta_init))
        self.eta_v = nn.Parameter(torch.tensor(eta_init))
        self.eta_s = nn.Parameter(torch.tensor(eta_init))

    def forward(
        self,
        opt: Tensor,
        sar: Tensor,
        shared: Tensor,
        opt_private: Tensor,
        sar_private: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        eta_p = F.softplus(self.eta_p)
        eta_v = F.softplus(self.eta_v)
        eta_s = F.softplus(self.eta_s)

        grad_p = self.x_p_t(self.x_s(shared) + self.x_p(opt_private) - opt.float())
        opt_private = self.prox_p(opt_private - eta_p * grad_p)

        grad_v = self.y_v_t(self.y_s(shared) + self.y_v(sar_private) - sar.float())
        sar_private = self.prox_v(sar_private - eta_v * grad_v)

        opt_private, sar_private, aux = self.glfcr(opt_private, sar_private)

        opt_residual = opt.float() - self.x_p(opt_private)
        sar_residual = sar.float() - self.y_v(sar_private)
        joint_observation = torch.cat([opt_residual, sar_residual], dim=1)
        grad_s = self.l_s_t(self.l_s(shared) - joint_observation)
        shared = self.prox_s(shared - eta_s * grad_s)
        return shared, opt_private, sar_private, aux


class GLFCRCoupledDDIN(nn.Module):
    """DDIN variant whose every unfolded step contains a GLF-CR P/V fuse."""

    def __init__(self, channels: int, steps: int = 3, prox_blocks: int = 2, kernel_size: int = 5) -> None:
        super().__init__()
        self.init_p = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.init_v = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.init_s = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.steps = nn.ModuleList(
            [
                GLFCRCoupledDDINStep(channels, prox_blocks=prox_blocks, kernel_size=kernel_size)
                for _ in range(max(1, int(steps)))
            ]
        )

    def forward(self, opt: Tensor, sar: Tensor) -> dict[str, Tensor]:
        opt = opt.float()
        sar = sar.float()
        opt_private = self.init_p(opt)
        sar_private = self.init_v(sar)
        shared = self.init_s(torch.cat([opt, sar], dim=1))
        aux: dict[str, Tensor] = {}
        for step in self.steps:
            shared, opt_private, sar_private, aux = step(opt, sar, shared, opt_private, sar_private)
        return {"shared": shared, "opt_private": opt_private, "sar_private": sar_private, **aux}


class SAGateFilterLayer(nn.Module):
    """SA-Gate channel filter layer from the public RGB-D implementation."""

    def __init__(self, in_planes: int, out_planes: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, out_planes // max(1, int(reduction)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_planes),
            nn.Sigmoid(),
        )
        self.out_planes = int(out_planes)

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return self.fc(y).view(b, self.out_planes, 1, 1)


class SAGateFSP(nn.Module):
    """SA-Gate Feature Separation Part."""

    def __init__(self, in_planes: int, out_planes: int, reduction: int = 16) -> None:
        super().__init__()
        self.filter = SAGateFilterLayer(2 * in_planes, out_planes, reduction)

    def forward(self, guide_path: Tensor, main_path: Tensor) -> Tensor:
        combined = torch.cat((guide_path, main_path), dim=1)
        channel_weight = self.filter(combined)
        return main_path + channel_weight * guide_path


class SAGateFusionStep(nn.Module):
    """Official SA-Gate data flow adapted from RGB/HHA to OPT/SAR features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fsp_opt = SAGateFSP(channels, channels, reduction)
        self.fsp_sar = SAGateFSP(channels, channels, reduction)
        self.gate_opt = nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True)
        self.gate_sar = nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True)
        self.relu_opt = nn.ReLU()
        self.relu_sar = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, opt: Tensor, sar: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        rec_opt = self.fsp_opt(sar, opt)
        rec_sar = self.fsp_sar(opt, sar)
        cat_fea = torch.cat([rec_opt, rec_sar], dim=1)
        attention_vector = torch.cat([self.gate_opt(cat_fea), self.gate_sar(cat_fea)], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_opt = attention_vector[:, 0:1]
        attention_sar = attention_vector[:, 1:2]
        merge_feature = opt * attention_opt + sar * attention_sar
        opt = self.relu_opt((opt + merge_feature) / 2.0)
        sar = self.relu_sar((sar + merge_feature) / 2.0)
        return opt, sar, {
            "M_sagate_opt_weight": attention_opt,
            "M_sagate_sar_weight": attention_sar,
        }


class DDINPairFusionContext(nn.Module):
    """Cross-modal context inserted after DDIN and before DADIGAN PDAFM."""

    def __init__(
        self,
        channels: int,
        context: str = "none",
        blocks: int = 0,
        kernel_size: int = 5,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.context = str(context).lower()
        if self.context in {"none", "identity", ""}:
            self.blocks = nn.ModuleList()
            return
        num_blocks = max(1, int(blocks))
        if self.context in {"glf_cr", "glfcr", "glf-cr"}:
            self.blocks = nn.ModuleList(
                [GLFCRFusionStep(channels, kernel_size=kernel_size) for _ in range(num_blocks)]
            )
        elif self.context in {"sagate", "sa_gate", "sa-gate"}:
            self.blocks = nn.ModuleList(
                [SAGateFusionStep(channels, reduction=reduction) for _ in range(num_blocks)]
            )
        else:
            raise ValueError("prefusion_context must be one of: none, glf_cr, sagate")

    def forward(self, opt_private: Tensor, sar_private: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        aux: dict[str, Tensor] = {}
        for block in self.blocks:
            opt_private, sar_private, aux = block(opt_private, sar_private)
        return opt_private, sar_private, aux


class LaMaFFCMultiScaleContext(nn.Module):
    """Sequential LaMa-FFC synthesis context at multiple feature resolutions."""

    def __init__(
        self,
        channels: int,
        blocks_per_scale: tuple[int, ...],
        downsamples: tuple[int, ...],
        ratio_g: float = 0.75,
        enable_lfu: bool = False,
        residual_scales: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if len(blocks_per_scale) != len(downsamples):
            raise ValueError("blocks_per_scale and downsamples must have the same length")
        if not blocks_per_scale:
            raise ValueError("LaMaFFCMultiScaleContext requires at least one scale")
        if residual_scales is None:
            residual_scales = tuple(0.05 for _ in blocks_per_scale)
        if len(residual_scales) != len(blocks_per_scale):
            raise ValueError("residual_scales must be omitted or match blocks_per_scale length")
        self.contexts = nn.ModuleList(
            [
                LaMaFFCResidualContext(
                    channels,
                    blocks=max(1, int(blocks)),
                    ratio_g=float(ratio_g),
                    enable_lfu=bool(enable_lfu),
                    downsample=max(1, int(downsample)),
                    residual_scale=float(scale),
                )
                for blocks, downsample, scale in zip(blocks_per_scale, downsamples, residual_scales)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for context in self.contexts:
            x = context(x)
        return x


class DADIGANCloudBranch(nn.Module):
    """DADIGAN single-path SAR-optical cloud generation branch."""

    def __init__(
        self,
        s2_channels: int,
        feature_channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        sar_channels: int = 2,
        ddin_steps: int = 3,
        prox_blocks: int = 2,
        reconstruct_blocks: int = 2,
        bottleneck_context: str = "none",
        pre_pda_context: str = "none",
        pre_pda_ffc_blocks: int = 0,
        pre_pda_ffc_ratio: float = 0.75,
        pre_pda_ffc_enable_lfu: bool = False,
        pre_pda_ffc_downsample: int = 4,
        pre_pda_ffc_residual_scale: float = 0.05,
        prefusion_context: str = "none",
        prefusion_blocks: int = 0,
        prefusion_kernel_size: int = 5,
        prefusion_reduction: int = 16,
        lowres_glfcr_coupled: bool = False,
        lowres_factor: int = 2,
        lowres_opt_ffc_blocks: int = 0,
        lowres_opt_ffc_ratio: float = 0.75,
        lowres_opt_ffc_enable_lfu: bool = False,
        lowres_opt_ffc_spatial_transform_layers: tuple[int, ...] | None = None,
        lowres_opt_ffc_spatial_transform_pad_coef: float = 0.5,
        lowres_opt_ffc_spatial_transform_angle_init_range: float = 80.0,
        lowres_opt_ffc_spatial_transform_train_angle: bool = True,
        lowres_glfcr_kernel_size: int = 5,
        cab_sr_ratio: int = 8,
        cab_attention_mode: str = "standard",
        msab_mode: str = "efficient",
        ffc_blocks: int = 0,
        ffc_blocks_per_scale: tuple[int, ...] | None = None,
        ffc_ratio: float = 0.75,
        ffc_enable_lfu: bool = False,
        ffc_downsample: int = 1,
        ffc_downsamples: tuple[int, ...] | None = None,
        ffc_residual_scale: float = 0.1,
        ffc_residual_scales: tuple[float, ...] | None = None,
        ffc_spatial_transform_layers: tuple[int, ...] | None = None,
        ffc_spatial_transform_pad_coef: float = 0.5,
        ffc_spatial_transform_angle_init_range: float = 80.0,
        ffc_spatial_transform_train_angle: bool = True,
        mask_input_mode: str = "raw",
        append_cloud_mask: bool = False,
        mask_fill_value: float = 0.0,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        channels = int(feature_channels[0])
        heads = max(1, min(8, channels // 8))
        bottleneck_context = str(bottleneck_context).lower()
        self.s2_channels = int(s2_channels)
        self.mask_input_mode = str(mask_input_mode).lower()
        self.append_cloud_mask = bool(append_cloud_mask)
        self.output_activation = str(output_activation).lower()
        if self.mask_input_mode not in {"raw", "zero", "zero_mask", "lama_zero", "constant", "learned"}:
            raise ValueError("mask_input_mode must be one of: raw, zero, lama_zero, constant, learned")
        if self.output_activation not in {"none", "identity", "sigmoid", "clamp"}:
            raise ValueError("output_activation must be one of: none, identity, sigmoid, clamp")
        self.lowres_glfcr_coupled = bool(lowres_glfcr_coupled)
        self.lowres_factor = max(1, int(lowres_factor))
        if self.lowres_glfcr_coupled and self.lowres_factor < 2:
            raise ValueError("lowres_glfcr_coupled requires lowres_factor >= 2")
        self.pixel_unshuffle: nn.Module = nn.PixelUnshuffle(self.lowres_factor) if self.lowres_glfcr_coupled else nn.Identity()
        if self.mask_input_mode == "learned":
            self.mask_token = nn.Parameter(torch.full((1, self.s2_channels, 1, 1), float(mask_fill_value)))
        else:
            self.register_buffer(
                "mask_token",
                torch.full((1, self.s2_channels, 1, 1), float(mask_fill_value)),
                persistent=False,
            )
        optical_in_channels = self.s2_channels + (1 if self.append_cloud_mask else 0)
        optical_stem_in_channels = optical_in_channels * (self.lowres_factor ** 2 if self.lowres_glfcr_coupled else 1)
        sar_stem_in_channels = sar_channels * (self.lowres_factor ** 2 if self.lowres_glfcr_coupled else 1)
        self.optical_stem = nn.Sequential(
            nn.Conv2d(optical_stem_in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.sar_stem = nn.Sequential(
            nn.Conv2d(sar_stem_in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        if self.lowres_glfcr_coupled:
            self.lowres_opt_context: nn.Module = (
                LaMaFFCFeatureContext(
                    channels,
                    blocks=max(1, int(lowres_opt_ffc_blocks)),
                    ratio_g=float(lowres_opt_ffc_ratio),
                    enable_lfu=bool(lowres_opt_ffc_enable_lfu),
                    spatial_transform_layers=lowres_opt_ffc_spatial_transform_layers,
                    spatial_transform_kwargs=_spatial_transform_kwargs(
                        lowres_opt_ffc_spatial_transform_pad_coef,
                        lowres_opt_ffc_spatial_transform_angle_init_range,
                        lowres_opt_ffc_spatial_transform_train_angle,
                    ),
                )
                if int(lowres_opt_ffc_blocks) > 0
                else nn.Identity()
            )
            self.ddin = GLFCRCoupledDDIN(
                channels,
                steps=ddin_steps,
                prox_blocks=prox_blocks,
                kernel_size=max(1, int(lowres_glfcr_kernel_size)),
            )
        else:
            self.lowres_opt_context = nn.Identity()
            self.ddin = DDIN(channels, steps=ddin_steps, prox_blocks=prox_blocks)
        self.pre_pda_context = DDINOutputContext(
            channels,
            context=pre_pda_context,
            blocks=max(1, int(pre_pda_ffc_blocks)),
            ratio_g=float(pre_pda_ffc_ratio),
            enable_lfu=bool(pre_pda_ffc_enable_lfu),
            downsample=max(1, int(pre_pda_ffc_downsample)),
            residual_scale=float(pre_pda_ffc_residual_scale),
        )
        self.prefusion_context = DDINPairFusionContext(
            channels,
            context=prefusion_context,
            blocks=int(prefusion_blocks),
            kernel_size=max(1, int(prefusion_kernel_size)),
            reduction=max(1, int(prefusion_reduction)),
        )
        # PDAFM already contains MSAB per DADIGAN Eq.(4):
        #   F = MSAB(CAB(CAB(P, S), V))
        # ``bottleneck_context`` is kept for ablation studies only.
        self.pdafm = PDAFMScale(
            channels,
            heads=heads,
            cab_sr_ratio=cab_sr_ratio,
            cab_attention_mode=cab_attention_mode,
            msab_mode=msab_mode,
        )
        if bottleneck_context == "none":
            self.bottleneck_context: nn.Module = nn.Identity()
        elif bottleneck_context == "msab":
            self.bottleneck_context = MSAB(channels, heads=heads, mode=msab_mode)
        elif bottleneck_context == "restormer_mdta":
            self.bottleneck_context = OfficialRestormerTransformerBlock(
                dim=channels,
                num_heads=heads,
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type="WithBias",
            )
        elif bottleneck_context == "restormer_block":
            self.bottleneck_context = OfficialRestormerTransformerBlock(
                dim=channels,
                num_heads=heads,
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type="WithBias",
            )
        elif bottleneck_context in {"lama_ffc", "ffc"}:
            self.bottleneck_context = LaMaFFCResidualContext(
                channels,
                blocks=max(1, int(ffc_blocks)),
                ratio_g=float(ffc_ratio),
                enable_lfu=bool(ffc_enable_lfu),
                downsample=max(1, int(ffc_downsample)),
                residual_scale=float(ffc_residual_scale),
            )
        elif bottleneck_context in {"lama_ffc_feature", "ffc_feature"}:
            self.bottleneck_context = LaMaFFCFeatureContext(
                channels,
                blocks=max(1, int(ffc_blocks)),
                ratio_g=float(ffc_ratio),
                enable_lfu=bool(ffc_enable_lfu),
                spatial_transform_layers=ffc_spatial_transform_layers,
                spatial_transform_kwargs=_spatial_transform_kwargs(
                    ffc_spatial_transform_pad_coef,
                    ffc_spatial_transform_angle_init_range,
                    ffc_spatial_transform_train_angle,
                ),
            )
        elif bottleneck_context in {"lama_ffc_multiscale", "ffc_multiscale", "fcf_synthesis"}:
            downsamples = tuple(int(v) for v in (ffc_downsamples or (ffc_downsample,)))
            blocks_per_scale = tuple(int(v) for v in (ffc_blocks_per_scale or tuple(int(ffc_blocks) for _ in downsamples)))
            residual_scales = (
                tuple(float(v) for v in ffc_residual_scales)
                if ffc_residual_scales is not None
                else tuple(float(ffc_residual_scale) for _ in downsamples)
            )
            self.bottleneck_context = LaMaFFCMultiScaleContext(
                channels,
                blocks_per_scale=blocks_per_scale,
                downsamples=downsamples,
                ratio_g=float(ffc_ratio),
                enable_lfu=bool(ffc_enable_lfu),
                residual_scales=residual_scales,
            )
        elif bottleneck_context == "identity":
            self.bottleneck_context = nn.Identity()
        else:
            raise ValueError(
                "bottleneck_context must be one of: 'none', 'msab', 'restormer_mdta', "
                "'restormer_block', 'lama_ffc', 'lama_ffc_feature', 'lama_ffc_multiscale', or 'identity'"
            )
        if self.lowres_glfcr_coupled:
            self.reconstruct = nn.Sequential(
                *[DADIGANResidualBlock(channels) for _ in range(max(1, int(reconstruct_blocks)))],
                nn.Conv2d(channels, channels * self.lowres_factor * self.lowres_factor, kernel_size=3, padding=1),
                nn.PixelShuffle(self.lowres_factor),
                nn.Conv2d(channels, s2_channels, kernel_size=3, padding=1),
            )
        else:
            self.reconstruct = nn.Sequential(
                *[DADIGANResidualBlock(channels) for _ in range(max(1, int(reconstruct_blocks)))],
                nn.Conv2d(channels, s2_channels, kernel_size=3, padding=1),
            )

    def _optical_condition(self, s2: Tensor, cloud_mask: Tensor) -> Tensor:
        m1 = cloud_mask.float().clamp(0, 1)
        s2_float = s2.float()
        if self.mask_input_mode == "raw":
            optical = s2_float
        elif self.mask_input_mode in {"zero", "zero_mask", "lama_zero"}:
            optical = (1.0 - m1) * s2_float
        else:
            token = self.mask_token.to(device=s2.device, dtype=s2_float.dtype)
            optical = (1.0 - m1) * s2_float + m1 * token
        if self.append_cloud_mask:
            optical = torch.cat([optical, m1], dim=1)
        return optical

    def _activate_fill(self, fill: Tensor) -> Tensor:
        if self.output_activation in {"none", "identity"}:
            return fill
        if self.output_activation == "sigmoid":
            return torch.sigmoid(fill)
        return fill.clamp(0.0, 1.0)

    def forward(self, s2: Tensor, sar: Tensor | None, cloud_mask: Tensor) -> dict[str, Tensor]:
        if sar is None:
            raise ValueError("DADIGAN cloud branch requires SAR/S1 input, but no SAR tensor was provided.")
        m1 = cloud_mask.float().clamp(0, 1)
        optical_input = self._optical_condition(s2, m1)
        sar_input = sar.float()
        if self.lowres_glfcr_coupled:
            optical_input = self.pixel_unshuffle(optical_input)
            sar_input = self.pixel_unshuffle(sar_input)
        opt_feat = self.optical_stem(optical_input)
        opt_feat = self.lowres_opt_context(opt_feat)
        sar_feat = self.sar_stem(sar_input)
        d = self.ddin(opt_feat, sar_feat)
        d = self.pre_pda_context(d)
        opt_private, sar_private, prefusion_aux = self.prefusion_context(d["opt_private"], d["sar_private"])
        d = dict(d)
        d["opt_private"] = opt_private
        d["sar_private"] = sar_private
        fused = self.pdafm(d["shared"], d["opt_private"], d["sar_private"])
        feature = self.bottleneck_context(fused)
        fill = self._activate_fill(self.reconstruct(feature))
        # Only cloud regions should be generated; outside cloud the candidate keeps input.
        image = (1.0 - m1) * s2.float() + m1 * fill
        return {
            "I_cloud": image,
            "I_cloud_raw": fill,
            "F_cloud": feature,
            "shared_s1": d["shared"],
            "opt_private_s1": d["opt_private"],
            "sar_private_s1": d["sar_private"],
            **{k: v for k, v in d.items() if k.startswith("M_glfcr_")},
            **prefusion_aux,
        }


def make_cloud_discriminator(
    s2_channels: int,
    base_channels: int = 32,
    num_layers: int = 5,
    condition_channels: int = 1,
    norm_type: str = "batch",
    output_mode: str = "patch",
) -> PatchDiscriminator:
    return PatchDiscriminator(
        s2_channels * 2 + int(condition_channels),
        base_channels=base_channels,
        num_layers=num_layers,
        norm_type=norm_type,
        output_mode=output_mode,
    )
