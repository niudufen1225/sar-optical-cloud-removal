"""PVT SRA cross-attention used as the DADIGAN CAB replacement.

This module is adapted from the public Pyramid Vision Transformer code:
https://github.com/whai362/PVT/blob/master/classification/pvt.py

The copied source pattern is PVT's ``Mlp``, ``Attention`` and ``Block``:
full-resolution Q, spatial-reduced K/V through a strided convolution,
LayerNorm after spatial reduction, softmax attention, output projection,
then residual MLP.  The first structural change is that K/V come from a
separate reference feature map, so the block can replace DADIGAN's CAB
positions as cross-attention.  The second optional change is
``attention_mode="complement"``, which keeps PVT's spatial reduction but uses
the stable normalized complement of the attention output to approximate
DADIGAN's discrepancy-aware ``1 - Attention`` CAB.
"""

from __future__ import annotations

import math

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _to_tokens(x: Tensor) -> Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def _to_nchw(x: Tensor, h: int, w: int) -> Tensor:
    b, _, c = x.shape
    return x.transpose(1, 2).reshape(b, c, h, w).contiguous()


class PVTMLP(nn.Module):
    """Official PVT token MLP."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = int(out_features or in_features)
        hidden_features = int(hidden_features or in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PVTSRACrossAttention(nn.Module):
    """PVT spatial-reduction attention adapted from self- to cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 8,
        attention_mode: str = "standard",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.attention_mode = str(attention_mode).lower()
        if self.attention_mode not in {"standard", "complement"}:
            raise ValueError("attention_mode must be one of: standard, complement")
        head_dim = self.dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim, self.dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = max(1, int(sr_ratio))
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(self.dim, self.dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(self.dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, query_feat: Tensor, reference_feat: Tensor) -> Tensor:
        b, c, h, w = query_feat.shape
        if reference_feat.shape != query_feat.shape:
            raise ValueError(
                "PVTSRACrossAttention expects query/reference with identical "
                f"shape, got {tuple(query_feat.shape)} and {tuple(reference_feat.shape)}"
            )

        n = h * w
        query = _to_tokens(query_feat)
        q = self.q(query).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None and self.norm is not None:
            reference = self.sr(reference_feat).reshape(b, c, -1).permute(0, 2, 1).contiguous()
            reference = self.norm(reference)
        else:
            reference = _to_tokens(reference_feat)
        kv = self.kv(reference).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # PVT's public code computes:
        #   attn = softmax((q @ k.transpose(-2, -1)) * scale)
        #   x = attn @ v
        # Calling PyTorch SDPA keeps the same scaled dot-product attention
        # formula but avoids explicitly materialising the large
        # [B, heads, HW, HW / sr_ratio^2] logits tensor during training.
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=self.scale)
        if self.attention_mode == "complement":
            # DADIGAN's CAB uses V * (1 - A) to emphasize discrepancy rather
            # than feature similarity.  A literal sum_j(1-A_ij)V_j grows with
            # the number of K/V tokens, so we use the normalized complement:
            #   (sum(V) - A@V) / max(Nkv - 1, 1)
            # This preserves the discrepancy-aware semantics while keeping the
            # activation scale stable for SRA-reduced and unreduced K/V.
            kv_tokens = max(1, int(v.shape[-2]))
            v_sum = v.sum(dim=-2, keepdim=True)
            if kv_tokens > 1:
                x = (v_sum - x) / float(kv_tokens - 1)
            else:
                x = v_sum - x
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SRACAB(nn.Module):
    """SRA-CAB: PVT Block mechanics with cross-attention K/V.

    DADIGAN uses two CAB calls in PDAFM.  This block is intended to be used at
    those same two positions: CAB(P,S) and CAB(V,FM).  ``attention_mode`` can
    keep standard SRA cross-attention or use the stable normalized complement
    approximation of DADIGAN's discrepancy-aware CAB.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        sr_ratio: int = 8,
        attention_mode: str = "standard",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.norm_query = nn.LayerNorm(self.channels)
        self.norm_reference = nn.LayerNorm(self.channels)
        self.attn = PVTSRACrossAttention(
            self.channels,
            num_heads=heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            attention_mode=attention_mode,
        )
        self.norm2 = nn.LayerNorm(self.channels)
        mlp_hidden_dim = int(self.channels * mlp_ratio)
        self.mlp = PVTMLP(in_features=self.channels, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(
        self,
        private_query: Tensor,
        reference_feat: Tensor,
        *,
        residual_base: Tensor | None = None,
        update_scale: float = 1.0,
    ) -> Tensor:
        b, c, h, w = private_query.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")
        if reference_feat.shape != private_query.shape:
            raise ValueError(
                "SRACAB expects query/reference with identical shape, got "
                f"{tuple(private_query.shape)} and {tuple(reference_feat.shape)}"
            )
        if residual_base is not None and residual_base.shape != private_query.shape:
            raise ValueError(
                "SRACAB residual_base must have the same shape as private_query, got "
                f"{tuple(residual_base.shape)} and {tuple(private_query.shape)}"
            )
        update_scale = float(update_scale)
        if not math.isfinite(update_scale) or update_scale < 0.0:
            raise ValueError("SRACAB update_scale must be a finite non-negative scalar")
        query_tokens = _to_tokens(private_query)
        reference_tokens = _to_tokens(reference_feat)
        query_norm = _to_nchw(self.norm_query(query_tokens), h, w)
        reference_norm = _to_nchw(self.norm_reference(reference_tokens), h, w)

        residual_tokens = query_tokens if residual_base is None else _to_tokens(residual_base)
        x = residual_tokens + update_scale * self.attn(query_norm, reference_norm)
        x = x + self.mlp(self.norm2(x))
        return _to_nchw(x, h, w)
