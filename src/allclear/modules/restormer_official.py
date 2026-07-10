"""Official Restormer MDTA blocks.

This file is adapted from the public Restormer implementation:
https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py

Only the modules needed to replace DADIGAN's MSAB position are included:
LayerNorm, GDFN, MDTA Attention, and TransformerBlock.
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def to_3d(x: Tensor) -> Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: Tensor, h: int, w: int) -> Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: Tensor) -> Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1.0e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1.0e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, LayerNorm_type: str) -> None:
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Official Restormer block: MDTA + GDFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
