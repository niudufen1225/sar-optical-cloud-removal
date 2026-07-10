"""LaMa high-receptive-field perceptual loss.

This is a local, dependency-light port of LaMa's
``saicinpainting.training.losses.perceptual.ResNetPL``.  The encoder
architecture follows the ADE20K ResNet-50 dilated model used by LaMa; only the
feature-extractor path required for HRF perceptual loss is included here.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BatchNorm2d


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _ADE20KResNet(nn.Module):
    """ResNet stem used by CSAIL ADE20K semantic-segmentation models."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 128
        self.conv1 = _conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = _conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * _Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * _Bottleneck.expansion),
            )
        layers: list[nn.Module] = [_Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * _Bottleneck.expansion
        layers.extend(_Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)


class _ResNetDilatedEncoder(nn.Module):
    """ADE20K ResNet-50 encoder with dilation scale 8."""

    def __init__(self) -> None:
        super().__init__()
        orig = _ADE20KResNet()
        orig.layer3.apply(partial(self._nostride_dilate, dilate=2))
        orig.layer4.apply(partial(self._nostride_dilate, dilate=4))

        self.conv1 = orig.conv1
        self.bn1 = orig.bn1
        self.relu1 = orig.relu1
        self.conv2 = orig.conv2
        self.bn2 = orig.bn2
        self.relu2 = orig.relu2
        self.conv3 = orig.conv3
        self.bn3 = orig.bn3
        self.relu3 = orig.relu3
        self.maxpool = orig.maxpool
        self.layer1 = orig.layer1
        self.layer2 = orig.layer2
        self.layer3 = orig.layer3
        self.layer4 = orig.layer4

    @staticmethod
    def _nostride_dilate(module: nn.Module, dilate: int) -> None:
        if not isinstance(module, nn.Conv2d):
            return
        if module.stride == (2, 2):
            module.stride = (1, 1)
            if module.kernel_size == (3, 3):
                module.dilation = (dilate // 2, dilate // 2)
                module.padding = (dilate // 2, dilate // 2)
        elif module.kernel_size == (3, 3):
            module.dilation = (dilate, dilate)
            module.padding = (dilate, dilate)

    def forward(self, x: Tensor, return_feature_maps: bool = False) -> list[Tensor]:
        features: list[Tensor] = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features if return_feature_maps else [x]


def _resolve_encoder_weights(weights_path: str | Path) -> Path:
    path = Path(weights_path).expanduser()
    if path.is_file():
        return path
    candidate = path / "ade20k" / "ade20k-resnet50dilated-ppm_deepsup" / "encoder_epoch_20.pth"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "Missing LaMa/ADE20K encoder weights. Expected either a direct .pth file "
        f"or {candidate}"
    )


class ResNetPL(nn.Module):
    """LaMa HRF perceptual loss using a frozen ADE20K ResNet-50 dilated encoder."""

    def __init__(self, weight: float = 1.0, weights_path: str | Path | None = None) -> None:
        super().__init__()
        if weights_path is None:
            raise ValueError("ResNetPL requires weights_path")
        self.impl = _ResNetDilatedEncoder()
        state = torch.load(_resolve_encoder_weights(weights_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported ADE20K encoder checkpoint format: {weights_path}")
        self.impl.load_state_dict(state, strict=False)
        self.impl.eval()
        for param in self.impl.parameters():
            param.requires_grad_(False)
        self.weight = float(weight)
        self.register_buffer("mean", IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer("std", IMAGENET_STD.clone(), persistent=False)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # The encoder is frozen, but gradients still flow to pred. Running this
        # high-receptive-field perceptual path in fp32 avoids bf16 autocast
        # instability while keeping the trainable restoration model under AMP.
        autocast_off = torch.amp.autocast(device_type=pred.device.type, enabled=False) if pred.is_cuda else nullcontext()
        with autocast_off:
            pred = (pred.float() - self.mean.to(pred.device, dtype=torch.float32)) / self.std.to(pred.device, dtype=torch.float32)
            target = (target.float() - self.mean.to(target.device, dtype=torch.float32)) / self.std.to(target.device, dtype=torch.float32)
            pred_feats = self.impl(pred, return_feature_maps=True)
            target_feats = self.impl(target, return_feature_maps=True)
            loss = pred.new_zeros(())
            for cur_pred, cur_target in zip(pred_feats, target_feats):
                loss = loss + F.mse_loss(cur_pred, cur_target)
            return loss * self.weight
