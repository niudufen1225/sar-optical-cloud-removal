"""SoftShadow-constrained shadow branch.

SoftShadow's essential pieces are preserved:

- predict a continuous soft shadow mask rather than consume a hard mask;
    - supervise it with a soft target derived from shadow/clear brightness ratio;
    - use penumbra formation constraints and shadow-removal reconstruction loss.

    The optional external SAM-LoRA backend can import the official SoftShadow
    repository when its dependencies/checkpoints are available.  ALLClear hard
    shadow labels may still be used to form shadow_case metadata, but the
    official-style SAM/SoftShadow path must not crop predicted soft masks by
    those hard labels.
"""

from __future__ import annotations

import sys
import importlib.machinery
import types
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.allclear.modules.common import (
    NAFBlock,
    RestorationBlock,
    bbox_from_mask,
    low_pass,
    resize_like,
    sobel_xy,
    y_luma_from_s2,
)
from src.allclear.modules.restormer import RestormerTransformerBlock


def soft_shadow_division_target(
    cloudy: Tensor,
    target: Tensor,
    rgb_indices: tuple[int, int, int] = (3, 2, 1),
    low_pass_kernel: int = 5,
    threshold: float = 0.05,
    eps: float = 1.0e-3,
) -> Tensor:
    """Build a SoftShadow paper-style division mask from paired brightness.

    The paper derives the soft-mask supervision from the Y-channel ratio of a
    shadow image and its shadow-free counterpart, then applies a low-pass filter
    and a threshold to suppress outliers. No external hard shadow support is
    applied here.
    """

    y_shadow = y_luma_from_s2(cloudy, rgb_indices=rgb_indices).clamp_min(eps)
    y_clear = y_luma_from_s2(target, rgb_indices=rgb_indices).clamp_min(eps)
    ratio = (y_shadow / y_clear).clamp(0.0, 1.25)
    # In SoftShadow notation, lit ~= 0 and umbra ~= 1.
    soft = (1.0 - ratio).clamp(0.0, 1.0)
    soft = low_pass(soft, kernel_size=low_pass_kernel).clamp(0.0, 1.0)
    if threshold > 0:
        soft = torch.where(soft >= float(threshold), soft, soft.new_zeros(()))
    return soft


def soft_shadow_target(
    cloudy: Tensor,
    target: Tensor,
    shadow_support: Tensor,
    rgb_indices: tuple[int, int, int] = (3, 2, 1),
    low_pass_kernel: int = 5,
    eps: float = 1.0e-3,
) -> Tensor:
    """Legacy ALLClear soft target constrained by hard shadow support.

    This is not the strict SoftShadow preprocessing path. It is kept as a
    fallback for old experiments that only trust ALLClear's hard shadow labels.
    """

    soft = soft_shadow_division_target(
        cloudy,
        target,
        rgb_indices=rgb_indices,
        low_pass_kernel=low_pass_kernel,
        threshold=0.0,
        eps=eps,
    )
    trust = dilated_shadow_support(shadow_support, kernel_size=9)
    return (soft * trust).clamp(0.0, 1.0)


def dilated_shadow_support(shadow_support: Tensor, kernel_size: int = 9) -> Tensor:
    pad = kernel_size // 2
    return F.max_pool2d(shadow_support.float().clamp(0, 1), kernel_size=kernel_size, stride=1, padding=pad)


class ConvSoftMaskPredictor(nn.Module):
    """Fallback soft mask predictor with the same training objective as SoftShadow."""

    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            RestorationBlock(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            RestorationBlock(hidden_channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, s2: Tensor, shadow_support: Tensor) -> Tensor:
        logits = self.net(torch.cat([s2.float(), shadow_support.float()], dim=1))
        return torch.sigmoid(logits).clamp(0.0, 1.0)


class MaskPriorSoftMaskPredictor(nn.Module):
    """Parameter-free soft shadow prior from ALLClear's known shadow support."""

    def __init__(self, low_pass_kernel: int = 7) -> None:
        super().__init__()
        self.low_pass_kernel = int(low_pass_kernel)

    def forward(self, s2: Tensor, shadow_support: Tensor) -> Tensor:
        del s2
        support = shadow_support.float().clamp(0.0, 1.0)
        if support.shape[1] != 1:
            support = support[:, :1]
        soft = low_pass(support, kernel_size=self.low_pass_kernel)
        return soft.clamp(0.0, 1.0)


class ExternalSoftShadowSAM(nn.Module):
    """Wrapper around the official SoftShadow SAM-LoRA code path."""

    def __init__(
        self,
        softshadow_repo: str | Path,
        sam_checkpoint: str | Path,
        model_type: str = "vit_h",
        rank: int = 8,
        input_size: int = 1024,
        checkpoint_blocks: bool = False,
        lora_layers: object | None = None,
    ) -> None:
        super().__init__()
        repo = Path(softshadow_repo)
        if not repo.exists():
            raise FileNotFoundError(f"SoftShadow repo not found: {repo}")
        sys.path.insert(0, str(repo))
        try:
            from model.segment_anything import sam_model_registry  # type: ignore
            from model.segment_anything.sam_lora import LoRA_Sam  # type: ignore
        except Exception as exc:  # pragma: no cover - optional external dependency
            raise RuntimeError("Could not import official SoftShadow SAM modules") from exc

        class SAMWithBBox(nn.Module):
            def __init__(self, image_encoder: nn.Module, mask_decoder: nn.Module, prompt_encoder: nn.Module) -> None:
                super().__init__()
                self.image_encoder = image_encoder
                self.mask_decoder = mask_decoder
                self.prompt_encoder = prompt_encoder

            def forward(self, image: Tensor, bbox: Tensor) -> Tensor:
                features = self.image_encoder(image)
                sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=bbox, masks=None)
                low_res_masks, _ = self.mask_decoder(
                    image_embeddings=features,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                return low_res_masks

        sam_model = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
        sam = SAMWithBBox(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder)
        self.sam_lora = LoRA_Sam(sam, rank, lora_layer=self._resolve_lora_layers(lora_layers, len(sam_model.image_encoder.blocks)))
        self.sam = self.sam_lora.sam
        if checkpoint_blocks:
            self._enable_image_encoder_checkpointing(self.sam.image_encoder)
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        self.input_size = int(input_size)
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def _resolve_lora_layers(spec: object | None, num_blocks: int) -> list[int] | None:
        if spec is None:
            return None
        if isinstance(spec, str):
            text = spec.strip().lower()
            if not text or text in {"all", "none", "default"}:
                return None
            if text.startswith("last_"):
                count = int(text.split("_", 1)[1])
                return list(range(max(0, num_blocks - count), num_blocks))
            if text.startswith("last"):
                count = int(text.replace("last", ""))
                return list(range(max(0, num_blocks - count), num_blocks))
            return [int(part.strip()) for part in text.split(",") if part.strip()]
        if isinstance(spec, int):
            return list(range(max(0, num_blocks - int(spec)), num_blocks))
        if isinstance(spec, (list, tuple)):
            values = [int(value) for value in spec]
            return [value for value in values if 0 <= value < num_blocks]
        raise ValueError("softshadow_sam_lora_layers must be all, last_N, an int, or a list of block indices")

    @staticmethod
    def _enable_image_encoder_checkpointing(image_encoder: nn.Module) -> None:
        """Checkpoint SAM ViT blocks without changing the forward function."""

        def forward_with_checkpoint(self: nn.Module, x: Tensor) -> Tensor:
            from torch.utils.checkpoint import checkpoint

            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            for blk in self.blocks:
                if torch.is_grad_enabled():
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            return self.neck(x.permute(0, 3, 1, 2))

        image_encoder.forward = MethodType(forward_with_checkpoint, image_encoder)

    def forward(self, s2_rgb: Tensor, shadow_support: Tensor, bbox: Tensor | None = None, bbox_space: str = "image") -> Tensor:
        b, _, h, w = s2_rgb.shape
        # SAM ViT attention/LoRA backward is numerically fragile under bf16
        # autocast at 1024 input size. Keep this branch in fp32 while the rest
        # of Stage1 can still use AMP.
        autocast_off = torch.amp.autocast(device_type=s2_rgb.device.type, enabled=False) if s2_rgb.is_cuda else nullcontext()
        with autocast_off:
            # SoftShadow's official dataloader builds sam_SR with Resize(1024),
            # ToTensor, and ImageNet normalization before calling LoRA_Sam.
            sam_input = F.interpolate(
                s2_rgb.float().clamp(0.0, 1.0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            mean = self.pixel_mean.to(device=sam_input.device, dtype=torch.float32)
            std = self.pixel_std.to(device=sam_input.device, dtype=torch.float32)
            sam_input = (sam_input - mean) / std
            if bbox is None:
                boxes = bbox_from_mask(shadow_support, pad=4).to(device=sam_input.device, dtype=torch.float32)
                scale_x = self.input_size / float(w)
                scale_y = self.input_size / float(h)
                boxes = boxes.clone()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
            else:
                boxes = bbox.to(device=sam_input.device, dtype=torch.float32)
                if boxes.ndim == 1:
                    boxes = boxes.unsqueeze(0)
                if boxes.ndim == 3:
                    boxes = boxes.reshape(boxes.shape[0], -1, 4)[:, 0, :]
                if boxes.shape[-1] != 4:
                    raise ValueError(f"SoftShadow bbox must have shape [B,4], got {tuple(boxes.shape)}")
                boxes = boxes.reshape(-1, 4).clone()
                if boxes.shape[0] != b:
                    raise ValueError(f"SoftShadow bbox batch size {boxes.shape[0]} does not match image batch size {b}")
                if str(bbox_space).lower() in {"image", "original", "source"}:
                    scale_x = self.input_size / float(w)
                    scale_y = self.input_size / float(h)
                    boxes[:, [0, 2]] *= scale_x
                    boxes[:, [1, 3]] *= scale_y
                elif str(bbox_space).lower() not in {"sam", "sam_input", "input", "input_size"}:
                    raise ValueError("softshadow_bbox_space must be 'image' or 'sam_input'")
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(self.input_size - 1))
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(self.input_size - 1))
            invalid = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
            if invalid.any():
                boxes[invalid] = boxes.new_tensor([0.0, 0.0, float(self.input_size - 1), float(self.input_size - 1)])
            mask = torch.sigmoid(self.sam(sam_input, boxes).float())
        return F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)


class _LoRAConvQKV(nn.Module):
    """SoftShadow-style low-rank q/v adapter for EfficientViT LiteMLA qkv conv.

    EfficientViT-SAM does not expose SAM ViT's ``attn.qkv`` linear layer. Its
    global-context blocks use LiteMLA, where q/k/v are produced by a 1x1
    ``ConvLayer``. This wrapper mirrors SoftShadow's LoRA intent by freezing
    the original qkv projection and adding trainable low-rank residuals only
    to the q and v channel groups.
    """

    def __init__(self, qkv: nn.Module, rank: int) -> None:
        super().__init__()
        if not hasattr(qkv, "conv"):
            raise TypeError("EfficientViT LiteMLA qkv module must expose a ConvLayer.conv")
        conv = qkv.conv
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("EfficientViT LiteMLA qkv ConvLayer.conv must be nn.Conv2d")
        if conv.kernel_size != (1, 1) or conv.groups != 1:
            raise ValueError("Only 1x1 dense qkv conv is supported for EfficientViT LoRA adapters")
        out_channels = int(conv.out_channels)
        if out_channels % 3 != 0:
            raise ValueError(f"qkv out_channels must be divisible by 3, got {out_channels}")
        self.qkv = qkv
        self.qv_channels = out_channels // 3
        in_channels = int(conv.in_channels)
        rank = int(rank)
        if rank <= 0:
            raise ValueError("EfficientViT adapter rank must be positive")
        self.q_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.q_up = nn.Conv2d(rank, self.qv_channels, kernel_size=1, bias=False)
        self.v_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.v_up = nn.Conv2d(rank, self.qv_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.q_down.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.v_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q_delta = self.q_up(self.q_down(x))
        v_delta = self.v_up(self.v_down(x))
        q, k, v = qkv.split(self.qv_channels, dim=1)
        return torch.cat([q + q_delta, k, v + v_delta], dim=1)


class ExternalEfficientViTSoftShadowSAM(nn.Module):
    """EfficientViT-SAM mask predictor adapted to SoftShadow training.

    The base EfficientViT-SAM image encoder is initialized from official
    checkpoints and frozen. Trainable parameters are limited to LiteMLA q/v
    low-rank adapters plus the SAM mask decoder, while the prompt encoder stays
    frozen as in the official SoftShadow SAM-LoRA path.
    """

    _MODEL_BUILDERS = {
        "efficientvit-sam-l0": "efficientvit_sam_l0",
        "efficientvit-sam-l1": "efficientvit_sam_l1",
        "efficientvit-sam-l2": "efficientvit_sam_l2",
        "efficientvit-sam-xl0": "efficientvit_sam_xl0",
        "efficientvit-sam-xl1": "efficientvit_sam_xl1",
    }

    def __init__(
        self,
        efficientvit_repo: str | Path,
        softshadow_repo: str | Path,
        checkpoint: str | Path | None,
        model_name: str = "efficientvit-sam-xl0",
        input_size: int = 1024,
        adapter_rank: int = 8,
        adapter_layers: object | None = None,
        train_mask_decoder: bool = True,
        force_fp32: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.force_fp32 = bool(force_fp32)
        self.model_name = str(model_name).lower()
        if self.model_name not in self._MODEL_BUILDERS:
            raise ValueError(
                f"Unsupported EfficientViT-SAM model {model_name!r}; "
                f"expected one of {sorted(self._MODEL_BUILDERS)}"
            )
        self._prepare_imports(efficientvit_repo, softshadow_repo)
        from efficientvit.models.efficientvit import sam as evit_sam  # type: ignore
        from efficientvit.models.nn.norm import set_norm_eps  # type: ignore

        builder = getattr(evit_sam, self._MODEL_BUILDERS[self.model_name])
        self.sam = builder(image_size=self.input_size)
        set_norm_eps(self.sam, 1.0e-6)
        if checkpoint:
            self._load_efficientvit_checkpoint(checkpoint)

        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = bool(train_mask_decoder)

        self.adapter_names = self._inject_litemla_adapters(
            self.sam.image_encoder,
            rank=int(adapter_rank),
            layer_spec=adapter_layers,
        )
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def _prepare_imports(efficientvit_repo: str | Path, softshadow_repo: str | Path) -> None:
        efficientvit_repo = Path(efficientvit_repo)
        softshadow_repo = Path(softshadow_repo)
        if not efficientvit_repo.exists():
            raise FileNotFoundError(f"EfficientViT repo not found: {efficientvit_repo}")
        if not softshadow_repo.exists():
            raise FileNotFoundError(f"SoftShadow repo not found: {softshadow_repo}")
        # EfficientViT-SAM imports ``segment_anything``. Reuse the SAM package
        # bundled by SoftShadow instead of requiring a second installation.
        sys.path.insert(0, str(softshadow_repo / "model"))
        sys.path.insert(0, str(efficientvit_repo))

        # Official EfficientViT imports ONNX export helpers through package
        # initializers even when only PyTorch training code is used. Provide
        # inert modules so training does not depend on ONNX packages.
        for name in ("onnx", "onnxsim"):
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
                sys.modules[name] = module
        if not hasattr(sys.modules["onnx"], "load_model"):
            sys.modules["onnx"].load_model = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        if not hasattr(sys.modules["onnx"], "save"):
            sys.modules["onnx"].save = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        if not hasattr(sys.modules["onnxsim"], "simplify"):
            sys.modules["onnxsim"].simplify = lambda model, *args, **kwargs: (model, True)  # type: ignore[attr-defined]

    def _load_efficientvit_checkpoint(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Missing EfficientViT-SAM checkpoint: {path}. "
                "Download the official weight, e.g. efficientvit_sam_xl0.pt, "
                "and set model.softshadow_efficientvit_checkpoint."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported EfficientViT-SAM checkpoint format: {path}")
        missing, unexpected = self.sam.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected EfficientViT-SAM checkpoint keys: {unexpected[:10]}")
        # Missing adapter keys are expected only when adapters are injected
        # after this load. At this point a clean official checkpoint should
        # load all base keys.
        if missing:
            raise RuntimeError(f"Missing EfficientViT-SAM checkpoint keys before adapter injection: {missing[:10]}")

    @staticmethod
    def _resolve_layers(spec: object | None, total: int) -> set[int]:
        if spec is None:
            return set(range(total))
        if isinstance(spec, str):
            text = spec.strip().lower()
            if not text or text in {"all", "default"}:
                return set(range(total))
            if text in {"none", "off", "disabled"}:
                return set()
            if text.startswith("last_"):
                count = int(text.split("_", 1)[1])
                return set(range(max(0, total - count), total))
            if text.startswith("last"):
                count = int(text.replace("last", ""))
                return set(range(max(0, total - count), total))
            return {int(part.strip()) for part in text.split(",") if part.strip()}
        if isinstance(spec, int):
            return set(range(max(0, total - int(spec)), total))
        if isinstance(spec, (list, tuple)):
            return {int(value) for value in spec if 0 <= int(value) < total}
        raise ValueError("softshadow_efficientvit_adapter_layers must be all, last_N, an int, or a list of indices")

    def _inject_litemla_adapters(self, image_encoder: nn.Module, rank: int, layer_spec: object | None) -> list[str]:
        candidates = []
        for name, module in image_encoder.named_modules():
            if module.__class__.__name__ == "LiteMLA" and hasattr(module, "qkv"):
                candidates.append((name, module))
        selected = self._resolve_layers(layer_spec, len(candidates))
        adapter_names: list[str] = []
        for idx, (name, module) in enumerate(candidates):
            if idx not in selected:
                continue
            module.qkv = _LoRAConvQKV(module.qkv, rank=rank)
            adapter_names.append(name)
        return adapter_names

    def forward(self, s2_rgb: Tensor, shadow_support: Tensor, bbox: Tensor | None = None, bbox_space: str = "image") -> Tensor:
        b, _, h, w = s2_rgb.shape
        autocast_off = (
            torch.amp.autocast(device_type=s2_rgb.device.type, enabled=False)
            if self.force_fp32 and s2_rgb.is_cuda
            else nullcontext()
        )
        with autocast_off:
            sam_input = F.interpolate(
                s2_rgb.float().clamp(0.0, 1.0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            mean = self.pixel_mean.to(device=sam_input.device, dtype=torch.float32)
            std = self.pixel_std.to(device=sam_input.device, dtype=torch.float32)
            sam_input = (sam_input - mean) / std
            if bbox is None:
                boxes = bbox_from_mask(shadow_support, pad=4).to(device=sam_input.device, dtype=torch.float32)
                scale_x = self.input_size / float(w)
                scale_y = self.input_size / float(h)
                boxes = boxes.clone()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
            else:
                boxes = bbox.to(device=sam_input.device, dtype=torch.float32)
                if boxes.ndim == 1:
                    boxes = boxes.unsqueeze(0)
                if boxes.ndim == 3:
                    boxes = boxes.reshape(boxes.shape[0], -1, 4)[:, 0, :]
                if boxes.shape[-1] != 4:
                    raise ValueError(f"SoftShadow bbox must have shape [B,4], got {tuple(boxes.shape)}")
                boxes = boxes.reshape(-1, 4).clone()
                if boxes.shape[0] != b:
                    raise ValueError(f"SoftShadow bbox batch size {boxes.shape[0]} does not match image batch size {b}")
                if str(bbox_space).lower() in {"image", "original", "source"}:
                    scale_x = self.input_size / float(w)
                    scale_y = self.input_size / float(h)
                    boxes[:, [0, 2]] *= scale_x
                    boxes[:, [1, 3]] *= scale_y
                elif str(bbox_space).lower() not in {"sam", "sam_input", "input", "input_size"}:
                    raise ValueError("softshadow_bbox_space must be 'image' or 'sam_input'")
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(self.input_size - 1))
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(self.input_size - 1))
            invalid = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
            if invalid.any():
                boxes[invalid] = boxes.new_tensor([0.0, 0.0, float(self.input_size - 1), float(self.input_size - 1)])

            features = self.sam.image_encoder(sam_input)
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=features,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            mask = torch.sigmoid(low_res_masks.float())
        return F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)


class ShadowRemovalHead(nn.Module):
    """Soft-mask guided shadow removal decoder."""

    def __init__(self, channels: int, hidden_channels: int = 96, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.body = nn.Sequential(*[RestorationBlock(hidden_channels) for _ in range(blocks)])
        self.affine = nn.Conv2d(hidden_channels, channels * 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, s2: Tensor, soft_mask: Tensor) -> Tensor:
        feat = self.body(self.stem(torch.cat([s2.float(), soft_mask.float()], dim=1)))
        gain, bias = self.affine(feat).chunk(2, dim=1)
        # Shadow is mostly illumination degradation, so use affine correction.
        return s2.float() * (1.0 + soft_mask * torch.tanh(gain)) + soft_mask * torch.tanh(bias)


class RestormerShadowRemovalHead(nn.Module):
    """Soft-mask guided shadow removal with official Restormer blocks."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        blocks: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            *[
                RestormerTransformerBlock(
                    dim=hidden_channels,
                    num_heads=heads,
                    ffn_expansion_factor=2.66,
                    bias=False,
                    layer_norm_type="WithBias",
                )
                for _ in range(blocks)
            ]
        )
        self.affine = nn.Conv2d(hidden_channels, channels * 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, s2: Tensor, soft_mask: Tensor) -> Tensor:
        feat = self.body(self.stem(torch.cat([s2.float(), soft_mask.float()], dim=1)))
        gain, bias = self.affine(feat).chunk(2, dim=1)
        return s2.float() * (1.0 + soft_mask * torch.tanh(gain)) + soft_mask * torch.tanh(bias)


class NAFShadowRemovalHead(nn.Module):
    """Soft-mask guided shadow removal with NAFNet NAFBlocks.

    Body  —  NAFBlock × 3  (NAFNet, Chen et al. ECCV 2022).
    Head  —  single 3×3 conv predicting a 13-ch residual, gated by the
             soft shadow mask  (``I = s2 + soft_mask * correction``).

    The residual-output pattern follows NAFNet's ``self.ending`` +
    ``x = x + inp``, while the soft-mask gating follows the inpainting
    mask-composite paradigm.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        blocks: int = 3,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            *[NAFBlock(hidden_channels) for _ in range(blocks)]
        )
        # NAFNet-style output: single 3×3 conv → residual
        self.ending = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, s2: Tensor, soft_mask: Tensor) -> Tensor:
        feat = self.body(self.stem(torch.cat([s2.float(), soft_mask.float()], dim=1)))
        correction = self.ending(feat)
        return s2.float() + soft_mask * correction


class SoftShadowBranch(nn.Module):
    """Predict soft shadow mask and a shadow-free candidate image."""

    def __init__(
        self,
        channels: int,
        backend: str = "conv",
        removal_backend: str | None = None,
        hidden_channels: int = 96,
        softshadow_repo: str | None = None,
        sam_checkpoint: str | None = None,
        softshadow_checkpoint: str | None = None,
        rgb_indices: tuple[int, int, int] = (3, 2, 1),
        sam_model_type: str = "vit_h",
        sam_lora_rank: int = 8,
        sam_lora_layers: object | None = None,
        sam_input_size: int = 1024,
        sam_checkpoint_blocks: bool = False,
        sam_bbox_space: str = "image",
        efficientvit_repo: str | None = None,
        efficientvit_checkpoint: str | None = None,
        efficientvit_model: str = "efficientvit-sam-xl0",
        efficientvit_adapter_rank: int = 8,
        efficientvit_adapter_layers: object | None = None,
        efficientvit_train_mask_decoder: bool = True,
        efficientvit_force_fp32: bool = False,
        use_hard_support_gate: bool = True,
        forward_valid_only: bool = False,
        restormer_hidden_channels: int = 64,
        restormer_blocks: int = 2,
        restormer_heads: int = 4,
        nafnet_hidden_channels: int | None = None,
        nafnet_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.backend = str(backend)
        self.sam_bbox_space = str(sam_bbox_space)
        self.use_hard_support_gate = bool(use_hard_support_gate)
        self.forward_valid_only = bool(forward_valid_only)
        if removal_backend is None:
            removal_backend = "nafnet" if self.backend == "external_softshadow" else ("restormer" if self.backend == "restormer" else "affine")
        self.removal_backend = str(removal_backend)
        self.rgb_indices = tuple(rgb_indices)
        if backend == "external_softshadow":
            if softshadow_repo is None or sam_checkpoint is None:
                raise ValueError("external_softshadow backend requires softshadow_repo and sam_checkpoint")
            self.mask_predictor: nn.Module = ExternalSoftShadowSAM(
                softshadow_repo,
                sam_checkpoint,
                model_type=str(sam_model_type),
                rank=int(sam_lora_rank),
                input_size=int(sam_input_size),
                checkpoint_blocks=bool(sam_checkpoint_blocks),
                lora_layers=sam_lora_layers,
            )
        elif backend == "efficientvit_softshadow":
            if softshadow_repo is None or efficientvit_repo is None:
                raise ValueError("efficientvit_softshadow backend requires softshadow_repo and efficientvit_repo")
            self.mask_predictor = ExternalEfficientViTSoftShadowSAM(
                efficientvit_repo=efficientvit_repo,
                softshadow_repo=softshadow_repo,
                checkpoint=efficientvit_checkpoint,
                model_name=str(efficientvit_model),
                input_size=int(sam_input_size),
                adapter_rank=int(efficientvit_adapter_rank),
                adapter_layers=efficientvit_adapter_layers,
                train_mask_decoder=bool(efficientvit_train_mask_decoder),
                force_fp32=bool(efficientvit_force_fp32),
            )
        elif backend == "mask_prior":
            self.mask_predictor = MaskPriorSoftMaskPredictor()
        elif backend == "conv":
            self.mask_predictor = ConvSoftMaskPredictor(channels, hidden_channels=hidden_channels)
        elif backend == "restormer":
            self.mask_predictor = ConvSoftMaskPredictor(channels, hidden_channels=hidden_channels)
        else:
            raise ValueError("backend must be 'mask_prior', 'conv', 'restormer', 'external_softshadow', or 'efficientvit_softshadow'")
        if self.removal_backend == "restormer":
            self.removal = RestormerShadowRemovalHead(
                channels,
                hidden_channels=int(restormer_hidden_channels),
                blocks=int(restormer_blocks),
                heads=int(restormer_heads),
            )
        elif self.removal_backend == "nafnet":
            self.removal = NAFShadowRemovalHead(
                channels,
                hidden_channels=int(nafnet_hidden_channels or restormer_hidden_channels),
                blocks=int(nafnet_blocks),
            )
        elif self.removal_backend in {"affine", "conv"}:
            self.removal = ShadowRemovalHead(channels, hidden_channels=hidden_channels)
        else:
            raise ValueError("removal_backend must be 'affine', 'conv', 'restormer', or 'nafnet'")
        if softshadow_checkpoint:
            self.load_pretrained(softshadow_checkpoint)

    def load_pretrained(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
        own = self.state_dict()
        filtered = {}
        for key, value in state.items():
            clean = key.replace("module.", "").replace("samshadow.", "")
            if clean in own and own[clean].shape == value.shape:
                filtered[clean] = value
        self.load_state_dict(filtered, strict=False)

    @staticmethod
    def _case_gate(shadow_case: Tensor | None, ref: Tensor) -> Tensor | None:
        if shadow_case is None:
            return None
        case = shadow_case.to(device=ref.device).long().view(-1)
        # Dataset convention: 1 = valid_shadow; 0 = no_shadow; 2 = ambiguous.
        return (case == 1).to(dtype=ref.dtype).view(-1, 1, 1, 1)

    def _param_anchor(self, ref: Tensor) -> Tensor:
        if not self.training:
            return ref.new_zeros(())
        anchor = ref.new_zeros(())
        for param in self.parameters():
            if param.requires_grad:
                anchor = anchor + param.float().sum() * 0.0
        return anchor

    def _predict_raw_mask(self, s2: Tensor, shadow_support: Tensor, bbox: Tensor | None) -> Tensor:
        if isinstance(self.mask_predictor, (ExternalSoftShadowSAM, ExternalEfficientViTSoftShadowSAM)):
            rgb = s2[:, list(self.rgb_indices)].float()
            raw_mask = self.mask_predictor(rgb, shadow_support, bbox=bbox, bbox_space=self.sam_bbox_space)
        else:
            raw_mask = self.mask_predictor(s2, shadow_support)
        if raw_mask.shape[-2:] != s2.shape[-2:]:
            raw_mask = F.interpolate(raw_mask, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        if self.backend in {"external_softshadow", "efficientvit_softshadow", "mask_prior"}:
            return raw_mask.clamp(0.0, 1.0)
        trust = dilated_shadow_support(shadow_support, kernel_size=9)
        return (raw_mask * trust).clamp(0.0, 1.0)

    def forward(
        self,
        s2: Tensor,
        shadow_support: Tensor,
        bbox: Tensor | None = None,
        shadow_case: Tensor | None = None,
    ) -> dict[str, Tensor]:
        case_gate = self._case_gate(shadow_case, s2)
        if shadow_case is None and self.use_hard_support_gate and shadow_support.detach().sum().item() < 1.0:
            soft_mask = shadow_support.new_zeros((shadow_support.shape[0], 1, s2.shape[-2], s2.shape[-1]))
            return {
                "I_shadow": s2.float(),
                "M_shadow_soft": soft_mask,
                "M_shadow_soft_raw": soft_mask,
                "M_shadow_soft_eff": soft_mask,
            }

        if self.forward_valid_only and case_gate is not None:
            valid = case_gate.view(-1) > 0.5
            raw_mask = shadow_support.new_zeros((shadow_support.shape[0], 1, s2.shape[-2], s2.shape[-1]))
            raw_mask = raw_mask + self._param_anchor(s2)
            if valid.any():
                bbox_valid = bbox[valid] if bbox is not None else None
                raw_valid = self._predict_raw_mask(s2[valid], shadow_support[valid], bbox_valid)
                raw_mask = raw_mask.clone()
                raw_mask[valid] = raw_valid
        else:
            raw_mask = self._predict_raw_mask(s2, shadow_support, bbox)
        if case_gate is not None:
            raw_mask = raw_mask * case_gate

        # SoftShadow's official path learns a continuous soft mask from paired
        # image-derived division masks and bbox prompts.  Cropping by an
        # external hard mask reintroduces the hard boundary artifact that the
        # paper is designed to avoid, so the effective mask is the gated raw
        # prediction itself.
        effective_mask = raw_mask.clamp(0.0, 1.0)
        image = self.removal(s2, effective_mask)
        if case_gate is not None:
            image = case_gate * image + (1.0 - case_gate) * s2.float()
        return {
            "I_shadow": image,
            "M_shadow_soft": effective_mask,
            "M_shadow_soft_raw": raw_mask.clamp(0.0, 1.0),
            "M_shadow_soft_eff": effective_mask,
        }


def softshadow_mask_loss(
    pred_mask: Tensor,
    soft_target: Tensor,
    support: Tensor | None = None,
    outside_weight: float = 0.05,
) -> Tensor:
    error = (pred_mask.float() - soft_target.float()).pow(2)
    if support is None:
        return error.mean()
    if support.shape[-2:] != pred_mask.shape[-2:]:
        support = F.interpolate(support.float(), size=pred_mask.shape[-2:], mode="nearest")
    trust = dilated_shadow_support(support.float(), kernel_size=9)
    if trust.shape[1] != error.shape[1]:
        trust = trust.expand(-1, error.shape[1], -1, -1)
    inside = (error * trust).sum() / trust.sum().clamp_min(1.0)
    outside = 1.0 - trust
    if outside.sum() < 1.0:
        return inside
    outside_loss = (error * outside).sum() / outside.sum().clamp_min(1.0)
    return inside + float(outside_weight) * outside_loss


def penumbra_constraint_loss(
    pred_mask: Tensor,
    threshold_low: float = 50.0 / 255.0,
    threshold_high: float = 220.0 / 255.0,
    mode: str = "softshadow_no_penumbra",
) -> Tensor:
    """SoftShadow contour-gradient constraint.

    ``mode=softshadow_no_penumbra`` mirrors the official SoftShadow joint
    training path: the penumbra range is used only to compute the contour
    center, while Sobel penalties are applied to the full predicted mask.
    ``mode=softshadow_penumbra`` keeps the official ablation that also masks
    Sobel responses by the penumbra region.
    """

    mode = str(mode).lower()
    gray_tensor = pred_mask.float()
    penumbra = ((gray_tensor > float(threshold_low)) & (gray_tensor < float(threshold_high))).float()
    gx, gy = sobel_xy(gray_tensor)
    b, _, h, w = gray_tensor.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=gray_tensor.device, dtype=gray_tensor.dtype),
        torch.arange(w, device=gray_tensor.device, dtype=gray_tensor.dtype),
        indexing="ij",
    )
    losses = []
    smooth = 1.0e-7
    for i in range(b):
        coords = torch.nonzero(penumbra[i, 0] > 0.5)
        if coords.numel() > 0:
            center = coords.float().mean(dim=0)
            c_y, c_x = center[0], center[1]
        else:
            c_y = gray_tensor.new_tensor(0.0)
            c_x = gray_tensor.new_tensor(0.0)
        weight_x = torch.ones((h, w), device=gray_tensor.device, dtype=gray_tensor.dtype)
        weight_y = torch.ones_like(weight_x)
        weight_x = weight_x * (xx < c_x).float() * -2.0 + 1.0
        weight_y = weight_y * (yy < c_y).float() * -2.0 + 1.0
        sobel_x = gx[i : i + 1]
        sobel_y = gy[i : i + 1]
        if mode in {"softshadow_penumbra", "penumbra"}:
            sobel_x = sobel_x * penumbra[i : i + 1]
            sobel_y = sobel_y * penumbra[i : i + 1]
            denom = 2.0 * penumbra[i : i + 1].sum() + 10.0 * smooth
        elif mode in {"softshadow_no_penumbra", "no_penumbra", "official"}:
            denom = gray_tensor.new_tensor(2.0 * float(h * w))
        else:
            raise ValueError("penumbra mode must be one of: softshadow_no_penumbra, softshadow_penumbra")
        mod_x = sobel_x * weight_x.view(1, 1, h, w)
        mod_y = sobel_y * weight_y.view(1, 1, h, w)
        losses.append((F.relu(mod_x).sum() + F.relu(mod_y).sum() + smooth) / denom)
    if not losses:
        return pred_mask.new_zeros(())
    return torch.stack(losses).mean()
