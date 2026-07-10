"""Losses for ALLClear Stage1 training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.allclear.modules.common import gradient_loss, masked_l1
from src.allclear.modules.softshadow import (
    penumbra_constraint_loss,
    soft_shadow_division_target,
    soft_shadow_target,
    softshadow_mask_loss,
)


@dataclass
class LossWeights:
    final_l1: float = 0.0
    grad: float = 0.0
    shadow_removal: float = 1.0
    shadow_mask: float = 0.5
    shadow_penumbra: float = 0.05
    cloud_l1: float = 1.0
    cloud_l1_missing: float = 0.0
    cloud_l1_known: float = 0.0
    cloud_kl: float = 0.05
    cloud_adv: float = 0.02
    feature_matching: float = 0.0
    perceptual: float = 0.0


def spectral_kl_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    reduction: str = "image_mean",
    mode: str = "softmax",
    eps: float = 1.0e-8,
) -> Tensor:
    """DADIGAN-style spectral regularization for multispectral distributions.

    DADIGAN defines ``KL(p(G(m)) || q(n))`` but does not specify how p/q are
    formed from Sentinel-2 bands.  ``softmax`` preserves the historical
    implementation.  ``sum_normalized`` and ``softplus_sum`` form explicit
    per-pixel band distributions; ``softplus_sum`` is safer for unconstrained
    generator outputs.  ``sam_angle`` is a mature remote-sensing spectral-shape
    alternative kept under the same weight for ablation.
    """

    mode = str(mode).lower()
    pred_f = pred.float()
    target_f = target.float()
    if mode == "softmax":
        p = F.softmax(pred_f, dim=1).clamp_min(eps)
        q = F.softmax(target_f, dim=1).clamp_min(eps)
        kl_map = (p * (p.log() - q.log())).sum(dim=1, keepdim=True)
    elif mode in {"sum_normalized", "reflectance_sum", "positive_sum"}:
        p_pos = pred_f.clamp_min(eps)
        q_pos = target_f.clamp_min(eps)
        p = p_pos / p_pos.sum(dim=1, keepdim=True).clamp_min(eps)
        q = q_pos / q_pos.sum(dim=1, keepdim=True).clamp_min(eps)
        kl_map = (p * (p.log() - q.log())).sum(dim=1, keepdim=True)
    elif mode in {"softplus_sum", "softplus_normalized"}:
        p_pos = F.softplus(pred_f).clamp_min(eps)
        q_pos = target_f.clamp_min(eps)
        p = p_pos / p_pos.sum(dim=1, keepdim=True).clamp_min(eps)
        q = q_pos / q_pos.sum(dim=1, keepdim=True).clamp_min(eps)
        kl_map = (p * (p.log() - q.log())).sum(dim=1, keepdim=True)
    elif mode in {"sam", "sam_angle", "spectral_angle"}:
        dot = (pred_f * target_f).sum(dim=1, keepdim=True)
        pred_norm = pred_f.pow(2).sum(dim=1, keepdim=True).sqrt()
        target_norm = target_f.pow(2).sum(dim=1, keepdim=True).sqrt()
        cos = dot / (pred_norm * target_norm).clamp_min(eps)
        kl_map = torch.acos(cos.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
    else:
        raise ValueError("spectral_kl_loss mode must be one of: softmax, sum_normalized, softplus_sum, sam_angle")
    if mask is None:
        return kl_map.mean()
    mask = mask.float().clamp(0.0, 1.0)
    if mask.shape[-2:] != kl_map.shape[-2:]:
        mask = F.interpolate(mask, size=kl_map.shape[-2:], mode="nearest")
    if mask.shape[1] != kl_map.shape[1]:
        mask = mask[:, :1]
    masked_error = (kl_map * mask).mean()
    mask_fraction = mask.mean().clamp_min(1.0e-6)
    mode = str(reduction).lower()
    if mode in {"mask_mean", "mask_normalized", "region_mean"}:
        return masked_error / mask_fraction
    if mode in {"image_mean", "full_mean"}:
        return masked_error
    if mode in {"sqrt_area", "sqrt_mask"}:
        return masked_error / mask_fraction.sqrt()
    if mode == "hybrid":
        return 0.5 * (masked_error / mask_fraction + masked_error)
    raise ValueError("spectral_kl_loss reduction must be one of: mask_mean, image_mean, sqrt_area, hybrid")


class AllClearStageLoss(nn.Module):
    """Generator-side loss for Stage1."""

    def __init__(
        self,
        weights: LossWeights,
        rgb_indices: tuple[int, int, int] = (3, 2, 1),
        final_mask_mode: str = "degraded",
        adversarial_loss: str = "bce",
        perceptual_type: str = "rgb_l1",
        perceptual_lama_repo: str | None = None,
        perceptual_weights_path: str | None = None,
        cloud_l1_reduction: str = "mask_mean",
        cloud_l1_region: str = "cloud",
        cloud_kl_reduction: str = "image_mean",
        cloud_kl_mode: str = "softmax",
        perceptual_input: str = "cloud_context",
        feature_matching_loss_type: str = "mse",
        shadow_mask_outside_weight: float = 0.05,
        shadow_mask_region: str = "support",
        shadow_removal_region: str = "shadow",
        shadow_removal_loss_type: str = "l1",
        shadow_penumbra_mode: str = "softshadow_no_penumbra",
        shadow_soft_target_low_pass_kernel: int = 5,
        shadow_soft_target_mode: str = "hard_support",
        shadow_soft_target_division_threshold: float = 0.05,
        shadow_case_gating: bool = False,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.rgb_indices = tuple(rgb_indices)
        self.final_mask_mode = str(final_mask_mode)
        self.adversarial_loss = str(adversarial_loss)
        self.perceptual_type = str(perceptual_type)
        self.perceptual_lama_repo = perceptual_lama_repo
        self.perceptual_weights_path = perceptual_weights_path
        self.cloud_l1_reduction = str(cloud_l1_reduction)
        self.cloud_l1_region = str(cloud_l1_region).lower()
        self.cloud_kl_reduction = str(cloud_kl_reduction)
        self.cloud_kl_mode = str(cloud_kl_mode).lower()
        self.perceptual_input = str(perceptual_input).lower()
        self.feature_matching_loss_type = str(feature_matching_loss_type).lower()
        self.shadow_mask_outside_weight = float(shadow_mask_outside_weight)
        self.shadow_mask_region = str(shadow_mask_region).lower()
        self.shadow_removal_region = str(shadow_removal_region).lower()
        self.shadow_removal_loss_type = str(shadow_removal_loss_type).lower()
        self.shadow_penumbra_mode = str(shadow_penumbra_mode).lower()
        self.shadow_soft_target_low_pass_kernel = int(shadow_soft_target_low_pass_kernel)
        self.shadow_soft_target_mode = str(shadow_soft_target_mode).lower()
        self.shadow_soft_target_division_threshold = float(shadow_soft_target_division_threshold)
        self.shadow_case_gating = bool(shadow_case_gating)
        self.perceptual_model = self._build_perceptual_model()

    def _shadow_removal_loss(self, pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
        mode = self.shadow_removal_loss_type
        if mask is not None:
            mask = mask.float().clamp(0.0, 1.0)
            if mask.shape[-2:] != pred.shape[-2:]:
                mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
            if mask.shape[1] == 1 and pred.shape[1] != 1:
                mask = mask.expand(-1, pred.shape[1], -1, -1)
        if mode in {"l1", "mae"}:
            if mask is None:
                return F.l1_loss(pred.float(), target.float())
            return (pred.float() - target.float()).abs().mul(mask).sum() / mask.sum().clamp_min(1.0)
        if mode in {"mse", "l2", "frobenius"}:
            if mask is None:
                return F.mse_loss(pred.float(), target.float())
            return (pred.float() - target.float()).square().mul(mask).sum() / mask.sum().clamp_min(1.0)
        raise ValueError("shadow_removal_loss_type must be one of: l1, mse")

    @staticmethod
    def _zero_shadow_anchor(outputs: dict[str, Tensor], pred: Tensor) -> Tensor:
        anchor = pred.new_zeros(())
        for key in ("M_shadow_soft", "M_shadow_soft_raw", "M_shadow_soft_eff", "I_shadow"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                anchor = anchor + value.float().sum() * 0.0
        return anchor

    def _shadow_valid_index(self, batch: dict[str, Tensor], pred: Tensor) -> Tensor | None:
        if not self.shadow_case_gating or "shadow_case" not in batch:
            return None
        case = batch["shadow_case"].to(device=pred.device).long().view(-1)
        return case == 1

    @staticmethod
    def _select_valid(x: Tensor, valid: Tensor | None) -> Tensor:
        return x if valid is None else x[valid]

    def _build_perceptual_model(self) -> nn.Module | None:
        if self.perceptual_type in {"", "none", "rgb_l1"}:
            return None
        if self.perceptual_type not in {"hrf_resnet", "lama_hrf"}:
            raise ValueError("perceptual_type must be one of: rgb_l1, hrf_resnet, lama_hrf")
        try:
            from src.allclear.modules.lama_perceptual import ResNetPL
        except Exception as exc:  # pragma: no cover - environment error path
            raise RuntimeError(
                "perceptual_type=hrf_resnet requires the local LaMa HRF perceptual module."
            ) from exc
        weights_path = self.perceptual_weights_path or self.perceptual_lama_repo
        if not weights_path:
            raise ValueError("perceptual_type=hrf_resnet requires loss.perceptual_weights_path")
        model = ResNetPL(weight=1.0, weights_path=weights_path)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def _final_mask(self, outputs: dict[str, Tensor], pred: Tensor) -> Tensor | None:
        if self.final_mask_mode == "all":
            return None
        if self.final_mask_mode == "none":
            return pred.new_zeros((pred.shape[0], 1, pred.shape[-2], pred.shape[-1]))
        if self.final_mask_mode == "degraded":
            mask = outputs["M_shadow"].float() + outputs["M_cloud"].float()
            return mask.clamp(0.0, 1.0)
        if self.final_mask_mode == "cloud_shadow":
            return (outputs["M_shadow"].float() + outputs["M_cloud"].float()).clamp(0.0, 1.0)
        if self.final_mask_mode == "boundary":
            return outputs.get(
                "M_boundary",
                pred.new_zeros((pred.shape[0], 1, pred.shape[-2], pred.shape[-1])),
            ).float()
        raise ValueError("final_mask_mode must be 'all', 'none', 'degraded', 'cloud_shadow', or 'boundary'")

    def forward(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        fake_logits: Tensor | None = None,
        real_features: list[Tensor] | None = None,
        fake_features: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pred = outputs["I_hat"]
        target = batch["target"].float()
        cloudy = batch["s2_toa"].float()
        m_shadow = outputs["M_shadow"].float()
        m_cloud = outputs["M_cloud"].float()
        w = self.weights
        shadow_valid = self._shadow_valid_index(batch, pred)
        shadow_has_valid = True if shadow_valid is None else bool(shadow_valid.any().item())

        terms: dict[str, Tensor] = {}
        if self.shadow_case_gating and "shadow_case" in batch:
            case = batch["shadow_case"].to(device=pred.device).long().view(-1)
            terms["shadow_valid_frac"] = (case == 1).float().mean()
            terms["shadow_no_shadow_frac"] = (case == 0).float().mean()
            terms["shadow_ambiguous_frac"] = (case == 2).float().mean()
        else:
            terms["shadow_valid_frac"] = pred.new_zeros(())
            terms["shadow_no_shadow_frac"] = pred.new_zeros(())
            terms["shadow_ambiguous_frac"] = pred.new_zeros(())
        final_mask = self._final_mask(outputs, pred)
        if w.final_l1 <= 0:
            terms["final_l1"] = pred.new_zeros(())
        elif final_mask is None:
            terms["final_l1"] = F.l1_loss(pred.float(), target)
        else:
            terms["final_l1"] = masked_l1(pred.float(), target, final_mask, weight=1.0)

        if w.grad <= 0:
            terms["grad"] = pred.new_zeros(())
        elif final_mask is None:
            terms["grad"] = gradient_loss(pred.float(), target)
        else:
            terms["grad"] = gradient_loss(pred.float(), target, final_mask)

        if w.shadow_removal <= 0:
            terms["shadow_removal"] = pred.new_zeros(())
        elif not shadow_has_valid:
            terms["shadow_removal"] = self._zero_shadow_anchor(outputs, pred)
        elif self.shadow_removal_region in {"all", "full", "softshadow"}:
            terms["shadow_removal"] = self._shadow_removal_loss(
                self._select_valid(outputs["I_shadow"], shadow_valid),
                self._select_valid(target, shadow_valid),
                None,
            )
        elif self.shadow_removal_region in {"soft", "soft_shadow", "pred_mask"}:
            terms["shadow_removal"] = self._shadow_removal_loss(
                self._select_valid(outputs["I_shadow"], shadow_valid),
                self._select_valid(target, shadow_valid),
                self._select_valid(outputs["M_shadow_soft"].float(), shadow_valid),
            )
        elif self.shadow_removal_region in {"restore", "restoration", "degraded", "cloud_shadow"}:
            restore_mask = outputs.get("M_restore", (m_shadow + m_cloud).clamp(0.0, 1.0)).float()
            terms["shadow_removal"] = self._shadow_removal_loss(
                self._select_valid(outputs["I_shadow"], shadow_valid),
                self._select_valid(target, shadow_valid),
                self._select_valid(restore_mask, shadow_valid),
            )
        elif self.shadow_removal_region in {"shadow", "hard", "support"}:
            terms["shadow_removal"] = self._shadow_removal_loss(
                self._select_valid(outputs["I_shadow"], shadow_valid),
                self._select_valid(target, shadow_valid),
                self._select_valid(m_shadow, shadow_valid),
            )
        else:
            raise ValueError("shadow_removal_region must be one of: shadow, soft_shadow, restore, all")
        if w.shadow_mask <= 0:
            terms["shadow_mask"] = pred.new_zeros(())
        elif not shadow_has_valid:
            terms["shadow_mask"] = self._zero_shadow_anchor(outputs, pred)
        else:
            if "sam_mask" in batch:
                shadow_soft_gt = batch["sam_mask"].float().to(device=pred.device, dtype=pred.dtype)
                if shadow_soft_gt.shape[-2:] != pred.shape[-2:]:
                    shadow_soft_gt = F.interpolate(shadow_soft_gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
                if shadow_soft_gt.shape[1] != 1:
                    shadow_soft_gt = shadow_soft_gt[:, :1]
                shadow_soft_gt = shadow_soft_gt.clamp(0.0, 1.0)
            elif self.shadow_soft_target_mode in {"offline", "offline_required", "strict_offline"}:
                raise KeyError(
                    "loss.shadow_soft_target_mode=offline_required but batch has no sam_mask. "
                    "Generate SoftShadow division_filter_results/bounding_boxes.yaml and set "
                    "data.softshadow_mask_dir[_split]/data.softshadow_bbox_path[_split]."
                )
            elif self.shadow_soft_target_mode in {"paper", "paper_ratio", "division", "softshadow", "offline_or_paper"}:
                shadow_soft_gt = soft_shadow_division_target(
                    cloudy,
                    target,
                    rgb_indices=self.rgb_indices,
                    low_pass_kernel=self.shadow_soft_target_low_pass_kernel,
                    threshold=self.shadow_soft_target_division_threshold,
                )
            elif self.shadow_soft_target_mode in {"hard", "hard_support", "legacy"}:
                shadow_soft_gt = soft_shadow_target(
                    cloudy,
                    target,
                    m_shadow,
                    rgb_indices=self.rgb_indices,
                    low_pass_kernel=self.shadow_soft_target_low_pass_kernel,
                )
            else:
                raise ValueError(
                    "shadow_soft_target_mode must be one of: offline_required, paper_ratio, hard_support"
                )
            support = None if self.shadow_mask_region in {"all", "full", "softshadow"} else m_shadow
            if self.shadow_mask_region not in {"all", "full", "softshadow", "support", "shadow", "hard"}:
                raise ValueError("shadow_mask_region must be one of: all, support")
            terms["shadow_mask"] = softshadow_mask_loss(
                self._select_valid(outputs["M_shadow_soft"], shadow_valid),
                self._select_valid(shadow_soft_gt, shadow_valid),
                support=None if support is None else self._select_valid(support, shadow_valid),
                outside_weight=self.shadow_mask_outside_weight,
            )
        if w.shadow_penumbra <= 0:
            terms["shadow_penumbra"] = pred.new_zeros(())
        elif not shadow_has_valid:
            terms["shadow_penumbra"] = self._zero_shadow_anchor(outputs, pred)
        else:
            terms["shadow_penumbra"] = penumbra_constraint_loss(
                self._select_valid(outputs["M_shadow_soft"], shadow_valid),
                mode=self.shadow_penumbra_mode,
            )

        cloud_raw = outputs.get("I_cloud_raw", outputs["I_cloud"])
        if w.cloud_l1 <= 0:
            terms["cloud_l1"] = pred.new_zeros(())
        else:
            if self.cloud_l1_region in {"known", "valid", "unmasked", "lama_known"}:
                l1_mask = 1.0 - m_cloud
            elif self.cloud_l1_region in {"cloud", "missing", "masked", "hole"}:
                l1_mask = m_cloud
            elif self.cloud_l1_region in {"all", "full"}:
                l1_mask = torch.ones_like(m_cloud)
            else:
                raise ValueError("cloud_l1_region must be one of: cloud, known, all")
            terms["cloud_l1"] = masked_l1(cloud_raw, target, l1_mask, weight=1.0, reduction=self.cloud_l1_reduction)
        if w.cloud_kl <= 0:
            terms["cloud_kl"] = pred.new_zeros(())
        else:
            terms["cloud_kl"] = spectral_kl_loss(
                cloud_raw,
                target,
                mask=m_cloud,
                reduction=self.cloud_kl_reduction,
                mode=self.cloud_kl_mode,
            )
        if fake_logits is None:
            terms["cloud_adv"] = pred.new_zeros(())
        elif self.adversarial_loss == "hinge":
            terms["cloud_adv"] = -fake_logits.mean()
        elif self.adversarial_loss in {"r1", "non_saturating", "non_saturating_r1", "softplus"}:
            terms["cloud_adv"] = F.softplus(-fake_logits.float()).mean()
        else:
            terms["cloud_adv"] = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

        # Feature matching loss between discriminator intermediate features.
        # Fake features must keep the generator graph; real features are detached
        # by the training loop.
        if w.feature_matching <= 0 or real_features is None or fake_features is None:
            terms["feature_matching"] = pred.new_zeros(())
        else:
            fm = pred.new_zeros(())
            count = 0
            for r, f in zip(real_features, fake_features):
                if self.feature_matching_loss_type in {"l1", "mae"}:
                    fm = fm + F.l1_loss(f, r)
                elif self.feature_matching_loss_type in {"mse", "l2"}:
                    fm = fm + F.mse_loss(f, r)
                else:
                    raise ValueError("feature_matching_loss_type must be one of: l1, mse")
                count += 1
            if count > 0:
                fm = fm / float(count)
            terms["feature_matching"] = fm

        # Perceptual loss.
        # rgb_l1 is the legacy proxy. hrf_resnet is LaMa's high-receptive-field
        # ResNet perceptual loss. ``perceptual_input=raw`` matches Big-LaMa's
        # predicted_image loss; ``cloud_context`` keeps the older ALLClear
        # cloud-composite adaptation.
        if w.perceptual <= 0:
            terms["perceptual"] = pred.new_zeros(())
        elif self.perceptual_model is not None:
            rgb_idxs = list(self.rgb_indices)
            if self.perceptual_input in {"raw", "cloud_raw", "predicted", "predicted_image", "lama_raw"}:
                perceptual_pred = cloud_raw
            elif self.perceptual_input in {"cloud_context", "composite", "inpainted"}:
                perceptual_pred = m_cloud * cloud_raw + (1.0 - m_cloud) * target
            elif self.perceptual_input in {"stage1", "final"}:
                perceptual_pred = pred
            else:
                raise ValueError("perceptual_input must be one of: raw, cloud_context, stage1")
            pred_rgb = perceptual_pred[:, rgb_idxs].clamp(0.0, 1.0)
            target_rgb = target[:, rgb_idxs].clamp(0.0, 1.0)
            terms["perceptual"] = self.perceptual_model(pred_rgb, target_rgb)
        else:
            rgb_idxs = list(self.rgb_indices)
            pred_rgb = pred[:, rgb_idxs]          # [B, 3, H, W]
            target_rgb = target[:, rgb_idxs]      # [B, 3, H, W]
            terms["perceptual"] = masked_l1(pred_rgb, target_rgb, m_cloud, weight=1.0)

        recon_total = (
            w.final_l1 * terms["final_l1"]
            + w.grad * terms["grad"]
            + w.shadow_removal * terms["shadow_removal"]
            + w.shadow_mask * terms["shadow_mask"]
            + w.shadow_penumbra * terms["shadow_penumbra"]
            + w.cloud_l1 * terms["cloud_l1"]
            + w.cloud_kl * terms["cloud_kl"]
        )
        gan_total = (
            w.cloud_adv * terms["cloud_adv"]
            + w.feature_matching * terms["feature_matching"]
            + w.perceptual * terms["perceptual"]
        )
        total = recon_total + gan_total
        terms["recon_total"] = recon_total
        terms["gan_total"] = gan_total
        terms["total"] = total
        return total, terms


class CloudOnlyRestorationLoss(nn.Module):
    """Cloud/degraded-region loss for the DaDiGAN+LaMa-FFC baseline.

    This profile intentionally excludes final-image, SoftShadow, and stage2
    losses so the DaDiGAN+FFC run logs contain only the terms that actually
    update the single restoration branch.
    """

    def __init__(
        self,
        weights: LossWeights,
        rgb_indices: tuple[int, int, int] = (3, 2, 1),
        adversarial_loss: str = "bce",
        perceptual_type: str = "rgb_l1",
        perceptual_lama_repo: str | None = None,
        perceptual_weights_path: str | None = None,
        cloud_l1_reduction: str = "mask_mean",
        cloud_l1_region: str = "cloud",
        cloud_kl_reduction: str = "image_mean",
        cloud_kl_mode: str = "softmax",
        perceptual_input: str = "cloud_context",
        feature_matching_loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.weights = weights
        self.rgb_indices = tuple(rgb_indices)
        self.adversarial_loss = str(adversarial_loss)
        self.perceptual_type = str(perceptual_type)
        self.perceptual_lama_repo = perceptual_lama_repo
        self.perceptual_weights_path = perceptual_weights_path
        self.cloud_l1_reduction = str(cloud_l1_reduction)
        self.cloud_l1_region = str(cloud_l1_region).lower()
        self.cloud_kl_reduction = str(cloud_kl_reduction)
        self.cloud_kl_mode = str(cloud_kl_mode).lower()
        self.perceptual_input = str(perceptual_input).lower()
        self.feature_matching_loss_type = str(feature_matching_loss_type).lower()
        self.perceptual_model: nn.Module | None = None
        self._perceptual_uses_model = self.perceptual_type not in {"", "none", "rgb_l1"}

    def _build_perceptual_model(self) -> nn.Module | None:
        if self.perceptual_type in {"", "none", "rgb_l1"}:
            return None
        if self.perceptual_type not in {"hrf_resnet", "lama_hrf"}:
            raise ValueError("perceptual_type must be one of: rgb_l1, hrf_resnet, lama_hrf")
        try:
            from src.allclear.modules.lama_perceptual import ResNetPL
        except Exception as exc:  # pragma: no cover - environment error path
            raise RuntimeError("perceptual_type=hrf_resnet requires the local LaMa HRF perceptual module.") from exc
        weights_path = self.perceptual_weights_path or self.perceptual_lama_repo
        if not weights_path:
            raise ValueError("perceptual_type=hrf_resnet requires loss.perceptual_weights_path")
        model = ResNetPL(weight=1.0, weights_path=weights_path)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def _ensure_perceptual_model(self, device: torch.device) -> nn.Module | None:
        if not self._perceptual_uses_model:
            return None
        if self.perceptual_model is None:
            self.perceptual_model = self._build_perceptual_model()
            if self.perceptual_model is not None:
                self.perceptual_model.to(device)
        return self.perceptual_model

    def forward(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        fake_logits: Tensor | None = None,
        real_features: list[Tensor] | None = None,
        fake_features: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pred = outputs["I_hat"]
        target = batch["target"].float()
        m_cloud = outputs["M_cloud"].float().clamp(0.0, 1.0)
        cloud_raw = outputs.get("I_cloud_raw", outputs["I_cloud"])
        w = self.weights

        terms: dict[str, Tensor] = {}
        terms["cloud_l1"] = pred.new_zeros(())
        terms["cloud_l1_missing"] = pred.new_zeros(())
        terms["cloud_l1_known"] = pred.new_zeros(())

        if w.cloud_l1 > 0:
            if self.cloud_l1_region in {"known", "valid", "unmasked", "lama_known"}:
                l1_mask = 1.0 - m_cloud
            elif self.cloud_l1_region in {"cloud", "missing", "masked", "hole", "degraded"}:
                l1_mask = m_cloud
            elif self.cloud_l1_region in {"all", "full"}:
                l1_mask = torch.ones_like(m_cloud)
            else:
                raise ValueError("cloud_l1_region must be one of: cloud, known, all")
            terms["cloud_l1"] = masked_l1(cloud_raw, target, l1_mask, weight=1.0, reduction=self.cloud_l1_reduction)

        if w.cloud_l1_missing > 0:
            terms["cloud_l1_missing"] = masked_l1(
                cloud_raw,
                target,
                m_cloud,
                weight=1.0,
                reduction=self.cloud_l1_reduction,
            )
        if w.cloud_l1_known > 0:
            terms["cloud_l1_known"] = masked_l1(
                cloud_raw,
                target,
                1.0 - m_cloud,
                weight=1.0,
                reduction=self.cloud_l1_reduction,
            )

        if w.cloud_kl <= 0:
            terms["cloud_kl"] = pred.new_zeros(())
        else:
            terms["cloud_kl"] = spectral_kl_loss(
                cloud_raw,
                target,
                mask=m_cloud,
                reduction=self.cloud_kl_reduction,
                mode=self.cloud_kl_mode,
            )

        if fake_logits is None:
            terms["cloud_adv"] = pred.new_zeros(())
        elif self.adversarial_loss == "hinge":
            terms["cloud_adv"] = -fake_logits.mean()
        elif self.adversarial_loss in {"r1", "non_saturating", "non_saturating_r1", "softplus"}:
            terms["cloud_adv"] = F.softplus(-fake_logits.float()).mean()
        else:
            terms["cloud_adv"] = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

        if w.feature_matching <= 0 or real_features is None or fake_features is None:
            terms["feature_matching"] = pred.new_zeros(())
        else:
            fm = pred.new_zeros(())
            count = 0
            for real_feat, fake_feat in zip(real_features, fake_features):
                if self.feature_matching_loss_type in {"l1", "mae"}:
                    fm = fm + F.l1_loss(fake_feat, real_feat)
                elif self.feature_matching_loss_type in {"mse", "l2"}:
                    fm = fm + F.mse_loss(fake_feat, real_feat)
                else:
                    raise ValueError("feature_matching_loss_type must be one of: l1, mse")
                count += 1
            terms["feature_matching"] = fm / float(max(1, count))

        if w.perceptual <= 0:
            terms["perceptual"] = pred.new_zeros(())
        else:
            perceptual_model = self._ensure_perceptual_model(pred.device)
            if perceptual_model is None:
                rgb_idxs = list(self.rgb_indices)
                terms["perceptual"] = masked_l1(pred[:, rgb_idxs], target[:, rgb_idxs], m_cloud, weight=1.0)
            else:
                if self.perceptual_input in {"raw", "cloud_raw", "predicted", "predicted_image", "lama_raw"}:
                    perceptual_pred = cloud_raw
                elif self.perceptual_input in {"cloud_context", "composite", "inpainted"}:
                    perceptual_pred = m_cloud * cloud_raw + (1.0 - m_cloud) * target
                elif self.perceptual_input in {"stage1", "final"}:
                    perceptual_pred = pred
                else:
                    raise ValueError("perceptual_input must be one of: raw, cloud_context, stage1")
                rgb_idxs = list(self.rgb_indices)
                terms["perceptual"] = perceptual_model(
                    perceptual_pred[:, rgb_idxs].clamp(0.0, 1.0),
                    target[:, rgb_idxs].clamp(0.0, 1.0),
                )

        pixel_total = (
            w.cloud_l1 * terms["cloud_l1"]
            + w.cloud_l1_missing * terms["cloud_l1_missing"]
            + w.cloud_l1_known * terms["cloud_l1_known"]
            + w.cloud_kl * terms["cloud_kl"]
        )
        perceptual_total = w.perceptual * terms["perceptual"]
        gan_total = w.cloud_adv * terms["cloud_adv"] + w.feature_matching * terms["feature_matching"]
        recon_total = pixel_total + perceptual_total
        total = recon_total + gan_total

        terms["pixel_total"] = pixel_total
        terms["perceptual_total"] = perceptual_total
        terms["recon_total"] = recon_total
        terms["gan_total"] = gan_total
        terms["total"] = total
        return total, terms


class CloudDiscriminatorLoss(nn.Module):
    """DADIGAN cGAN discriminator loss — BCE for PatchGAN."""

    def forward(self, real_logits: Tensor, fake_logits: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        loss = real_loss + fake_loss
        return loss, {
            "disc_total": loss,
            "disc_real_loss": real_loss.detach(),
            "disc_fake_loss": fake_loss.detach(),
            "disc_real_logit": real_logits.mean(),
            "disc_fake_logit": fake_logits.mean(),
        }


class HingeDiscriminatorLoss(nn.Module):
    """SN-PatchGAN Hinge loss (ICCV 2019, DeepFill v2).

    D_loss = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))
    G_loss = -mean(D(fake))
    """

    def __init__(
        self,
        mask_as_fake_target: bool = False,
        allow_scale_mask: bool = True,
        mask_scale_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.mask_as_fake_target = bool(mask_as_fake_target)
        self.allow_scale_mask = bool(allow_scale_mask)
        self.mask_scale_mode = str(mask_scale_mode).lower()

    def _score_mask(self, mask: Tensor | None, logits: Tensor) -> Tensor | None:
        if mask is None:
            return None
        if mask.shape[-2:] != logits.shape[-2:]:
            if not self.allow_scale_mask:
                raise ValueError(
                    f"mask shape {tuple(mask.shape[-2:])} does not match discriminator logits "
                    f"{tuple(logits.shape[-2:])}"
                )
            if self.mask_scale_mode == "maxpool":
                mask = F.adaptive_max_pool2d(mask.float(), output_size=logits.shape[-2:])
            else:
                mask = F.interpolate(mask.float(), size=logits.shape[-2:], mode=self.mask_scale_mode)
        if mask.shape[1] != logits.shape[1]:
            mask = mask.expand(-1, logits.shape[1], -1, -1)
        return mask.float().clamp(0.0, 1.0)

    def forward(
        self,
        real_logits: Tensor,
        fake_logits: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss_map = F.relu(1.0 + fake_logits)
        score_mask = self._score_mask(mask, fake_logits)
        if score_mask is not None and self.mask_as_fake_target:
            # LaMa semantics: generated pixels outside the hole are copied from
            # the real image, so those discriminator locations should be treated
            # as real rather than fake.
            fake_loss_map = fake_loss_map * score_mask + F.relu(1.0 - fake_logits) * (1.0 - score_mask)
        fake_loss = fake_loss_map.mean()
        loss = real_loss + fake_loss
        return loss, {
            "disc_total": loss,
            "disc_real_loss": real_loss.detach(),
            "disc_fake_loss": fake_loss.detach(),
            "disc_real_logit": real_logits.mean(),
            "disc_fake_logit": fake_logits.mean(),
        }


class R1DiscriminatorLoss(nn.Module):
    """LaMa/Big-LaMa non-saturating GAN loss with R1 regularization.

    This matches the core semantics of
    ``saicinpainting.training.losses.adversarial.NonSaturatingWithR1``:
    generator uses ``softplus(-D(fake))``; discriminator uses
    ``softplus(-D(real)) + softplus(D(fake)) + gp_coef * R1``.  When
    ``mask_as_fake_target`` is enabled, fake scores outside the cloud mask are
    treated as real targets because those pixels are copied context.
    """

    def __init__(
        self,
        gp_coef: float = 0.001,
        mask_as_fake_target: bool = True,
        allow_scale_mask: bool = True,
        mask_scale_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.gp_coef = float(gp_coef)
        self.mask_as_fake_target = bool(mask_as_fake_target)
        self.allow_scale_mask = bool(allow_scale_mask)
        self.mask_scale_mode = str(mask_scale_mode).lower()

    def _score_mask(self, mask: Tensor | None, logits: Tensor) -> Tensor | None:
        if mask is None:
            return None
        if mask.shape[-2:] != logits.shape[-2:]:
            if not self.allow_scale_mask:
                raise ValueError(
                    f"mask shape {tuple(mask.shape[-2:])} does not match discriminator logits "
                    f"{tuple(logits.shape[-2:])}"
                )
            if self.mask_scale_mode == "maxpool":
                mask = F.adaptive_max_pool2d(mask.float(), output_size=logits.shape[-2:])
            else:
                mask = F.interpolate(mask.float(), size=logits.shape[-2:], mode=self.mask_scale_mode)
        if mask.shape[1] != logits.shape[1]:
            mask = mask.expand(-1, logits.shape[1], -1, -1)
        return mask.float().clamp(0.0, 1.0)

    @staticmethod
    def _r1_penalty(real_logits: Tensor, real_image: Tensor) -> Tensor:
        if not torch.is_grad_enabled() or not real_image.requires_grad:
            return real_logits.new_zeros(())
        grad_real = torch.autograd.grad(
            outputs=real_logits.sum(),
            inputs=real_image,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad_real.reshape(grad_real.shape[0], -1).pow(2).sum(dim=1).mean()

    def forward(
        self,
        real_logits: Tensor,
        fake_logits: Tensor,
        real_image: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        real_logits = real_logits.float()
        fake_logits = fake_logits.float()
        real_loss = F.softplus(-real_logits).mean()
        fake_loss_map = F.softplus(fake_logits)
        score_mask = self._score_mask(mask, fake_logits)
        if score_mask is not None and self.mask_as_fake_target:
            fake_loss_map = fake_loss_map * score_mask + F.softplus(-fake_logits) * (1.0 - score_mask)
        fake_loss = fake_loss_map.mean()
        gp = self._r1_penalty(real_logits, real_image) * self.gp_coef
        loss = real_loss + fake_loss + gp
        return loss, {
            "disc_total": loss,
            "disc_real_loss": real_loss.detach(),
            "disc_fake_loss": fake_loss.detach(),
            "disc_real_gp": gp.detach(),
            "disc_real_logit": real_logits.mean(),
            "disc_fake_logit": fake_logits.mean(),
        }
