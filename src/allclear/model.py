"""Main Stage1 ALLClear model — three-candidate cloud/shadow restoration."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.allclear.modules.common import RegionMasks, masks_from_cld_shdw
from src.allclear.modules.dadigan import DADIGANCloudBranch
from src.allclear.modules.lama_cloud import UFFCCloudBranch
from src.allclear.modules.softshadow import ExternalSoftShadowSAM, SoftShadowBranch


class AllClearTGDADSoftShadow(nn.Module):
    """Stage-1 three-candidate restoration model.

    Three independent branches produce candidate images for clear, shadow
    and cloud regions.  A hard-mask composition stitches them into one image.
    """

    def __init__(
        self,
        s2_channels: int = 13,
        sar_channels: int = 2,
        dim: int = 48,
        shadow_backend: str = "conv",
        shadow_removal_backend: str | None = None,
        shadow_hidden_channels: int | None = None,
        shadow_restormer_hidden_channels: int = 64,
        shadow_restormer_blocks: int = 2,
        shadow_restormer_heads: int = 4,
        shadow_nafnet_hidden_channels: int | None = None,
        shadow_nafnet_blocks: int = 3,
        softshadow_repo: str | None = None,
        sam_checkpoint: str | None = None,
        softshadow_checkpoint: str | None = None,
        softshadow_sam_model_type: str = "vit_h",
        softshadow_sam_lora_rank: int = 8,
        softshadow_sam_lora_layers: object | None = None,
        softshadow_sam_input_size: int = 1024,
        softshadow_sam_checkpoint_blocks: bool = False,
        softshadow_bbox_space: str = "image",
        softshadow_efficientvit_repo: str | None = None,
        softshadow_efficientvit_checkpoint: str | None = None,
        softshadow_efficientvit_model: str = "efficientvit-sam-xl0",
        softshadow_efficientvit_adapter_rank: int = 8,
        softshadow_efficientvit_adapter_layers: object | None = None,
        softshadow_efficientvit_train_mask_decoder: bool = True,
        softshadow_efficientvit_force_fp32: bool = False,
        softshadow_use_hard_support_gate: bool = True,
        softshadow_forward_valid_only: bool = False,
        rgb_indices: tuple[int, int, int] = (3, 2, 1),
        shadow_index: int = 3,
        cloud_index: int = 1,
        cloud_backend: str = "dadigan",
        cloud_bottleneck_context: str = "none",
        cloud_pre_pda_context: str = "none",
        cloud_pre_pda_ffc_blocks: int = 0,
        cloud_pre_pda_ffc_ratio: float = 0.75,
        cloud_pre_pda_ffc_enable_lfu: bool = False,
        cloud_pre_pda_ffc_downsample: int = 4,
        cloud_pre_pda_ffc_residual_scale: float = 0.05,
        cloud_prefusion_context: str = "none",
        cloud_prefusion_blocks: int = 0,
        cloud_prefusion_kernel_size: int = 5,
        cloud_prefusion_reduction: int = 16,
        cloud_lowres_glfcr_coupled: bool = False,
        cloud_lowres_enabled: bool | None = None,
        cloud_ddin_glfcr_coupled: bool | None = None,
        cloud_lowres_factor: int = 2,
        cloud_lowres_opt_ffc_blocks: int = 0,
        cloud_lowres_opt_ffc_ratio: float = 0.75,
        cloud_lowres_opt_ffc_enable_lfu: bool = False,
        cloud_lowres_opt_ffc_spatial_transform_layers: tuple[int, ...] | None = None,
        cloud_lowres_opt_ffc_spatial_transform_pad_coef: float = 0.5,
        cloud_lowres_opt_ffc_spatial_transform_angle_init_range: float = 80.0,
        cloud_lowres_opt_ffc_spatial_transform_train_angle: bool = True,
        cloud_lowres_glfcr_kernel_size: int = 5,
        cloud_ddin_steps: int = 3,
        cloud_prox_blocks: int = 2,
        cloud_reconstruct_blocks: int = 2,
        cloud_cab_sr_ratio: int = 8,
        cloud_cab_attention_mode: str = "standard",
        cloud_msab_mode: str = "efficient",
        cloud_cab2_residual_source: str = "query",
        cloud_cab2_update_scale: float = 1.0,
        cloud_post_ddin_sar_filter: str = "none",
        cloud_post_ddin_sar_filter_kernel_size: int | None = None,
        cloud_ffc_blocks: int = 0,
        cloud_ffc_blocks_per_scale: tuple[int, ...] | None = None,
        cloud_mask_input_mode: str = "raw",
        cloud_append_mask: bool = False,
        cloud_mask_fill_value: float = 0.0,
        cloud_output_activation: str = "none",
        cloud_ffc_ratio: float = 0.75,
        cloud_ffc_enable_lfu: bool = False,
        cloud_ffc_downsample: int = 1,
        cloud_ffc_downsamples: tuple[int, ...] | None = None,
        cloud_ffc_residual_scale: float = 0.1,
        cloud_ffc_residual_scales: tuple[float, ...] | None = None,
        cloud_ffc_spatial_transform_layers: tuple[int, ...] | None = None,
        cloud_ffc_spatial_transform_pad_coef: float = 0.5,
        cloud_ffc_spatial_transform_angle_init_range: float = 80.0,
        cloud_ffc_spatial_transform_train_angle: bool = True,
        cloud_lama_ngf: int = 64,
        cloud_lama_downs: int = 3,
        cloud_lama_blocks: int = 9,
        cloud_lama_pretrained: str | None = None,
        cloud_lama_use_sar: bool = False,
        cloud_lama_mask_input: bool = True,
        cloud_lama_enable_lfu: bool = False,
    ) -> None:
        super().__init__()
        self.s2_channels = int(s2_channels)
        self.shadow_index = int(shadow_index)
        self.cloud_index = int(cloud_index)
        self.feature_channels = (dim, dim * 2, dim * 4, dim * 8)

        self.shadow_branch = SoftShadowBranch(
            channels=s2_channels,
            backend=shadow_backend,
            removal_backend=shadow_removal_backend,
            hidden_channels=int(shadow_hidden_channels or dim * 2),
            softshadow_repo=softshadow_repo,
            sam_checkpoint=sam_checkpoint,
            softshadow_checkpoint=softshadow_checkpoint,
            rgb_indices=rgb_indices,
            sam_model_type=softshadow_sam_model_type,
            sam_lora_rank=softshadow_sam_lora_rank,
            sam_lora_layers=softshadow_sam_lora_layers,
            sam_input_size=softshadow_sam_input_size,
            sam_checkpoint_blocks=softshadow_sam_checkpoint_blocks,
            sam_bbox_space=softshadow_bbox_space,
            efficientvit_repo=softshadow_efficientvit_repo,
            efficientvit_checkpoint=softshadow_efficientvit_checkpoint,
            efficientvit_model=softshadow_efficientvit_model,
            efficientvit_adapter_rank=softshadow_efficientvit_adapter_rank,
            efficientvit_adapter_layers=softshadow_efficientvit_adapter_layers,
            efficientvit_train_mask_decoder=softshadow_efficientvit_train_mask_decoder,
            efficientvit_force_fp32=softshadow_efficientvit_force_fp32,
            use_hard_support_gate=softshadow_use_hard_support_gate,
            forward_valid_only=softshadow_forward_valid_only,
            restormer_hidden_channels=shadow_restormer_hidden_channels,
            restormer_blocks=shadow_restormer_blocks,
            restormer_heads=shadow_restormer_heads,
            nafnet_hidden_channels=shadow_nafnet_hidden_channels,
            nafnet_blocks=shadow_nafnet_blocks,
        )
        if cloud_backend == "uffc":
            self.cloud_branch = UFFCCloudBranch(
                s2_channels=s2_channels, sar_channels=sar_channels,
                ngf=cloud_lama_ngf, n_downsampling=cloud_lama_downs,
                n_blocks=cloud_lama_blocks,
                ratio_gin=cloud_ffc_ratio, ratio_gout=cloud_ffc_ratio,
                enable_lfu=bool(cloud_lama_enable_lfu),
                pretrained_checkpoint=cloud_lama_pretrained,
                use_sar=cloud_lama_use_sar,
                mask_cloud_input=cloud_lama_mask_input,
                rgb_indices=rgb_indices,
            )
        else:
            self.cloud_branch = DADIGANCloudBranch(
                s2_channels=s2_channels,
                feature_channels=self.feature_channels,
                sar_channels=sar_channels,
                ddin_steps=cloud_ddin_steps,
                prox_blocks=cloud_prox_blocks,
                reconstruct_blocks=cloud_reconstruct_blocks,
                bottleneck_context=cloud_bottleneck_context,
                pre_pda_context=cloud_pre_pda_context,
                pre_pda_ffc_blocks=cloud_pre_pda_ffc_blocks,
                pre_pda_ffc_ratio=cloud_pre_pda_ffc_ratio,
                pre_pda_ffc_enable_lfu=cloud_pre_pda_ffc_enable_lfu,
                pre_pda_ffc_downsample=cloud_pre_pda_ffc_downsample,
                pre_pda_ffc_residual_scale=cloud_pre_pda_ffc_residual_scale,
                prefusion_context=cloud_prefusion_context,
                prefusion_blocks=cloud_prefusion_blocks,
                prefusion_kernel_size=cloud_prefusion_kernel_size,
                prefusion_reduction=cloud_prefusion_reduction,
                lowres_glfcr_coupled=cloud_lowres_glfcr_coupled,
                lowres_enabled=cloud_lowres_enabled,
                ddin_glfcr_coupled=cloud_ddin_glfcr_coupled,
                lowres_factor=cloud_lowres_factor,
                lowres_opt_ffc_blocks=cloud_lowres_opt_ffc_blocks,
                lowres_opt_ffc_ratio=cloud_lowres_opt_ffc_ratio,
                lowres_opt_ffc_enable_lfu=cloud_lowres_opt_ffc_enable_lfu,
                lowres_opt_ffc_spatial_transform_layers=cloud_lowres_opt_ffc_spatial_transform_layers,
                lowres_opt_ffc_spatial_transform_pad_coef=cloud_lowres_opt_ffc_spatial_transform_pad_coef,
                lowres_opt_ffc_spatial_transform_angle_init_range=cloud_lowres_opt_ffc_spatial_transform_angle_init_range,
                lowres_opt_ffc_spatial_transform_train_angle=cloud_lowres_opt_ffc_spatial_transform_train_angle,
                lowres_glfcr_kernel_size=cloud_lowres_glfcr_kernel_size,
                cab_sr_ratio=cloud_cab_sr_ratio,
                cab_attention_mode=cloud_cab_attention_mode,
                msab_mode=cloud_msab_mode,
                cab2_residual_source=cloud_cab2_residual_source,
                cab2_update_scale=cloud_cab2_update_scale,
                post_ddin_sar_filter=cloud_post_ddin_sar_filter,
                post_ddin_sar_filter_kernel_size=cloud_post_ddin_sar_filter_kernel_size,
                ffc_blocks=cloud_ffc_blocks,
                ffc_blocks_per_scale=cloud_ffc_blocks_per_scale,
                ffc_ratio=cloud_ffc_ratio,
                ffc_enable_lfu=cloud_ffc_enable_lfu,
                ffc_downsample=cloud_ffc_downsample,
                ffc_downsamples=cloud_ffc_downsamples,
                ffc_residual_scale=cloud_ffc_residual_scale,
                ffc_residual_scales=cloud_ffc_residual_scales,
                ffc_spatial_transform_layers=cloud_ffc_spatial_transform_layers,
                ffc_spatial_transform_pad_coef=cloud_ffc_spatial_transform_pad_coef,
                ffc_spatial_transform_angle_init_range=cloud_ffc_spatial_transform_angle_init_range,
                ffc_spatial_transform_train_angle=cloud_ffc_spatial_transform_train_angle,
                mask_input_mode=cloud_mask_input_mode,
                append_cloud_mask=cloud_append_mask,
                mask_fill_value=cloud_mask_fill_value,
                output_activation=cloud_output_activation,
            )

    def _masks(self, cld_shdw: Tensor) -> RegionMasks:
        return masks_from_cld_shdw(cld_shdw, shadow_index=self.shadow_index, cloud_index=self.cloud_index)

    def forward(
        self,
        s2_toa: Tensor,
        s1: Tensor | None,
        cld_shdw: Tensor,
        softshadow_bbox: Tensor | None = None,
        softshadow_case: Tensor | None = None,
        return_intermediates: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        masks = self._masks(cld_shdw)

        i_clear = s2_toa.float()
        shadow = self.shadow_branch(s2_toa, masks.shadow, bbox=softshadow_bbox, shadow_case=softshadow_case)
        cloud = self.cloud_branch(s2_toa, s1, masks.cloud)
        i_shadow = shadow["I_shadow"]
        i_cloud = cloud["I_cloud"]
        i_cloud_raw = cloud["I_cloud_raw"]

        output = masks.clear * i_clear + masks.shadow * i_shadow + masks.cloud * i_cloud_raw

        if not return_intermediates:
            return output
        return {
            "I_hat": output,
            "I_stage1": output,
            "I_clear": i_clear,
            "I_shadow": i_shadow,
            "I_cloud": i_cloud,
            "I_cloud_raw": i_cloud_raw,
            "M_clear": masks.clear,
            "M_shadow": masks.shadow,
            "M_cloud": masks.cloud,
            "M_shadow_soft": shadow["M_shadow_soft"],
            "M_shadow_soft_raw": shadow.get("M_shadow_soft_raw", shadow["M_shadow_soft"]),
            "M_shadow_soft_eff": shadow.get("M_shadow_soft_eff", shadow["M_shadow_soft"]),
            "M_shadow_case_valid": (
                (softshadow_case.to(device=s2_toa.device).long().view(-1, 1, 1, 1) == 1).to(dtype=s2_toa.dtype)
                if softshadow_case is not None
                else torch.ones((s2_toa.shape[0], 1, 1, 1), device=s2_toa.device, dtype=s2_toa.dtype)
            ),
            "F_cloud": cloud.get("F_cloud", i_cloud_raw),
            **{k: v for k, v in cloud.items() if k.startswith("M_glfcr_") or k.startswith("M_sagate_")},
        }


class DADIGANBaseline(nn.Module):
    """Strict full-image DADIGAN baseline for ALLClear.

    This wrapper disables the Stage1 clear/shadow routing and trains one
    DADIGAN generator to produce the complete 13-channel cloud-free image
    G(m, SAR).  It still returns the common intermediate keys so the existing
    trainer, discriminator, visualizer, and loss code can be reused with
    full-image DADIGAN loss settings.
    """

    def __init__(
        self,
        s2_channels: int = 13,
        sar_channels: int = 2,
        dim: int = 64,
        shadow_index: int = 3,
        cloud_index: int = 1,
        cloud_ddin_steps: int = 3,
        cloud_prox_blocks: int = 2,
        cloud_reconstruct_blocks: int = 2,
        cloud_bottleneck_context: str = "none",
        cloud_pre_pda_context: str = "none",
        cloud_pre_pda_ffc_blocks: int = 0,
        cloud_pre_pda_ffc_ratio: float = 0.75,
        cloud_pre_pda_ffc_enable_lfu: bool = False,
        cloud_pre_pda_ffc_downsample: int = 4,
        cloud_pre_pda_ffc_residual_scale: float = 0.05,
        cloud_prefusion_context: str = "none",
        cloud_prefusion_blocks: int = 0,
        cloud_prefusion_kernel_size: int = 5,
        cloud_prefusion_reduction: int = 16,
        cloud_lowres_glfcr_coupled: bool = False,
        cloud_lowres_enabled: bool | None = None,
        cloud_ddin_glfcr_coupled: bool | None = None,
        cloud_lowres_factor: int = 2,
        cloud_lowres_opt_ffc_blocks: int = 0,
        cloud_lowres_opt_ffc_ratio: float = 0.75,
        cloud_lowres_opt_ffc_enable_lfu: bool = False,
        cloud_lowres_opt_ffc_spatial_transform_layers: tuple[int, ...] | None = None,
        cloud_lowres_opt_ffc_spatial_transform_pad_coef: float = 0.5,
        cloud_lowres_opt_ffc_spatial_transform_angle_init_range: float = 80.0,
        cloud_lowres_opt_ffc_spatial_transform_train_angle: bool = True,
        cloud_lowres_glfcr_kernel_size: int = 5,
        cloud_cab_sr_ratio: int = 8,
        cloud_cab_attention_mode: str = "standard",
        cloud_msab_mode: str = "efficient",
        cloud_cab2_residual_source: str = "query",
        cloud_cab2_update_scale: float = 1.0,
        cloud_post_ddin_sar_filter: str = "none",
        cloud_post_ddin_sar_filter_kernel_size: int | None = None,
        cloud_ffc_blocks: int = 0,
        cloud_ffc_blocks_per_scale: tuple[int, ...] | None = None,
        cloud_ffc_ratio: float = 0.75,
        cloud_ffc_enable_lfu: bool = False,
        cloud_ffc_downsample: int = 1,
        cloud_ffc_downsamples: tuple[int, ...] | None = None,
        cloud_ffc_residual_scale: float = 0.1,
        cloud_ffc_residual_scales: tuple[float, ...] | None = None,
        cloud_ffc_spatial_transform_layers: tuple[int, ...] | None = None,
        cloud_ffc_spatial_transform_pad_coef: float = 0.5,
        cloud_ffc_spatial_transform_angle_init_range: float = 80.0,
        cloud_ffc_spatial_transform_train_angle: bool = True,
        cloud_mask_input_mode: str = "raw",
        cloud_append_mask: bool = False,
        cloud_mask_fill_value: float = 0.0,
        cloud_output_activation: str = "none",
        baseline_mask_mode: str = "full",
        baseline_output_mode: str = "raw",
    ) -> None:
        super().__init__()
        self.shadow_index = int(shadow_index)
        self.cloud_index = int(cloud_index)
        self.baseline_mask_mode = str(baseline_mask_mode).lower()
        self.baseline_output_mode = str(baseline_output_mode).lower()
        if self.baseline_mask_mode not in {"full", "cloud", "shadow", "degraded", "cloud_shadow", "none"}:
            raise ValueError("baseline_mask_mode must be one of: full, cloud, shadow, degraded, cloud_shadow, none")
        if self.baseline_output_mode not in {"raw", "composite"}:
            raise ValueError("baseline_output_mode must be one of: raw, composite")
        self.cloud_branch = DADIGANCloudBranch(
            s2_channels=s2_channels,
            feature_channels=(dim, dim * 2, dim * 4, dim * 8),
            sar_channels=sar_channels,
            ddin_steps=cloud_ddin_steps,
            prox_blocks=cloud_prox_blocks,
            reconstruct_blocks=cloud_reconstruct_blocks,
            bottleneck_context=cloud_bottleneck_context,
            pre_pda_context=cloud_pre_pda_context,
            pre_pda_ffc_blocks=cloud_pre_pda_ffc_blocks,
            pre_pda_ffc_ratio=cloud_pre_pda_ffc_ratio,
            pre_pda_ffc_enable_lfu=cloud_pre_pda_ffc_enable_lfu,
            pre_pda_ffc_downsample=cloud_pre_pda_ffc_downsample,
            pre_pda_ffc_residual_scale=cloud_pre_pda_ffc_residual_scale,
            prefusion_context=cloud_prefusion_context,
            prefusion_blocks=cloud_prefusion_blocks,
            prefusion_kernel_size=cloud_prefusion_kernel_size,
            prefusion_reduction=cloud_prefusion_reduction,
            lowres_glfcr_coupled=cloud_lowres_glfcr_coupled,
            lowres_enabled=cloud_lowres_enabled,
            ddin_glfcr_coupled=cloud_ddin_glfcr_coupled,
            lowres_factor=cloud_lowres_factor,
            lowres_opt_ffc_blocks=cloud_lowres_opt_ffc_blocks,
            lowres_opt_ffc_ratio=cloud_lowres_opt_ffc_ratio,
            lowres_opt_ffc_enable_lfu=cloud_lowres_opt_ffc_enable_lfu,
            lowres_opt_ffc_spatial_transform_layers=cloud_lowres_opt_ffc_spatial_transform_layers,
            lowres_opt_ffc_spatial_transform_pad_coef=cloud_lowres_opt_ffc_spatial_transform_pad_coef,
            lowres_opt_ffc_spatial_transform_angle_init_range=cloud_lowres_opt_ffc_spatial_transform_angle_init_range,
            lowres_opt_ffc_spatial_transform_train_angle=cloud_lowres_opt_ffc_spatial_transform_train_angle,
            lowres_glfcr_kernel_size=cloud_lowres_glfcr_kernel_size,
            cab_sr_ratio=cloud_cab_sr_ratio,
            cab_attention_mode=cloud_cab_attention_mode,
            msab_mode=cloud_msab_mode,
            cab2_residual_source=cloud_cab2_residual_source,
            cab2_update_scale=cloud_cab2_update_scale,
            post_ddin_sar_filter=cloud_post_ddin_sar_filter,
            post_ddin_sar_filter_kernel_size=cloud_post_ddin_sar_filter_kernel_size,
            ffc_blocks=cloud_ffc_blocks,
            ffc_blocks_per_scale=cloud_ffc_blocks_per_scale,
            ffc_ratio=cloud_ffc_ratio,
            ffc_enable_lfu=cloud_ffc_enable_lfu,
            ffc_downsample=cloud_ffc_downsample,
            ffc_downsamples=cloud_ffc_downsamples,
            ffc_residual_scale=cloud_ffc_residual_scale,
            ffc_residual_scales=cloud_ffc_residual_scales,
            ffc_spatial_transform_layers=cloud_ffc_spatial_transform_layers,
            ffc_spatial_transform_pad_coef=cloud_ffc_spatial_transform_pad_coef,
            ffc_spatial_transform_angle_init_range=cloud_ffc_spatial_transform_angle_init_range,
            ffc_spatial_transform_train_angle=cloud_ffc_spatial_transform_train_angle,
            mask_input_mode=cloud_mask_input_mode,
            append_cloud_mask=cloud_append_mask,
            mask_fill_value=cloud_mask_fill_value,
            output_activation=cloud_output_activation,
        )

    def _baseline_mask(self, s2_toa: Tensor, cld_shdw: Tensor) -> tuple[Tensor, RegionMasks]:
        masks = masks_from_cld_shdw(cld_shdw, shadow_index=self.shadow_index, cloud_index=self.cloud_index)
        full = torch.ones((s2_toa.shape[0], 1, s2_toa.shape[-2], s2_toa.shape[-1]), device=s2_toa.device, dtype=s2_toa.dtype)
        zero = torch.zeros_like(full)
        if self.baseline_mask_mode == "full":
            return full, masks
        if self.baseline_mask_mode == "none":
            return zero, masks
        if self.baseline_mask_mode == "cloud":
            return masks.cloud.to(device=s2_toa.device, dtype=s2_toa.dtype), masks
        if self.baseline_mask_mode == "shadow":
            return masks.shadow.to(device=s2_toa.device, dtype=s2_toa.dtype), masks
        return (masks.cloud + masks.shadow).clamp(0.0, 1.0).to(device=s2_toa.device, dtype=s2_toa.dtype), masks

    def forward(
        self,
        s2_toa: Tensor,
        s1: Tensor | None,
        cld_shdw: Tensor,
        softshadow_bbox: Tensor | None = None,
        softshadow_case: Tensor | None = None,
        return_intermediates: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        del softshadow_bbox, softshadow_case
        mask, masks = self._baseline_mask(s2_toa, cld_shdw)
        cloud = self.cloud_branch(s2_toa, s1, mask)
        pred = cloud["I_cloud"] if self.baseline_output_mode == "composite" else cloud["I_cloud_raw"]
        if not return_intermediates:
            return pred
        zero_mask = torch.zeros_like(mask)
        return {
            "I_hat": pred,
            "I_stage1": pred,
            "I_clear": s2_toa.float(),
            "I_shadow": s2_toa.float(),
            "I_cloud": cloud["I_cloud"],
            "I_cloud_raw": cloud["I_cloud_raw"],
            "M_clear": zero_mask,
            "M_shadow": masks.shadow.to(device=s2_toa.device, dtype=s2_toa.dtype),
            "M_cloud": mask,
            "M_cloud_vis": mask if self.baseline_mask_mode != "full" else masks.cloud,
            "M_shadow_soft": zero_mask,
            "F_cloud": cloud.get("F_cloud", pred),
            "shared_s1": cloud.get("shared_s1", pred),
            "opt_private_s1": cloud.get("opt_private_s1", pred),
            "sar_private_s1": cloud.get("sar_private_s1", pred),
            **{k: v for k, v in cloud.items() if k.startswith("M_glfcr_") or k.startswith("M_sagate_")},
        }


class SoftShadowDADIGANBaseline(nn.Module):
    """SoftShadow SAM-LoRA mask learner with a DADIGAN restoration backbone.

    SoftShadow's official SAM-LoRA branch operates on RGB images and bbox
    prompts. DADIGAN keeps the multispectral/SAR restoration path. The two
    parts are joined by a single restoration mask:

        M_restore = M_cloud OR M_shadow_soft

    This preserves SoftShadow's soft-mask supervision while avoiding a fragile
    13-channel SAM adaptation.
    """

    def __init__(
        self,
        s2_channels: int = 13,
        sar_channels: int = 2,
        dim: int = 64,
        shadow_index: int = 3,
        cloud_index: int = 1,
        rgb_indices: tuple[int, int, int] = (3, 2, 1),
        softshadow_repo: str | None = None,
        sam_checkpoint: str | None = None,
        softshadow_checkpoint: str | None = None,
        softshadow_sam_model_type: str = "vit_h",
        softshadow_sam_lora_rank: int = 8,
        softshadow_sam_lora_layers: object | None = None,
        softshadow_sam_input_size: int = 1024,
        softshadow_sam_checkpoint_blocks: bool = True,
        softshadow_bbox_space: str = "image",
        softshadow_use_hard_support_gate: bool = True,
        softshadow_forward_valid_only: bool = False,
        restore_mask_mode: str = "cloud_plus_soft_shadow",
        cloud_ddin_steps: int = 3,
        cloud_prox_blocks: int = 5,
        cloud_reconstruct_blocks: int = 2,
        cloud_cab_sr_ratio: int = 8,
        cloud_cab_attention_mode: str = "standard",
        cloud_msab_mode: str = "restormer_mdta",
        cloud_mask_input_mode: str = "learned",
        cloud_append_mask: bool = True,
        cloud_mask_fill_value: float = 0.0,
        cloud_output_activation: str = "none",
    ) -> None:
        super().__init__()
        if softshadow_repo is None or sam_checkpoint is None:
            raise ValueError("softshadow_dadigan_baseline requires softshadow_repo and sam_checkpoint")
        self.shadow_index = int(shadow_index)
        self.cloud_index = int(cloud_index)
        self.rgb_indices = tuple(rgb_indices)
        self.restore_mask_mode = str(restore_mask_mode).lower()
        self.softshadow_use_hard_support_gate = bool(softshadow_use_hard_support_gate)
        self.softshadow_forward_valid_only = bool(softshadow_forward_valid_only)
        if self.restore_mask_mode not in {"cloud_plus_soft_shadow", "soft_shadow", "cloud", "hard_degraded"}:
            raise ValueError("restore_mask_mode must be one of: cloud_plus_soft_shadow, soft_shadow, cloud, hard_degraded")
        self.softshadow_mask = ExternalSoftShadowSAM(
            softshadow_repo=softshadow_repo,
            sam_checkpoint=sam_checkpoint,
            model_type=str(softshadow_sam_model_type),
            rank=int(softshadow_sam_lora_rank),
            input_size=int(softshadow_sam_input_size),
            checkpoint_blocks=bool(softshadow_sam_checkpoint_blocks),
            lora_layers=softshadow_sam_lora_layers,
        )
        self.softshadow_bbox_space = str(softshadow_bbox_space)
        if softshadow_checkpoint:
            self._load_softshadow_checkpoint(softshadow_checkpoint)
        self.cloud_branch = DADIGANCloudBranch(
            s2_channels=s2_channels,
            feature_channels=(dim, dim * 2, dim * 4, dim * 8),
            sar_channels=sar_channels,
            ddin_steps=cloud_ddin_steps,
            prox_blocks=cloud_prox_blocks,
            reconstruct_blocks=cloud_reconstruct_blocks,
            bottleneck_context="none",
            cab_sr_ratio=cloud_cab_sr_ratio,
            cab_attention_mode=cloud_cab_attention_mode,
            msab_mode=cloud_msab_mode,
            mask_input_mode=cloud_mask_input_mode,
            append_cloud_mask=cloud_append_mask,
            mask_fill_value=cloud_mask_fill_value,
            output_activation=cloud_output_activation,
        )

    def _softshadow_param_anchor(self, ref: Tensor) -> Tensor:
        if not self.training:
            return ref.new_zeros(())
        anchor = ref.new_zeros(())
        for param in self.softshadow_mask.parameters():
            if param.requires_grad:
                anchor = anchor + param.float().sum() * 0.0
        return anchor

    def _predict_softshadow_raw(self, s2_toa: Tensor, hard_shadow: Tensor, bbox: Tensor | None) -> Tensor:
        rgb = s2_toa[:, list(self.rgb_indices)].float()
        soft_mask = self.softshadow_mask(rgb, hard_shadow, bbox=bbox, bbox_space=self.softshadow_bbox_space)
        if soft_mask.shape[-2:] != s2_toa.shape[-2:]:
            soft_mask = torch.nn.functional.interpolate(
                soft_mask,
                size=s2_toa.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return soft_mask.clamp(0.0, 1.0)

    def _masks(self, cld_shdw: Tensor) -> RegionMasks:
        return masks_from_cld_shdw(cld_shdw, shadow_index=self.shadow_index, cloud_index=self.cloud_index)

    def _load_softshadow_checkpoint(self, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported SoftShadow checkpoint format: {checkpoint}")
        own = self.softshadow_mask.state_dict()
        filtered = {}
        for key, value in state.items():
            clean = key.replace("module.", "").replace("samshadow.", "").replace("lora.", "sam_lora.")
            if clean in own and own[clean].shape == value.shape:
                filtered[clean] = value
        self.softshadow_mask.load_state_dict(filtered, strict=False)

    def _soft_shadow_mask(
        self,
        s2_toa: Tensor,
        hard_shadow: Tensor,
        bbox: Tensor | None = None,
        shadow_case: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        has_shadow = (hard_shadow.flatten(1).sum(dim=1) > 0).to(dtype=s2_toa.dtype).view(-1, 1, 1, 1)
        if shadow_case is None and self.softshadow_use_hard_support_gate and has_shadow.sum().item() < 1.0:
            zero = hard_shadow.new_zeros((hard_shadow.shape[0], 1, s2_toa.shape[-2], s2_toa.shape[-1]))
            return zero, zero
        case_gate = None
        if shadow_case is not None:
            case_gate = (shadow_case.to(device=s2_toa.device).long().view(-1, 1, 1, 1) == 1).to(dtype=s2_toa.dtype)
        if self.softshadow_forward_valid_only and case_gate is not None:
            valid = case_gate.view(-1) > 0.5
            soft_mask = hard_shadow.new_zeros((hard_shadow.shape[0], 1, s2_toa.shape[-2], s2_toa.shape[-1]))
            soft_mask = soft_mask + self._softshadow_param_anchor(s2_toa)
            if valid.any():
                bbox_valid = bbox[valid] if bbox is not None else None
                soft_valid = self._predict_softshadow_raw(s2_toa[valid], hard_shadow[valid], bbox_valid)
                soft_mask = soft_mask.clone()
                soft_mask[valid] = soft_valid
        else:
            soft_mask = self._predict_softshadow_raw(s2_toa, hard_shadow, bbox)
        if self.softshadow_use_hard_support_gate:
            soft_mask = soft_mask * has_shadow
        if case_gate is not None:
            soft_mask = soft_mask * case_gate
        # Keep the official SoftShadow semantics: the soft mask is learned from
        # the division target and bbox prompt.  ALLClear hard-shadow labels are
        # not used to crop it, otherwise the branch inherits hard mask edges.
        effective = soft_mask.clamp(0.0, 1.0)
        return soft_mask.clamp(0.0, 1.0), effective

    def _restore_mask(self, masks: RegionMasks, soft_shadow: Tensor) -> Tensor:
        cloud = masks.cloud.to(device=soft_shadow.device, dtype=soft_shadow.dtype)
        hard_shadow = masks.shadow.to(device=soft_shadow.device, dtype=soft_shadow.dtype)
        if self.restore_mask_mode == "soft_shadow":
            return soft_shadow
        if self.restore_mask_mode == "cloud":
            return cloud
        if self.restore_mask_mode == "hard_degraded":
            return (cloud + hard_shadow).clamp(0.0, 1.0)
        return (cloud + (1.0 - cloud) * soft_shadow).clamp(0.0, 1.0)

    def forward(
        self,
        s2_toa: Tensor,
        s1: Tensor | None,
        cld_shdw: Tensor,
        softshadow_bbox: Tensor | None = None,
        softshadow_case: Tensor | None = None,
        return_intermediates: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        masks = self._masks(cld_shdw)
        soft_shadow_raw, soft_shadow = self._soft_shadow_mask(
            s2_toa,
            masks.shadow,
            bbox=softshadow_bbox,
            shadow_case=softshadow_case,
        )
        restore_mask = self._restore_mask(masks, soft_shadow)
        cloud = self.cloud_branch(s2_toa, s1, restore_mask)
        pred = cloud["I_cloud"]
        if not return_intermediates:
            return pred
        return {
            "I_hat": pred,
            "I_stage1": pred,
            "I_clear": s2_toa.float(),
            "I_shadow": pred,
            "I_cloud": cloud["I_cloud"],
            "I_cloud_raw": cloud["I_cloud_raw"],
            "M_clear": (1.0 - restore_mask).clamp(0.0, 1.0),
            "M_shadow": masks.shadow,
            "M_cloud": restore_mask,
            "M_cloud_vis": masks.cloud,
            "M_restore": restore_mask,
            "M_shadow_soft": soft_shadow,
            "M_shadow_soft_raw": soft_shadow_raw,
            "M_shadow_soft_eff": soft_shadow,
            "M_shadow_case_valid": (
                (softshadow_case.to(device=s2_toa.device).long().view(-1, 1, 1, 1) == 1).to(dtype=s2_toa.dtype)
                if softshadow_case is not None
                else torch.ones((s2_toa.shape[0], 1, 1, 1), device=s2_toa.device, dtype=s2_toa.dtype)
            ),
            "F_cloud": cloud.get("F_cloud", pred),
            "shared_s1": cloud.get("shared_s1", pred),
            "opt_private_s1": cloud.get("opt_private_s1", pred),
            "sar_private_s1": cloud.get("sar_private_s1", pred),
        }
