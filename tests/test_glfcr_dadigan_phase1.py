from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import Tensor, nn

from src.allclear.config import load_config
from src.allclear.model import DADIGANBaseline
from src.allclear.modules import dadigan as dadigan_module
from src.allclear.modules.dadigan import (
    DADIGANCloudBranch,
    DDIN,
    GLFCRCoupledDDIN,
    GLFCRCoupledDDINStep,
    GLFCRFusionStep,
    GLFCRPostDDINSARFilter,
)
from src.allclear.modules.lama_ffc import FFCResnetBlock, LearnableSpatialTransformWrapper
from src.allclear.modules.pvt_sra_cab import SRACAB
from src.allclear.train import build_model


ROOT = Path(__file__).resolve().parents[1]
OLD_CONFIGS = (
    ROOT / "configs/allclear_dadigan_lama_ffc_stage1_rgb_lowres_glfcr_coupled_spatial_main_v2.yaml",
    ROOT / "configs/allclear_dadigan_lama_ffc_stage1_rgb_lowres_glfcr_coupled_spatial_v3_loss_calibration.yaml",
    ROOT / "configs/allclear_dadigan_lama_ffc_stage1_rgb_lowres_glfcr_coupled_spatial_v4_structure.yaml",
)


def tiny_branch(**overrides: object) -> DADIGANCloudBranch:
    kwargs: dict[str, object] = {
        "s2_channels": 3,
        "feature_channels": (8, 16, 32, 64),
        "sar_channels": 2,
        "ddin_steps": 1,
        "prox_blocks": 1,
        "reconstruct_blocks": 1,
        "lowres_factor": 2,
        "lowres_opt_ffc_blocks": 0,
        "cab_sr_ratio": 2,
        "cab_attention_mode": "standard",
        "msab_mode": "restormer_mdta",
        "mask_input_mode": "raw",
        "output_activation": "none",
    }
    kwargs.update(overrides)
    return DADIGANCloudBranch(**kwargs)


class _ZeroAttention(nn.Module):
    def forward(self, query_feat: Tensor, reference_feat: Tensor) -> Tensor:
        del reference_feat
        b, c, h, w = query_feat.shape
        return torch.zeros(b, h * w, c, dtype=query_feat.dtype, device=query_feat.device)


class _ZeroMLP(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


class TestGLFCRDADIGANPhase1(unittest.TestCase):
    def test_old_configs_build_with_legacy_coupled_structure(self) -> None:
        for config_path in OLD_CONFIGS:
            with self.subTest(config_path=config_path.name):
                model = build_model(load_config(config_path))
                self.assertIsInstance(model, DADIGANBaseline)
                self.assertIsInstance(model.cloud_branch.ddin, GLFCRCoupledDDIN)
                self.assertTrue(model.cloud_branch.lowres_enabled)
                self.assertTrue(model.cloud_branch.ddin_glfcr_coupled)
                self.assertIsNone(model.cloud_branch.post_ddin_sar_filter)

    def test_explicit_legacy_switches_preserve_state_and_forward(self) -> None:
        kwargs = {
            "lowres_glfcr_coupled": True,
            "lowres_factor": 2,
            "lowres_opt_ffc_blocks": 0,
            "cab_sr_ratio": 2,
            "msab_mode": "restormer_mdta",
        }
        torch.manual_seed(20260710)
        legacy = tiny_branch(**kwargs).eval()
        torch.manual_seed(20260710)
        explicit = tiny_branch(
            **kwargs,
            lowres_enabled=True,
            ddin_glfcr_coupled=True,
        ).eval()

        self.assertEqual(list(legacy.state_dict()), list(explicit.state_dict()))
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)

        torch.manual_seed(20260711)
        s2 = torch.randn(1, 3, 16, 16)
        sar = torch.randn(1, 2, 16, 16)
        mask = torch.rand(1, 1, 16, 16)
        with torch.inference_mode():
            left = legacy(s2, sar, mask)
            right = explicit(s2, sar, mask)
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_lowres_and_ddin_coupling_are_independent(self) -> None:
        lowres_plain = tiny_branch(
            lowres_glfcr_coupled=False,
            lowres_enabled=True,
            ddin_glfcr_coupled=False,
        )
        self.assertIsInstance(lowres_plain.pixel_unshuffle, nn.PixelUnshuffle)
        self.assertIsInstance(lowres_plain.ddin, DDIN)
        self.assertFalse(any(isinstance(module, GLFCRCoupledDDINStep) for module in lowres_plain.modules()))
        self.assertEqual(lowres_plain.optical_stem[0].in_channels, 3 * 4)
        self.assertEqual(lowres_plain.sar_stem[0].in_channels, 2 * 4)
        self.assertTrue(any(isinstance(module, nn.PixelShuffle) for module in lowres_plain.reconstruct.modules()))

        fullres_plain = tiny_branch(
            lowres_glfcr_coupled=False,
            lowres_enabled=False,
            ddin_glfcr_coupled=False,
        )
        self.assertIsInstance(fullres_plain.pixel_unshuffle, nn.Identity)
        self.assertIsInstance(fullres_plain.ddin, DDIN)
        self.assertEqual(fullres_plain.optical_stem[0].in_channels, 3)
        self.assertEqual(fullres_plain.sar_stem[0].in_channels, 2)
        self.assertFalse(any(isinstance(module, nn.PixelShuffle) for module in fullres_plain.reconstruct.modules()))

    def test_cab2_query_mode_is_default_and_reference_mode_uses_fm_base(self) -> None:
        torch.manual_seed(20260710)
        default = dadigan_module.PDAFMScale(
            8,
            heads=1,
            cab_sr_ratio=2,
            cab_attention_mode="standard",
            msab_mode="restormer_mdta",
        ).eval()
        torch.manual_seed(20260710)
        explicit_query = dadigan_module.PDAFMScale(
            8,
            heads=1,
            cab_sr_ratio=2,
            cab_attention_mode="standard",
            msab_mode="restormer_mdta",
            cab2_residual_source="query",
            cab2_update_scale=1.0,
        ).eval()
        self.assertEqual(list(default.state_dict()), list(explicit_query.state_dict()))
        explicit_query.load_state_dict(default.state_dict())
        shared = torch.randn(1, 8, 8, 8)
        optical = torch.randn(1, 8, 8, 8)
        sar = torch.randn(1, 8, 8, 8)
        with torch.inference_mode():
            self.assertTrue(torch.equal(default(shared, optical, sar), explicit_query(shared, optical, sar)))

        cab = SRACAB(8, heads=1, sr_ratio=2, attention_mode="standard").eval()
        cab.attn = _ZeroAttention()
        cab.mlp = _ZeroMLP()
        query = torch.randn(1, 8, 8, 8)
        fm = torch.randn(1, 8, 8, 8)
        with torch.inference_mode():
            output = cab(query, fm, residual_base=fm, update_scale=0.25)
        self.assertTrue(torch.equal(output, fm))

    def test_cab2_reference_mode_keeps_sar_query_gradient(self) -> None:
        torch.manual_seed(20260710)
        cab = SRACAB(8, heads=1, sr_ratio=2, attention_mode="standard")
        query = torch.randn(1, 8, 8, 8, requires_grad=True)
        fm = torch.randn(1, 8, 8, 8)
        output = cab(query, fm, residual_base=fm, update_scale=0.25)
        output.square().mean().backward()
        self.assertIsNotNone(query.grad)
        assert query.grad is not None
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertGreater(query.grad.abs().sum().item(), 0.0)
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(output.dtype, query.dtype)
        self.assertEqual(output.device, query.device)
        self.assertFalse(any(name == "update_scale" for name, _ in cab.named_parameters()))

    def test_post_ddin_filter_is_one_shot_and_does_not_modify_optical(self) -> None:
        for kernel_size in (3, 5):
            with self.subTest(kernel_size=kernel_size):
                branch = tiny_branch(
                    lowres_glfcr_coupled=False,
                    lowres_enabled=True,
                    ddin_glfcr_coupled=False,
                    post_ddin_sar_filter="glfcr_dynamic",
                    post_ddin_sar_filter_kernel_size=kernel_size,
                ).eval()
                self.assertIsInstance(branch.post_ddin_sar_filter, GLFCRPostDDINSARFilter)
                assert branch.post_ddin_sar_filter is not None
                self.assertFalse(
                    any(isinstance(module, GLFCRFusionStep) for module in branch.post_ddin_sar_filter.modules())
                )

                dfg_calls = 0
                conv_calls = 0

                def on_dfg(*_args: object) -> None:
                    nonlocal dfg_calls
                    dfg_calls += 1

                original_conv = dadigan_module._glfcr_kernel2d_conv

                def counted_conv(*args: object, **kwargs: object) -> Tensor:
                    nonlocal conv_calls
                    conv_calls += 1
                    return original_conv(*args, **kwargs)

                handle = branch.post_ddin_sar_filter.dynamic_filter.register_forward_hook(
                    lambda *args: on_dfg(*args)
                )
                dadigan_module._glfcr_kernel2d_conv = counted_conv
                try:
                    s2 = torch.randn(1, 3, 16, 16)
                    sar = torch.randn(1, 2, 16, 16)
                    mask = torch.rand(1, 1, 16, 16)
                    with torch.inference_mode():
                        outputs = branch(s2, sar, mask)
                finally:
                    dadigan_module._glfcr_kernel2d_conv = original_conv
                    handle.remove()
                self.assertEqual(dfg_calls, 1)
                self.assertEqual(conv_calls, 1)
                self.assertEqual(outputs["sar_private_s1"].shape, outputs["opt_private_s1"].shape)

                opt = torch.randn(1, 8, 8, 8, requires_grad=True)
                sar_private = torch.randn(1, 8, 8, 8, requires_grad=True)
                opt_before = opt.detach().clone()
                filtered = branch.post_ddin_sar_filter(opt, sar_private)
                self.assertEqual(filtered.shape, sar_private.shape)
                self.assertTrue(torch.equal(opt, opt_before))
                filtered.square().mean().backward()
                self.assertTrue(torch.isfinite(filtered).all())
                self.assertIsNotNone(opt.grad)
                self.assertIsNotNone(sar_private.grad)
                assert opt.grad is not None and sar_private.grad is not None
                self.assertTrue(torch.isfinite(opt.grad).all())
                self.assertTrue(torch.isfinite(sar_private.grad).all())

    def test_post_ddin_none_has_no_filter_module(self) -> None:
        branch = tiny_branch(lowres_enabled=True, ddin_glfcr_coupled=False, post_ddin_sar_filter="none")
        self.assertIsNone(branch.post_ddin_sar_filter)
        self.assertFalse(any(isinstance(module, GLFCRPostDDINSARFilter) for module in branch.modules()))

    def test_post_ddin_kernel_size_null_falls_back_to_legacy_kernel(self) -> None:
        branch = tiny_branch(
            lowres_glfcr_kernel_size=3,
            post_ddin_sar_filter="glfcr_dynamic",
            post_ddin_sar_filter_kernel_size=None,
        )
        self.assertIsNotNone(branch.post_ddin_sar_filter)
        assert branch.post_ddin_sar_filter is not None
        self.assertEqual(branch.post_ddin_sar_filter.kernel_size, 3)

    def test_smoke_forward_backward_has_finite_outputs_and_gradients(self) -> None:
        branch = tiny_branch(lowres_enabled=True, ddin_glfcr_coupled=False).train()
        s2 = torch.randn(1, 3, 16, 16, requires_grad=True)
        sar = torch.randn(1, 2, 16, 16, requires_grad=True)
        mask = torch.rand(1, 1, 16, 16)
        outputs = branch(s2, sar, mask)
        scalar = sum(value.square().mean() for value in outputs.values() if isinstance(value, Tensor))
        scalar.backward()
        self.assertTrue(torch.isfinite(scalar))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values() if isinstance(value, Tensor)))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in branch.parameters()
            )
        )
        self.assertTrue(any(parameter.grad is not None for parameter in branch.parameters()))

    def test_s1_shape_has_no_glfcr_ffc_or_spatial_modules(self) -> None:
        branch = tiny_branch(
            lowres_enabled=True,
            ddin_glfcr_coupled=False,
            lowres_opt_ffc_blocks=0,
            bottleneck_context="none",
            post_ddin_sar_filter="none",
        )
        self.assertIsInstance(branch.ddin, DDIN)
        self.assertFalse(any(isinstance(module, GLFCRCoupledDDINStep) for module in branch.modules()))
        self.assertFalse(any(isinstance(module, GLFCRPostDDINSARFilter) for module in branch.modules()))
        self.assertFalse(any(isinstance(module, FFCResnetBlock) for module in branch.modules()))
        self.assertFalse(any(isinstance(module, LearnableSpatialTransformWrapper) for module in branch.modules()))


if __name__ == "__main__":
    unittest.main()
