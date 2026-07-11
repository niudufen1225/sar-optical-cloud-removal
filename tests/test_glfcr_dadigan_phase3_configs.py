from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.allclear.config import load_config
from src.allclear.train import build_model
from scripts.inspect_glfcr_s1_s2_configs import (
    config_differences,
    inspect_model,
    make_smoke_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
S1_CONFIG = ROOT / "configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml"
S2_CONFIG = ROOT / "configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml"
OLD_PHASE1_TEST = ROOT / "tests/test_glfcr_dadigan_phase1.py"


class TestGLFCRDADIGANPhase3Configs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = load_config(S1_CONFIG)
        cls.s2 = load_config(S2_CONFIG)

    def test_s1_s2_yaml_parse_and_only_post_filter_differs(self) -> None:
        differences = config_differences(self.s1, self.s2)
        self.assertEqual(set(differences), {
            "model.cloud_post_ddin_sar_filter",
            "model.cloud_post_ddin_sar_filter_kernel_size",
        })
        self.assertEqual(self.s1["model"]["cloud_post_ddin_sar_filter"], "none")
        self.assertIsNone(self.s1["model"]["cloud_post_ddin_sar_filter_kernel_size"])
        self.assertEqual(self.s2["model"]["cloud_post_ddin_sar_filter"], "glfcr_dynamic")
        self.assertEqual(self.s2["model"]["cloud_post_ddin_sar_filter_kernel_size"], 5)

    def test_shared_clean_core_configuration(self) -> None:
        for config in (self.s1, self.s2):
            model = config["model"]
            self.assertTrue(model["cloud_lowres_enabled"])
            self.assertFalse(model["cloud_ddin_glfcr_coupled"])
            self.assertEqual(model["cloud_ddin_steps"], 3)
            self.assertEqual(model["cloud_prox_blocks"], 2)
            self.assertEqual(model["cloud_reconstruct_blocks"], 2)
            self.assertEqual(model["cloud_cab_attention_mode"], "standard")
            self.assertEqual(model["cloud_cab2_residual_source"], "reference")
            self.assertEqual(model["cloud_cab2_update_scale"], 0.1)
            self.assertEqual(model["cloud_lowres_opt_ffc_blocks"], 0)
            self.assertEqual(model["cloud_bottleneck_context"], "none")
            self.assertEqual(model["cloud_ffc_blocks"], 0)
            self.assertEqual(model["cloud_ffc_spatial_transform_layers"], [])
            self.assertEqual(config["loss"]["cloud_kl"], 0.0)
            self.assertEqual(config["loss"]["cloud_adv"], 0.0)
            self.assertEqual(config["loss"]["feature_matching"], 0.0)
            self.assertEqual(config["loss"]["perceptual"], 0.0)

    def test_model_structure_counts(self) -> None:
        s1_stats = inspect_model(build_model(self.s1))
        s2_stats = inspect_model(build_model(self.s2))
        self.assertEqual(s1_stats["pixel_unshuffle_count"], 1)
        self.assertEqual(s1_stats["pixel_shuffle_count"], 1)
        self.assertEqual(s1_stats["ddin_type"], "DDIN")
        self.assertEqual(s1_stats["ordinary_ddin_step_count"], 3)
        self.assertEqual(s1_stats["glfcr_coupled_ddin_step_count"], 0)
        self.assertEqual(s1_stats["glfcr_fusion_step_count"], 0)
        self.assertEqual(s1_stats["post_ddin_dynamic_filter_generator_count"], 0)
        self.assertEqual(s1_stats["ffc_block_count"], 0)
        self.assertEqual(s1_stats["spatial_wrapper_count"], 0)
        self.assertEqual(s2_stats["post_ddin_dynamic_filter_generator_count"], 1)
        self.assertEqual(s2_stats["glfcr_coupled_ddin_step_count"], 0)
        self.assertEqual(s2_stats["glfcr_fusion_step_count"], 0)

    def test_cab2_and_no_forbidden_modules(self) -> None:
        for config in (self.s1, self.s2):
            stats = inspect_model(build_model(config))
            self.assertEqual(stats["cab_attention_mode"], "standard")
            self.assertEqual(stats["cab2_residual_source"], "reference")
            self.assertEqual(stats["cab2_update_scale"], 0.1)
            self.assertEqual(stats["ffc_block_count"], 0)
            self.assertEqual(stats["spatial_wrapper_count"], 0)

    def test_smoke_inputs_match_dadigan_interface(self) -> None:
        tensors = make_smoke_inputs(
            device=torch.device("cpu"),
            seed=2026,
            batch_size=2,
            height=16,
            width=16,
        )
        self.assertEqual(tensors["s2"].shape, (2, 3, 16, 16))
        self.assertEqual(tensors["sar"].shape, (2, 2, 16, 16))
        self.assertEqual(tensors["cld_shdw"].shape, (2, 4, 16, 16))

    def test_phase1_compatibility_test_remains_present(self) -> None:
        self.assertTrue(OLD_PHASE1_TEST.exists())
        source = OLD_PHASE1_TEST.read_text(encoding="utf-8")
        self.assertIn("test_old_configs_build_with_legacy_coupled_structure", source)


if __name__ == "__main__":
    unittest.main()
