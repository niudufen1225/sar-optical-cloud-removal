from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.evaluate_allclear_runs import paired_comparison
from src.allclear.eval_metrics import (
    channel_bias_stats,
    haar_swt2,
    paired_bootstrap,
    region_ssim,
    sar_counterfactuals,
    wavelet_metrics,
)


class _TinyCheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, s2: torch.Tensor, sar: torch.Tensor | None, cld_shdw: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        del sar, cld_shdw
        return {"I_hat": self.conv(s2)}


class TestGLFCRDADIGANEvaluatorPhase2(unittest.TestCase):
    def test_haar_constant_direction_checkerboard_and_identity(self) -> None:
        constant = torch.ones(1, 1, 8, 8, dtype=torch.float32)
        bands = haar_swt2(constant)
        for name in ("LH", "HL", "HH"):
            self.assertLess(float(bands[name].abs().max()), 1.0e-5)

        vertical = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        vertical_bands = haar_swt2(vertical)
        self.assertGreater(float(vertical_bands["LH"].square().mean()), float(vertical_bands["HL"].square().mean()) * 10.0)

        horizontal = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        horizontal_bands = haar_swt2(horizontal)
        self.assertGreater(float(horizontal_bands["HL"].square().mean()), float(horizontal_bands["LH"].square().mean()) * 10.0)

        checker = (torch.arange(8).view(1, 1, 8, 1) + torch.arange(8).view(1, 1, 1, 8)).remainder(2).float()
        checker_bands = haar_swt2(checker)
        self.assertGreater(float(checker_bands["HH"].square().mean()), float(checker_bands["LH"].square().mean()) + 1.0e-8)

        errors = wavelet_metrics(constant, constant)
        for value in errors.values():
            self.assertAlmostEqual(float(value), 0.0, places=6)

    def test_haar_cpu_gpu_fp32_finite(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=str(device)):
                x = torch.randn(1, 3, 8, 8, device=device, dtype=torch.float32)
                bands = haar_swt2(x)
                self.assertTrue(all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in bands.values()))

    def test_mask_metrics_rgb_bias_and_small_region_rule(self) -> None:
        target = torch.zeros(1, 3, 8, 8)
        pred = target.clone()
        pred[:, 0] += 0.1
        pred[:, 1] -= 0.2
        pred[:, 2] += 0.3
        full = torch.ones(1, 1, 8, 8)
        bias = channel_bias_stats(pred, target, full, (0, 1, 2))
        self.assertAlmostEqual(bias["bias_r"], 0.1, places=6)
        self.assertAlmostEqual(bias["bias_g"], -0.2, places=6)
        self.assertAlmostEqual(bias["bias_b"], 0.3, places=6)
        self.assertAlmostEqual(bias["mean_abs_channel_bias"], 0.2, places=6)
        self.assertTrue(math.isfinite(region_ssim(pred, target, full, rgb_indices=(0, 1, 2))))
        tiny = torch.zeros(1, 1, 8, 8)
        tiny[:, :, 0:2, 0:2] = 1.0
        self.assertTrue(math.isnan(region_ssim(pred, target, tiny, rgb_indices=(0, 1, 2))))

    def test_sar_counterfactuals_are_deterministic_and_unrenormalized(self) -> None:
        sar = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8) / 100.0
        first = sar_counterfactuals(sar, low_pass_kernel=5)
        second = sar_counterfactuals(sar, low_pass_kernel=5)
        for key in ("real", "zero", "shuffle", "low_pass", "high_pass"):
            self.assertTrue(torch.equal(first[key], second[key]))  # type: ignore[arg-type]
        self.assertTrue(torch.equal(first["real"], sar))  # type: ignore[arg-type]
        self.assertTrue(torch.equal(first["zero"], torch.zeros_like(sar)))  # type: ignore[arg-type]
        self.assertTrue(torch.allclose(first["low_pass"] + first["high_pass"], sar))  # type: ignore[operator]
        self.assertFalse(bool(first["shuffle_valid"]))

        batch = torch.cat([sar, sar + 1.0], dim=0)
        shuffled = sar_counterfactuals(batch, low_pass_kernel=5)
        self.assertTrue(bool(shuffled["shuffle_valid"]))
        self.assertTrue(torch.equal(shuffled["shuffle"][0], batch[1]))  # type: ignore[index]

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        kwargs = {"higher_is_better": False, "resamples": 2000, "seed": 20260710}
        first = paired_bootstrap([1.0, 2.0, 3.0], [0.5, 1.5, 2.5], **kwargs)
        second = paired_bootstrap([1.0, 2.0, 3.0], [0.5, 1.5, 2.5], **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["n_valid"], 3)
        self.assertEqual(first["direction"], "improved")
        self.assertEqual(first["mean_delta"], -0.5)

    def test_paired_comparison_joins_by_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv1 = root / "s1.csv"
            csv2 = root / "s2.csv"
            csv1.write_text("sample_id,final_full_mae,final_full_psnr\na,0.4,10\nb,0.2,20\n", encoding="utf-8")
            csv2.write_text("sample_id,final_full_mae,final_full_psnr\nb,0.1,21\na,0.3,11\n", encoding="utf-8")
            comparison = paired_comparison(
                csv1,
                csv2,
                run1=root / "run1",
                run2=root / "run2",
                split="test",
                resamples=2000,
                seed=7,
            )
            self.assertEqual(comparison["shared_samples"], 2)
            self.assertEqual(comparison["metrics"]["final_full_mae"]["n_valid"], 2)
            self.assertEqual(comparison["metrics"]["final_full_psnr"]["higher_is_better"], True)

    def test_tiny_model_checkpoint_like_smoke(self) -> None:
        model = _TinyCheckpointModel().eval()
        s2 = torch.rand(1, 3, 8, 8)
        sar = torch.rand(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        with torch.inference_mode():
            output = model(s2, sar, mask)["I_hat"]
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(math.isfinite(region_ssim(output, output, torch.ones_like(mask), rgb_indices=(0, 1, 2))))
        self.assertTrue(all(math.isfinite(value) for value in wavelet_metrics(output, output).values()))


if __name__ == "__main__":
    unittest.main()
