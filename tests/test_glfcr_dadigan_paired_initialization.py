"""Tests for strict GLF-CR S1/S2 model-only paired initialization.

These tests are intentionally added without being executed in the implementation
phase.  They use tiny CPU-only modules so that the later user-run test command
does not need the ALLClear dataset or a CUDA device.
"""

from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.allclear.paired_initialization import (
    MODEL_INIT_KIND,
    build_model_init_payload,
    config_sha256,
    load_model_initialization,
    model_structure_signature,
    pair_id_for,
    save_paired_initializations,
    synchronize_shared_state,
    validate_key_sets,
    validate_paired_metadata,
)
from src.allclear import train


class _TinyModel(nn.Module):
    """Small model whose S2-only module is under the production prefix."""

    def __init__(self, with_post_filter: bool) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 3)
        self.cloud_branch = nn.Module()
        if with_post_filter:
            self.cloud_branch.post_ddin_sar_filter = nn.Module()
            self.cloud_branch.post_ddin_sar_filter.extra = nn.Linear(3, 3)
        self.tail = nn.Linear(3, 3)


def _build_tiny_pair(seed: int = 20260710) -> tuple[_TinyModel, _TinyModel, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    s1 = _TinyModel(with_post_filter=False)
    torch.manual_seed(seed)
    s2 = _TinyModel(with_post_filter=True)
    state_s1 = {key: value.detach().cpu().clone() for key, value in s1.state_dict().items()}
    state_s2 = {key: value.detach().cpu().clone() for key, value in s2.state_dict().items()}
    return s1, s2, state_s1, state_s2


class PairedInitializationTest(unittest.TestCase):
    def test_same_seed_does_not_align_later_common_parameters(self) -> None:
        _s1, _s2, state_s1, state_s2 = _build_tiny_pair()
        shared, s1_only, s2_only = validate_key_sets(state_s1, state_s2)
        self.assertEqual(s1_only, [])
        self.assertTrue(s2_only)
        self.assertTrue(all(key.startswith("cloud_branch.post_ddin_sar_filter.") for key in s2_only))
        self.assertGreater(float((state_s1["tail.weight"] - state_s2["tail.weight"]).abs().max()), 0.0)
        self.assertGreater(len(shared), 0)

    def test_synchronize_shared_state_is_exact_and_preserves_s2_only(self) -> None:
        _s1, _s2, state_s1, state_s2 = _build_tiny_pair()
        shared, s1_only, s2_only = validate_key_sets(state_s1, state_s2)
        synced, maximum = synchronize_shared_state(state_s1, state_s2, shared)
        self.assertEqual(s1_only, [])
        self.assertEqual(maximum, 0.0)
        for key in shared:
            self.assertTrue(torch.equal(state_s1[key], synced[key]))
        for key in s2_only:
            self.assertTrue(torch.equal(state_s2[key], synced[key]))

    def test_full_model_only_payloads_load_strictly(self) -> None:
        s1, s2, state_s1, state_s2 = _build_tiny_pair()
        shared, s1_only, s2_only = validate_key_sets(state_s1, state_s2)
        synced_s2, maximum = synchronize_shared_state(state_s1, state_s2, shared)
        self.assertEqual(maximum, 0.0)
        config_s1 = {"model": {"post": "none"}}
        config_s2 = {"model": {"post": "glfcr_dynamic"}}
        hash_s1 = config_sha256(config_s1)
        hash_s2 = config_sha256(config_s2)
        signature_s1 = model_structure_signature(s1)
        signature_s2 = model_structure_signature(s2)
        pair_id = pair_id_for(
            seed=20260710,
            config_sha_s1=hash_s1,
            config_sha_s2=hash_s2,
            structure_s1=signature_s1,
            structure_s2=signature_s2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload_s1 = build_model_init_payload(
                state_dict=state_s1,
                config_path=root / "s1.yaml",
                config_hash=hash_s1,
                structure_signature=signature_s1,
                seed=20260710,
                pair_id=pair_id,
                shared_key_count=len(shared),
                s1_only_keys=s1_only,
                s2_only_keys=s2_only,
                project_root=root,
            )
            payload_s2 = build_model_init_payload(
                state_dict=synced_s2,
                config_path=root / "s2.yaml",
                config_hash=hash_s2,
                structure_signature=signature_s2,
                seed=20260710,
                pair_id=pair_id,
                shared_key_count=len(shared),
                s1_only_keys=s1_only,
                s2_only_keys=s2_only,
                project_root=root,
            )
            self.assertEqual(payload_s1["kind"], MODEL_INIT_KIND)
            self.assertNotIn("optimizer", payload_s1)
            self.assertNotIn("scheduler", payload_s1)
            saved = save_paired_initializations(
                s1_payload=payload_s1,
                s2_payload=payload_s2,
                output_dir=root / "pair",
            )
            loaded_s1 = load_model_initialization(saved["s1_path"], s1, config_s1, expected_seed=20260710)
            loaded_s2 = load_model_initialization(saved["s2_path"], s2, config_s2, expected_seed=20260710)
            validate_paired_metadata(loaded_s1, loaded_s2)
            self.assertTrue(loaded_s1["strict_load"])
            self.assertTrue(loaded_s2["strict_load"])

    def test_config_and_structure_mismatch_are_rejected(self) -> None:
        s1, _s2, state_s1, _state_s2 = _build_tiny_pair()
        signature = model_structure_signature(s1)
        config = {"model": {"post": "none"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "format_version": 1,
                "kind": MODEL_INIT_KIND,
                "model": state_s1,
                "config_sha256": config_sha256(config),
                "model_structure_signature": signature,
                "pair_id": "pair",
                "seed": 20260710,
            }
            path = root / "init.pt"
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_model_initialization(path, s1, {"model": {"post": "different"}}, expected_seed=20260710)
            broken = copy.deepcopy(payload)
            broken["model_structure_signature"] = "wrong"
            torch.save(broken, path)
            with self.assertRaises(ValueError):
                load_model_initialization(path, s1, config, expected_seed=20260710)

    def test_runtime_screening_config_differences_are_allowed(self) -> None:
        s1, _s2, state_s1, _state_s2 = _build_tiny_pair()
        reference_config = {
            "run_name": "base",
            "output_root": "outputs/allclear",
            "seed": 20260710,
            "model": {"post": "none"},
            "loss": {"cloud_l1_missing": 30.0},
            "train": {"epochs": 100, "val_every": 1, "keep_best": 1},
        }
        screening_config = copy.deepcopy(reference_config)
        screening_config["run_name"] = "screen10"
        screening_config["train"]["epochs"] = 10
        invalid_config = copy.deepcopy(screening_config)
        invalid_config["loss"]["cloud_l1_missing"] = 10.0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            reference_path.write_text(json.dumps(reference_config), encoding="utf-8")
            payload = {
                "format_version": 1,
                "kind": MODEL_INIT_KIND,
                "model": state_s1,
                "config_path": str(reference_path),
                "config_sha256": config_sha256(reference_config),
                "model_structure_signature": model_structure_signature(s1),
                "pair_id": "pair",
                "seed": 20260710,
            }
            path = root / "init.pt"
            torch.save(payload, path)
            metadata = load_model_initialization(path, s1, screening_config, expected_seed=20260710)
            self.assertEqual(metadata["config_compatibility"], "runtime_screening_fields")
            self.assertEqual(
                sorted(item["path"] for item in metadata["allowed_config_differences"]),
                ["run_name", "train.epochs"],
            )
            with self.assertRaises(ValueError):
                load_model_initialization(path, s1, invalid_config, expected_seed=20260710)

    def test_missing_and_unexpected_keys_are_rejected_strictly(self) -> None:
        s1, _s2, state_s1, _state_s2 = _build_tiny_pair()
        config = {"model": {"post": "none"}}
        payload = {
            "format_version": 1,
            "kind": MODEL_INIT_KIND,
            "model": state_s1,
            "config_sha256": config_sha256(config),
            "model_structure_signature": model_structure_signature(s1),
            "pair_id": "pair",
            "seed": 20260710,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "init.pt"
            missing = copy.deepcopy(payload)
            missing["model"].pop("tail.bias")
            torch.save(missing, path)
            with self.assertRaises(RuntimeError):
                load_model_initialization(path, s1, config, expected_seed=20260710)
            unexpected = copy.deepcopy(payload)
            unexpected["model"]["unexpected.weight"] = torch.zeros(1)
            torch.save(unexpected, path)
            with self.assertRaises(RuntimeError):
                load_model_initialization(path, s1, config, expected_seed=20260710)

    def test_pair_metadata_and_overwrite_rules(self) -> None:
        meta_s1 = {
            "kind": MODEL_INIT_KIND,
            "pair_id": "pair",
            "config_sha256": "s1",
            "model_structure_signature": "sig1",
            "seed": 1,
        }
        meta_s2 = {**meta_s1, "config_sha256": "s2", "model_structure_signature": "sig2"}
        validate_paired_metadata(meta_s1, meta_s2)
        with self.assertRaises(ValueError):
            validate_paired_metadata(meta_s1, {**meta_s2, "pair_id": "other"})
        with self.assertRaises(ValueError):
            validate_paired_metadata(meta_s1, {**meta_s2, "config_sha256": "s1"})

        s1, _s2, state_s1, _state_s2 = _build_tiny_pair()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = build_model_init_payload(
                state_dict=state_s1,
                config_path=root / "s1.yaml",
                config_hash="s1",
                structure_signature=model_structure_signature(s1),
                seed=1,
                pair_id="pair",
                shared_key_count=len(state_s1),
                s1_only_keys=[],
                s2_only_keys=[],
                project_root=root,
            )
            save_paired_initializations(
                s1_payload=payload,
                s2_payload=payload,
                output_dir=root / "pair",
            )
            with self.assertRaises(FileExistsError):
                save_paired_initializations(
                    s1_payload=payload,
                    s2_payload=payload,
                    output_dir=root / "pair",
                )

    def test_paired_loader_generator_is_deterministic(self) -> None:
        config = {
            "data": {
                "root": ".",
                "train_manifest": "unused-train.json",
                "val_manifest": "unused-val.json",
            },
            "train": {"batch_size": 4, "num_workers": 0},
        }
        dataset = TensorDataset(torch.arange(24))
        with patch.object(train, "AllClearDataset", return_value=dataset):
            first_loader = train.make_loader(config, "train", paired_seed=20260710)
            second_loader = train.make_loader(config, "train", paired_seed=20260710)
        first = torch.cat([batch[0] for batch in first_loader])
        second = torch.cat([batch[0] for batch in second_loader])
        self.assertTrue(torch.equal(first, second))

        train.reset_paired_runtime_seed(7)
        first_python = random.random()
        first_torch = torch.rand(1)
        train.reset_paired_runtime_seed(7)
        self.assertEqual(first_python, random.random())
        self.assertTrue(torch.equal(first_torch, torch.rand(1)))

    def test_legacy_loader_does_not_install_paired_rng_hooks(self) -> None:
        config = {
            "data": {
                "root": ".",
                "train_manifest": "unused-train.json",
                "val_manifest": "unused-val.json",
            },
            "train": {"batch_size": 4, "num_workers": 0},
        }
        with patch.object(train, "AllClearDataset", return_value=TensorDataset(torch.arange(8))):
            loader = train.make_loader(config, "train")
        self.assertIsNone(loader.generator)
        self.assertIsNone(loader.worker_init_fn)

    def test_resume_and_init_model_flags_are_mutually_exclusive(self) -> None:
        argv = [
            "train.py",
            "--config",
            "unused.yaml",
            "--resume",
            "resume.pt",
            "--init-model-checkpoint",
            "init.pt",
        ]
        with patch("sys.argv", argv):
            with self.assertRaises(SystemExit):
                train.main()

    def test_legacy_resume_still_restores_training_state(self) -> None:
        model = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        payload = train.checkpoint_payload(model, optimizer, epoch=4, best_metric=0.25)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resume.pt"
            torch.save(payload, path)
            restored_model = nn.Linear(3, 2)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1.0e-3)
            start_epoch, best_metric = train.load_checkpoint(path, restored_model, restored_optimizer)
        self.assertEqual(start_epoch, 5)
        self.assertEqual(best_metric, 0.25)
        self.assertTrue(restored_optimizer.state_dict()["state"])
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
