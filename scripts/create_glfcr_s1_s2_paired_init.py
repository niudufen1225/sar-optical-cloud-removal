#!/usr/bin/env python3
"""Create strict, full model-only paired initializations for Phase 3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inspect_glfcr_s1_s2_configs import config_differences
from src.allclear.config import load_config
from src.allclear.paired_initialization import (
    build_model_init_payload,
    config_sha256,
    model_structure_signature,
    pair_id_for,
    save_paired_initializations,
    synchronize_shared_state,
    validate_key_sets,
)
from src.allclear.train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-config", required=True, type=Path)
    parser.add_argument("--s2-config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_s1 = load_config(args.s1_config)
    config_s2 = load_config(args.s2_config)
    differences = config_differences(config_s1, config_s2)
    disallowed = {key: value for key, value in differences.items() if not value["allowed"]}
    if disallowed:
        raise ValueError(
            "S1/S2 configs differ outside the two post-DDIN fields:\n"
            + json.dumps(disallowed, indent=2, ensure_ascii=False)
        )
    device = torch.device(args.device)

    torch.manual_seed(int(args.seed))
    model_s1 = build_model(config_s1).to(device)
    torch.manual_seed(int(args.seed))
    model_s2 = build_model(config_s2).to(device)
    state_s1 = {key: value.detach().cpu().clone() for key, value in model_s1.state_dict().items()}
    state_s2 = {key: value.detach().cpu().clone() for key, value in model_s2.state_dict().items()}
    shared_keys, s1_only_keys, s2_only_keys = validate_key_sets(state_s1, state_s2)
    synced_s2, raw_global_max = synchronize_shared_state(state_s1, state_s2, shared_keys)
    if raw_global_max < 0.0:  # pragma: no cover - defensive invariant
        raise RuntimeError("Invalid negative initialization difference")
    synced_global_max = 0.0
    for key in shared_keys:
        difference = (state_s1[key].float() - synced_s2[key].float()).abs()
        if difference.numel():
            synced_global_max = max(synced_global_max, float(difference.max().item()))
    if synced_global_max != 0.0:
        raise RuntimeError(f"Shared initialization synchronization failed: max_abs_diff={synced_global_max}")

    structure_s1 = model_structure_signature(model_s1)
    structure_s2 = model_structure_signature(model_s2)
    config_hash_s1 = config_sha256(config_s1)
    config_hash_s2 = config_sha256(config_s2)
    pair_id = pair_id_for(
        seed=int(args.seed),
        config_sha_s1=config_hash_s1,
        config_sha_s2=config_hash_s2,
        structure_s1=structure_s1,
        structure_s2=structure_s2,
    )
    payload_s1 = build_model_init_payload(
        state_dict=state_s1,
        config_path=args.s1_config,
        config_hash=config_hash_s1,
        structure_signature=structure_s1,
        seed=int(args.seed),
        pair_id=pair_id,
        shared_key_count=len(shared_keys),
        s1_only_keys=s1_only_keys,
        s2_only_keys=s2_only_keys,
        project_root=PROJECT_ROOT,
    )
    payload_s2 = build_model_init_payload(
        state_dict=synced_s2,
        config_path=args.s2_config,
        config_hash=config_hash_s2,
        structure_signature=structure_s2,
        seed=int(args.seed),
        pair_id=pair_id,
        shared_key_count=len(shared_keys),
        s1_only_keys=s1_only_keys,
        s2_only_keys=s2_only_keys,
        project_root=PROJECT_ROOT,
    )
    result = save_paired_initializations(
        s1_payload=payload_s1,
        s2_payload=payload_s2,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    result.update(
        {
            "pair_id": pair_id,
            "seed": int(args.seed),
            "shared_key_count": len(shared_keys),
            "s1_only_keys": s1_only_keys,
            "s2_only_keys": s2_only_keys,
            "raw_shared_global_max_abs_diff": raw_global_max,
            "shared_global_max_abs_diff": synced_global_max,
            "config_sha256_s1": config_hash_s1,
            "config_sha256_s2": config_hash_s2,
            "model_structure_signature_s1": structure_s1,
            "model_structure_signature_s2": structure_s2,
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
