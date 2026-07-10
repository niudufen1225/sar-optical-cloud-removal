#!/usr/bin/env python3
"""Download and verify OmniCloudMask pretrained weights.

This script uses OmniCloudMask's public ``predict_from_array`` API.  Running a
small dummy inference forces the package to download the requested model
weights into ``--model-dir`` and verifies that the weights can be loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("pretrained/omnicloudmask/v4"))
    parser.add_argument("--model-version", type=float, default=4.0, help="Use 4.0 for the current best public model.")
    parser.add_argument("--download-source", choices=("hugging_face", "google_drive"), default="hugging_face")
    parser.add_argument("--device", default="cpu", help="Use cpu for download-only verification, cuda for GPU verification.")
    parser.add_argument("--dtype", default="float32", choices=("float32", "fp32", "float16", "fp16", "bfloat16", "bf16"))
    parser.add_argument("--dummy-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    from omnicloudmask import predict_from_array
    from omnicloudmask.__version__ import __version__ as omnicloudmask_version
    from omnicloudmask.download_models import get_models, get_latest_model_version

    # A nonzero synthetic Red/Green/NIR scene avoids the package's nodata short
    # circuit and verifies model loading without needing a real Sentinel scene.
    h = w = int(args.dummy_size)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dummy = np.stack(
        [
            0.10 + 0.05 * xx / max(w - 1, 1),
            0.12 + 0.05 * yy / max(h - 1, 1),
            0.20 + 0.03 * (xx + yy) / max(h + w - 2, 1),
        ],
        axis=0,
    ).astype(np.float32)

    pred = predict_from_array(
        dummy,
        patch_size=int(args.patch_size),
        patch_overlap=int(args.patch_overlap),
        batch_size=int(args.batch_size),
        inference_device=args.device,
        mosaic_device="cpu",
        inference_dtype=args.dtype,
        export_confidence=False,
        no_data_value=0,
        apply_no_data_mask=True,
        destination_model_dir=args.model_dir,
        model_download_source=args.download_source,
        model_version=float(args.model_version),
    )
    models = get_models(
        force_download=False,
        model_dir=args.model_dir,
        source=args.download_source,
        model_version=float(args.model_version),
    )
    report = {
        "omnicloudmask_version": omnicloudmask_version,
        "latest_available_model_version": get_latest_model_version(),
        "requested_model_version": float(args.model_version),
        "download_source": args.download_source,
        "model_dir": str(args.model_dir.resolve()),
        "model_files": [str(item["Path"]) for item in models],
        "dummy_input_shape": list(dummy.shape),
        "dummy_prediction_shape": list(pred.shape),
        "dummy_prediction_classes": sorted(int(x) for x in np.unique(pred)),
    }
    out = args.model_dir / "download_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
