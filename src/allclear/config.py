"""Small config helpers for the ALLClear training scripts."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "PyYAML is required for .yaml configs. Install pyyaml or pass a JSON config."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else dict(data)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def make_run_dir(root: str | Path, stage: str, name: str | None = None) -> Path:
    tag = name.strip() if name else "allclear_tgdad_softshadow"
    run_dir = Path(root) / f"{timestamp()}_{stage}_{tag}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(config: dict[str, Any], run_dir: str | Path) -> None:
    path = Path(run_dir) / "config.resolved.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
