"""ALLClear dataset loader for s2_toa / optional s1 / cld_shdw experiments.

The official ALLClear ``cld_shdw`` TIFF products are five-channel masks:
channel 0 is cloud probability, channel 1 is the binary cloud mask, and
channels 2/3/4 are binary shadow masks generated with different dark-pixel
thresholds.  The current Stage1 configs use channel 1 for cloud routing and
channel 3 for shadow routing.
"""

from __future__ import annotations

import csv
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from src.allclear.modules.common import masks_from_cld_shdw


SHADOW_CASE_NO_SHADOW = 0
SHADOW_CASE_VALID_SHADOW = 1
SHADOW_CASE_AMBIGUOUS = 2
SHADOW_CASE_NAMES = {
    SHADOW_CASE_NO_SHADOW: "no_shadow",
    SHADOW_CASE_VALID_SHADOW: "valid_shadow",
    SHADOW_CASE_AMBIGUOUS: "ambiguous",
}


def _read_array(path: Path) -> Tensor:
    """Read .pt/.npy/.npz/.tif/.tiff/.png into CHW float tensor."""

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("array", "tensor", "data", "s2_toa", "s1", "cld_shdw", "target"):
                if key in obj:
                    obj = obj[key]
                    break
        if not isinstance(obj, torch.Tensor):
            obj = torch.as_tensor(obj)
        return _to_chw(obj.float())
    if suffix in {".npy", ".npz"}:
        import numpy as np  # type: ignore

        obj = np.load(path)
        if suffix == ".npz":
            key = sorted(obj.files)[0]
            obj = obj[key]
        return _to_chw(torch.from_numpy(obj).float())
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile  # type: ignore

            arr = tifffile.imread(path)
            return _to_chw(torch.as_tensor(arr.copy()).float())
        except Exception:
            try:
                import rasterio  # type: ignore

                with rasterio.open(path) as ds:
                    return torch.from_numpy(ds.read()).float()
            except Exception as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(f"Could not read TIFF {path}; install tifffile or rasterio") from exc
    if suffix == ".png":
        try:
            from PIL import Image  # type: ignore

            import numpy as np  # type: ignore

            arr = np.asarray(Image.open(path))
            return _to_chw(torch.as_tensor(arr.copy()).float())
        except Exception:
            return _read_png_grayscale(path).float()
    raise ValueError(f"Unsupported file suffix for {path}")


def _read_png_grayscale(path: Path) -> Tensor:
    """Minimal PNG reader for ALLClear 8-bit grayscale cld_shdw masks."""

    raw = path.read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Not a PNG file: {path}")
    pos = 8
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        chunk_type = raw[pos + 4 : pos + 8]
        data = raw[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", data)
        elif chunk_type == b"IDAT":
            compressed.extend(data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type != 0:
        raise RuntimeError(f"PNG fallback only supports 8-bit grayscale masks: {path}")
    payload = zlib.decompress(bytes(compressed))
    stride = int(width)
    rows = []
    prev = [0] * stride
    i = 0
    for _ in range(int(height)):
        filt = payload[i]
        i += 1
        scan = list(payload[i : i + stride])
        i += stride
        recon = [0] * stride
        for x, value in enumerate(scan):
            left = recon[x - 1] if x > 0 else 0
            up = prev[x]
            up_left = prev[x - 1] if x > 0 else 0
            if filt == 0:
                px = value
            elif filt == 1:
                px = value + left
            elif filt == 2:
                px = value + up
            elif filt == 3:
                px = value + ((left + up) >> 1)
            elif filt == 4:
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                pred = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                px = value + pred
            else:
                raise RuntimeError(f"Unsupported PNG filter {filt} in {path}")
            recon[x] = px & 255
        rows.append(recon)
        prev = recon
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


def _to_chw(x: Tensor) -> Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got {tuple(x.shape)}")
    # Prefer CHW when first dimension is sensor-band-like.  Otherwise HWC -> CHW.
    if x.shape[0] <= 32:
        return x.contiguous()
    return x.permute(2, 0, 1).contiguous()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _derive_cld_shdw_from_s2(path: Path) -> Path | None:
    parts = list(path.parts)
    try:
        idx = parts.index("s2_toa")
    except ValueError:
        return None
    parts[idx] = "cld_shdw"
    derived = Path(*parts)
    derived = derived.with_name(derived.name.replace("_s2_toa_", "_cld_shdw_")).with_suffix(".tif")
    return derived


def _normalize_mask(mask: Tensor) -> Tensor:
    """Normalize ALLClear probability masks while preserving categorical maps."""

    mask = torch.nan_to_num(mask.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if mask.numel() == 0:
        return mask
    max_value = float(mask.max())
    if mask.shape[0] > 1:
        out = mask.clone()
        for channel in range(out.shape[0]):
            channel_max = float(out[channel].max())
            if channel_max > 1.0:
                denom = 100.0 if channel_max <= 100.0 else 255.0
                out[channel] = out[channel] / denom
        return out.clamp(0.0, 1.0)
    if mask.shape[0] == 1 and max_value > 4.0:
        denom = 100.0 if max_value <= 100.0 else 255.0
        return (mask / denom).clamp(0.0, 1.0)
    return mask


def _normalize_optical(optical: Tensor, optical_scale: float) -> Tensor:
    optical = torch.nan_to_num(optical.float(), nan=0.0, posinf=float(optical_scale), neginf=0.0)
    return optical.clamp(0.0, float(optical_scale)) / float(optical_scale)


class AllClearDataset(Dataset[dict[str, Any]]):
    """Manifest-driven loader.

    Required manifest columns can use either canonical or common alias names:

    - cloudy optical: ``s2_toa`` / ``s2_toa_path`` / ``cloudy_path`` / ``s2_cloudy_path`` / ``cloudy_s2_path``
    - target optical: ``target`` / ``target_path`` / ``clear_path`` / ``s2_clear_path`` / ``clear_s2_path``
    - SAR: ``s1`` / ``s1_path`` / ``sar_path`` / ``sar_s1_path`` when ``load_sar=True``
    - masks: ``cld_shdw`` / ``cld_shdw_path`` / ``mask_path`` / ``cloudy_mask_path``

    Preprocessing follows DADIGAN (He et al., 2025, Information Fusion) Section 4.1:

    - Optical (S2 TOA): clip [0, 10000], then divide by *optical_scale* (10000) → [0, 1].
    - SAR (S1 GRD, dB): per-channel clip and min–max normalise to [0, 1].
      VV: clip [-25, 0] dB → (x + 25) / 25
      VH: clip [-32, 0] dB → (x + 32) / 32
    """

    CLOUDY_KEYS = ("s2_toa", "s2_toa_path", "cloudy_path", "s2_cloudy_path", "cloudy_s2_path")
    TARGET_KEYS = ("target", "target_path", "clear_path", "s2_clear_path", "clear_s2_path")
    S1_KEYS = ("s1", "s1_path", "sar_path", "sar_s1_path")
    MASK_KEYS = ("cld_shdw", "cld_shdw_path", "mask_path", "cloudy_mask_path")
    SOFTSHADOW_MASK_KEYS = ("sam_mask", "sam_mask_path", "division_mask", "division_mask_path", "softshadow_mask_path")
    SOFTSHADOW_BBOX_KEYS = ("bbox", "sam_bbox", "softshadow_bbox")

    # DADIGAN per-channel SAR dB clipping ranges (Section 4.1)
    SAR_VV_CLIP = (-25.0, 0.0)
    SAR_VH_CLIP = (-32.0, 0.0)

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        optical_scale: float = 10000.0,
        image_size: int | None = None,
        shadow_index: int = 3,
        cloud_index: int = 1,
        prefer_original_cld_shdw: bool = True,
        load_sar: bool = True,
        cache_dir: str | Path | None = None,
        band_indices: tuple[int, ...] | list[int] | None = None,
        softshadow_mask_dir: str | Path | None = None,
        softshadow_bbox_path: str | Path | None = None,
        softshadow_bbox_space: str = "image",
        softshadow_sam_input_size: int = 1024,
        softshadow_shadow_case_enabled: bool = False,
        softshadow_shadow_case_positive_threshold: float = 0.05,
        softshadow_absent_shadow_threshold: float = 0.002,
        softshadow_absent_division_threshold: float = 0.002,
        softshadow_valid_shadow_threshold: float = 0.002,
        softshadow_valid_division_threshold: float = 0.005,
        softshadow_max_bbox_area: float = 0.95,
        softshadow_max_clear_leakage: float = 0.30,
        softshadow_max_cloud_leakage: float = 0.30,
        softshadow_min_division_shadow_precision_dilated: float = 0.30,
    ) -> None:
        self.root = Path(root)
        self.manifest = _resolve(self.root, str(manifest))
        self.optical_scale = float(optical_scale)
        self.image_size = image_size
        self.shadow_index = int(shadow_index)
        self.cloud_index = int(cloud_index)
        self.prefer_original_cld_shdw = bool(prefer_original_cld_shdw)
        self.load_sar = bool(load_sar)
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.band_indices = tuple(int(i) for i in band_indices) if band_indices is not None else None
        self.softshadow_mask_dir = Path(softshadow_mask_dir).expanduser().resolve() if softshadow_mask_dir else None
        self.softshadow_bbox_path = Path(softshadow_bbox_path).expanduser().resolve() if softshadow_bbox_path else None
        self.softshadow_bbox_data = self._load_bbox_yaml(self.softshadow_bbox_path) if self.softshadow_bbox_path else {}
        self.softshadow_bbox_space = str(softshadow_bbox_space).lower()
        self.softshadow_sam_input_size = int(softshadow_sam_input_size)
        self.softshadow_shadow_case_enabled = bool(softshadow_shadow_case_enabled)
        self.softshadow_shadow_case_positive_threshold = float(softshadow_shadow_case_positive_threshold)
        self.softshadow_absent_shadow_threshold = float(softshadow_absent_shadow_threshold)
        self.softshadow_absent_division_threshold = float(softshadow_absent_division_threshold)
        self.softshadow_valid_shadow_threshold = float(softshadow_valid_shadow_threshold)
        self.softshadow_valid_division_threshold = float(softshadow_valid_division_threshold)
        self.softshadow_max_bbox_area = float(softshadow_max_bbox_area)
        self.softshadow_max_clear_leakage = float(softshadow_max_clear_leakage)
        self.softshadow_max_cloud_leakage = float(softshadow_max_cloud_leakage)
        self.softshadow_min_division_shadow_precision_dilated = float(
            softshadow_min_division_shadow_precision_dilated
        )
        if self.softshadow_bbox_space not in {"image", "sam_input"}:
            raise ValueError("softshadow_bbox_space must be 'image' or 'sam_input'")
        if not self.manifest.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest}")
        with self.manifest.open("r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _value(row: dict[str, str], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = row.get(key, "")
            if value:
                return value
        raise KeyError(f"Manifest row is missing any of columns: {keys}")

    def _load(self, row: dict[str, str], keys: tuple[str, ...]) -> Tensor:
        return _read_array(_resolve(self.root, self._value(row, keys)))

    @staticmethod
    def _load_bbox_yaml(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        if not path.exists():
            raise FileNotFoundError(f"Missing SoftShadow bbox file: {path}")
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency error
            raise RuntimeError("Reading SoftShadow bbox YAML requires PyYAML") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _load_mask(self, row: dict[str, str]) -> Tensor:
        if self.prefer_original_cld_shdw:
            for key in self.CLOUDY_KEYS:
                value = row.get(key, "")
                if not value:
                    continue
                derived = _derive_cld_shdw_from_s2(_resolve(self.root, value))
                if derived is not None and derived.exists():
                    return _read_array(derived)
        return self._load(row, self.MASK_KEYS)

    def _sample_keys(self, row: dict[str, str], sample_id: str) -> list[str]:
        keys = [sample_id]
        for column in (*self.CLOUDY_KEYS, *self.TARGET_KEYS):
            value = row.get(column, "")
            if not value:
                continue
            path = Path(value)
            keys.extend([path.stem, path.name])
        seen = set()
        out = []
        for key in keys:
            if key and key not in seen:
                out.append(key)
                seen.add(key)
        return out

    def _find_softshadow_mask_path(self, row: dict[str, str], sample_id: str) -> Path | None:
        for key in self.SOFTSHADOW_MASK_KEYS:
            value = row.get(key, "")
            if value:
                path = _resolve(self.root, value)
                if path.exists():
                    return path
                raise FileNotFoundError(f"Missing SoftShadow division mask from column {key}: {path}")
        if self.softshadow_mask_dir is None:
            return None
        suffixes = (".png", ".tif", ".tiff", ".pt", ".pth", ".npy", ".npz")
        for key in self._sample_keys(row, sample_id):
            candidate = self.softshadow_mask_dir / key
            if candidate.exists() and candidate.is_file():
                return candidate
            for suffix in suffixes:
                candidate = self.softshadow_mask_dir / f"{Path(key).stem}{suffix}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"Could not find SoftShadow division mask for sample_id={sample_id!r} in {self.softshadow_mask_dir}"
        )

    def _load_softshadow_mask(self, row: dict[str, str], sample_id: str) -> Tensor | None:
        path = self._find_softshadow_mask_path(row, sample_id)
        if path is None:
            return None
        mask = _normalize_mask(self._resize(_read_array(path), mode="bilinear"))
        if mask.shape[0] > 1:
            mask = mask[:1]
        return mask.float().clamp(0.0, 1.0)

    @staticmethod
    def _parse_bbox_value(value: Any) -> Tensor | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = [float(part) for part in text.replace(";", ",").split(",") if part.strip()]
        bbox = torch.as_tensor(value, dtype=torch.float32)
        if bbox.numel() < 4:
            return None
        bbox = bbox.reshape(-1, 4)[0]
        return bbox

    def _load_softshadow_bbox(self, row: dict[str, str], sample_id: str) -> Tensor | None:
        for key in self.SOFTSHADOW_BBOX_KEYS:
            bbox = self._parse_bbox_value(row.get(key, ""))
            if bbox is not None:
                return bbox
        for key in self._sample_keys(row, sample_id):
            if key in self.softshadow_bbox_data:
                bbox = self._parse_bbox_value(self.softshadow_bbox_data[key])
                if bbox is not None:
                    return bbox
            stem = Path(key).stem
            if stem in self.softshadow_bbox_data:
                bbox = self._parse_bbox_value(self.softshadow_bbox_data[stem])
                if bbox is not None:
                    return bbox
        return None

    def _cache_path(self, sample_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_id = sample_id.replace("/", "_")
        return self.cache_dir / f"{safe_id}.pt"

    def _load_cache(self, sample_id: str) -> dict[str, Any] | None:
        # Manifest-level replacement masks, such as OmniCloudMask outputs, must
        # not be bypassed by older .pt caches that already contain cld_shdw.
        if not self.prefer_original_cld_shdw:
            return None
        path = self._cache_path(sample_id)
        if path is None or not path.exists():
            return None
        item = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(item, dict):
            return None
        required = {"s2_toa", "target", "cld_shdw"}
        if not required.issubset(item):
            return None
        if self.load_sar and "s1" not in item:
            return None
        s2_toa = item["s2_toa"].float()
        target = item["target"].float()
        if self.band_indices is not None:
            s2_toa = self._select_bands_from_cached(s2_toa)
            target = self._select_bands_from_cached(target)
            if s2_toa is None or target is None:
                return None
        if self.softshadow_mask_dir is not None and "sam_mask" not in item:
            return None
        if self.softshadow_bbox_path is not None and "bbox" not in item:
            return None
        out: dict[str, Any] = {
            "s2_toa": s2_toa,
            "target": target,
            "cld_shdw": item["cld_shdw"].float(),
            "sample_id": item.get("sample_id", sample_id),
        }
        if self.load_sar:
            out["s1"] = item["s1"].float()
        if "sam_mask" in item:
            out["sam_mask"] = item["sam_mask"].float()
        if "bbox" in item:
            out["bbox"] = item["bbox"].float()
        return self._attach_shadow_case(out)

    def _bbox_area_fraction(self, bbox: Tensor | None, h: int, w: int) -> tuple[bool, float]:
        if bbox is None:
            return False, 0.0
        box = bbox.float().reshape(-1, 4)[0].clone()
        if not torch.isfinite(box).all():
            return False, 0.0
        if self.softshadow_bbox_space == "sam_input":
            box[[0, 2]] *= float(w) / float(self.softshadow_sam_input_size)
            box[[1, 3]] *= float(h) / float(self.softshadow_sam_input_size)
        box[[0, 2]] = box[[0, 2]].clamp(0.0, float(w - 1))
        box[[1, 3]] = box[[1, 3]].clamp(0.0, float(h - 1))
        x1, y1, x2, y2 = [float(v.item()) for v in box]
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
        area = ((x2 - x1 + 1.0) * (y2 - y1 + 1.0)) / max(float(h * w), 1.0)
        return True, max(0.0, min(1.0, area))

    def _compute_shadow_case(self, item: dict[str, Any]) -> dict[str, Tensor]:
        cld_shdw = item["cld_shdw"].float().unsqueeze(0)
        masks = masks_from_cld_shdw(cld_shdw, shadow_index=self.shadow_index, cloud_index=self.cloud_index)
        shadow = masks.shadow[0].float()
        cloud = masks.cloud[0].float()
        clear = masks.clear[0].float()
        h, w = shadow.shape[-2:]

        division = item.get("sam_mask")
        if isinstance(division, Tensor):
            division = division.float()
            if division.shape[-2:] != (h, w):
                division = F.interpolate(
                    division.unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            division = division[:1].clamp(0.0, 1.0)
        else:
            division = shadow.new_zeros(shadow.shape)

        pos = (division > self.softshadow_shadow_case_positive_threshold).float()
        denom = float(h * w)
        shadow_frac = float(shadow.mean().item())
        cloud_frac = float(cloud.mean().item())
        clear_frac = float(clear.mean().item())
        division_pos_frac = float(pos.mean().item())
        bbox_valid, bbox_area_frac = self._bbox_area_fraction(item.get("bbox"), h, w)

        cloud_sum = float(cloud.sum().item())
        clear_sum = float(clear.sum().item())
        pos_sum = float(pos.sum().item())
        shadow_dilated = F.max_pool2d(shadow.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
        division_shadow_precision_dilated = (
            float((pos * shadow_dilated).sum().item()) / pos_sum if pos_sum > 1.0e-6 else 0.0
        )
        division_cloud_leakage = float((pos * cloud).sum().item()) / cloud_sum if cloud_sum > 1.0e-6 else 0.0
        division_clear_leakage = float((pos * clear).sum().item()) / clear_sum if clear_sum > 1.0e-6 else 0.0

        no_shadow = (
            shadow_frac < self.softshadow_absent_shadow_threshold
            and division_pos_frac < self.softshadow_absent_division_threshold
        )
        valid_shadow = (
            not no_shadow
            and bbox_valid
            and shadow_frac >= self.softshadow_valid_shadow_threshold
            and division_pos_frac >= self.softshadow_valid_division_threshold
            and bbox_area_frac <= self.softshadow_max_bbox_area
            and division_shadow_precision_dilated >= self.softshadow_min_division_shadow_precision_dilated
            and division_clear_leakage <= self.softshadow_max_clear_leakage
            and division_cloud_leakage <= self.softshadow_max_cloud_leakage
        )
        if no_shadow:
            case = SHADOW_CASE_NO_SHADOW
        elif valid_shadow:
            case = SHADOW_CASE_VALID_SHADOW
        else:
            case = SHADOW_CASE_AMBIGUOUS

        return {
            "shadow_case": torch.tensor(case, dtype=torch.long),
            "shadow_case_valid": torch.tensor(1.0 if case == SHADOW_CASE_VALID_SHADOW else 0.0, dtype=torch.float32),
            "shadow_case_no_shadow": torch.tensor(1.0 if case == SHADOW_CASE_NO_SHADOW else 0.0, dtype=torch.float32),
            "shadow_case_ambiguous": torch.tensor(1.0 if case == SHADOW_CASE_AMBIGUOUS else 0.0, dtype=torch.float32),
            "shadow_case_shadow_frac": torch.tensor(shadow_frac, dtype=torch.float32),
            "shadow_case_cloud_frac": torch.tensor(cloud_frac, dtype=torch.float32),
            "shadow_case_clear_frac": torch.tensor(clear_frac, dtype=torch.float32),
            "shadow_case_division_pos_frac": torch.tensor(division_pos_frac, dtype=torch.float32),
            "shadow_case_bbox_valid": torch.tensor(1.0 if bbox_valid else 0.0, dtype=torch.float32),
            "shadow_case_bbox_area_frac": torch.tensor(bbox_area_frac, dtype=torch.float32),
            "shadow_case_division_shadow_precision_dilated": torch.tensor(
                division_shadow_precision_dilated, dtype=torch.float32
            ),
            "shadow_case_division_clear_leakage": torch.tensor(division_clear_leakage, dtype=torch.float32),
            "shadow_case_division_cloud_leakage": torch.tensor(division_cloud_leakage, dtype=torch.float32),
        }

    def _attach_shadow_case(self, item: dict[str, Any]) -> dict[str, Any]:
        if not self.softshadow_shadow_case_enabled:
            return item
        item.update(self._compute_shadow_case(item))
        return item

    def _resize(self, x: Tensor, mode: str = "bilinear") -> Tensor:
        if self.image_size is None:
            return x
        if x.shape[-2:] == (self.image_size, self.image_size):
            return x
        y = x.unsqueeze(0)
        if mode == "nearest":
            y = F.interpolate(y, size=(self.image_size, self.image_size), mode=mode)
        else:
            y = F.interpolate(y, size=(self.image_size, self.image_size), mode=mode, align_corners=False)
        return y.squeeze(0)

    def _select_bands(self, x: Tensor) -> Tensor:
        if self.band_indices is None:
            return x
        max_idx = max(self.band_indices)
        if x.shape[0] <= max_idx:
            raise ValueError(f"Optical tensor has {x.shape[0]} channels, cannot select bands {self.band_indices}")
        return x[list(self.band_indices)].contiguous()

    def _select_bands_from_cached(self, x: Tensor) -> Tensor | None:
        if self.band_indices is None:
            return x
        if x.shape[0] == len(self.band_indices):
            return x
        max_idx = max(self.band_indices)
        if x.shape[0] <= max_idx:
            return None
        return x[list(self.band_indices)].contiguous()

    @staticmethod
    def _normalize_sar(sar: Tensor) -> Tensor:
        """DADIGAN Section 4.1: per-channel dB clip + min–max to [0, 1].

        Channel 0 (VV): clip [-25, 0]  → (x + 25) / 25
        Channel 1 (VH): clip [-32, 0]  → (x + 32) / 32
        """
        sar = torch.nan_to_num(sar.float(), nan=-50.0, posinf=0.0, neginf=-50.0).clone()
        # VV
        vv_lo, vv_hi = AllClearDataset.SAR_VV_CLIP
        sar[0].clamp_(vv_lo, vv_hi)
        sar[0].add_(-vv_lo).div_(vv_hi - vv_lo)
        # VH
        vh_lo, vh_hi = AllClearDataset.SAR_VH_CLIP
        sar[1].clamp_(vh_lo, vh_hi)
        sar[1].add_(-vh_lo).div_(vh_hi - vh_lo)
        return sar

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        sample_id = row.get("sample_id") or row.get("id") or f"sample_{idx:06d}"
        cached = self._load_cache(sample_id)
        if cached is not None:
            return cached
        cloudy = _normalize_optical(self._resize(self._load(row, self.CLOUDY_KEYS), mode="bilinear"), self.optical_scale)
        target = _normalize_optical(self._resize(self._load(row, self.TARGET_KEYS), mode="bilinear"), self.optical_scale)
        cloudy = self._select_bands(cloudy)
        target = self._select_bands(target)
        cld_shdw = _normalize_mask(self._resize(self._load_mask(row), mode="nearest"))
        item: dict[str, Any] = {
            "s2_toa": cloudy.float(),
            "target": target.float(),
            "cld_shdw": cld_shdw.float(),
            "sample_id": sample_id,
            "roi_id": row.get("roi_id", ""),
            "cloudy_date": row.get("cloudy_date", ""),
            "clear_date": row.get("clear_date", ""),
            "cloudy_s2_path": self._value(row, self.CLOUDY_KEYS),
            "clear_s2_path": self._value(row, self.TARGET_KEYS),
        }
        if self.load_sar:
            sar_raw = self._resize(self._load(row, self.S1_KEYS), mode="bilinear")
            item["s1"] = self._normalize_sar(sar_raw).float()
        sam_mask = self._load_softshadow_mask(row, sample_id)
        if sam_mask is not None:
            item["sam_mask"] = sam_mask
        bbox = self._load_softshadow_bbox(row, sample_id)
        if bbox is not None:
            item["bbox"] = bbox.float()
        return self._attach_shadow_case(item)


def cloud_fraction(cld_shdw: Tensor, cloud_index: int = 1) -> Tensor:
    if cld_shdw.ndim == 3:
        cld_shdw = cld_shdw.unsqueeze(0)
    if cld_shdw.shape[1] == 1:
        return (cld_shdw.round().long() == int(cloud_index)).float().flatten(1).mean(dim=1)
    if cld_shdw.shape[1] <= cloud_index:
        return cld_shdw.new_zeros((cld_shdw.shape[0],))
    return cld_shdw[:, cloud_index].float().clamp(0.0, 1.0).flatten(1).mean(dim=1)
