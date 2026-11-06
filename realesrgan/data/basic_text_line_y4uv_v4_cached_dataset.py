"""Cached native-scale crop dataset for the isolated V4 Y4UV training path."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from basicsr.utils.registry import DATASET_REGISTRY

from .basic_text_line_y4uv_v3 import (
    numpy_rgb_to_y4uv,
    y4uv_to_yuv444,
    yuv420_mask_to_y4uv,
)
from .basic_text_line_y4uv_v4_cached_augmentation import (
    AUGMENTATION_VERSION,
    CachedY4UVAugmentConfig,
    apply_cached_y4uv_augmentation,
    cached_y4uv_augment_config,
)
from .basic_text_line_y4uv_v4_geometry import DEFAULT_CANVAS_HW, is_hx_related
from ..utils import YUV2RGB


DATASET_VERSION = "y4uv_v4_cached_native_crop_aug_v1"
MANIFEST_VERSION = "basic_v2_0820_native_crop_manifest_v1"
MATERIALIZER_VERSION = "basic_v2_native_crop_materializer_v1"
GEOMETRY_VERSION = "y4uv_v4_native_crop_even_inset2_rotate_vertical_v1"
CANVAS_HW = DEFAULT_CANVAS_HW
REQUIRED_MANIFEST_FIELDS = (
    "lq_crop",
    "gt_crop",
    "text_mask_crop",
    "source_line_id",
    "segment_id",
    "native_crop_hw",
    "source_crop_xyxy_even",
    "inset_requested_px",
    "inset_applied_ltrb",
    "source_orientation",
    "rotated_to_horizontal",
    "resized_hw",
    "tile_start_x",
    "tile_width",
    "pad_right",
    "valid_hw",
    "polygon_local_native",
    "polygon_local_resized_tile",
    "cp_eligible",
    "transcription",
    "geometry_version",
    "materializer_version",
)


def _require_mapping(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    """@shanlan-comment Validate the crop-only schema and return a mutable row copy."""
    if not isinstance(record, Mapping):
        raise ValueError(f"manifest row {index} must be a JSON object")
    row = dict(record)
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"manifest row {row.get('segment_id', index)} is missing cached fields: {missing}"
        )
    for field in (
        "lq_crop",
        "gt_crop",
        "text_mask_crop",
        "source_line_id",
        "segment_id",
        "source_orientation",
        "geometry_version",
        "materializer_version",
        "transcription",
    ):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"manifest row {row.get('segment_id', index)} field {field!r} must be non-empty text")
    for field in ("geometry_version", "materializer_version"):
        row[field] = row[field].strip()
        if re.search(r"_v[0-9]+$", row[field]) is None:
            raise ValueError(
                f"cached row {row.get('segment_id', index)} field {field!r} must use a _vN version"
            )
    row["source_line_id"] = row["source_line_id"].strip()
    row["segment_id"] = row["segment_id"].strip()
    row["source_orientation"] = row["source_orientation"].strip().upper()
    if row["source_orientation"] not in {"HORIZONTAL", "VERTICAL", "OBLIQUE", "UNKNOWN"}:
        raise ValueError(f"unsupported source_orientation: {row['source_orientation']!r}")
    row["rotated_to_horizontal"] = _bool(row["rotated_to_horizontal"], "rotated_to_horizontal", row)
    row["transcription"] = row["transcription"]
    row["native_crop_hw"] = _int_pair(row["native_crop_hw"], "native_crop_hw", row)
    row["source_crop_xyxy_even"] = _int_quad(row["source_crop_xyxy_even"], "source_crop_xyxy_even", row)
    row["inset_applied_ltrb"] = _int_quad(row["inset_applied_ltrb"], "inset_applied_ltrb", row, nonnegative=True)
    row["inset_requested_px"] = _nonnegative_number_or_quad(row["inset_requested_px"], "inset_requested_px", row)
    row["resized_hw"] = _int_pair(row["resized_hw"], "resized_hw", row)
    row["valid_hw"] = _int_pair(row["valid_hw"], "valid_hw", row)
    if any(item % 2 for item in row["native_crop_hw"]):
        raise ValueError(f"cached row {row['segment_id']} native_crop_hw must be YUV420-even")
    left, top, right, bottom = row["source_crop_xyxy_even"]
    source_hw = (bottom - top, right - left)
    expected_native_hw = (source_hw[1], source_hw[0]) if row["rotated_to_horizontal"] else source_hw
    if tuple(row["native_crop_hw"]) != expected_native_hw:
        raise ValueError(
            f"cached row {row['segment_id']} native_crop_hw {row['native_crop_hw']} "
            f"does not match source crop/rotation {expected_native_hw}"
        )
    requested = row["inset_requested_px"]
    requested_sides = requested if isinstance(requested, list) else [requested] * 4
    if any(applied > int(requested_side) for applied, requested_side in zip(row["inset_applied_ltrb"], requested_sides)):
        raise ValueError(f"cached row {row['segment_id']} inset exceeds requested recovery")
    if row["resized_hw"][0] != CANVAS_HW[0]:
        raise ValueError(f"cached V4 rows require resized_hw height 96, got {row['resized_hw']}")
    expected_width = max(2, int(round(row["native_crop_hw"][1] * CANVAS_HW[0] / row["native_crop_hw"][0])))
    if expected_width % 2:
        expected_width += 1
    if row["resized_hw"][1] != expected_width:
        raise ValueError(
            f"cached row {row['segment_id']} resized width {row['resized_hw'][1]} "
            f"!= aspect-preserving width {expected_width}"
        )
    if row["valid_hw"][0] != CANVAS_HW[0]:
        raise ValueError(f"cached V4 rows require valid_hw height 96, got {row['valid_hw']}")
    row["tile_start_x"] = _positive_int(row["tile_start_x"], "tile_start_x", row, allow_zero=True)
    row["tile_width"] = _positive_int(row["tile_width"], "tile_width", row)
    row["pad_right"] = _positive_int(row["pad_right"], "pad_right", row, allow_zero=True)
    if row["tile_width"] > CANVAS_HW[1] or row["pad_right"] != CANVAS_HW[1] - row["tile_width"]:
        raise ValueError(
            f"cached row {row['segment_id']} has invalid tile width/padding: "
            f"{row['tile_width']}/{row['pad_right']}"
        )
    if row["valid_hw"][1] != row["tile_width"]:
        raise ValueError(f"cached row {row['segment_id']} valid_hw width must equal tile_width")
    if row["resized_hw"][1] < row["tile_start_x"] + row["tile_width"]:
        raise ValueError(f"cached row {row['segment_id']} tile exceeds resized width")
    if row["resized_hw"][1] % 2 or row["tile_start_x"] % 2 or row["tile_width"] % 2:
        raise ValueError(f"cached row {row['segment_id']} tile geometry must be YUV420-even")
    row["polygon_local_native"] = _polygon(row["polygon_local_native"], "polygon_local_native", row)
    row["polygon_local_resized_tile"] = _polygon(
        row["polygon_local_resized_tile"], "polygon_local_resized_tile", row
    )
    row["cp_eligible"] = _bool(row["cp_eligible"], "cp_eligible", row)
    if "resize_applied" in row and _bool(row["resize_applied"], "resize_applied", row):
        raise ValueError(f"cached row {row['segment_id']} must reference native, unresized crops")
    if "pixel_scale" in row and str(row["pixel_scale"]).strip().lower() != "native":
        raise ValueError(f"cached row {row['segment_id']} pixel_scale must be native")
    return row


def _int_pair(value: Any, field: str, row: Mapping[str, Any]) -> list[int]:
    """@shanlan-comment Validate a positive two-element integer geometry field."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"cached row {row.get('segment_id')} field {field} must have two integers")
    result = [int(item) for item in value]
    if any(float(item) != float(raw) for item, raw in zip(result, value)) or any(item <= 0 for item in result):
        raise ValueError(f"cached row {row.get('segment_id')} field {field} must be positive integers")
    return result


def _int_quad(value: Any, field: str, row: Mapping[str, Any], nonnegative: bool = False) -> list[int]:
    """@shanlan-comment Validate a four-element integer coordinate or inset field."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"cached row {row.get('segment_id')} field {field} must have four integers")
    result = [int(item) for item in value]
    if any(float(item) != float(raw) for item, raw in zip(result, value)):
        raise ValueError(f"cached row {row.get('segment_id')} field {field} must contain integers")
    if nonnegative and any(item < 0 for item in result):
        raise ValueError(f"cached row {row.get('segment_id')} field {field} must be non-negative")
    if field == "source_crop_xyxy_even":
        left, top, right, bottom = result
        if min(result) < 0 or right <= left or bottom <= top or any(item % 2 for item in result):
            raise ValueError(f"cached row {row.get('segment_id')} has invalid even source crop coordinates")
    return result


def _nonnegative_number_or_quad(value: Any, field: str, row: Mapping[str, Any]) -> float | list[int]:
    """@shanlan-comment Accept scalar or four-side native-crop inset forms."""
    if isinstance(value, (list, tuple)):
        return _int_quad(value, field, row, nonnegative=True)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cached row {row.get('segment_id')} field {field} must be numeric") from error
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"cached row {row.get('segment_id')} field {field} must be non-negative")
    return result


def _positive_int(value: Any, field: str, row: Mapping[str, Any], allow_zero: bool = False) -> int:
    """@shanlan-comment Validate one scalar tile geometry value."""
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cached row {row.get('segment_id')} field {field} must be an integer") from error
    if float(result) != float(value) or (result < 0 if allow_zero else result <= 0):
        raise ValueError(f"cached row {row.get('segment_id')} field {field} has invalid value {value!r}")
    return result


def _polygon(value: Any, field: str, row: Mapping[str, Any]) -> list[list[float]]:
    """@shanlan-comment Validate a polygon coordinate list while retaining JSON-safe floats."""
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"cached row {row.get('segment_id')} field {field} needs at least three points")
    result: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"cached row {row.get('segment_id')} field {field} has an invalid point")
        coords = [float(item) for item in point]
        if not np.isfinite(coords).all():
            raise ValueError(f"cached row {row.get('segment_id')} field {field} has non-finite coordinates")
        result.append(coords)
    return result


def _bool(value: Any, field: str, row: Mapping[str, Any]) -> bool:
    """@shanlan-comment Validate a JSON boolean while accepting conventional 0/1 encoding."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"cached row {row.get('segment_id')} field {field} must be boolean")


def _read_cached_rgb(path: str) -> np.ndarray:
    """@shanlan-comment Decode one lossless native RGB crop without touching a source frame."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"cached RGB crop must be HWC with three channels: {path}")
    if image.shape[2] > 3:
        image = image[..., :3]
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        image = image.astype(np.float32)
        scale = 1.0 if float(np.nanmax(image)) <= 1.0 else 255.0
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / scale
    return np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), dtype=np.float32)


def _read_cached_mask(path: str) -> np.ndarray:
    """@shanlan-comment Decode one lossless native text mask and normalize to float [0,1]."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        if mask.shape[2] < 3:
            mask = mask[..., 0]
        else:
            mask = cv2.cvtColor(mask[..., :3], cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2:
        raise ValueError(f"cached text mask must be HW: {path}")
    if np.issubdtype(mask.dtype, np.integer):
        scale = float(np.iinfo(mask.dtype).max)
    else:
        mask = mask.astype(np.float32)
        scale = 1.0 if float(np.nanmax(mask)) <= 1.0 else 255.0
    result = mask.astype(np.float32) / scale
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def _resize(image: np.ndarray, resized_hw: Sequence[int], interpolation: int) -> np.ndarray:
    """@shanlan-comment Perform the sole training-time height-96 resize recorded by the manifest."""
    height, width = (int(resized_hw[0]), int(resized_hw[1]))
    if height != CANVAS_HW[0] or width <= 0:
        raise ValueError(f"cached V4 resize must be [96,W], got {resized_hw}")
    return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=interpolation), dtype=np.float32)


def _pad_image(image: np.ndarray, pad_right: int) -> np.ndarray:
    """@shanlan-comment Replicate the final valid image column into fixed-canvas padding."""
    if pad_right <= 0:
        return np.ascontiguousarray(image, dtype=np.float32)
    if image.shape[1] <= 0:
        raise ValueError("cannot pad an empty cached crop")
    return np.ascontiguousarray(
        np.pad(image, ((0, 0), (0, int(pad_right)), (0, 0)), mode="edge"), dtype=np.float32
    )


def _pad_mask(mask: np.ndarray, pad_right: int) -> np.ndarray:
    """@shanlan-comment Zero-fill mask padding so padded pixels cannot contribute to loss."""
    if pad_right <= 0:
        return np.ascontiguousarray(mask, dtype=np.float32)
    return np.ascontiguousarray(
        np.pad(mask, ((0, 0), (0, int(pad_right))), mode="constant"), dtype=np.float32
    )


def _interpolation_value(value: Any) -> int:
    """@shanlan-comment Resolve OpenCV interpolation without a source-frame fallback."""
    normalized = str(value or "area").strip().lower()
    choices = {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "linear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "cubic": cv2.INTER_CUBIC,
    }
    if normalized not in choices:
        raise ValueError(f"unsupported resize_interpolation: {value!r}")
    return choices[normalized]


def _native_to_rgb(native: np.ndarray) -> torch.Tensor:
    """@shanlan-comment Regenerate an RGB CP target from the exact augmented native Y4UV tensor."""
    value = torch.from_numpy(np.ascontiguousarray(native, dtype=np.float32)).unsqueeze(0)
    yuv444 = y4uv_to_yuv444(value)
    rgb = YUV2RGB(yuv444, format="RGB", is_train=True, uv_normalized=True)[0]
    return rgb.clamp(0.0, 1.0).contiguous()


def _simulate_lq_from_gt(gt_tile: np.ndarray, rng: Any, config: Mapping[str, Any]) -> np.ndarray:
    """Create a deterministic-on-worker synthetic observation from the clean V5 GT tile."""
    downscale = float(config.get("downscale", 0.72))
    sigma = float(config.get("blur_sigma", 1.15))
    noise_std = float(config.get("noise_std", 0.012))
    height, width = gt_tile.shape[:2]
    small_w = max(2, int(round(width * downscale)) // 2 * 2)
    small_h = max(2, int(round(height * downscale)) // 2 * 2)
    blurred = cv2.GaussianBlur(gt_tile, (0, 0), sigmaX=sigma, sigmaY=sigma)
    down = cv2.resize(blurred, (small_w, small_h), interpolation=cv2.INTER_AREA)
    result = cv2.resize(down, (width, height), interpolation=cv2.INTER_CUBIC)
    if noise_std > 0:
        result = result + rng.normal(0.0, noise_std, size=result.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


@DATASET_REGISTRY.register()
class BasicTextLineSegmentDatasetY4UVV4CachedNativeCrop96x480(Dataset):
    """@shanlan-comment Read native crops, materialize 96x480 tiles, and emit V4 Y4UV samples."""

    def __init__(self, opt: Mapping[str, Any]):
        """@shanlan-comment Load/validate crop-only rows and augmentation options."""
        self.opt = dict(opt)
        manifest = self.opt.get("manifest") or self.opt.get("manifest_jsonl")
        if not manifest:
            raise ValueError("cached native-crop dataset requires manifest")
        self.manifest_path = Path(str(manifest))
        rows = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.canvas_hw = tuple(int(value) for value in self.opt.get("canvas_hw", CANVAS_HW))
        if self.canvas_hw != CANVAS_HW:
            raise ValueError("cached V4 dataset requires canvas_hw=[96,480]")
        self.resize_interpolation = _interpolation_value(self.opt.get("resize_interpolation", "area"))
        self.exclude_hx = bool(self.opt.get("exclude_hx", True))
        expected_geometry = self.opt.get("expected_geometry_version", GEOMETRY_VERSION)
        expected_materializer = self.opt.get("expected_materializer_version", MATERIALIZER_VERSION)
        expected_manifest = self.opt.get("expected_manifest_version", MANIFEST_VERSION)
        self.records: list[dict[str, Any]] = []
        self.filtered_records: list[dict[str, Any]] = []
        for index, raw in enumerate(rows):
            record = _require_mapping(raw, index)
            if expected_geometry is not None and record["geometry_version"] != str(expected_geometry):
                raise ValueError(
                    f"cached row {record['segment_id']} geometry_version {record['geometry_version']!r} "
                    f"!= expected {expected_geometry!r}"
                )
            if expected_materializer is not None and record["materializer_version"] != str(expected_materializer):
                raise ValueError(
                    f"cached row {record['segment_id']} materializer_version "
                    f"{record['materializer_version']!r} != expected {expected_materializer!r}"
                )
            if "manifest_version" in record and str(record["manifest_version"]) != str(expected_manifest):
                raise ValueError(
                    f"cached row {record['segment_id']} manifest_version {record['manifest_version']!r} "
                    f"!= expected {expected_manifest!r}"
                )
            if self.exclude_hx and is_hx_related(record):
                self.filtered_records.append({"segment_id": record["segment_id"], "reason": "HX_RELATED"})
                continue
            self.records.append(record)
        if not self.records:
            raise ValueError("cached native-crop manifest is empty after filtering")
        self.augmentation = cached_y4uv_augment_config(self.opt)
        simulated = self.opt.get("simulated_degradation") or {}
        if not isinstance(simulated, Mapping):
            raise ValueError("simulated_degradation must be a mapping")
        self.simulated_degradation = dict(simulated)
        self.simulated_degradation_enabled = bool(self.simulated_degradation.get("enabled", False))
        self._worker_rng: Any = None
        self._worker_seed: int | None = None
        self.size_filter_stats = {
            "dataset_version": DATASET_VERSION,
            "augmentation_version": AUGMENTATION_VERSION,
            "input_records": len(rows),
            "accepted_records": len(self.records),
            "runtime_filtered_records": len(self.filtered_records),
            "canvas_hw": list(self.canvas_hw),
            "resize_policy": "native_crop_then_aspect_preserving_height_96_then_tile_pad_480",
            "source_frame_decode": False,
            "geometry_versions": sorted({record["geometry_version"] for record in self.records}),
            "materializer_versions": sorted({record["materializer_version"] for record in self.records}),
            "manifest_versions": sorted(
                {str(record["manifest_version"]) for record in self.records if "manifest_version" in record}
            ),
            "augmentation": {
                "inversion_probability": self.augmentation.inversion_probability,
                "contrast_probability": self.augmentation.contrast_probability,
                "contrast_factor_range": list(self.augmentation.contrast_factor_range),
            },
            "simulated_degradation": dict(self.simulated_degradation),
        }

    def __len__(self) -> int:
        """@shanlan-comment Return the number of accepted cached native-crop tiles."""
        return len(self.records)

    def _resolve_asset_path(self, value: str) -> str:
        """@shanlan-comment Resolve absolute or cache-relative crop paths beside the manifest."""
        path = Path(value)
        if path.is_absolute():
            return str(path)
        candidates = [self.manifest_path.parent / path]
        if self.manifest_path.parent.name == "manifests":
            candidates.append(self.manifest_path.parent.parent / path)
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return str(candidates[-1])

    def _rng(self) -> Any:
        """@shanlan-comment Return one worker-local RNG seeded from DataLoader worker state."""
        worker = get_worker_info()
        if worker is None:
            seed = self.opt.get("augmentation_seed")
            if seed is not None:
                if self._worker_rng is None or self._worker_seed != int(seed):
                    self._worker_seed = int(seed)
                    self._worker_rng = np.random.default_rng(self._worker_seed)
                return self._worker_rng
            return np.random
        seed = int(worker.seed)
        if self._worker_rng is None or self._worker_seed != seed:
            self._worker_seed = seed
            self._worker_rng = np.random.default_rng(seed)
        return self._worker_rng

    def __getitem__(self, index: int) -> dict[str, Any]:
        """@shanlan-comment Materialize one crop row, augment, and emit masks/CP metadata."""
        record = self.records[int(index)]
        lq_crop_path = self._resolve_asset_path(record["lq_crop"])
        gt_crop_path = self._resolve_asset_path(record["gt_crop"])
        text_mask_crop_path = self._resolve_asset_path(record["text_mask_crop"])
        lq_crop = _read_cached_rgb(lq_crop_path)
        gt_crop = _read_cached_rgb(gt_crop_path)
        text_mask_crop = _read_cached_mask(text_mask_crop_path)
        expected_native_hw = tuple(record["native_crop_hw"])
        if tuple(lq_crop.shape[:2]) != expected_native_hw:
            raise ValueError(
                f"cached row {record['segment_id']} lq_crop shape {lq_crop.shape[:2]} "
                f"!= native_crop_hw {expected_native_hw}"
            )
        if tuple(gt_crop.shape[:2]) != expected_native_hw or gt_crop.shape != lq_crop.shape:
            raise ValueError(f"cached row {record['segment_id']} LQ/GT native crop shapes differ")
        if tuple(text_mask_crop.shape) != expected_native_hw:
            raise ValueError(
                f"cached row {record['segment_id']} text mask shape {text_mask_crop.shape} "
                f"!= native_crop_hw {expected_native_hw}"
            )

        resized_hw = tuple(record["resized_hw"])
        lq_resized = _resize(lq_crop, resized_hw, self.resize_interpolation)
        gt_resized = _resize(gt_crop, resized_hw, self.resize_interpolation)
        mask_resized = _resize(text_mask_crop, resized_hw, cv2.INTER_NEAREST)
        tile_start = int(record["tile_start_x"])
        tile_width = int(record["tile_width"])
        pad_right = int(record["pad_right"])
        tile_slice = slice(tile_start, tile_start + tile_width)
        lq_tile = _pad_image(lq_resized[:, tile_slice], pad_right)
        gt_tile = _pad_image(gt_resized[:, tile_slice], pad_right)
        mask_tile = _pad_mask(mask_resized[:, tile_slice], pad_right)
        if lq_tile.shape[:2] != self.canvas_hw or gt_tile.shape[:2] != self.canvas_hw:
            raise RuntimeError(
                f"cached row {record['segment_id']} image tile shape "
                f"{lq_tile.shape[:2]}/{gt_tile.shape[:2]} != {self.canvas_hw}"
            )
        if mask_tile.shape != self.canvas_hw:
            raise RuntimeError(
                f"cached row {record['segment_id']} mask tile shape {mask_tile.shape} "
                f"!= {self.canvas_hw}"
            )

        valid_mask = np.zeros(self.canvas_hw, dtype=np.float32)
        valid_mask[:, :tile_width] = 1.0
        if self.simulated_degradation_enabled:
            lq_tile = _simulate_lq_from_gt(gt_tile, self._rng(), self.simulated_degradation)
        lq_native = numpy_rgb_to_y4uv(lq_tile).numpy()
        gt_native = numpy_rgb_to_y4uv(gt_tile).numpy()
        lq_augmented, gt_augmented, augmentation_metadata = apply_cached_y4uv_augmentation(
            lq_native,
            gt_native,
            valid_mask,
            mask_tile,
            rng=self._rng(),
            config=self.augmentation,
        )
        lq_rgb = _native_to_rgb(lq_augmented) if augmentation_metadata["lq_rgb_regenerated"] else (
            torch.from_numpy(np.ascontiguousarray(lq_tile)).permute(2, 0, 1).contiguous()
        )
        gt_rgb = _native_to_rgb(gt_augmented) if augmentation_metadata["gt_rgb_regenerated"] else (
            torch.from_numpy(np.ascontiguousarray(gt_tile)).permute(2, 0, 1).contiguous()
        )
        text_mask = np.clip(mask_tile, 0.0, 1.0)
        polygon = record["polygon_local_resized_tile"]
        result = {
            "lq": torch.from_numpy(lq_augmented),
            "gt": torch.from_numpy(gt_augmented),
            "lq_rgb": lq_rgb,
            "gt_rgb": gt_rgb,
            "lq_crop": record["lq_crop"],
            "gt_crop": record["gt_crop"],
            "text_mask_crop": record["text_mask_crop"],
            "lq_path": lq_crop_path,
            "gt_path": gt_crop_path,
            "valid_mask": yuv420_mask_to_y4uv(torch.from_numpy(valid_mask)),
            "text_mask": yuv420_mask_to_y4uv(torch.from_numpy(text_mask)),
            # Full-resolution GT masks are retained for training-only RGB
            # auxiliary losses; padded columns are zero in valid_mask.
            "valid_mask_rgb": torch.from_numpy(valid_mask[None]),
            "text_mask_rgb": torch.from_numpy(text_mask[None]),
            "line_id": record["source_line_id"],
            "source_line_id": record["source_line_id"],
            "segment_id": record["segment_id"],
            "source_orientation": record["source_orientation"],
            "orientation": "HORIZONTAL" if record["rotated_to_horizontal"] else record["source_orientation"],
            "rotated_to_horizontal": record["rotated_to_horizontal"],
            "rotated_ccw": record["rotated_to_horizontal"],
            "native_crop_hw": list(record["native_crop_hw"]),
            "source_crop_xyxy_even": list(record["source_crop_xyxy_even"]),
            "crop_xyxy_even": tuple(record["source_crop_xyxy_even"]),
            "inset_requested_px": record["inset_requested_px"],
            "inset_applied_ltrb": list(record["inset_applied_ltrb"]),
            "resized_hw": list(record["resized_hw"]),
            "tile_start_x": tile_start,
            "tile_width": tile_width,
            "pad_right": pad_right,
            "valid_hw": list(record["valid_hw"]),
            "polygon_local_native": record["polygon_local_native"],
            "polygon_local_resized_tile": polygon,
            "polygon_local": polygon,
            "polygon_valid": True,
            "cp_eligible": record["cp_eligible"],
            "cp_skip_reason": str(record.get("cp_skip_reason", "")),
            "is_crop_truncated": bool(record.get("is_crop_truncated", not record["cp_eligible"])),
            "transcription": record["transcription"],
            "geometry_version": record["geometry_version"],
            "materializer_version": record["materializer_version"],
            "manifest_version": str(record.get("manifest_version", MANIFEST_VERSION)),
            "resize_dataset_version": DATASET_VERSION,
            "source_frame_decode": False,
            "simulated_degradation": self.simulated_degradation_enabled,
        }
        result.update(augmentation_metadata)
        return result

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """@shanlan-comment Stack fixed tensors and preserve geometry/augmentation metadata."""
        if not batch:
            raise ValueError("cached native-crop collate cannot receive an empty batch")
        tensor_keys = ("lq", "gt", "lq_rgb", "gt_rgb", "valid_mask", "text_mask", "valid_mask_rgb", "text_mask_rgb")
        result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result


# Short alias for configs and callers that use the crop-cache terminology.
BasicTextLineSegmentDatasetY4UVV4CachedCrop96x480 = BasicTextLineSegmentDatasetY4UVV4CachedNativeCrop96x480


__all__ = [
    "CANVAS_HW",
    "DATASET_VERSION",
    "GEOMETRY_VERSION",
    "MANIFEST_VERSION",
    "MATERIALIZER_VERSION",
    "REQUIRED_MANIFEST_FIELDS",
    "BasicTextLineSegmentDatasetY4UVV4CachedCrop96x480",
    "BasicTextLineSegmentDatasetY4UVV4CachedNativeCrop96x480",
]
