"""Resize-to-canvas dataset for the isolated BASIC V2 Y4UV V3 experiments.

The existing Y4UV V3 dataset preserves native crop geometry and pads into a
canvas.  This module is deliberately separate: every pre-expanded text-line
crop is resized to the requested canvas before Y4UV420 conversion.  No legacy
V2/V3 dataset behavior is changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

from .basic_text_line_y4uv_v3 import numpy_rgb_to_y4uv, yuv420_mask_to_y4uv
from .text_line_crop_v3 import SIZE_FILTER_VERSION


RESIZE_DATASET_VERSION = "y4uv_resize_to_canvas_v1"
DEFAULT_CANVAS_HW = (96, 480)


def _read_rgb(path: str) -> np.ndarray:
    """@luna-comment Decode a paired image as float32 RGB HWC data in the ``[0,1]`` range."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _resize_image(image: np.ndarray, canvas_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    """@luna-comment Resize an HWC RGB crop to the exact training canvas without padding."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"RGB crop must be HWC with three channels, got {tuple(image.shape)}")
    height, width = (int(canvas_hw[0]), int(canvas_hw[1]))
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image, dtype=np.float32)
    return np.ascontiguousarray(
        cv2.resize(image, (width, height), interpolation=interpolation), dtype=np.float32
    )


def _resize_mask(mask: np.ndarray, canvas_hw: tuple[int, int]) -> np.ndarray:
    """@luna-comment Resize a binary/soft line mask with nearest-neighbor geometry preservation."""
    height, width = (int(canvas_hw[0]), int(canvas_hw[1]))
    if mask.ndim != 2:
        raise ValueError(f"mask must be HW, got {tuple(mask.shape)}")
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)


def _crop_image(image: np.ndarray, record: Dict[str, Any], key: str) -> np.ndarray:
    """@luna-comment Extract one manifest crop and reject empty or invalid paired regions."""
    crop = record.get("crop_xyxy_even")
    if not crop or len(crop) != 4:
        raise ValueError(f"record {record.get('segment_id', '<unknown>')} is missing crop_xyxy_even")
    left, top, right, bottom = (int(value) for value in crop)
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f"record {record.get('segment_id', '<unknown>')} has invalid crop_xyxy_even")
    if bottom > image.shape[0] or right > image.shape[1]:
        raise ValueError(f"{key} crop {crop} exceeds image shape {image.shape[:2]}")
    result = image[top:bottom, left:right]
    if result.size == 0:
        raise ValueError(f"{key} crop is empty for {record.get('segment_id', '<unknown>')}")
    return result


def _polygon_mask_and_scaled_polygon(
    record: Dict[str, Any], source_hw: tuple[int, int], canvas_hw: tuple[int, int]
) -> tuple[np.ndarray, list[list[float]]]:
    """@luna-comment Rasterize a source-local polygon, resize it, and return scaled geometry."""
    source_height, source_width = (int(source_hw[0]), int(source_hw[1]))
    canvas_height, canvas_width = (int(canvas_hw[0]), int(canvas_hw[1]))
    mask = np.zeros((source_height, source_width), dtype=np.float32)
    polygon = np.asarray(record.get("polygon_local", []), dtype=np.float32).reshape(-1, 2)
    if polygon.shape[0] < 3:
        return np.zeros((canvas_height, canvas_width), dtype=np.float32), []
    clipped = polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, source_width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, source_height - 1))
    cv2.fillPoly(mask, [np.rint(clipped).astype(np.int32)], 1.0)
    scaled = clipped * np.asarray(
        [canvas_width / float(source_width), canvas_height / float(source_height)], dtype=np.float32
    )
    return _resize_mask(mask, canvas_hw), scaled.astype(np.float32).tolist()


def _interpolation_value(value: Any) -> int:
    """@luna-comment Resolve a stable OpenCV interpolation name from YAML configuration."""
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


@DATASET_REGISTRY.register()
class BasicTextLineSegmentDatasetY4UVV3Resize96x480(Dataset):
    """Return every pre-expanded line crop resized to ``[6,48,240]`` Y4UV."""

    def __init__(self, opt: Dict[str, Any]):
        """@luna-comment Load all segment records and establish the exact resize-only contract."""
        self.opt = dict(opt)
        manifest = self.opt.get("manifest") or self.opt.get("manifest_jsonl")
        if not manifest:
            raise ValueError("manifest is required")
        self.records = [
            json.loads(line)
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError("segment manifest is empty")
        canvas = self.opt.get("canvas_hw", list(DEFAULT_CANVAS_HW))
        if len(canvas) != 2:
            raise ValueError("canvas_hw must contain height and width")
        self.canvas_hw = (int(canvas[0]), int(canvas[1]))
        if self.canvas_hw != DEFAULT_CANVAS_HW:
            raise ValueError(
                "BasicTextLineSegmentDatasetY4UVV3Resize96x480 requires canvas_hw=[96,480]"
            )
        if any(value <= 0 or value % 2 for value in self.canvas_hw):
            raise ValueError("canvas_hw values must be positive even numbers")
        self.resize_interpolation = _interpolation_value(self.opt.get("resize_interpolation", "area"))
        self.filtered_records: list[dict[str, Any]] = []
        self.size_filter_stats = {
            "filter_version": f"{SIZE_FILTER_VERSION}+{RESIZE_DATASET_VERSION}",
            "input_records": len(self.records),
            "accepted_records": len(self.records),
            "filtered_records": 0,
            "resize_policy": "direct_resize_no_padding",
            "canvas_hw": list(self.canvas_hw),
        }
        self.source_line_ids = [
            str(record.get("source_line_id", record.get("line_id", index)))
            for index, record in enumerate(self.records)
        ]

    def __len__(self) -> int:
        """@luna-comment Return the number of source text-line segments retained for training."""
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """@luna-comment Resize paired crops/masks, convert to Y4UV420, and preserve CP metadata."""
        record = self.records[index]
        lq_full = _read_rgb(str(record["lq"]))
        gt_full = _read_rgb(str(record["gt"]))
        lq_crop = _crop_image(lq_full, record, "lq")
        gt_crop = _crop_image(gt_full, record, "gt")
        if lq_crop.shape != gt_crop.shape:
            raise ValueError(f"LQ/GT crop mismatch at {record.get('segment_id', index)}")
        source_hw = (int(lq_crop.shape[0]), int(lq_crop.shape[1]))
        lq = _resize_image(lq_crop, self.canvas_hw, self.resize_interpolation)
        gt = _resize_image(gt_crop, self.canvas_hw, self.resize_interpolation)
        text, scaled_polygon = _polygon_mask_and_scaled_polygon(
            record, source_hw, self.canvas_hw
        )
        valid = np.ones(self.canvas_hw, dtype=np.float32)
        lq_rgb = torch.from_numpy(np.ascontiguousarray(lq)).permute(2, 0, 1)
        gt_rgb = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)
        return {
            "lq": numpy_rgb_to_y4uv(lq),
            "gt": numpy_rgb_to_y4uv(gt),
            "lq_rgb": lq_rgb,
            "gt_rgb": gt_rgb,
            "lq_path": str(record["lq"]),
            "gt_path": str(record["gt"]),
            "valid_mask": yuv420_mask_to_y4uv(torch.from_numpy(valid)),
            "text_mask": yuv420_mask_to_y4uv(torch.from_numpy(text)),
            "line_id": str(record.get("source_line_id", record.get("line_id", index))),
            "segment_id": str(record.get("segment_id", index)),
            "tile_index": int(record.get("tile_index", 0)),
            "tile_count": int(record.get("tile_count", 1)),
            "orientation": str(record.get("orientation", "")),
            "crop_xyxy_even": tuple(int(value) for value in record["crop_xyxy_even"]),
            "source_valid_hw": list(source_hw),
            "valid_hw": list(self.canvas_hw),
            "resize_applied": True,
            "resize_dataset_version": RESIZE_DATASET_VERSION,
            "cp_eligible": bool(record.get("cp_eligible", False)),
            "cp_skip_reason": str(record.get("cp_skip_reason", "")),
            "polygon_local": scaled_polygon,
            "polygon_valid": bool(record.get("polygon_valid", True)),
            "is_crop_truncated": bool(record.get("is_crop_truncated", False)),
            "transcription": str(record.get("transcription", "")),
            "size_filter_version": str(record.get("size_filter_version", SIZE_FILTER_VERSION)),
            "short_side_ratio": float(record.get("short_side_ratio", 0.0)),
            "long_side_ratio": float(record.get("long_side_ratio", 0.0)),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """@luna-comment Stack fixed tensors and retain per-segment metadata as batch lists."""
        tensor_keys = ("lq", "gt", "lq_rgb", "gt_rgb", "valid_mask", "text_mask")
        result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result


__all__ = [
    "RESIZE_DATASET_VERSION",
    "DEFAULT_CANVAS_HW",
    "BasicTextLineSegmentDatasetY4UVV3Resize96x480",
]
