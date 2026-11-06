"""Aspect-preserving, vertical-normalizing V4 Y4UV420 training dataset."""

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
from .basic_text_line_y4uv_v4_geometry import (
    DEFAULT_CANVAS_HW,
    DEFAULT_MIN_TEXT_SIZE,
    GEOMETRY_VERSION,
    is_hx_related,
)


DATASET_VERSION = "y4uv_v4_aspect_resize96_pad480_v1"


def _read_rgb(path: str) -> np.ndarray:
    """@shanlan-comment Decode one paired image as contiguous float32 RGB data in [0,1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _crop_image(image: np.ndarray, record: Dict[str, Any], key: str) -> np.ndarray:
    """@shanlan-comment Extract the recorded even source crop and reject invalid geometry."""
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


def _interpolation_value(value: Any) -> int:
    """@shanlan-comment Resolve a stable OpenCV interpolation name from the experiment YAML."""
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


def _rotate_ccw(image: np.ndarray) -> np.ndarray:
    """@shanlan-comment Rotate a vertical source crop counter-clockwise into reading-order horizontal form."""
    return np.ascontiguousarray(np.rot90(image, k=1))


def _resize(image: np.ndarray, resized_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    """@shanlan-comment Resize a rotated crop to height 96 while preserving its recorded width."""
    height, width = (int(resized_hw[0]), int(resized_hw[1]))
    return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=interpolation), dtype=np.float32)


def _pad_right(image: np.ndarray, pad_right: int) -> np.ndarray:
    """@shanlan-comment Replicate the last valid column into the right-side fixed-canvas padding."""
    pad_right = int(pad_right)
    if pad_right <= 0:
        return np.ascontiguousarray(image, dtype=np.float32)
    if image.shape[1] <= 0:
        raise ValueError("cannot pad an empty resized crop")
    return np.ascontiguousarray(
        np.pad(image, ((0, 0), (0, pad_right), (0, 0)), mode="edge"), dtype=np.float32
    )


def _polygon_mask(polygon: Any, canvas_hw: tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Rasterize final tile polygon coordinates into a clipped text mask."""
    height, width = (int(canvas_hw[0]), int(canvas_hw[1]))
    mask = np.zeros((height, width), dtype=np.float32)
    points = np.asarray(polygon or [], dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return mask
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, max(0.0, float(width - 1)))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, max(0.0, float(height - 1)))
    cv2.fillPoly(mask, [np.rint(clipped).astype(np.int32)], 1.0)
    return mask


@DATASET_REGISTRY.register()
class BasicTextLineSegmentDatasetY4UVV4Aspect96x480(Dataset):
    """Return V4 fixed-canvas Y4UV tiles with aspect-preserving resize and vertical normalization."""

    def __init__(self, opt: Dict[str, Any]):
        """@shanlan-comment Load a filtered V4 manifest and recheck its geometry/filter contract."""
        self.opt = dict(opt)
        manifest = self.opt.get("manifest") or self.opt.get("manifest_jsonl")
        if not manifest:
            raise ValueError("manifest is required")
        records = [
            json.loads(line)
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.canvas_hw = tuple(int(value) for value in self.opt.get("canvas_hw", DEFAULT_CANVAS_HW))
        if self.canvas_hw != DEFAULT_CANVAS_HW:
            raise ValueError("V4 dataset requires canvas_hw=[96,480]")
        self.min_text_size = float(self.opt.get("min_text_size", DEFAULT_MIN_TEXT_SIZE))
        self.exclude_hx = bool(self.opt.get("exclude_hx", True))
        self.resize_interpolation = _interpolation_value(self.opt.get("resize_interpolation", "area"))
        self.records: list[dict[str, Any]] = []
        self.filtered_records: list[dict[str, Any]] = []
        for record in records:
            if str(record.get("geometry_version", "")) != GEOMETRY_VERSION:
                raise ValueError(
                    f"record {record.get('segment_id', '<unknown>')} is not a V4 geometry manifest row"
                )
            if self.exclude_hx and is_hx_related(record):
                self.filtered_records.append(
                    {"segment_id": record.get("segment_id", "<unknown>"), "reason": "HX_RELATED"}
                )
                continue
            if float(record.get("text_size_px", 0.0)) < self.min_text_size:
                self.filtered_records.append(
                    {"segment_id": record.get("segment_id", "<unknown>"), "reason": "TEXT_SIZE_BELOW_MIN"}
                )
                continue
            self.records.append(record)
        if not self.records:
            raise ValueError("V4 aspect-resize manifest is empty after filtering")
        self.size_filter_stats = {
            "dataset_version": DATASET_VERSION,
            "geometry_version": GEOMETRY_VERSION,
            "input_records": len(records),
            "accepted_records": len(self.records),
            "runtime_filtered_records": len(self.filtered_records),
            "min_text_size": self.min_text_size,
            "exclude_hx": self.exclude_hx,
            "resize_policy": "aspect_preserving_height_96_then_split_or_right_pad_480",
            "vertical_policy": "counter_clockwise_rotate_to_horizontal",
            "canvas_hw": list(self.canvas_hw),
        }
        self.source_line_ids = [
            str(record.get("source_line_id", record.get("line_id", index)))
            for index, record in enumerate(self.records)
        ]

    def __len__(self) -> int:
        """@shanlan-comment Return the number of accepted V4 tiles."""
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """@shanlan-comment Transform paired crops, build masks, and emit native six-channel tensors."""
        record = self.records[index]
        lq_full = _read_rgb(str(record["lq"]))
        gt_full = _read_rgb(str(record["gt"]))
        lq_crop = _crop_image(lq_full, record, "lq")
        gt_crop = _crop_image(gt_full, record, "gt")
        if lq_crop.shape != gt_crop.shape:
            raise ValueError(f"LQ/GT crop mismatch at {record.get('segment_id', index)}")
        if bool(record.get("rotated_ccw", False)):
            lq_crop = _rotate_ccw(lq_crop)
            gt_crop = _rotate_ccw(gt_crop)

        resized_hw = tuple(int(value) for value in record["resized_hw"])
        lq_resized = _resize(lq_crop, resized_hw, self.resize_interpolation)
        gt_resized = _resize(gt_crop, resized_hw, self.resize_interpolation)
        tile_start = int(record.get("tile_start_x", 0))
        tile_width = int(record["tile_width"])
        pad_right = int(record.get("pad_right", 0))
        lq_tile = _pad_right(lq_resized[:, tile_start : tile_start + tile_width], pad_right)
        gt_tile = _pad_right(gt_resized[:, tile_start : tile_start + tile_width], pad_right)
        if lq_tile.shape[:2] != self.canvas_hw or gt_tile.shape[:2] != self.canvas_hw:
            raise RuntimeError(
                f"V4 tile shape failed for {record.get('segment_id', index)}: "
                f"{lq_tile.shape[:2]} and {gt_tile.shape[:2]}"
            )

        valid = np.zeros(self.canvas_hw, dtype=np.float32)
        valid[:, :tile_width] = 1.0
        text = _polygon_mask(record.get("polygon_local", []), self.canvas_hw)
        lq_rgb = torch.from_numpy(np.ascontiguousarray(lq_tile)).permute(2, 0, 1)
        gt_rgb = torch.from_numpy(np.ascontiguousarray(gt_tile)).permute(2, 0, 1)
        return {
            "lq": numpy_rgb_to_y4uv(lq_tile),
            "gt": numpy_rgb_to_y4uv(gt_tile),
            "lq_rgb": lq_rgb,
            "gt_rgb": gt_rgb,
            "lq_path": str(record["lq"]),
            "gt_path": str(record["gt"]),
            "valid_mask": yuv420_mask_to_y4uv(torch.from_numpy(valid)),
            "text_mask": yuv420_mask_to_y4uv(torch.from_numpy(text)),
            # Keep direct full-resolution masks alongside the legacy packed
            # masks.  Training-only RGB auxiliary losses consume these fields;
            # native Y4UV reconstruction continues to use the packed contract.
            "valid_mask_rgb": torch.from_numpy(valid[None]),
            "text_mask_rgb": torch.from_numpy(text[None]),
            "line_id": str(record.get("source_line_id", record.get("line_id", index))),
            "segment_id": str(record.get("segment_id", index)),
            "tile_index": int(record.get("tile_index", 0)),
            "tile_count": int(record.get("tile_count", 1)),
            "orientation": str(record.get("orientation", "HORIZONTAL")),
            "source_orientation": str(record.get("source_orientation", record.get("orientation", ""))),
            "crop_xyxy_even": tuple(int(value) for value in record["crop_xyxy_even"]),
            "source_valid_hw": list(record.get("source_crop_hw", [])),
            "valid_hw": [self.canvas_hw[0], tile_width],
            "resize_applied": True,
            "rotated_ccw": bool(record.get("rotated_ccw", False)),
            "resized_hw": list(resized_hw),
            "tile_start_x": tile_start,
            "pad_right": pad_right,
            "text_size_px": float(record.get("text_size_px", 0.0)),
            "resize_dataset_version": DATASET_VERSION,
            "geometry_version": GEOMETRY_VERSION,
            "cp_eligible": bool(record.get("cp_eligible", False)),
            "cp_skip_reason": str(record.get("cp_skip_reason", "")),
            "polygon_local": record.get("polygon_local", []),
            "polygon_valid": bool(record.get("polygon_valid", True)),
            "is_crop_truncated": bool(record.get("is_crop_truncated", False)),
            "transcription": str(record.get("transcription", "")),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """@shanlan-comment Stack fixed tensors and retain per-tile metadata as batch lists."""
        tensor_keys = ("lq", "gt", "lq_rgb", "gt_rgb", "valid_mask", "text_mask", "valid_mask_rgb", "text_mask_rgb")
        result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result


__all__ = ["DATASET_VERSION", "BasicTextLineSegmentDatasetY4UVV4Aspect96x480"]
