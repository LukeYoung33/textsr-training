"""Isolated segment dataset for the six-channel BASIC V2 Y4UV path."""

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
from .text_line_crop_v3 import SIZE_FILTER_VERSION, assess_crop_size


def _read_rgb(path: str) -> np.ndarray:
    """@luna-comment Decode an RGB image into float32 ``[0,1]`` HWC data."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _pad(image: np.ndarray, canvas_hw: tuple[int, int]) -> np.ndarray:
    """@luna-comment Place a segment in the fixed canvas without rescaling it."""
    h, w = canvas_hw
    if image.shape[0] > h or image.shape[1] > w:
        raise ValueError(f"crop {image.shape[:2]} exceeds canvas {canvas_hw}")
    output = np.zeros((h, w, image.shape[2]), dtype=np.float32)
    output[: image.shape[0], : image.shape[1]] = image
    return output


@DATASET_REGISTRY.register()
class BasicTextLineSegmentDatasetY4UVV3(Dataset):
    """Read pre-expanded OCR segments and return a six-channel Y4UV420 sample."""

    def __init__(self, opt: Dict[str, Any]):
        """@luna-comment Load records, apply the existing size filter, and validate an even canvas."""
        self.opt = dict(opt)
        manifest = self.opt.get("manifest") or self.opt.get("manifest_jsonl")
        if not manifest:
            raise ValueError("manifest is required")
        self.records = [
            json.loads(line)
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        canvas = self.opt.get("canvas_hw", [96, 320])
        if len(canvas) != 2:
            raise ValueError("canvas_hw must contain height and width")
        self.canvas_hw = (int(canvas[0]), int(canvas[1]))
        if any(value <= 0 or value % 32 for value in self.canvas_hw):
            raise ValueError("canvas_hw must be positive multiples of 32")
        self.min_short_side_ratio = float(self.opt.get("min_short_side_ratio", 0.5))
        self.min_long_side_ratio = float(self.opt.get("min_long_side_ratio", 0.5))
        self.filtered_records: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        for record in self.records:
            valid_hw = record.get("valid_hw")
            if not valid_hw or len(valid_hw) < 2:
                raise ValueError(f"segment {record.get('segment_id', '<unknown>')} is missing valid_hw")
            decision = assess_crop_size(
                (int(valid_hw[0]), int(valid_hw[1])),
                self.canvas_hw,
                str(record.get("orientation", "")),
                min_short_side_ratio=self.min_short_side_ratio,
                min_long_side_ratio=self.min_long_side_ratio,
            )
            if not bool(decision["accepted"]):
                self.filtered_records.append(
                    {
                        "segment_id": str(record.get("segment_id", "<unknown>")),
                        "reason": str(decision["reason"]),
                        "valid_hw": list(valid_hw),
                    }
                )
                continue
            accepted.append(record)
        self.records = accepted
        self.size_filter_stats = {
            "filter_version": SIZE_FILTER_VERSION,
            "min_short_side_ratio": self.min_short_side_ratio,
            "min_long_side_ratio": self.min_long_side_ratio,
            "input_records": len(self.records) + len(self.filtered_records),
            "accepted_records": len(self.records),
            "filtered_records": len(self.filtered_records),
        }
        if not self.records:
            raise ValueError("segment manifest is empty")
        self.source_line_ids = [
            str(record.get("source_line_id", record.get("line_id", index)))
            for index, record in enumerate(self.records)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """@luna-comment Crop, pad, convert both images to Y4UV, and emit six-channel masks."""
        record = self.records[index]
        crop = tuple(int(value) for value in record["crop_xyxy_even"])
        left, top, right, bottom = crop
        lq = _read_rgb(record["lq"])[top:bottom, left:right]
        gt = _read_rgb(record["gt"])[top:bottom, left:right]
        if lq.shape != gt.shape:
            raise ValueError(f"LQ/GT crop mismatch at {record.get('segment_id', index)}")
        if lq.shape[0] % 2 or lq.shape[1] % 2:
            raise ValueError("Y4UV segments require even crop height and width")
        canvas_h, canvas_w = self.canvas_hw
        lq = _pad(lq, self.canvas_hw)
        gt = _pad(gt, self.canvas_hw)

        valid = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        valid[: bottom - top, : right - left] = 1.0
        text = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        polygon = np.asarray(record.get("polygon_local", []), dtype=np.float32).reshape(-1, 2)
        if polygon.shape[0] >= 3:
            cv2.fillPoly(text, [np.rint(polygon).astype(np.int32)], 1.0)
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
            "crop_xyxy_even": crop,
            "valid_hw": [bottom - top, right - left],
            "cp_eligible": bool(record.get("cp_eligible", False)),
            "cp_skip_reason": str(record.get("cp_skip_reason", "")),
            "polygon_local": record.get("polygon_local", []),
            "polygon_valid": bool(record.get("polygon_valid", True)),
            "is_crop_truncated": bool(record.get("is_crop_truncated", False)),
            "transcription": str(record.get("transcription", "")),
            "size_filter_version": str(record.get("size_filter_version", SIZE_FILTER_VERSION)),
            "short_side_ratio": float(record.get("short_side_ratio", 0.0)),
            "long_side_ratio": float(record.get("long_side_ratio", 0.0)),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """@luna-comment Stack six-channel tensors and preserve per-segment metadata as lists."""
        tensor_keys = ("lq", "gt", "lq_rgb", "gt_rgb", "valid_mask", "text_mask")
        result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result


__all__ = ["BasicTextLineSegmentDatasetY4UVV3"]
