"""Segment-level BASIC V2 dataset for pre-expanded OCR-line manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

from .text_line_crop_v3 import assess_crop_size, SIZE_FILTER_VERSION


def _read_rgb(path: str) -> np.ndarray:
    """@shanlan-comment Decode an RGB image in float32 [0,1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _pad(image: np.ndarray, canvas_hw: tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Native-scale pad an HWC image into the fixed canvas."""
    h, w = canvas_hw
    if image.shape[0] > h or image.shape[1] > w:
        raise ValueError(f"crop {image.shape[:2]} exceeds canvas {canvas_hw}")
    result = np.zeros((h, w, image.shape[2]), np.float32)
    result[:image.shape[0], :image.shape[1]] = image
    return result


def _resize_stretch(image: np.ndarray, canvas_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[float, float]]:
    """@shanlan-comment Resize one segment directly to the complete model canvas."""
    height, width = (int(canvas_hw[0]), int(canvas_hw[1]))
    source_h, source_w = image.shape[:2]
    if source_h <= 0 or source_w <= 0:
        raise ValueError("segment image must have positive dimensions")
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.float32), (height / source_h, width / source_w)


def rgb_to_nv21(rgb: np.ndarray) -> torch.Tensor:
    """@shanlan-comment Convert even RGB HWC data into canonical packed NV21 [1,H+H/2,W]."""
    h, w = rgb.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("NV21 requires even crop dimensions")
    yuv = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    u = cv2.resize(yuv[..., 1], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(yuv[..., 2], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    vu = np.empty((h // 2, w), np.uint8)
    vu[:, 0::2], vu[:, 1::2] = v, u
    return torch.from_numpy(np.concatenate([yuv[..., 0], vu]).astype(np.float32) / 255.0)[None]


def rgb_to_yuv444(rgb: np.ndarray) -> torch.Tensor:
    """@shanlan-comment Convert RGB HWC data into planar YUV444 CHW."""
    yuv = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    return torch.from_numpy(yuv.astype(np.float32) / 255.0).permute(2, 0, 1)


@DATASET_REGISTRY.register()
class BasicTextLineSegmentDatasetV3(Dataset):
    """Read one manifest record per OCR line segment; never silently selects tile zero."""

    def __init__(self, opt: Dict[str, Any]):
        """@shanlan-comment Load pre-expanded segment records and enforce the fixed shape contract."""
        self.opt = dict(opt)
        manifest = self.opt.get("manifest") or self.opt.get("manifest_jsonl")
        if not manifest:
            raise ValueError("manifest is required")
        self.records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        canvas = self.opt.get("canvas_hw", [96, 320])
        self.canvas_hw = (int(canvas[0]), int(canvas[1]))
        if any(v <= 0 or v % 32 for v in self.canvas_hw):
            raise ValueError("canvas_hw must be positive multiples of 32")
        self.resize_mode = str(self.opt.get("resize_mode", "aspect_fit")).strip().lower()
        if self.resize_mode not in ("aspect_fit", "stretch"):
            raise ValueError("resize_mode must be 'aspect_fit' or 'stretch'")
        self.min_short_side_ratio = float(self.opt.get("min_short_side_ratio", 0.5))
        self.min_long_side_ratio = float(self.opt.get("min_long_side_ratio", 0.5))
        self.filtered_records = []
        accepted_records = []
        for record in self.records:
            valid_hw = record.get("valid_hw")
            if not valid_hw or len(valid_hw) < 2:
                raise ValueError(f"segment {record.get('segment_id', '<unknown>')} is missing valid_hw")
            if self.resize_mode == "stretch":
                # The full-canvas route normalizes every positive crop to the
                # model geometry, so native-size thresholds must not discard
                # records before the resize is applied.
                accepted_records.append(record)
                continue
            decision = assess_crop_size(
                (int(valid_hw[0]), int(valid_hw[1])),
                self.canvas_hw,
                str(record.get("orientation", "")),
                min_short_side_ratio=self.min_short_side_ratio,
                min_long_side_ratio=self.min_long_side_ratio,
            )
            if not bool(decision["accepted"]):
                self.filtered_records.append({
                    "segment_id": str(record.get("segment_id", "<unknown>")),
                    "reason": str(decision["reason"]),
                    "valid_hw": list(valid_hw),
                })
                continue
            accepted_records.append(record)
        self.records = accepted_records
        self.size_filter_stats = {
            "filter_version": SIZE_FILTER_VERSION,
            "filter_applied": self.resize_mode != "stretch",
            "min_short_side_ratio": self.min_short_side_ratio,
            "min_long_side_ratio": self.min_long_side_ratio,
            "input_records": len(self.records) + len(self.filtered_records),
            "accepted_records": len(self.records),
            "filtered_records": len(self.filtered_records),
        }
        if not self.records:
            raise ValueError("segment manifest is empty")
        self.source_line_ids = [str(r.get("source_line_id", r.get("line_id", i))) for i, r in enumerate(self.records)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """@shanlan-comment Crop one pre-expanded segment, convert colors, and return masks/metadata."""
        record = self.records[index]
        crop = tuple(int(v) for v in record["crop_xyxy_even"])
        left, top, right, bottom = crop
        lq = _read_rgb(record["lq"])[top:bottom, left:right]
        gt = _read_rgb(record["gt"])[top:bottom, left:right]
        if lq.shape != gt.shape:
            raise ValueError(f"LQ/GT crop mismatch at {record.get('segment_id', index)}")
        h, w = self.canvas_hw
        source_hw = (bottom - top, right - left)
        if self.resize_mode == "stretch":
            lq, resize_scale = _resize_stretch(lq, (h, w)); gt, _ = _resize_stretch(gt, (h, w))
            valid = np.ones((h, w), np.float32)
        else:
            lq = _pad(lq, (h, w)); gt = _pad(gt, (h, w))
            resize_scale = (1.0, 1.0)
            valid = np.zeros((h, w), np.float32); valid[:source_hw[0], :source_hw[1]] = 1.0
        text = np.zeros((h, w), np.float32)
        polygon = np.asarray(record.get("polygon_local", []), dtype=np.float32).reshape(-1, 2)
        if polygon.shape[0] >= 3:
            if self.resize_mode == "stretch":
                polygon = polygon * np.asarray([resize_scale[1], resize_scale[0]], dtype=np.float32)
            cv2.fillPoly(text, [np.rint(polygon).astype(np.int32)], 1.0)
        return {
            "lq": rgb_to_nv21(lq), "gt": rgb_to_yuv444(gt),
            "lq_rgb": torch.from_numpy(lq).permute(2, 0, 1),
            "gt_rgb": torch.from_numpy(gt).permute(2, 0, 1),
            # BasicSR's generic validation path uses these names even when
            # the actual model input is a packed tensor rather than a file.
            "lq_path": str(record["lq"]), "gt_path": str(record["gt"]),
            "valid_mask": torch.from_numpy(valid[None]), "text_mask": torch.from_numpy(text[None]),
            "line_id": str(record.get("source_line_id", record.get("line_id", index))),
            "segment_id": str(record.get("segment_id", index)),
            "tile_index": int(record.get("tile_index", 0)), "tile_count": int(record.get("tile_count", 1)),
            "orientation": str(record.get("orientation", "")), "crop_xyxy_even": crop,
            "cp_eligible": bool(record.get("cp_eligible", False)),
            "cp_skip_reason": str(record.get("cp_skip_reason", "")),
            "polygon_local": record.get("polygon_local", []),
            "polygon_valid": bool(record.get("polygon_valid", True)),
            "is_crop_truncated": bool(record.get("is_crop_truncated", False)),
            "transcription": str(record.get("transcription", "")),
            "resize_mode": self.resize_mode, "resize_scale": list(resize_scale),
            "size_filter_version": str(record.get("size_filter_version", SIZE_FILTER_VERSION)),
            "short_side_ratio": float(record.get("short_side_ratio", 0.0)),
            "long_side_ratio": float(record.get("long_side_ratio", 0.0)),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """@shanlan-comment Stack fixed tensors and retain segment metadata as lists."""
        tensor_keys = ("lq", "gt", "lq_rgb", "gt_rgb", "valid_mask", "text_mask")
        result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result
