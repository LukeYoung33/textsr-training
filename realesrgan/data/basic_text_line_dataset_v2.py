"""Manifest-driven BASIC 2.0 single-text-line dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

from .text_line_geometry_v2 import make_line_region, make_segment, split_horizontal_line, split_vertical_line, capacity_decision


@DATASET_REGISTRY.register()
class BasicTextLineDatasetV2(Dataset):
    """Load pre-paired line records and emit fixed-size tensor/mask samples."""

    def __init__(self, opt: Dict[str, Any]):
        """@shanlan-comment Read JSONL records and freeze the canvas contract."""
        self.opt = dict(opt)
        manifest = self.opt.get("manifest") or self.opt.get("meta_info")
        if not manifest:
            raise ValueError("BasicTextLineDatasetV2 requires manifest")
        self.records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        canvas = self.opt.get("canvas_hw", [96, 320])
        if len(canvas) != 2 or any(int(value) <= 0 or int(value) % 32 for value in canvas):
            raise ValueError("canvas_hw must contain positive multiples of 32")
        self.canvas_hw = (int(canvas[0]), int(canvas[1]))
        self.margin_ratio = float(self.opt.get("margin_ratio", 0.15))
        self.overlap = int(self.opt.get("overlap", self.canvas_hw[1] // 4))
        self.safe_short_side = float(self.opt.get("safe_short_side", self.canvas_hw[0] - 16))
        self.min_context = float(self.opt.get("min_context", 8))

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        """@shanlan-comment Decode an image path to float32 RGB in [0,1]."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    @staticmethod
    def _rgb_to_yuv444(image: np.ndarray) -> np.ndarray:
        """@shanlan-comment Convert RGB float HWC data to planar YUV444 float CHW data."""
        rgb_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
        yuv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
        return torch.from_numpy(yuv).permute(2, 0, 1)

    @staticmethod
    def _rgb_to_nv21(image: np.ndarray) -> torch.Tensor:
        """@shanlan-comment Convert RGB HWC data to canonical packed NV21 [1,H+H/2,W]."""
        height, width = image.shape[:2]
        if height % 2 or width % 2:
            raise ValueError("NV21 conversion requires even image height and width")
        rgb_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
        yuv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YUV)
        y_plane = yuv[..., 0]
        u_plane = cv2.resize(yuv[..., 1], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        v_plane = cv2.resize(yuv[..., 2], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        vu = np.empty((height // 2, width), dtype=np.uint8)
        vu[:, 0::2] = v_plane
        vu[:, 1::2] = u_plane
        packed = np.concatenate([y_plane, vu], axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(packed[None])

    @staticmethod
    def _pad_hwc(image: np.ndarray, canvas_hw: tuple[int, int]) -> np.ndarray:
        """@shanlan-comment Constant-pad an HWC ROI to the fixed canvas without resizing."""
        canvas_h, canvas_w = canvas_hw
        height, width = image.shape[:2]
        if height > canvas_h or width > canvas_w:
            raise ValueError(f"ROI {height}x{width} exceeds canvas {canvas_h}x{canvas_w}")
        output = np.zeros((canvas_h, canvas_w, image.shape[2]), dtype=np.float32)
        output[:height, :width] = image
        return output

    def _record_segments(self, record: Dict[str, Any]) -> List[Any]:
        """@shanlan-comment Convert one manifest line polygon into deterministic segments."""
        polygon = record.get("polygon_xy", record.get("polygon", record.get("poly")))
        region = make_line_region(record.get("frame_id", ""), record.get("line_id", "0"), polygon,
                                  det_score=record.get("det_score", 1.0), transcription=record.get("transcription"))
        reason, _ = capacity_decision(region.short_side_px, self.canvas_hw[0], self.safe_short_side, self.min_context)
        if reason == "OVERSIZE_TEXT_BYPASS":
            return []
        ranges = (split_vertical_line(region, self.canvas_hw[0], min(self.overlap, self.canvas_hw[0]-1))
                  if region.orientation.value == "VERTICAL" else
                  split_horizontal_line(region, self.canvas_hw[1], min(self.overlap, self.canvas_hw[1]-1)))
        image_path = record.get("lq_path") or record.get("image_path") or record.get("lq")
        image = self._read_image(image_path)
        image_hw = image.shape[:2]
        return [make_segment(region, axis_range, self.canvas_hw, image_hw, idx, len(ranges), context=self.min_context)
                for idx, axis_range in enumerate(ranges)]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """@shanlan-comment Crop/pad one line sample and return tensors plus auditable metadata."""
        record = self.records[index]
        lq_path = record.get("lq_path") or record.get("image_path") or record.get("lq")
        gt_path = record.get("gt_path") or record.get("gt") or lq_path
        lq = self._read_image(lq_path)
        gt = self._read_image(gt_path)
        segments = self._record_segments(record)
        if not segments:
            raise IndexError(f"record {index} has no BASIC-compatible segment")
        segment = segments[int(record.get("tile_index", 0)) % len(segments)]
        x0, y0, x1, y1 = segment.crop_xyxy_even
        lq_crop = lq[y0:y1, x0:x1]
        gt_crop = gt[y0:y1, x0:x1]
        ch, cw = self.canvas_hw
        lq_canvas = self._pad_hwc(lq_crop, self.canvas_hw)
        gt_canvas = self._pad_hwc(gt_crop, self.canvas_hw)
        return {
            "lq": self._rgb_to_nv21(lq_canvas), "gt": self._rgb_to_yuv444(gt_canvas),
            "lq_rgb": torch.from_numpy(lq_canvas).permute(2, 0, 1),
            "gt_rgb": torch.from_numpy(gt_canvas).permute(2, 0, 1),
            "input_yuv420": self._rgb_to_nv21(lq_canvas),
            "target_yuv444": self._rgb_to_yuv444(gt_canvas),
            "valid_mask": torch.from_numpy(segment.valid_mask[None]),
            "text_mask": torch.from_numpy(segment.text_mask[None]),
            "blend_mask": torch.from_numpy(segment.blend_mask[None]),
            "line_id": segment.source_line_id, "frame_id": record.get("frame_id", ""),
            "crop_xyxy_even": segment.crop_xyxy_even, "valid_hw": segment.valid_hw,
            "pad_ltrb": segment.pad_ltrb, "tile_index": segment.tile_index,
            "tile_count": segment.tile_count, "orientation": record.get("orientation", ""),
            "is_border_clipped": segment.is_border_clipped,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """@shanlan-comment Stack fixed tensor fields while preserving auditable metadata lists."""
        if not batch:
            raise ValueError("cannot collate an empty batch")
        tensor_keys = ("lq", "gt", "lq_rgb", "gt_rgb", "input_yuv420", "target_yuv444", "valid_mask", "text_mask", "blend_mask")
        result = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result


# V2 samples expose canonical packed NV21 input and planar YUV444 target.
