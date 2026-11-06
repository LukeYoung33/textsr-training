"""Geometry contract for the isolated V4 Y4UV text-line training dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .text_line_crop_v3 import classify_line_orientation, even_bbox_crop, polygon_to_array


GEOMETRY_VERSION = "y4uv_v4_aspect96_rotate_vertical_v1"
DEFAULT_CANVAS_HW = (96, 480)
DEFAULT_MIN_TEXT_SIZE = 50.0
DEFAULT_SPLIT_OVERLAP = 64


@dataclass(frozen=True)
class V4LineGeometry:
    """One filtered text-line tile with all source-to-canvas geometry metadata."""

    tile_index: int
    tile_count: int
    source_orientation: str
    orientation: str
    rotated_ccw: bool
    crop_xyxy_even: tuple[int, int, int, int]
    source_crop_hw: tuple[int, int]
    rotated_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    tile_start_x: int
    tile_width: int
    pad_right: int
    valid_hw: tuple[int, int]
    polygon_local: np.ndarray
    text_size_px: float
    is_crop_truncated: bool
    cp_eligible: bool
    cp_skip_reason: str

    def as_record_fields(self) -> dict[str, Any]:
        """@shanlan-comment Serialize geometry into JSON-safe manifest fields."""
        return {
            "geometry_version": GEOMETRY_VERSION,
            "source_orientation": self.source_orientation,
            "orientation": self.orientation,
            "rotated_ccw": bool(self.rotated_ccw),
            "crop_xyxy_even": list(self.crop_xyxy_even),
            "source_crop_hw": list(self.source_crop_hw),
            "rotated_hw": list(self.rotated_hw),
            "resized_hw": list(self.resized_hw),
            "tile_start_x": int(self.tile_start_x),
            "tile_width": int(self.tile_width),
            "pad_right": int(self.pad_right),
            "valid_hw": list(self.valid_hw),
            "polygon_local": self.polygon_local.astype(float).tolist(),
            "text_size_px": float(self.text_size_px),
            "is_crop_truncated": bool(self.is_crop_truncated),
            "cp_eligible": bool(self.cp_eligible),
            "cp_skip_reason": str(self.cp_skip_reason),
        }


def is_hx_related(record: dict[str, Any]) -> bool:
    """@shanlan-comment Detect HX-related frames from dataset, stem, and asset identifiers."""
    values = [
        str(record.get("dataset", "")),
        str(record.get("stem", "")),
        str(record.get("frame_id", "")),
        str(record.get("lq", "")),
        str(record.get("gt", "")),
        str(record.get("ocr", "")),
    ]
    for raw in values:
        normalized = raw.replace("\\", "/").upper()
        if "HX_TESTSET" in normalized or "/HX/" in normalized:
            return True
        for token in normalized.replace("/", "_").replace("-", "_").split("_"):
            if token.startswith("HX") and len(token) >= 2:
                return True
    return False


def polygon_text_size_px(polygon: Sequence[Sequence[float]]) -> float:
    """@shanlan-comment Return the short side of the minimum-area text rectangle in source pixels."""
    points = polygon_to_array(polygon)
    _, dimensions, _ = cv2.minAreaRect(points)
    width, height = (float(dimensions[0]), float(dimensions[1]))
    return min(width, height)


def _even_width(value: int) -> int:
    """@shanlan-comment Round a positive resized width up to the even YUV420 alignment."""
    value = max(2, int(value))
    return value if value % 2 == 0 else value + 1


def _rotate_polygon_ccw(points: np.ndarray, source_width: int) -> np.ndarray:
    """@shanlan-comment Rotate local polygon coordinates counter-clockwise with the image crop."""
    rotated = np.empty_like(points, dtype=np.float32)
    rotated[:, 0] = points[:, 1]
    # Polygon coordinates describe pixel boundaries, so the right boundary of
    # a width-W crop maps to y=0 rather than y=-1 after the quarter turn.
    rotated[:, 1] = float(source_width) - points[:, 0]
    return rotated


def _split_ranges(width: int, canvas_width: int, overlap: int) -> list[tuple[int, int]]:
    """@shanlan-comment Split an aspect-resized line into even overlapping canvas-width windows."""
    width = int(width)
    canvas_width = int(canvas_width)
    overlap = int(overlap)
    if width <= canvas_width:
        return [(0, width)]
    if canvas_width <= 0 or overlap < 0 or overlap >= canvas_width:
        raise ValueError("invalid canvas width or split overlap")
    stride = max(2, canvas_width - overlap)
    stride -= stride % 2
    ranges: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(width, start + canvas_width)
        if end - start < canvas_width and ranges:
            start = max(0, width - canvas_width)
            end = width
        start -= start % 2
        end = min(width, _even_width(end))
        if end > width:
            end = width
        pair = (start, end)
        if not ranges or pair != ranges[-1]:
            ranges.append(pair)
        if end >= width:
            break
        start += stride
    return ranges


def build_line_geometries(
    polygon: Sequence[Sequence[float]],
    image_hw: tuple[int, int],
    *,
    canvas_hw: tuple[int, int] = DEFAULT_CANVAS_HW,
    min_text_size: float = DEFAULT_MIN_TEXT_SIZE,
    split_overlap: int = DEFAULT_SPLIT_OVERLAP,
    line_index: int = 0,
) -> tuple[list[V4LineGeometry], dict[str, Any] | None]:
    """@shanlan-comment Filter, rotate, aspect-resize, and tile one OCR polygon."""
    del line_index  # The caller owns stable source-line identifiers.
    points = polygon_to_array(polygon)
    image_height, image_width = (int(image_hw[0]), int(image_hw[1]))
    canvas_height, canvas_width = (int(canvas_hw[0]), int(canvas_hw[1]))
    if canvas_height != 96 or canvas_width != 480:
        raise ValueError("V4 geometry requires canvas_hw=(96,480)")
    text_size = polygon_text_size_px(points)
    if text_size < float(min_text_size):
        return [], {
            "reason": "TEXT_SIZE_BELOW_MIN",
            "text_size_px": float(text_size),
            "min_text_size": float(min_text_size),
        }
    crop = even_bbox_crop(points, (image_height, image_width))
    left, top, right, bottom = crop
    source_crop_hw = (bottom - top, right - left)
    local = points - np.asarray([left, top], dtype=np.float32)
    source_orientation = classify_line_orientation(points)
    rotated_ccw = source_orientation == "VERTICAL"
    if rotated_ccw:
        local = _rotate_polygon_ccw(local, source_crop_hw[1])
        rotated_hw = (source_crop_hw[1], source_crop_hw[0])
    else:
        rotated_hw = source_crop_hw
    if rotated_hw[0] <= 0 or rotated_hw[1] <= 0:
        return [], {"reason": "CROP_SIZE_MISMATCH", "text_size_px": float(text_size)}

    resize_scale = float(canvas_height) / float(rotated_hw[0])
    resized_width = _even_width(int(round(rotated_hw[1] * resize_scale)))
    resized_hw = (canvas_height, resized_width)
    local[:, 0] *= float(resized_width) / float(rotated_hw[1])
    local[:, 1] *= float(canvas_height) / float(rotated_hw[0])
    ranges = _split_ranges(resized_width, canvas_width, split_overlap)
    geometries: list[V4LineGeometry] = []
    for tile_index, (tile_start, tile_end) in enumerate(ranges):
        tile_width = int(tile_end - tile_start)
        pad_right = max(0, canvas_width - tile_width)
        tile_polygon = local.copy()
        tile_polygon[:, 0] -= float(tile_start)
        truncated = bool(
            np.any(tile_polygon[:, 0] < 0.0)
            or np.any(tile_polygon[:, 0] > float(tile_width))
        )
        cp_eligible = not truncated
        cp_skip_reason = "" if cp_eligible else "SPLIT_TILE_TRUNCATED"
        geometries.append(
            V4LineGeometry(
                tile_index=tile_index,
                tile_count=len(ranges),
                source_orientation=source_orientation,
                orientation="HORIZONTAL" if rotated_ccw else source_orientation,
                rotated_ccw=rotated_ccw,
                crop_xyxy_even=crop,
                source_crop_hw=source_crop_hw,
                rotated_hw=rotated_hw,
                resized_hw=resized_hw,
                tile_start_x=int(tile_start),
                tile_width=tile_width,
                pad_right=pad_right,
                valid_hw=(canvas_height, tile_width),
                polygon_local=tile_polygon,
                text_size_px=float(text_size),
                is_crop_truncated=truncated,
                cp_eligible=cp_eligible,
                cp_skip_reason=cp_skip_reason,
            )
        )
    return geometries, None


__all__ = [
    "DEFAULT_CANVAS_HW",
    "DEFAULT_MIN_TEXT_SIZE",
    "DEFAULT_SPLIT_OVERLAP",
    "GEOMETRY_VERSION",
    "V4LineGeometry",
    "build_line_geometries",
    "is_hx_related",
    "polygon_text_size_px",
]
