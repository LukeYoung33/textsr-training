"""Single-OCR-line crop contract for BASIC text-line restoration.

The primary unit is one OCR polygon. A crop is created from that polygon's
axis-aligned bounding box at native pixel scale. Long horizontal and vertical
lines are split only when that one-line crop cannot fit the fixed canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


SIZE_FILTER_VERSION = "native_capacity_v1"


@dataclass(frozen=True)
class TextLineCropV3:
    """One native-scale OCR line crop and its fixed-canvas mapping metadata."""

    line_index: int
    tile_index: int
    tile_count: int
    orientation: str
    crop_xyxy_even: tuple[int, int, int, int]
    polygon_local: np.ndarray
    canvas_hw: tuple[int, int]
    valid_hw: tuple[int, int]
    valid_mask: np.ndarray
    text_mask: np.ndarray


def polygon_to_array(polygon: Sequence[Sequence[float]]) -> np.ndarray:
    """@shanlan-comment Validate one OCR polygon and return finite float32 Nx2 coordinates."""
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 4 or not np.isfinite(points).all():
        raise ValueError("OCR polygon must contain at least four finite points")
    return points


def polygon_to_quad(polygon: Sequence[Sequence[float]]) -> np.ndarray:
    """@shanlan-comment Convert an OCR polygon to a finite four-corner quadrilateral for CP crops."""
    points = polygon_to_array(polygon)
    if points.shape[0] == 4:
        return points
    rectangle = cv2.minAreaRect(points)
    return cv2.boxPoints(rectangle).astype(np.float32)


def polygon_aabb(polygon: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """@shanlan-comment Return the tight axis-aligned bounding box of one OCR polygon."""
    points = polygon_to_array(polygon)
    return (float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()))


def classify_line_orientation(polygon: Sequence[Sequence[float]]) -> str:
    """@shanlan-comment Classify one polygon as horizontal, vertical, or oblique by its long edge."""
    points = polygon_to_array(polygon)
    rectangle = cv2.minAreaRect(points)
    corners = cv2.boxPoints(rectangle).astype(np.float32)
    edges = np.roll(corners, -1, axis=0) - corners
    axis = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    angle = abs(float(np.degrees(np.arctan2(axis[1], axis[0])))) % 180.0
    if angle <= 20.0 or angle >= 160.0:
        return "HORIZONTAL"
    if 70.0 <= angle <= 110.0:
        return "VERTICAL"
    return "OBLIQUE"


def assess_crop_size(
    valid_hw: tuple[int, int],
    canvas_hw: tuple[int, int],
    orientation: str,
    *,
    min_short_side_ratio: float = 0.5,
    min_long_side_ratio: float = 0.5,
) -> dict[str, object]:
    """@shanlan-comment
    Decide whether a native text-line crop has sufficient fixed-canvas capacity.

    Input: native valid crop height/width, fixed canvas height/width, orientation,
    and minimum orientation-aware dimension ratios.
    Output: an auditable decision dictionary with acceptance, reason and ratios.
    Key Steps: reject invalid or over-capacity crops, map the long axis to the
    canvas width for horizontal/oblique rows or canvas height for vertical rows,
    then compare both ratios with the configured lower bounds.
    Key Params: default ratios are 0.5; they are training-data capacity controls,
    not image-resizing parameters.
    State/Dependencies: pure function; it does not alter crop coordinates or masks.
    """
    height, width = (int(valid_hw[0]), int(valid_hw[1]))
    canvas_height, canvas_width = (int(canvas_hw[0]), int(canvas_hw[1]))
    short_ratio = 0.0
    long_ratio = 0.0
    result = {
        "accepted": False,
        "reason": "",
        "filter_version": SIZE_FILTER_VERSION,
        "valid_hw": [height, width],
        "short_side_ratio": short_ratio,
        "long_side_ratio": long_ratio,
        "min_short_side_ratio": float(min_short_side_ratio),
        "min_long_side_ratio": float(min_long_side_ratio),
    }
    if height <= 0 or width <= 0:
        result["reason"] = "CROP_SIZE_MISMATCH"
        return result
    if height > canvas_height or width > canvas_width:
        result["reason"] = "CROP_EXCEEDS_CANVAS"
        return result
    if min_short_side_ratio < 0.0 or min_long_side_ratio < 0.0:
        raise ValueError("minimum crop-size ratios must be non-negative")

    short_side = min(height, width)
    long_side = max(height, width)
    long_target = canvas_height if str(orientation).upper() == "VERTICAL" else canvas_width
    short_ratio = short_side / float(canvas_height)
    long_ratio = long_side / float(long_target)
    result["short_side_ratio"] = short_ratio
    result["long_side_ratio"] = long_ratio
    if short_ratio < float(min_short_side_ratio) or long_ratio < float(min_long_side_ratio):
        result["reason"] = "CROP_SIZE_MISMATCH"
        return result
    result["accepted"] = True
    return result


def even_bbox_crop(
    polygon: Sequence[Sequence[float]], image_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """@shanlan-comment Round a tight polygon AABB outward to even in-bounds YUV420 coordinates."""
    x0, y0, x1, y1 = polygon_aabb(polygon)
    height, width = map(int, image_hw)
    left = max(0, int(np.floor(x0)) & ~1)
    top = max(0, int(np.floor(y0)) & ~1)
    right = min(width, int(np.ceil(x1)) + (int(np.ceil(x1)) & 1))
    bottom = min(height, int(np.ceil(y1)) + (int(np.ceil(y1)) & 1))
    if right <= left or bottom <= top:
        raise ValueError("polygon AABB is empty after clipping")
    if (right - left) % 2 and right < width:
        right += 1
    if (bottom - top) % 2 and bottom < height:
        bottom += 1
    return left, top, right, bottom


def _axis_ranges(start: int, end: int, limit: int, overlap: int) -> list[tuple[int, int]]:
    """@shanlan-comment Split one axis into deterministic overlapping native-scale windows."""
    if end <= start or limit <= 0 or overlap < 0 or overlap >= limit:
        raise ValueError("invalid axis split parameters")
    if end - start <= limit:
        return [(start, end)]
    stride = limit - overlap
    ranges = []
    cursor = start
    while cursor < end:
        right = min(end, cursor + limit)
        if right - cursor < limit and ranges:
            cursor = max(start, end - limit)
            right = end
        ranges.append((cursor, right))
        if right >= end:
            break
        cursor += stride
    return ranges


def rasterize_polygon_mask(
    polygon_local: np.ndarray, valid_hw: tuple[int, int], canvas_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """@shanlan-comment Rasterize a line polygon and valid ROI into fixed canvas masks."""
    valid_h, valid_w = valid_hw
    canvas_h, canvas_w = canvas_hw
    valid_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    valid_mask[:valid_h, :valid_w] = 1.0
    text_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    clipped = polygon_local.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, valid_w - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, valid_h - 1))
    cv2.fillPoly(text_mask, [np.rint(clipped).astype(np.int32)], 1.0)
    return valid_mask, text_mask


def build_line_crops(
    polygon: Sequence[Sequence[float]], image_hw: tuple[int, int],
    canvas_hw: tuple[int, int] = (96, 320), overlap: int = 64,
    line_index: int = 0,
) -> list[TextLineCropV3]:
    """@shanlan-comment Build tight one-line bbox crops and split only oversized long axes."""
    points = polygon_to_array(polygon)
    canvas_h, canvas_w = map(int, canvas_hw)
    x0, y0, x1, y1 = even_bbox_crop(points, image_hw)
    orientation = classify_line_orientation(points)
    if (x1 - x0) > canvas_w:
        ranges = _axis_ranges(x0, x1, canvas_w, min(overlap, canvas_w - 1))
        crop_ranges = [(a, y0, b, y1) for a, b in ranges]
    elif (y1 - y0) > canvas_h:
        ranges = _axis_ranges(y0, y1, canvas_h, min(overlap, canvas_h - 1))
        crop_ranges = [(x0, a, x1, b) for a, b in ranges]
    else:
        crop_ranges = [(x0, y0, x1, y1)]
    result = []
    for tile_index, crop in enumerate(crop_ranges):
        left, top, right, bottom = crop
        local = points - np.array([left, top], dtype=np.float32)
        valid_hw = (bottom - top, right - left)
        if valid_hw[0] > canvas_h or valid_hw[1] > canvas_w:
            raise ValueError("one-line crop still exceeds fixed canvas")
        valid_mask, text_mask = rasterize_polygon_mask(local, valid_hw, canvas_hw)
        result.append(TextLineCropV3(
            line_index=line_index, tile_index=tile_index, tile_count=len(crop_ranges),
            orientation=orientation, crop_xyxy_even=crop, polygon_local=local,
            canvas_hw=canvas_hw, valid_hw=valid_hw, valid_mask=valid_mask,
            text_mask=text_mask,
        ))
    return result


__all__ = [
    "SIZE_FILTER_VERSION", "TextLineCropV3", "assess_crop_size", "build_line_crops", "classify_line_orientation",
    "even_bbox_crop", "polygon_aabb", "polygon_to_array", "polygon_to_quad", "rasterize_polygon_mask",
]
