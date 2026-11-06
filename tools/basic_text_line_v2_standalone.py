"""Standalone BASIC Text Line Restore V2 geometry and fusion primitives.

This module intentionally has no dependency on the existing training registry.  It
is useful for SHAPE-S0/P0 contract tests and can be used by a future dataset or
inference adapter after the static shape is frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import ceil, cos, degrees, hypot, radians, sin
from typing import Iterable, Mapping, Sequence

import numpy as np


class Orientation(str, Enum):
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"
    OBLIQUE = "OBLIQUE"
    INVALID = "INVALID"


class BypassReason(str, Enum):
    NONE = ""
    CONSTRAINED_CONTEXT = "CONSTRAINED_CONTEXT"
    OVERSIZE_TEXT_BYPASS = "OVERSIZE_TEXT_BYPASS"
    MULTILINE_UNRESOLVED = "MULTILINE_UNRESOLVED"
    INVALID_POLYGON = "INVALID_POLYGON"


@dataclass(frozen=True)
class ShapeConfigV2:
    """Frozen-shape contract used by both the offline simulator and adapters."""

    canvas_h: int = 96
    canvas_w: int = 384
    margin_ratio: float = 0.25
    min_context: int = 4
    overlap: int = 64
    vertical_group_size: int = 2

    def __post_init__(self) -> None:
        if self.canvas_h <= 0 or self.canvas_w <= 0:
            raise ValueError("canvas dimensions must be positive")
        if self.canvas_h % 32 or self.canvas_w % 32:
            raise ValueError("canvas dimensions must be multiples of 32")
        if not 0 <= self.margin_ratio <= 1:
            raise ValueError("margin_ratio must be in [0, 1]")
        if self.min_context < 0 or self.overlap < 0:
            raise ValueError("context and overlap must be non-negative")
        if self.overlap >= self.canvas_w:
            raise ValueError("overlap must be smaller than canvas_w")
        if self.vertical_group_size < 1:
            raise ValueError("vertical_group_size must be positive")


@dataclass(frozen=True)
class TextLineRegionV2:
    frame_id: str
    line_id: str
    polygon_xy: tuple[tuple[float, float], ...]
    det_score: float = 1.0
    orientation: Orientation = Orientation.INVALID
    long_axis_xy: tuple[float, float] = (0.0, 0.0)
    short_side_px: float = 0.0
    aabb_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    neighbor_line_ids: tuple[str, ...] = ()
    transcription: str | None = None
    char_polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    ignore_reason: str = ""

    def to_dict(self) -> dict:
        value = asdict(self)
        value["orientation"] = self.orientation.value
        return value


@dataclass(frozen=True)
class TextLineSegmentV2:
    segment_id: str
    source_line_id: str
    crop_xyxy_even: tuple[int, int, int, int]
    line_polygon_local: tuple[tuple[float, float], ...]
    valid_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    pad_ltrb: tuple[int, int, int, int]
    segment_axis_range: tuple[float, float]
    tile_index: int
    tile_count: int
    valid_mask: np.ndarray = field(repr=False, compare=False)
    text_mask: np.ndarray = field(repr=False, compare=False)
    blend_mask: np.ndarray = field(repr=False, compare=False)
    bypass_reason: BypassReason = BypassReason.NONE

    def to_dict(self, include_masks: bool = False) -> dict:
        value = asdict(self)
        for key in ("valid_mask", "text_mask", "blend_mask"):
            value.pop(key, None)
        value["bypass_reason"] = self.bypass_reason.value
        if include_masks:
            value.update(
                valid_mask=self.valid_mask.astype(np.uint8).tolist(),
                text_mask=self.text_mask.astype(np.uint8).tolist(),
                blend_mask=self.blend_mask.tolist(),
            )
        return value


def _as_polygon(polygon: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("polygon must be an Nx2 array with N >= 3")
    if not np.isfinite(points).all():
        raise ValueError("polygon contains non-finite coordinates")
    return points


def _crosses(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    points = _as_polygon(polygon)
    return abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))) / 2.0)


def _minimum_rect_axes(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return long axis, long side, short side without requiring OpenCV."""
    best = None
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 1e-12:
            continue
        axis = edge / length
        normal = np.array([-axis[1], axis[0]])
        along = points @ axis
        across = points @ normal
        width = float(across.max() - across.min())
        height = float(along.max() - along.min())
        if height < width:
            axis = normal
            height, width = width, height
        candidate = (width * height, axis, height, width)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("polygon has no non-zero edge")
    _, axis, long_side, short_side = best
    if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
        axis = -axis
    return axis, long_side, short_side


def describe_region(
    frame_id: str,
    line_id: str,
    polygon: Sequence[Sequence[float]],
    *,
    det_score: float = 1.0,
    char_polygons: Iterable[Sequence[Sequence[float]]] = (),
) -> TextLineRegionV2:
    points = _as_polygon(polygon)
    axis, _, short_side = _minimum_rect_axes(points)
    angle = abs(degrees(np.arctan2(axis[1], axis[0])))
    if short_side <= 0 or polygon_area(points) <= 0:
        orientation = Orientation.INVALID
    elif angle <= 15 or angle >= 165:
        orientation = Orientation.HORIZONTAL
    elif 75 <= angle <= 105:
        orientation = Orientation.VERTICAL
    else:
        orientation = Orientation.OBLIQUE
    aabb = (float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max()))
    chars = tuple(tuple(tuple(float(v) for v in p) for p in _as_polygon(char)) for char in char_polygons)
    return TextLineRegionV2(
        frame_id=frame_id,
        line_id=line_id,
        polygon_xy=tuple(tuple(float(v) for v in p) for p in points),
        det_score=float(det_score),
        orientation=orientation,
        long_axis_xy=(float(axis[0]), float(axis[1])),
        short_side_px=float(short_side),
        aabb_xyxy=aabb,
        char_polygons=chars,
    )


def even_crop_xyxy(xyxy: Sequence[float], image_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """Round outward, then clamp to the image while retaining even boundaries."""
    if len(xyxy) != 4:
        raise ValueError("xyxy must contain four values")
    height, width = image_hw
    x0, y0, x1, y1 = map(float, xyxy)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("crop must have positive area")
    ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
    ix1, iy1 = int(np.ceil(x1)), int(np.ceil(y1))
    ix0 -= ix0 % 2
    iy0 -= iy0 % 2
    ix1 += ix1 % 2
    iy1 += iy1 % 2
    ix0, iy0 = max(0, ix0), max(0, iy0)
    ix1, iy1 = min(width, ix1), min(height, iy1)
    if ix1 % 2 and ix1 < width:
        ix1 += 1
    if iy1 % 2 and iy1 < height:
        iy1 += 1
    if ix1 <= ix0 or iy1 <= iy0:
        raise ValueError("crop is empty after image clipping")
    return ix0, iy0, ix1, iy1


def _axis_projection(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    projection = points @ axis
    return float(projection.min()), float(projection.max())


def axis_segments(start: float, end: float, limit: int, overlap: int) -> list[tuple[float, float]]:
    if end <= start or limit <= 0 or overlap < 0 or overlap >= limit:
        raise ValueError("invalid axis segmentation arguments")
    length = end - start
    if length <= limit:
        return [(start, end)]
    stride = limit - overlap
    count = int(ceil((length - overlap) / stride))
    segments = []
    for index in range(count):
        left = start + index * stride
        right = min(left + limit, end)
        if right - left < limit and index:
            left = max(start, end - limit)
            right = end
        segments.append((left, right))
    return segments


def neighbor_limited_margin(short_side_px: float, margin_ratio: float, nearest_distance_px: float | None, min_context: float = 0.0) -> tuple[float, BypassReason]:
    requested = max(0.0, margin_ratio * short_side_px)
    if nearest_distance_px is None or not np.isfinite(nearest_distance_px):
        return requested, BypassReason.NONE
    allowed = min(requested, max(0.0, 0.5 * nearest_distance_px))
    if allowed + 1e-9 < min_context:
        return allowed, BypassReason.CONSTRAINED_CONTEXT
    return allowed, BypassReason.NONE


def build_segments(region: TextLineRegionV2, image_hw: tuple[int, int], config: ShapeConfigV2, *, nearest_distance_px: float | None = None) -> list[TextLineSegmentV2]:
    if region.orientation == Orientation.INVALID:
        return []
    points = np.asarray(region.polygon_xy, dtype=np.float64)
    axis = np.asarray(region.long_axis_xy, dtype=np.float64)
    normal = np.array([-axis[1], axis[0]])
    lo, hi = _axis_projection(points, axis)
    cross_lo, cross_hi = _axis_projection(points, normal)
    margin, reason = neighbor_limited_margin(region.short_side_px, config.margin_ratio, nearest_distance_px, config.min_context)
    cross_lo -= margin
    cross_hi += margin
    hard_limit = config.canvas_h if region.orientation == Orientation.VERTICAL else config.canvas_h
    if region.short_side_px > 1.5 * max(config.min_context, hard_limit - 2 * config.min_context):
        return []
    if region.orientation == Orientation.VERTICAL:
        ranges = axis_segments(lo, hi, config.canvas_h - 2 * config.min_context, max(1, config.overlap // 2))
    else:
        ranges = axis_segments(lo, hi, config.canvas_w, config.overlap)
    segments: list[TextLineSegmentV2] = []
    for index, (axis_lo, axis_hi) in enumerate(ranges):
        corners = np.array([
            axis * axis_lo + normal * cross_lo,
            axis * axis_hi + normal * cross_lo,
            axis * axis_hi + normal * cross_hi,
            axis * axis_lo + normal * cross_hi,
        ])
        crop = even_crop_xyxy((corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()), image_hw)
        x0, y0, x1, y1 = crop
        valid_h, valid_w = y1 - y0, x1 - x0
        if valid_h > config.canvas_h or valid_w > config.canvas_w:
            return []
        pad_l, pad_t = 0, 0
        pad_r, pad_b = config.canvas_w - valid_w, config.canvas_h - valid_h
        local = points - np.array([x0, y0])
        valid_mask = np.ones((valid_h, valid_w), dtype=np.float32)
        text_mask = _polygon_mask(local, (valid_h, valid_w))
        blend_mask = text_mask.copy()
        segments.append(TextLineSegmentV2(
            segment_id=f"{region.line_id}:{index}", source_line_id=region.line_id,
            crop_xyxy_even=crop, line_polygon_local=tuple(tuple(float(v) for v in p) for p in local),
            valid_hw=(valid_h, valid_w), canvas_hw=(config.canvas_h, config.canvas_w),
            pad_ltrb=(pad_l, pad_t, pad_r, pad_b), segment_axis_range=(axis_lo, axis_hi),
            tile_index=index, tile_count=len(ranges), valid_mask=valid_mask,
            text_mask=text_mask, blend_mask=blend_mask, bypass_reason=reason,
        ))
    return segments


def _polygon_mask(polygon_local: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    height, width = hw
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([xx + 0.5, yy + 0.5], axis=-1)
    inside = np.zeros((height, width), dtype=bool)
    for index in range(len(polygon_local)):
        a = polygon_local[index]
        b = polygon_local[(index + 1) % len(polygon_local)]
        crossing = ((a[1] > points[..., 1]) != (b[1] > points[..., 1]))
        x_intersection = (b[0] - a[0]) * (points[..., 1] - a[1]) / ((b[1] - a[1]) + 1e-12) + a[0]
        inside ^= crossing & (points[..., 0] < x_intersection)
    return inside.astype(np.float32)


def pad_to_canvas(roi: np.ndarray, config: ShapeConfigV2, value: float = 0.0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if roi.ndim not in (2, 3):
        raise ValueError("roi must be HxW or HxWxC")
    height, width = roi.shape[:2]
    if height > config.canvas_h or width > config.canvas_w:
        raise ValueError("roi exceeds static canvas")
    padding = ((0, config.canvas_h - height), (0, config.canvas_w - width))
    if roi.ndim == 3:
        padding += ((0, 0),)
    return np.pad(roi, padding, mode="constant", constant_values=value), (0, 0, config.canvas_w - width, config.canvas_h - height)


def merge_segments(segments: Sequence[np.ndarray], *, overlap: int) -> np.ndarray:
    if not segments:
        raise ValueError("segments cannot be empty")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if len(segments) == 1:
        return np.asarray(segments[0]).copy()
    arrays = [np.asarray(item) for item in segments]
    if any(item.ndim != arrays[0].ndim or item.shape[1:] != arrays[0].shape[1:] for item in arrays):
        raise ValueError("all segments must have matching non-spatial dimensions")
    first = arrays[0]
    width = first.shape[1]
    if overlap >= width:
        raise ValueError("overlap must be smaller than segment width")
    output = first.copy()
    for current in arrays[1:]:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        left = output[:, -overlap:]
        right = current[:, :overlap]
        if output.ndim == 3:
            ramp = ramp[None, :, None]
        else:
            ramp = ramp[None, :]
        output = np.concatenate([output[:, :-overlap], left * (1.0 - ramp) + right * ramp, current[:, overlap:]], axis=1)
    return output


def alpha_blend(base: np.ndarray, restored: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    base_array = np.asarray(base, dtype=np.float32)
    restored_array = np.asarray(restored, dtype=np.float32)
    weights = np.asarray(alpha, dtype=np.float32)
    if base_array.shape != restored_array.shape:
        raise ValueError("base and restored shapes must match")
    if weights.shape not in (base_array.shape[:2], base_array.shape):
        raise ValueError("alpha must be spatial or full-shape")
    if weights.ndim == 2:
        weights = weights[..., None]
    return weights * restored_array + (1.0 - weights) * base_array


def yuv420_to_yuv444(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-neighbor chroma expansion for a planar 4:2:0 frame."""
    y_array = np.asarray(y, dtype=np.float32)
    u_array, v_array = np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32)
    if y_array.ndim != 2 or u_array.ndim != 2 or v_array.ndim != 2:
        raise ValueError("Y/U/V planes must be two-dimensional")
    if u_array.shape != v_array.shape or y_array.shape != (u_array.shape[0] * 2, u_array.shape[1] * 2):
        raise ValueError("invalid YUV420 plane shapes")
    return np.stack([y_array, u_array.repeat(2, axis=0).repeat(2, axis=1), v_array.repeat(2, axis=0).repeat(2, axis=1)], axis=-1)


def candidate_shapes() -> tuple[tuple[int, int], ...]:
    return tuple((height, width) for height in (64, 96, 128) for width in (256, 320, 384, 512))


def shape_summary(regions: Sequence[TextLineRegionV2], config: ShapeConfigV2, image_hw: tuple[int, int]) -> dict:
    line_rows = []
    for region in regions:
        segments = build_segments(region, image_hw, config)
        line_rows.append({
            "frame_id": region.frame_id,
            "line_id": region.line_id,
            "orientation": region.orientation.value,
            "short_side_px": region.short_side_px,
            "tiles_per_line": len(segments),
            "seam_count_per_line": max(0, len(segments) - 1),
            "reason": (segments[0].bypass_reason.value if segments else BypassReason.OVERSIZE_TEXT_BYPASS.value),
        })
    return {"canvas_hw": [config.canvas_h, config.canvas_w], "line_count": len(line_rows), "lines": line_rows}


__all__ = [
    "BypassReason", "Orientation", "ShapeConfigV2", "TextLineRegionV2", "TextLineSegmentV2",
    "alpha_blend", "axis_segments", "build_segments", "candidate_shapes", "describe_region",
    "even_crop_xyxy", "merge_segments", "neighbor_limited_margin", "pad_to_canvas", "polygon_area",
    "shape_summary", "yuv420_to_yuv444",
]
