"""Deterministic geometry contracts for BASIC 2.0 text-line strips."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


class OrientationV2(str, Enum):
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"
    OBLIQUE = "OBLIQUE"
    INVALID = "INVALID"


class BypassReasonV2(str, Enum):
    NONE = ""
    INVALID = "INVALID"
    OVERSIZE_TEXT_BYPASS = "OVERSIZE_TEXT_BYPASS"
    MULTILINE_UNRESOLVED = "MULTILINE_UNRESOLVED"
    VERTICAL_SPLIT_UNRELIABLE = "VERTICAL_SPLIT_UNRELIABLE"


@dataclass
class TextLineRegionV2:
    """Validated line-level polygon and derived geometry in frame coordinates."""

    frame_id: str
    line_id: str
    polygon_xy: np.ndarray
    det_score: float = 1.0
    transcription: Optional[str] = None
    char_polygons: Optional[List[np.ndarray]] = None
    neighbor_line_ids: List[str] = field(default_factory=list)
    orientation: OrientationV2 = OrientationV2.INVALID
    long_axis_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    short_side_px: float = 0.0
    aabb_xyxy: Tuple[float, float, float, float] = (0, 0, 0, 0)
    ignore_reason: str = ""


@dataclass
class TextLineSegmentV2:
    """Mapping and masks for one fixed-size BASIC canvas segment."""

    segment_id: str
    source_line_id: str
    crop_xyxy_even: Tuple[int, int, int, int]
    line_polygon_local: np.ndarray
    valid_hw: Tuple[int, int]
    canvas_hw: Tuple[int, int]
    pad_ltrb: Tuple[int, int, int, int]
    segment_axis_range: Tuple[float, float]
    tile_index: int
    tile_count: int
    valid_mask: np.ndarray
    text_mask: np.ndarray
    blend_mask: np.ndarray
    bypass_reason: str = ""
    is_crop_truncated: bool = False
    is_border_clipped: bool = False


def _as_polygon(polygon: Sequence[Sequence[float]]) -> np.ndarray:
    """@shanlan-comment Input polygon-like coordinates; output finite float32 Nx2 array."""
    array = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] < 4 or not np.isfinite(array).all():
        raise ValueError("polygon must contain at least four finite points")
    return array


def classify_orientation(polygon: Sequence[Sequence[float]], oblique_tolerance: float = 20.0) -> OrientationV2:
    """@shanlan-comment Classify a polygon by its PCA long-axis angle in degrees."""
    points = _as_polygon(polygon)
    centered = points - points.mean(axis=0, keepdims=True)
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    axis = vectors[:, -1]
    angle = abs(np.degrees(np.arctan2(float(axis[1]), float(axis[0])))) % 180.0
    if min(angle, 180.0 - angle) <= oblique_tolerance:
        return OrientationV2.HORIZONTAL
    if abs(angle - 90.0) <= oblique_tolerance:
        return OrientationV2.VERTICAL
    return OrientationV2.OBLIQUE


def make_line_region(
    frame_id: str, line_id: str, polygon: Sequence[Sequence[float]], *, det_score: float = 1.0,
    transcription: Optional[str] = None, char_polygons: Optional[Iterable[Sequence[Sequence[float]]]] = None,
) -> TextLineRegionV2:
    """@shanlan-comment Validate polygon and derive AABB, PCA axis, and short side."""
    points = _as_polygon(polygon)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axis = vectors[:, order[0]].astype(np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    long_extent = float(np.ptp(centered @ axis))
    short_extent = float(np.ptp(centered @ normal))
    if long_extent < 1.0 or short_extent < 1.0:
        raise ValueError("degenerate text-line polygon")
    char_list = None if char_polygons is None else [_as_polygon(item) for item in char_polygons]
    return TextLineRegionV2(
        frame_id=str(frame_id), line_id=str(line_id), polygon_xy=points,
        det_score=float(det_score), transcription=transcription, char_polygons=char_list,
        orientation=classify_orientation(points), long_axis_xy=axis,
        short_side_px=short_extent,
        aabb_xyxy=(float(points[:, 0].min()), float(points[:, 1].min()),
                   float(points[:, 0].max()), float(points[:, 1].max())),
    )


def even_crop_xyxy(xyxy: Sequence[float], image_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """@shanlan-comment Expand a crop outward to even YUV420 coordinates and image bounds."""
    height, width = (int(image_hw[0]), int(image_hw[1]))
    x0, y0, x1, y1 = map(float, xyxy)
    ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
    ix1, iy1 = int(np.ceil(x1)), int(np.ceil(y1))
    ix0 = max(0, ix0 - (ix0 % 2)); iy0 = max(0, iy0 - (iy0 % 2))
    ix1 = min(width, ix1 + (ix1 % 2)); iy1 = min(height, iy1 + (iy1 % 2))
    if ix1 <= ix0 or iy1 <= iy0:
        raise ValueError("crop does not intersect image")
    if (ix1 - ix0) % 2: ix1 = min(width, ix1 + 1)
    if (iy1 - iy0) % 2: iy1 = min(height, iy1 + 1)
    return ix0, iy0, ix1, iy1


def neighbor_limited_margin(region: TextLineRegionV2, regions: Iterable[TextLineRegionV2], margin_ratio: float) -> Tuple[float, float, float, float]:
    """@shanlan-comment Return AABB margins limited by the nearest non-overlapping line."""
    x0, y0, x1, y1 = region.aabb_xyxy
    requested = max(0.0, float(margin_ratio)) * region.short_side_px
    margins = [requested] * 4
    for other in regions:
        if other.line_id == region.line_id:
            continue
        ox0, oy0, ox1, oy1 = other.aabb_xyxy
        if ox1 <= x0 and y0 < oy1 and y1 > oy0: margins[0] = min(margins[0], max(0.0, x0 - ox1) * 0.5)
        if ox0 >= x1 and y0 < oy1 and y1 > oy0: margins[2] = min(margins[2], max(0.0, ox0 - x1) * 0.5)
        if oy1 <= y0 and x0 < ox1 and x1 > ox0: margins[1] = min(margins[1], max(0.0, y0 - oy1) * 0.5)
        if oy0 >= y1 and x0 < ox1 and x1 > ox0: margins[3] = min(margins[3], max(0.0, oy0 - y1) * 0.5)
    return tuple(margins)


def capacity_decision(short_side_px: float, canvas_height: int, safe_short_side: float, min_context: float) -> Tuple[str, float]:
    """@shanlan-comment Decide standard, constrained, or bypass handling using fixed canvas thresholds."""
    hard = min(float(canvas_height) - 2.0 * float(min_context), 1.5 * float(safe_short_side))
    if short_side_px > hard:
        return BypassReasonV2.OVERSIZE_TEXT_BYPASS.value, hard
    if short_side_px > safe_short_side:
        return "CONSTRAINED_CONTEXT", hard
    return "", hard


def _axis_projection(points: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    values = points @ axis
    return float(values.min()), float(values.max())


def split_horizontal_line(region: TextLineRegionV2, canvas_width: int, overlap: int) -> List[Tuple[float, float]]:
    """@shanlan-comment Produce deterministic overlapping long-axis ranges for a line."""
    if canvas_width <= 0 or overlap < 0 or overlap >= canvas_width:
        raise ValueError("canvas_width must exceed non-negative overlap")
    lo, hi = _axis_projection(region.polygon_xy, region.long_axis_xy)
    span = hi - lo
    if span <= canvas_width:
        return [(lo, hi)]
    stride = canvas_width - overlap
    ranges = []
    start = lo
    while start < hi:
        end = min(hi, start + canvas_width)
        ranges.append((start, end))
        if end >= hi: break
        start += stride
    return ranges


def split_vertical_line(region: TextLineRegionV2, canvas_height: int, overlap: int = 8) -> List[Tuple[float, float]]:
    """@shanlan-comment Split a vertical line along its long axis without rotating characters."""
    ranges = split_horizontal_line(region, canvas_height, overlap)
    return ranges


def make_segment(
    region: TextLineRegionV2, axis_range: Tuple[float, float], canvas_hw: Tuple[int, int],
    image_hw: Tuple[int, int], tile_index: int, tile_count: int, *, context: float = 0.0,
) -> TextLineSegmentV2:
    """@shanlan-comment Build an even-aligned segment and canvas masks from an axis range."""
    ch, cw = map(int, canvas_hw)
    axis = region.long_axis_xy
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    p = region.polygon_xy
    along = p @ axis; across = p @ normal
    lo, hi = float(along.min()), float(along.max())
    nlo, nhi = float(across.min()), float(across.max())
    crop_points = np.array([
        axis * (axis_range[0] - context) + normal * (nlo - context),
        axis * (axis_range[1] + context) + normal * (nlo - context),
        axis * (axis_range[1] + context) + normal * (nhi + context),
        axis * (axis_range[0] - context) + normal * (nhi + context),
    ])
    aabb = (crop_points[:, 0].min(), crop_points[:, 1].min(), crop_points[:, 0].max(), crop_points[:, 1].max())
    crop = even_crop_xyxy(aabb, image_hw)
    x0, y0, x1, y1 = crop
    vh, vw = y1 - y0, x1 - x0
    if vh > ch or vw > cw:
        raise ValueError("segment exceeds fixed canvas; apply capacity decision first")
    local = p - np.array([x0, y0], dtype=np.float32)
    valid = np.ones((vh, vw), dtype=np.float32)
    valid_canvas = np.zeros((ch, cw), dtype=np.float32); valid_canvas[:vh, :vw] = valid
    text = np.zeros((ch, cw), dtype=np.float32)
    # Polygon rasterization is deliberately dependency-free; point-in-polygon uses scanline rows.
    for yy in range(vh):
        xs = local[:, 0]; ys = local[:, 1]
        inside = np.zeros(vw, dtype=bool)
        for xx in range(vw):
            j = len(local) - 1; hit = False
            for i in range(len(local)):
                if ((ys[i] > yy) != (ys[j] > yy)) and (xx < (xs[j]-xs[i])*(yy-ys[i])/(ys[j]-ys[i]+1e-12)+xs[i]): hit = not hit
                j = i
            inside[xx] = hit
        text[yy, :vw] = inside.astype(np.float32)
    blend = text.copy()
    if vw > 2:
        blend[:, 0] *= 0.25; blend[:, -1] *= 0.25
    return TextLineSegmentV2(
        segment_id=f"{region.line_id}_{tile_index:04d}", source_line_id=region.line_id,
        crop_xyxy_even=crop, line_polygon_local=local, valid_hw=(vh, vw), canvas_hw=(ch, cw),
        pad_ltrb=(0, 0, cw-vw, ch-vh), segment_axis_range=axis_range,
        tile_index=tile_index, tile_count=tile_count, valid_mask=valid_canvas,
        text_mask=text, blend_mask=blend,
        is_border_clipped=bool(x0 == 0 or y0 == 0 or x1 == image_hw[1] or y1 == image_hw[0]),
    )
