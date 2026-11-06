"""Full-frame geometry and fusion helpers for the BASIC text-line model.

The trained BASIC V2/R4 Student is a native-scale fixed-canvas model.  This
module therefore deliberately does not resize or perspective-warp OCR lines:
it expands the polygon AABB when capacity allows, splits only along the long
axis when necessary, and returns mappings back to the original frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


CANVAS_HW = (96, 320)


@dataclass(frozen=True)
class OCRLine:
    """One validated OCR line in full-frame coordinates."""

    line_index: int
    polygon_xy: np.ndarray
    score: float = 1.0
    text: str = ""
    source: str = "ocr"


@dataclass(frozen=True)
class FullFrameSegment:
    """One fixed-canvas segment and its full-frame mapping metadata."""

    line_index: int
    tile_index: int
    tile_count: int
    orientation: str
    crop_xyxy: tuple[int, int, int, int]
    polygon_local: np.ndarray
    valid_hw: tuple[int, int]
    axis: str
    resize_mode: str = "native"
    resize_scale: float = 1.0
    model_valid_hw: tuple[int, int] | None = None
    model_canvas_hw: tuple[int, int] | None = None
    core_mask_local: np.ndarray | None = None


def _finite_score(value: Any, default: float = 1.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(default)
    return score if np.isfinite(score) else float(default)


def normalize_ocr_polygon(raw: Any) -> np.ndarray:
    """Normalize a polygon or ``[x0,y0,x1,y1]`` box into an ``Nx2`` array."""
    array = np.asarray(raw, dtype=np.float32)
    if array.size == 4 and (array.ndim == 1 or array.shape == (1, 4)):
        x0, y0, x1, y1 = map(float, array.reshape(-1))
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        array = np.asarray(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )
    else:
        if array.size % 2:
            raise ValueError("OCR polygon coordinate count must be even")
        array = array.reshape(-1, 2)
    if array.shape[0] < 3 or not np.isfinite(array).all():
        raise ValueError("OCR polygon must contain at least three finite points")
    return array


def _field(data: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def parse_ocr_json(data: Mapping[str, Any] | Sequence[Mapping[str, Any]], image_hw: tuple[int, int]) -> list[OCRLine]:
    """Parse common PPOCR-style JSON fields and clip lines to the frame.

    Supported frame-level fields are ``rec_polys``, ``dt_polys`` and
    ``rec_boxes``.  Text and score fields are optional.  A list of per-line
    dictionaries with ``polygon``/``points``/``bbox`` is accepted as well.
    """
    height, width = map(int, image_hw)
    if height <= 0 or width <= 0:
        raise ValueError("image_hw must be positive")

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        entries = list(data)
        polygons = []
        texts = []
        scores = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            raw = _field(entry, ("polygon", "points", "poly", "bbox", "box"))
            if raw is None:
                continue
            polygons.append(raw)
            texts.append(_field(entry, ("text", "transcription", "rec_text"), ""))
            scores.append(_field(entry, ("score", "rec_score", "det_score"), 1.0))
    else:
        if not isinstance(data, Mapping):
            raise ValueError("OCR JSON must be an object or a list of line objects")
        polygon_field = None
        polygons = []
        for name in ("rec_polys", "dt_polys", "rec_boxes"):
            if data.get(name) is not None:
                polygon_field = name
                polygons = data[name]
                break
        if polygon_field is None:
            polygons = []
        texts = _field(data, ("rec_texts", "rec_text", "text_word", "texts"), []) or []
        scores = _field(data, ("rec_scores", "rec_score", "det_scores", "dt_scores"), []) or []

    lines: list[OCRLine] = []
    for index, raw in enumerate(polygons):
        try:
            polygon = normalize_ocr_polygon(raw).copy()
        except (TypeError, ValueError):
            continue
        polygon[:, 0] = np.clip(polygon[:, 0], 0.0, max(0.0, width - 1.0))
        polygon[:, 1] = np.clip(polygon[:, 1], 0.0, max(0.0, height - 1.0))
        x0, y0 = polygon.min(axis=0)
        x1, y1 = polygon.max(axis=0)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            continue
        text = str(texts[index]).strip() if index < len(texts) else ""
        score = _finite_score(scores[index], 1.0) if index < len(scores) else 1.0
        lines.append(OCRLine(index, polygon, score=max(0.0, score), text=text))
    return lines


def parse_p17_ocr_annotations(data: Mapping[str, Any], image_hw: tuple[int, int]) -> list[OCRLine]:
    """Parse P17 manifest OCR annotations without invoking an OCR detector.

    P17 stores crop-local geometry under ``ocr.annotations``.  The polygon is
    used for compositing while its AABB is used by the P17 crop builder, which
    matches the training dataset contract.
    """
    if not isinstance(data, Mapping):
        raise ValueError("P17 OCR JSON must be an object")
    annotations = data.get("ocr", {}).get("annotations", [])
    if not annotations and isinstance(data.get("lines"), Sequence):
        annotations = [
            {
                "polygon": line.get("polygon") or line.get("bbox_xyxy"),
                "score": line.get("rec_score", 1.0),
                "text": line.get("text", ""),
                "ignore": line.get("ignore", False),
            }
            for line in data["lines"] if isinstance(line, Mapping)
        ]
    if not isinstance(annotations, Sequence):
        raise ValueError("P17 OCR annotations must be a sequence")
    height, width = map(int, image_hw)
    repaired_geometry = bool(data.get("ocr_geometry_repair"))
    lines: list[OCRLine] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping) or annotation.get("ignore") or annotation.get("visible") is False:
            continue
        raw = _field(annotation, ("points", "polygon", "bbox", "box"))
        # The OCR-fix data set contains per-character font runs.  Some crop
        # boundaries leave early characters outside the crop while later ones
        # remain visible.  An envelope over both sets becomes a very tall
        # pseudo-line.  Rebuild repaired geometry from only the runs that
        # intersect this crop; without a visible run, skip the unsafe line.
        if repaired_geometry and isinstance(annotation.get("font_runs"), Sequence):
            visible_runs: list[np.ndarray] = []
            for run in annotation["font_runs"]:
                if not isinstance(run, Mapping):
                    continue
                run_raw = _field(run, ("bbox", "points", "polygon", "box"))
                if run_raw is None:
                    continue
                try:
                    run_polygon = normalize_ocr_polygon(run_raw).copy()
                except (TypeError, ValueError):
                    continue
                run_x0, run_y0 = run_polygon.min(axis=0)
                run_x1, run_y1 = run_polygon.max(axis=0)
                if run_x1 <= 0.0 or run_y1 <= 0.0 or run_x0 >= width or run_y0 >= height:
                    continue
                run_polygon[:, 0] = np.clip(run_polygon[:, 0], 0.0, max(0.0, width - 1.0))
                run_polygon[:, 1] = np.clip(run_polygon[:, 1], 0.0, max(0.0, height - 1.0))
                if np.ptp(run_polygon[:, 0]) >= 1.0 and np.ptp(run_polygon[:, 1]) >= 1.0:
                    visible_runs.append(run_polygon)
            if visible_runs:
                all_runs = np.concatenate(visible_runs, axis=0)
                x0, y0 = all_runs.min(axis=0)
                x1, y1 = all_runs.max(axis=0)
                raw = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            else:
                # The original repaired envelope is not trustworthy if none of
                # its component glyphs is visible in the current crop.
                continue
        if raw is None:
            continue
        try:
            polygon = normalize_ocr_polygon(raw).copy()
        except (TypeError, ValueError):
            continue
        polygon[:, 0] = np.clip(polygon[:, 0], 0.0, max(0.0, width - 1.0))
        polygon[:, 1] = np.clip(polygon[:, 1], 0.0, max(0.0, height - 1.0))
        x0, y0 = polygon.min(axis=0)
        x1, y1 = polygon.max(axis=0)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            continue
        lines.append(OCRLine(
            line_index=index,
            polygon_xy=polygon,
            score=max(0.0, _finite_score(_field(annotation, ("score", "confidence"), 1.0))),
            text=str(_field(annotation, ("transcription", "text", "raw_text"), "")).strip(),
        ))
    return lines


def pad_even_rgb(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad right/bottom by edge replication for an even YUV420 working frame."""
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")
    original_hw = (int(image.shape[0]), int(image.shape[1]))
    pad_bottom = image.shape[0] % 2
    pad_right = image.shape[1] % 2
    if not pad_bottom and not pad_right:
        return image.copy(), original_hw
    padded = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    return padded.astype(np.float32, copy=False), original_hw


def _even_bounds(x0: float, y0: float, x1: float, y1: float, image_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = map(int, image_hw)
    left = max(0, int(np.floor(x0)) & ~1)
    top = max(0, int(np.floor(y0)) & ~1)
    right = min(width, int(np.ceil(x1)))
    bottom = min(height, int(np.ceil(y1)))
    if right % 2:
        right = min(width, right + 1)
    if bottom % 2:
        bottom = min(height, bottom + 1)
    if right - left <= 0 or bottom - top <= 0:
        raise ValueError("empty crop")
    if (right - left) % 2:
        if right < width:
            right += 1
        elif left >= 2:
            left -= 2
    if (bottom - top) % 2:
        if bottom < height:
            bottom += 1
        elif top >= 2:
            top -= 2
    return left, top, right, bottom


def _axis_ranges(start: int, end: int, limit: int, overlap: int) -> list[tuple[int, int]]:
    if end <= start or limit <= 0 or overlap < 0 or overlap >= limit:
        raise ValueError("invalid axis split parameters")
    if end - start <= limit:
        return [(start, end)]
    stride = limit - overlap
    ranges: list[tuple[int, int]] = []
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


def _orientation(width: int, height: int) -> tuple[str, str]:
    if width >= height:
        return "HORIZONTAL", "x"
    return "VERTICAL", "y"


def build_full_frame_segments(
    polygon: Sequence[Sequence[float]],
    image_hw: tuple[int, int],
    canvas_hw: tuple[int, int] = CANVAS_HW,
    *,
    context_ratio: float = 0.15,
    overlap: int = 64,
) -> tuple[list[FullFrameSegment], str]:
    """Build native-scale, even-aligned segments from one OCR polygon AABB.

    The context is reduced only when the fixed canvas cannot hold the expanded
    box.  If the short axis itself exceeds the canvas, the line is bypassed;
    no geometric resize is introduced at inference time.
    """
    points = normalize_ocr_polygon(polygon)
    canvas_h, canvas_w = map(int, canvas_hw)
    if canvas_h <= 0 or canvas_w <= 0 or canvas_h % 2 or canvas_w % 2:
        raise ValueError("canvas dimensions must be positive and even")
    height, width = map(int, image_hw)
    base = _even_bounds(float(points[:, 0].min()), float(points[:, 1].min()),
                        float(points[:, 0].max()), float(points[:, 1].max()), image_hw)
    base_left, base_top, base_right, base_bottom = base
    base_w, base_h = base_right - base_left, base_bottom - base_top
    orientation, axis = _orientation(base_w, base_h)
    short_capacity = canvas_h if axis == "x" else canvas_w
    short_size = base_h if axis == "x" else base_w
    if short_size > short_capacity:
        return [], "SHORT_AXIS_EXCEEDS_CANVAS"

    requested_margin = int(round(min(base_w, base_h) * float(context_ratio)))
    if requested_margin < 0:
        margin = min(-requested_margin, max(0, min(base_w, base_h) // 2 - 1))
        margin -= margin % 2
        expanded_left, expanded_top = base_left + margin, base_top + margin
        expanded_right, expanded_bottom = base_right - margin, base_bottom - margin
    else:
        margin = requested_margin - (requested_margin % 2)
        if axis == "x":
            margin = min(margin, max(0, (canvas_h - base_h) // 2))
        else:
            margin = min(margin, max(0, (canvas_w - base_w) // 2))
        margin -= margin % 2
        expanded_left = max(0, base_left - margin)
        expanded_top = max(0, base_top - margin)
        expanded_right = min(width, base_right + margin)
        expanded_bottom = min(height, base_bottom + margin)
    expanded_left, expanded_top, expanded_right, expanded_bottom = _even_bounds(
        expanded_left, expanded_top, expanded_right, expanded_bottom, image_hw
    )
    expanded_w = expanded_right - expanded_left
    expanded_h = expanded_bottom - expanded_top

    if axis == "x":
        if expanded_h > canvas_h:
            return [], "SHORT_AXIS_EXCEEDS_CANVAS"
        ranges = _axis_ranges(expanded_left, expanded_right, canvas_w, min(overlap, canvas_w - 2))
    else:
        if expanded_w > canvas_w:
            return [], "SHORT_AXIS_EXCEEDS_CANVAS"
        ranges = _axis_ranges(expanded_top, expanded_bottom, canvas_h, min(overlap, canvas_h - 2))

    crops: list[tuple[int, int, int, int]] = []
    for start, end in ranges:
        if axis == "x":
            crop = _even_bounds(start, expanded_top, end, expanded_bottom, image_hw)
        else:
            crop = _even_bounds(expanded_left, start, expanded_right, end, image_hw)
        if crop[2] - crop[0] > canvas_w or crop[3] - crop[1] > canvas_h:
            return [], "CROP_EXCEEDS_CANVAS"
        crops.append(crop)

    segments: list[FullFrameSegment] = []
    for tile_index, crop in enumerate(crops):
        left, top, right, bottom = crop
        local = points - np.asarray([left, top], dtype=np.float32)
        segments.append(FullFrameSegment(
            line_index=-1,
            tile_index=tile_index,
            tile_count=len(crops),
            orientation=orientation,
            crop_xyxy=crop,
            polygon_local=local,
            valid_hw=(bottom - top, right - left),
            axis=axis,
        ))
    return segments, ""


def build_native_tiled_full_frame_segments(
    polygon: Sequence[Sequence[float]],
    image_hw: tuple[int, int],
    canvas_hw: tuple[int, int] = CANVAS_HW,
    *,
    overlap_x: int = 64,
    overlap_y: int = 24,
) -> tuple[list[FullFrameSegment], str]:
    """@luna-comment Split an oversized OCR region into native-scale 2-D tiles.

    Input: OCR polygon, source HW and model canvas HW in pixels.
    Output: Native segments and failure reason; source pixels are not modified.
    Key Steps: Even-align bounds and enumerate overlapping x/y ranges.
    Key Params: Defaults use 96x480 canvas and 64/24 horizontal/vertical overlap.
    State/Dependencies: Pure geometry using the shared bounds/range helpers.
    This route is for genuinely large glyphs and wide vertical text.  It never
    rescales source pixels: every crop is at most the fixed model canvas and
    overlapping predictions are fused by the caller in frame coordinates.
    """
    points = normalize_ocr_polygon(polygon)
    canvas_h, canvas_w = map(int, canvas_hw)
    left, top, right, bottom = _even_bounds(
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()), image_hw,
    )
    x_ranges = _axis_ranges(left, right, canvas_w, min(int(overlap_x), canvas_w - 2))
    y_ranges = _axis_ranges(top, bottom, canvas_h, min(int(overlap_y), canvas_h - 2))
    orientation, axis = _orientation(right - left, bottom - top)
    crops: list[tuple[int, int, int, int]] = []
    for y0, y1 in y_ranges:
        for x0, x1 in x_ranges:
            crop = _even_bounds(x0, y0, x1, y1, image_hw)
            if crop[2] - crop[0] > canvas_w or crop[3] - crop[1] > canvas_h:
                return [], "NATIVE_TILE_EXCEEDS_CANVAS"
            crops.append(crop)
    segments: list[FullFrameSegment] = []
    for tile_index, crop in enumerate(crops):
        left, top, right, bottom = crop
        segments.append(FullFrameSegment(
            line_index=-1,
            tile_index=tile_index,
            # Each 2-D tile is independently weighted; overlapping tiles are
            # normalized by the compositor's accumulated line weight.
            tile_count=1,
            orientation=orientation,
            crop_xyxy=crop,
            polygon_local=points - np.asarray([left, top], dtype=np.float32),
            valid_hw=(bottom - top, right - left),
            axis=axis,
            model_canvas_hw=(canvas_h, canvas_w),
        ))
    return segments, ""


def _even_scaled_dimension(value: int, scale: float) -> int:
    """Return a positive even dimension after a uniform resize."""
    result = max(2, int(round(float(value) * float(scale))))
    return result if result % 2 == 0 else result + 1


def build_resized_full_frame_segments(
    polygon: Sequence[Sequence[float]],
    image_hw: tuple[int, int],
    canvas_hw: tuple[int, int] = CANVAS_HW,
    *,
    context_ratio: float = 0.15,
    overlap: int = 64,
    min_ratio: float = 0.5,
    max_ratio: float = 1.5,
    target_short_ratio: float = 0.75,
) -> tuple[list[FullFrameSegment], str]:
    """Build optional uniform-resize segments for native short-axis bypasses.

    The eligibility ratio is measured on the polygon AABB before context
    expansion.  A ratio in ``[0.5, 1.5]`` maps to a uniform resize scale in
    ``[0.5, 1.5]`` by placing the core short axis at 75% of the model short
    capacity.  The remaining 25% provides a small context band.  The long
    axis is then split in source coordinates so each resized tile remains
    inside the fixed model canvas.  All coordinates and masks remain in the
    original frame; only the model input/output tile is resized.
    """
    points = normalize_ocr_polygon(polygon)
    canvas_h, canvas_w = map(int, canvas_hw)
    if canvas_h <= 0 or canvas_w <= 0 or canvas_h % 2 or canvas_w % 2:
        raise ValueError("canvas dimensions must be positive and even")
    if not (0.0 < float(target_short_ratio) <= 1.0):
        raise ValueError("target_short_ratio must be in (0, 1]")
    if min_ratio < 0.0 or max_ratio < min_ratio:
        raise ValueError("invalid resize ratio range")

    height, width = map(int, image_hw)
    base_left, base_top, base_right, base_bottom = _even_bounds(
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()), image_hw
    )
    base_w, base_h = base_right - base_left, base_bottom - base_top
    orientation, axis = _orientation(base_w, base_h)
    short_capacity = canvas_h if axis == "x" else canvas_w
    short_size = base_h if axis == "x" else base_w
    ratio = short_size / float(short_capacity)
    if ratio < float(min_ratio) or ratio > float(max_ratio):
        return [], "RESIZE_RATIO_OUT_OF_RANGE"

    target_short = max(2, int(round(short_capacity * float(target_short_ratio))))
    target_short -= target_short % 2
    scale = target_short / float(short_size)
    if scale <= 0.0 or not np.isfinite(scale):
        return [], "RESIZE_SCALE_INVALID"

    requested_margin = int(round(min(base_w, base_h) * float(context_ratio)))
    if requested_margin < 0:
        margin = min(-requested_margin, max(0, min(base_w, base_h) // 2 - 1))
        margin -= margin % 2
        expanded = _even_bounds(base_left + margin, base_top + margin, base_right - margin, base_bottom - margin, image_hw)
        expanded_short = (expanded[3] - expanded[1]) if axis == "x" else (expanded[2] - expanded[0])
        scale = target_short / float(short_size)
        if _even_scaled_dimension(expanded_short, scale) > short_capacity:
            return [], "RESIZE_CONTEXT_EXCEEDS_CANVAS"
        margin = -margin
    else:
        margin = requested_margin - (requested_margin % 2)
        expanded = None
    max_source_short = canvas_h / scale if axis == "x" else canvas_w / scale
    max_margin = max(0, int(np.floor((max_source_short - short_size) / 2.0)))
    margin = min(margin, max_margin)
    margin -= margin % 2

    expanded_short = 0
    # Account for even-coordinate rounding after the context is clipped.
    # Reducing the margin by two is deterministic and keeps the resized crop
    # within the fixed canvas without silently changing the core scale.
    while expanded is None and margin >= 0:
        expanded_left = max(0, base_left - margin)
        expanded_top = max(0, base_top - margin)
        expanded_right = min(width, base_right + margin)
        expanded_bottom = min(height, base_bottom + margin)
        candidate = _even_bounds(
            expanded_left, expanded_top, expanded_right, expanded_bottom, image_hw
        )
        candidate_short = (candidate[3] - candidate[1]) if axis == "x" else (candidate[2] - candidate[0])
        if _even_scaled_dimension(candidate_short, scale) <= short_capacity:
            expanded = candidate
            expanded_short = candidate_short
            break
        margin -= 2
    if expanded is None:
        return [], "RESIZE_CONTEXT_EXCEEDS_CANVAS"

    long_capacity = canvas_w if axis == "x" else canvas_h
    source_limit = int(np.floor(long_capacity / scale))
    source_limit -= source_limit % 2
    if source_limit < 2:
        return [], "RESIZE_LONG_AXIS_INVALID"
    model_overlap = min(int(overlap), long_capacity - 2)
    source_overlap = int(round(model_overlap / scale))
    source_overlap = min(source_overlap, source_limit - 2)
    source_overlap -= source_overlap % 2

    if axis == "x":
        ranges = _axis_ranges(expanded[0], expanded[2], source_limit, source_overlap)
    else:
        ranges = _axis_ranges(expanded[1], expanded[3], source_limit, source_overlap)

    segments: list[FullFrameSegment] = []
    for tile_index, (start, end) in enumerate(ranges):
        if axis == "x":
            crop = _even_bounds(start, expanded[1], end, expanded[3], image_hw)
        else:
            crop = _even_bounds(expanded[0], start, expanded[2], end, image_hw)
        left, top, right, bottom = crop
        valid_hw = (bottom - top, right - left)
        model_valid_hw = (
            _even_scaled_dimension(valid_hw[0], scale),
            _even_scaled_dimension(valid_hw[1], scale),
        )
        if model_valid_hw[0] > canvas_h or model_valid_hw[1] > canvas_w:
            return [], "RESIZE_CROP_EXCEEDS_CANVAS"
        local = points - np.asarray([left, top], dtype=np.float32)
        segments.append(FullFrameSegment(
            line_index=-1,
            tile_index=tile_index,
            tile_count=len(ranges),
            orientation=orientation,
            crop_xyxy=crop,
            polygon_local=local,
            valid_hw=valid_hw,
            axis=axis,
            resize_mode="uniform_short_axis",
            resize_scale=float(scale),
            model_valid_hw=model_valid_hw,
        ))
    return segments, ""


def attach_line_index(segments: Sequence[FullFrameSegment], line_index: int) -> list[FullFrameSegment]:
    """Attach the source OCR line index without mutating segment geometry."""
    return [
        FullFrameSegment(
            line_index=int(line_index), tile_index=item.tile_index, tile_count=item.tile_count,
            orientation=item.orientation, crop_xyxy=item.crop_xyxy,
            polygon_local=item.polygon_local, valid_hw=item.valid_hw, axis=item.axis,
            resize_mode=item.resize_mode, resize_scale=item.resize_scale,
            model_valid_hw=item.model_valid_hw, model_canvas_hw=item.model_canvas_hw,
            core_mask_local=item.core_mask_local,
        )
        for item in segments
    ]


def build_height96_tiled_full_frame_segments(polygon: Sequence[Sequence[float]], image_hw: tuple[int, int], canvas_hw: tuple[int, int] = CANVAS_HW, *, overlap: int = 48) -> tuple[list[FullFrameSegment], str]:
    """Build no-context OCR crops resized to height 96 and tiled along width."""
    points = normalize_ocr_polygon(polygon)
    canvas_h, canvas_w = map(int, canvas_hw)
    left, top, right, bottom = _even_bounds(float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max()), image_hw)
    source_h, source_w = bottom - top, right - left
    if source_h < 2 or source_w < 2:
        return [], "EMPTY_OCR_BOUNDS"
    scale = canvas_h / float(source_h)
    source_limit = max(2, int(np.floor(canvas_w / scale)) // 2 * 2)
    source_overlap = min(max(0, int(round(overlap / scale)) // 2 * 2), source_limit - 2)
    ranges = _axis_ranges(left, right, source_limit, source_overlap)
    segments: list[FullFrameSegment] = []
    for tile_index, (start, end) in enumerate(ranges):
        crop = _even_bounds(start, top, end, bottom, image_hw)
        valid_h, valid_w = crop[3] - crop[1], crop[2] - crop[0]
        model_w = min(canvas_w, max(2, int(round(valid_w * scale))))
        segments.append(FullFrameSegment(line_index=-1, tile_index=tile_index, tile_count=len(ranges), orientation="HORIZONTAL", crop_xyxy=crop, polygon_local=points - np.asarray([crop[0], crop[1]], dtype=np.float32), valid_hw=(valid_h, valid_w), axis="x", resize_mode="height96_tile_pad", resize_scale=float(scale), model_valid_hw=(canvas_h, model_w)))
    return segments, ""


def build_p17_height96_tiled_full_frame_segments(
    polygon: Sequence[Sequence[float]], image_hw: tuple[int, int], canvas_hw: tuple[int, int] = CANVAS_HW,
    *, vertical_context: float = 0.12, horizontal_context: float = 0.03, overlap: int = 64,
) -> tuple[list[FullFrameSegment], str]:
    """Build P17's OCR-AABB crops with training-matched context and tiling.

    The short axis is resized to 96 px exactly as P17 training's
    ``height_crop_pad`` route.  Long OCR lines are tiled only after that
    scaling so full-frame inference covers content that the fixed canvas
    cannot hold in one pass.
    """
    points = normalize_ocr_polygon(polygon)
    canvas_h, canvas_w = map(int, canvas_hw)
    if canvas_h <= 0 or canvas_w <= 0 or canvas_h % 2 or canvas_w % 2:
        raise ValueError("canvas dimensions must be positive and even")
    if vertical_context < 0.0 or horizontal_context < 0.0:
        raise ValueError("P17 context ratios must be non-negative")
    left, top, right, bottom = _even_bounds(
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()), image_hw,
    )
    line_h, line_w = bottom - top, right - left
    if line_h < 2 or line_w < 2:
        return [], "EMPTY_OCR_BOUNDS"
    pad_y = max(2, int(round(line_h * float(vertical_context))))
    pad_x = max(2, int(round(line_w * float(horizontal_context))))
    expanded_left, expanded_top, expanded_right, expanded_bottom = _even_bounds(
        left - pad_x, top - pad_y, right + pad_x, bottom + pad_y, image_hw,
    )
    source_h = expanded_bottom - expanded_top
    if source_h < 2:
        return [], "EMPTY_CONTEXT_BOUNDS"
    scale = canvas_h / float(source_h)
    source_limit = max(2, int(np.floor(canvas_w / scale)) // 2 * 2)
    source_overlap = min(max(0, int(round(overlap / scale)) // 2 * 2), source_limit - 2)
    ranges = _axis_ranges(expanded_left, expanded_right, source_limit, source_overlap)
    segments: list[FullFrameSegment] = []
    for tile_index, (start, end) in enumerate(ranges):
        crop = _even_bounds(start, expanded_top, end, expanded_bottom, image_hw)
        valid_h, valid_w = crop[3] - crop[1], crop[2] - crop[0]
        model_w = min(canvas_w, max(2, int(round(valid_w * scale))))
        segments.append(FullFrameSegment(
            line_index=-1, tile_index=tile_index, tile_count=len(ranges),
            orientation="HORIZONTAL", crop_xyxy=crop,
            polygon_local=points - np.asarray([crop[0], crop[1]], dtype=np.float32),
            valid_hw=(valid_h, valid_w), axis="x", resize_mode="height96_tile_pad",
            resize_scale=float(scale), model_valid_hw=(canvas_h, model_w),
        ))
    return segments, ""


def rasterize_polygon_mask(polygon_local: np.ndarray, valid_hw: tuple[int, int]) -> np.ndarray:
    """Rasterize the OCR core inside a segment's valid crop."""
    height, width = map(int, valid_hw)
    mask = np.zeros((height, width), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return mask.astype(np.float32)
    clipped = np.asarray(polygon_local, dtype=np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, height - 1))
    cv2.fillPoly(mask, [np.rint(clipped).astype(np.int32)], 255)
    return mask.astype(np.float32) / 255.0


def feather_alpha(core_mask: np.ndarray, radius: int) -> np.ndarray:
    """Create an opaque core and an outward ring that reaches zero at crop edges."""
    core = (np.asarray(core_mask, dtype=np.float32) > 0.5).astype(np.uint8)
    if not np.any(core):
        return np.zeros_like(core, dtype=np.float32)
    if int(radius) <= 0:
        return core.astype(np.float32)
    outside = (1 - core).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    alpha = np.clip(1.0 - distance / float(radius), 0.0, 1.0).astype(np.float32)
    # The model has no prediction outside its crop.  Taper the non-core ring
    # again toward the crop boundary so a small available context never ends
    # with a non-zero candidate abruptly switching back to the original frame.
    height, width = core.shape
    yy, xx = np.mgrid[:height, :width]
    edge_distance = np.minimum.reduce([yy, xx, height - 1 - yy, width - 1 - xx]).astype(np.float32)
    alpha *= np.clip(edge_distance / float(max(1, radius)), 0.0, 1.0)
    alpha[core > 0] = 1.0
    return alpha


def tile_axis_weight(valid_hw: tuple[int, int], axis: str, tile_count: int) -> np.ndarray:
    """Return a non-zero center-weight window for same-line tile fusion."""
    height, width = map(int, valid_hw)
    if tile_count <= 1:
        return np.ones((height, width), dtype=np.float32)
    length = width if axis == "x" else height
    curve = np.sin(np.linspace(0.0, np.pi, max(length, 2), dtype=np.float32))
    curve = np.maximum(curve, 0.1)
    if axis == "x":
        return np.broadcast_to(curve[None, :], (height, width)).copy()
    return np.broadcast_to(curve[:, None], (height, width)).copy()


__all__ = [
    "CANVAS_HW", "FullFrameSegment", "OCRLine", "attach_line_index", "build_full_frame_segments", "build_native_tiled_full_frame_segments", "build_height96_tiled_full_frame_segments", "build_p17_height96_tiled_full_frame_segments",
    "build_resized_full_frame_segments",
    "feather_alpha", "normalize_ocr_polygon", "pad_even_rgb", "parse_ocr_json", "parse_p17_ocr_annotations",
    "rasterize_polygon_mask", "tile_axis_weight",
]
