"""Materialize native-scale paired OCR line crops for the 20260820 split.

The source manifests are frame-level JSONL files.  This tool reads each unique
frame once, validates the source text mask, writes one lossless LQ/GT/mask crop
per accepted OCR line, and emits one or more metadata-only 96x480 tile rows
that reference that crop.  The crop pixels are never resized by this tool.

The final output directory is installed by an atomic rename from a staging
directory.  An existing final directory is never overwritten.  This keeps the
new cache isolated from the historical V3/V4 manifests and checkpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np


MATERIALIZER_VERSION = "basic_v2_native_crop_materializer_v1"
GEOMETRY_VERSION = "y4uv_v4_native_crop_even_inset2_rotate_vertical_v1"
MANIFEST_VERSION = "basic_v2_0820_native_crop_manifest_v1"
DEFAULT_CANVAS_HW = (96, 480)
DEFAULT_SPLIT_OVERLAP = 64
DEFAULT_MIN_TEXT_SIZE = 50.0
DEFAULT_INSET_PX = 2
REQUIRED_RECORD_KEYS = ("lq", "gt", "text_mask", "ocr")
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.=-]+")


def _sha256(path: Path) -> str:
    """@shanlan-comment Hash one source or derived file in binary mode.

    Input: an existing filesystem path.  Output: a lowercase SHA-256 digest;
    the function has no side effects beyond reading the file.  Hashing is kept
    in one helper so manifest and summary evidence use the same algorithm.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    """@shanlan-comment Hash a JSON-serializable value deterministically."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_token(value: str) -> str:
    """@shanlan-comment Convert a source identifier into a bounded path token."""
    token = _SAFE_TOKEN.sub("_", str(value)).strip("._") or "line"
    return token[:96]


def _source_line_token(source_line_id: str) -> str:
    """@shanlan-comment Build a collision-resistant stable crop filename token."""
    digest = hashlib.sha1(source_line_id.encode("utf-8")).hexdigest()[:16]
    return f"{_safe_token(source_line_id)}--{digest}"


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """@shanlan-comment Yield numbered non-empty JSONL records without rewriting the source."""
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: source row is not an object")
            yield line_number, value


def _read_bgr(path: Path) -> np.ndarray:
    """@shanlan-comment Decode one source image as uint8 BGR pixels."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(str(path))
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"unsupported source image format for {path}: {image.shape}, {image.dtype}")
    return np.ascontiguousarray(image)


def _read_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Decode and normalize the source binary text mask."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(str(path))
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or image.shape[:2] != expected_hw:
        raise ValueError(f"text-mask shape mismatch for {path}: {image.shape}, expected {expected_hw}")
    if image.size == 0:
        raise ValueError(f"empty source text mask: {path}")
    return np.ascontiguousarray(image > 0, dtype=bool)


def _points(value: Any) -> np.ndarray:
    """@shanlan-comment Validate an OCR polygon and return finite float32 Nx2 points."""
    points = np.asarray(value, dtype=np.float32)
    if points.ndim == 1 and points.size % 2 == 0:
        points = points.reshape(-1, 2)
    else:
        points = points.reshape(-1, 2)
    if points.shape[0] < 4 or not np.isfinite(points).all():
        raise ValueError("OCR polygon must contain at least four finite points")
    return points


def _ocr_lines(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """@shanlan-comment Parse the supported OCR line schemas into polygon/text records.

    Input: one OCR JSON object in the current ``lines`` schema or a legacy
    ``rec_polys``/``dt_polys``/``rec_boxes`` object.  Output: ordered records
    with source OCR indices; malformed polygons are represented as rejected
    records so the caller can retain an auditable reason.  No image data is
    decoded here.
    """
    parsed: list[dict[str, Any]] = []
    lines = payload.get("lines") or []
    if isinstance(lines, list) and lines:
        for index, line in enumerate(lines):
            if not isinstance(line, Mapping):
                parsed.append({"index": index, "error": "OCR_LINE_NOT_OBJECT"})
                continue
            value = line.get("polygon")
            if value is None:
                value = line.get("original_polygon")
            try:
                polygon = _points(value)
                parsed.append(
                    {
                        "index": index,
                        "line_id": line.get("source_line_id", line.get("line_id", index)),
                        "polygon": polygon,
                        "text": str(line.get("text", "")),
                        "score": line.get("rec_score"),
                        "reject_reasons": list(line.get("reject_reasons") or []),
                    }
                )
            except (TypeError, ValueError):
                parsed.append({"index": index, "error": "OCR_POLYGON_INVALID"})
        return parsed

    for field in ("rec_polys", "dt_polys", "rec_boxes"):
        values = payload.get(field) or []
        if not isinstance(values, list) or not values:
            continue
        texts = payload.get("rec_texts") or payload.get("text_word") or []
        for index, value in enumerate(values):
            try:
                raw = np.asarray(value, dtype=np.float32)
                # Legacy ``rec_boxes`` can be [x0, y0, x1, y1].  Expand it to
                # a polygon while keeping the more general polygon schemas
                # unchanged.
                if field == "rec_boxes" and raw.size == 4:
                    x0, y0, x1, y1 = (float(item) for item in raw.reshape(-1))
                    polygon = _points([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                else:
                    polygon = _points(value)
                error = None
            except (TypeError, ValueError):
                polygon = None
                error = "OCR_POLYGON_INVALID"
            parsed.append(
                {
                    "index": index,
                    "line_id": index,
                    "polygon": polygon,
                    "text": str(texts[index]) if index < len(texts) else "",
                    "score": None,
                    "reject_reasons": [],
                    "polygon_source": field,
                    **({"error": error} if error else {}),
                }
            )
        return parsed
    return parsed


def _is_hx_related(record: Mapping[str, Any]) -> bool:
    """@shanlan-comment Detect HX frames in dataset, identity, or asset paths."""
    values = (
        record.get("dataset", ""),
        record.get("stem", ""),
        record.get("frame_id", ""),
        record.get("lq", ""),
        record.get("gt", ""),
        record.get("ocr", ""),
    )
    for raw in values:
        normalized = str(raw).replace("\\", "/").upper()
        if "HX_TESTSET" in normalized or "/HX/" in normalized:
            return True
        for token in normalized.replace("/", "_").replace("-", "_").split("_"):
            if token.startswith("HX") and len(token) >= 2:
                return True
    return False


def _min_area_short_side(points: np.ndarray) -> float:
    """@shanlan-comment Return the short side of an OCR polygon's minimum-area rectangle."""
    _, dimensions, _ = cv2.minAreaRect(points.astype(np.float32))
    return float(min(float(dimensions[0]), float(dimensions[1])))


def _line_id(record: Mapping[str, Any], line: Mapping[str, Any], line_index: int) -> str:
    """@shanlan-comment Construct a stable source-line identifier from frame identity and OCR index."""
    dataset = str(record.get("dataset", ""))
    stem = str(record.get("stem", record.get("frame_id", "frame")))
    source_id = str(line.get("line_id", line_index))
    return f"{dataset}::{stem}::line{source_id}"


def _even_base_bbox(points: np.ndarray, image_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """@shanlan-comment Build the outward, in-bounds, even-coordinate polygon AABB."""
    height, width = (int(image_hw[0]), int(image_hw[1]))
    x0 = max(0, int(math.floor(float(points[:, 0].min()))))
    y0 = max(0, int(math.floor(float(points[:, 1].min()))))
    x1 = min(width, int(math.ceil(float(points[:, 0].max()))))
    y1 = min(height, int(math.ceil(float(points[:, 1].max()))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("POLYGON_AABB_EMPTY")
    x0 -= x0 % 2
    y0 -= y0 % 2
    if x1 % 2:
        x1 += 1
    if y1 % 2:
        y1 += 1
    if x1 > width or y1 > height:
        # YUV420 end coordinates must remain even.  An odd source edge cannot
        # be represented without clipping one source column/row, so reject it
        # rather than silently dropping a protected pixel.
        raise ValueError("ODD_SOURCE_EDGE_CANNOT_ALIGN_EVEN")
    if x1 <= x0 or y1 <= y0 or (x1 - x0) % 2 or (y1 - y0) % 2:
        raise ValueError("POLYGON_AABB_EMPTY_AFTER_EVEN_ALIGNMENT")
    return x0, y0, x1, y1


def _slice_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """@shanlan-comment Slice a boolean source mask by an exclusive XYXY crop."""
    left, top, right, bottom = bbox
    return mask[top:bottom, left:right]


def _strip_has_positive(mask: np.ndarray, bbox: tuple[int, int, int, int], side: str, width: int) -> bool:
    """@shanlan-comment Check whether an inward candidate would remove text-mask pixels."""
    left, top, right, bottom = bbox
    amount = int(width)
    if side == "left":
        strip = mask[top:bottom, left:min(right, left + amount)]
    elif side == "top":
        strip = mask[top:min(bottom, top + amount), left:right]
    elif side == "right":
        strip = mask[top:bottom, max(left, right - amount):right]
    elif side == "bottom":
        strip = mask[max(top, bottom - amount):bottom, left:right]
    else:
        raise ValueError(f"unknown crop side: {side}")
    return bool(np.any(strip))


def _line_border_sides(mask: np.ndarray, points: np.ndarray) -> list[str]:
    """@shanlan-comment Identify positive source-mask pixels at frame borders for one OCR polygon."""
    height, width = mask.shape
    line_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(line_mask, [np.rint(points).astype(np.int32)], 1)
    protected = mask & (line_mask > 0)
    sides: list[str] = []
    if bool(np.any(protected[:, 0])):
        sides.append("left")
    if bool(np.any(protected[:, width - 1])):
        sides.append("right")
    if bool(np.any(protected[0, :])):
        sides.append("top")
    if bool(np.any(protected[height - 1, :])):
        sides.append("bottom")
    return sides


def _apply_inset(
    base_bbox: tuple[int, int, int, int],
    source_mask: np.ndarray,
    *,
    requested_px: int,
) -> tuple[tuple[int, int, int, int], list[int], dict[str, Any]]:
    """@shanlan-comment Apply safe inward recovery per side without losing source-mask pixels.

    Input: an even-coordinate source AABB and the frame text mask.  Output:
    final even-coordinate crop, actual LTRB insets, and per-side audit details.
    Each side is considered independently; a two-pixel strip is removed only
    when it contains no positive source text-mask pixel.  The final mask count
    is checked again by the caller for a complete no-loss assertion.
    """
    if requested_px < 0:
        raise ValueError("requested inset must be non-negative")
    left, top, right, bottom = base_bbox
    original = base_bbox
    details: dict[str, Any] = {}
    applied = {"left": 0, "top": 0, "right": 0, "bottom": 0}
    for side in ("left", "top", "right", "bottom"):
        amount = min(2, int(requested_px))
        if amount <= 0:
            details[side] = {"requested": 0, "applied": 0, "status": "disabled"}
            continue
        current = (left, top, right, bottom)
        if _strip_has_positive(source_mask, current, side, amount):
            details[side] = {
                "requested": amount,
                "applied": 0,
                "status": "protected_text_in_strip",
            }
            continue
        candidate = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        }
        if side == "left":
            candidate[side] += amount
        elif side == "top":
            candidate[side] += amount
        else:
            candidate[side] -= amount
        if candidate["right"] <= candidate["left"] or candidate["bottom"] <= candidate["top"]:
            details[side] = {"requested": amount, "applied": 0, "status": "empty_candidate"}
            continue
        left, top, right, bottom = (
            int(candidate["left"]),
            int(candidate["top"]),
            int(candidate["right"]),
            int(candidate["bottom"]),
        )
        applied[side] += amount
        details[side] = {"requested": amount, "applied": amount, "status": "safe_no_text_strip"}

    result = (left, top, right, bottom)
    if any(value % 2 for value in result) or (right - left) % 2 or (bottom - top) % 2:
        raise ValueError("ODD_FINAL_CROP_ALIGNMENT")
    if result[0] < 0 or result[1] < 0 or result[2] > source_mask.shape[1] or result[3] > source_mask.shape[0]:
        raise ValueError("FINAL_CROP_OUT_OF_BOUNDS")
    actual = [left - original[0], top - original[1], original[2] - right, original[3] - bottom]
    return result, actual, details


def _rotate_points_ccw(points: np.ndarray, crop_width: int) -> np.ndarray:
    """@shanlan-comment Rotate crop-local polygon boundary coordinates CCW without scaling."""
    rotated = np.empty_like(points, dtype=np.float32)
    rotated[:, 0] = points[:, 1]
    rotated[:, 1] = float(crop_width) - points[:, 0]
    return rotated


def _orientation(points: np.ndarray) -> str:
    """@shanlan-comment Classify an OCR polygon by the long edge of its minimum-area rectangle."""
    rectangle = cv2.minAreaRect(points.astype(np.float32))
    corners = cv2.boxPoints(rectangle).astype(np.float32)
    edges = np.roll(corners, -1, axis=0) - corners
    axis = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    angle = abs(float(np.degrees(np.arctan2(float(axis[1]), float(axis[0]))))) % 180.0
    if angle <= 20.0 or angle >= 160.0:
        return "HORIZONTAL"
    if 70.0 <= angle <= 110.0:
        return "VERTICAL"
    return "OBLIQUE"


def _even_width(value: int) -> int:
    """@shanlan-comment Round a positive resized width up to even YUV420 metadata."""
    value = max(2, int(value))
    return value if value % 2 == 0 else value + 1


def _split_ranges(width: int, canvas_width: int, overlap: int) -> list[tuple[int, int]]:
    """@shanlan-comment Generate deterministic even overlapping metadata tile windows."""
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


def _resize_metadata(native_hw: tuple[int, int], canvas_hw: tuple[int, int]) -> tuple[int, int]:
    """@shanlan-comment Compute aspect-preserving height-96 metadata without touching pixels."""
    native_height, native_width = (int(native_hw[0]), int(native_hw[1]))
    canvas_height = int(canvas_hw[0])
    if native_height <= 0 or native_width <= 0 or canvas_height <= 0:
        raise ValueError("NATIVE_CROP_EMPTY")
    resized_width = _even_width(int(round(float(native_width) * canvas_height / native_height)))
    return canvas_height, resized_width


def _relative_or_absolute(path: Path, root: Path, *, relative: bool) -> str:
    """@shanlan-comment Serialize a path using the requested manifest path convention."""
    if relative:
        return path.relative_to(root).as_posix()
    # The writer operates below ``<output>.staging`` but manifests must remain
    # valid after the atomic rename to ``<output>``.
    staging_root = root.with_name(root.name + ".staging")
    try:
        return str(root / path.relative_to(staging_root))
    except ValueError:
        pass
    return str(path)


def _write_png(path: Path, image: np.ndarray) -> str:
    """@shanlan-comment Write one lossless PNG and return its SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # PNG compression level changes CPU cost and file size, not losslessness.
    # Level 3 keeps NAS materialization practical while preserving every pixel.
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"failed to write PNG: {path}")
    return _sha256(path)


def _frame_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """@shanlan-comment Identify a unique source frame by its immutable asset paths."""
    return tuple(str(record.get(key, "")) for key in REQUIRED_RECORD_KEYS)


def _record_identity(record: Mapping[str, Any], line_number: int) -> str:
    """@shanlan-comment Return the best stable human-readable frame identity."""
    return str(record.get("stem", record.get("frame_id", f"frame_{line_number}")))


class NativeCropMaterializer:
    """Build one staged native crop cache and split/val metadata manifests."""

    def __init__(
        self,
        *,
        source_root: Path,
        output_root: Path,
        splits: Sequence[str],
        min_text_size: float,
        inset_px: int,
        canvas_hw: tuple[int, int],
        split_overlap: int,
        exclude_hx: bool,
        max_frames: int | None,
        relative_paths: bool,
        keep_stage: bool = False,
    ) -> None:
        """@shanlan-comment Validate immutable run parameters and initialize counters/state."""
        self.source_root = source_root
        self.output_root = output_root
        self.stage_root = output_root.with_name(output_root.name + ".staging")
        self.splits = tuple(splits)
        self.min_text_size = float(min_text_size)
        self.inset_px = int(inset_px)
        self.canvas_hw = tuple(int(value) for value in canvas_hw)
        self.split_overlap = int(split_overlap)
        self.exclude_hx = bool(exclude_hx)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.relative_paths = bool(relative_paths)
        self.keep_stage = bool(keep_stage)
        if self.canvas_hw != DEFAULT_CANVAS_HW:
            raise ValueError("native crop tile metadata requires canvas_hw=(96,480)")
        if self.inset_px < 0:
            raise ValueError("inset_px must be non-negative")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.counts: Counter[str] = Counter()
        self.rejections: dict[str, list[dict[str, Any]]] = {split: [] for split in self.splits}
        self.rows: dict[str, list[dict[str, Any]]] = {split: [] for split in self.splits}
        self.frame_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.processed_frame_splits: set[tuple[tuple[str, str, str, str], str]] = set()
        self.source_line_cache: dict[str, dict[str, Any]] = {}
        self.accepted_line_ids: dict[str, set[str]] = {split: set() for split in self.splits}
        self.orientation_counts: Counter[str] = Counter()
        self.inset_counts: Counter[str] = Counter()

    def _stage_prepare(self) -> None:
        """@shanlan-comment Create a fresh staging tree without touching an existing final root."""
        if self.output_root.exists():
            raise FileExistsError(f"refusing to overwrite existing final output root: {self.output_root}")
        if self.stage_root.exists():
            raise FileExistsError(
                f"staging root already exists; inspect or remove it explicitly before retrying: {self.stage_root}"
            )
        for relative in (
            "crops/lq",
            "crops/gt",
            "crops/text_mask",
            "manifests",
            "summary",
            "audit",
        ):
            (self.stage_root / relative).mkdir(parents=True, exist_ok=False)

    def _source_record(self, record: Mapping[str, Any], line_number: int) -> dict[str, Any]:
        """@shanlan-comment Decode and hash one unique frame's paired assets exactly once."""
        missing = [key for key in REQUIRED_RECORD_KEYS if not record.get(key)]
        if missing:
            raise ValueError(f"FRAME_MISSING_KEYS:{','.join(missing)}")
        paths = {key: Path(str(record[key])) for key in REQUIRED_RECORD_KEYS}
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"FRAME_MISSING_{key}:{path}")
        lq = _read_bgr(paths["lq"])
        gt = _read_bgr(paths["gt"])
        if lq.shape != gt.shape:
            raise ValueError(f"LQ_GT_SHAPE_MISMATCH:{lq.shape[:2]}!={gt.shape[:2]}")
        height, width = (int(lq.shape[0]), int(lq.shape[1]))
        if height % 2 or width % 2:
            raise ValueError(f"ODD_SOURCE_FRAME_SHAPE:{height}x{width}")
        text_mask = _read_mask(paths["text_mask"], (height, width))
        payload = json.loads(paths["ocr"].read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("OCR_PAYLOAD_NOT_OBJECT")
        return {
            "lq": lq,
            "gt": gt,
            "text_mask": text_mask,
            "ocr_payload": payload,
            "paths": paths,
            "hashes": {key: _sha256(path) for key, path in paths.items()},
            "image_hw": [height, width],
            "source_record": dict(record),
            "source_line_number": int(line_number),
        }

    def _cache_frame_metadata(
        self,
        frame: Mapping[str, Any],
        accepted_line_ids: Sequence[str],
        key: tuple[str, str, str, str],
    ) -> None:
        """@shanlan-comment Retain only reusable frame metadata after releasing decoded source arrays.

        Input: one fully processed frame, its accepted source-line identifiers,
        and the immutable frame-cache key.  Output: a compact cache entry used
        when the same frame appears in another split; decoded LQ/GT/mask arrays
        and the OCR payload are deliberately not retained.  This preserves the
        one-decode-per-frame contract without accumulating full-resolution
        image buffers across thousands of frames.
        """
        self.frame_cache[key] = {
            "source_record": dict(frame["source_record"]),
            "source_line_number": int(frame["source_line_number"]),
            "paths": dict(frame["paths"]),
            "hashes": dict(frame["hashes"]),
            "image_hw": list(frame["image_hw"]),
            "accepted_line_ids": list(accepted_line_ids),
        }

    def _reject(self, split: str, **entry: Any) -> None:
        """@shanlan-comment Append one auditable frame/line rejection and update counters."""
        entry.setdefault("split", split)
        reason = str(entry.get("reason", "UNKNOWN"))
        self.rejections[split].append(entry)
        self.counts[f"rejected_{reason}"] += 1

    def _emit_line(self, split: str, frame: Mapping[str, Any], line: Mapping[str, Any], line_index: int) -> None:
        """@shanlan-comment Validate, crop, rotate, write, and tile one OCR line exactly once."""
        source_record = frame["source_record"]
        source_line_id = _line_id(source_record, line, line_index)
        common_rejection = {
            "source_line_id": source_line_id,
            "stem": _record_identity(source_record, int(frame["source_line_number"])),
            "source_manifest_line": int(frame["source_line_number"]),
            "line_index": int(line_index),
        }
        if line.get("error"):
            self._reject(split, **common_rejection, reason=str(line["error"]))
            return
        points = line.get("polygon")
        if not isinstance(points, np.ndarray):
            self._reject(split, **common_rejection, reason="OCR_POLYGON_INVALID")
            return
        reject_reasons = list(line.get("reject_reasons") or [])
        if reject_reasons:
            self._reject(
                split,
                **common_rejection,
                reason="OCR_REJECTED_BY_SOURCE",
                detail=reject_reasons,
            )
            return
        text_size = _min_area_short_side(points)
        # OpenCV's minimum-area rectangle can return 49.999996 for an exact
        # 50-pixel side.  Treat that floating-point round-off as the inclusive
        # ``>= 50`` policy promised by the task.
        if text_size + 1e-4 < self.min_text_size:
            self._reject(
                split,
                **common_rejection,
                reason="TEXT_SIZE_BELOW_MIN",
                text_size_px=float(text_size),
                min_text_size=float(self.min_text_size),
            )
            return
        source_mask = frame["text_mask"]
        image_hw = tuple(int(value) for value in frame["image_hw"])
        try:
            base_bbox = _even_base_bbox(points, image_hw)
        except ValueError as exc:
            self._reject(split, **common_rejection, reason=str(exc), text_size_px=float(text_size))
            return
        base_mask = _slice_mask(source_mask, base_bbox)
        if not np.any(base_mask):
            self._reject(
                split,
                **common_rejection,
                reason="EMPTY_SOURCE_TEXT_MASK_IN_AABB",
                base_crop_xyxy_even=list(base_bbox),
            )
            return
        border_sides = _line_border_sides(source_mask, points)
        if border_sides:
            self._reject(
                split,
                **common_rejection,
                reason="SOURCE_BORDER_TEXT_PROTECTED",
                source_border_sides=border_sides,
                base_crop_xyxy_even=list(base_bbox),
            )
            return
        try:
            crop_bbox, actual_inset, inset_detail = _apply_inset(
                base_bbox,
                source_mask,
                requested_px=self.inset_px,
            )
        except ValueError as exc:
            self._reject(
                split,
                **common_rejection,
                reason=str(exc),
                base_crop_xyxy_even=list(base_bbox),
            )
            return
        crop_mask = _slice_mask(source_mask, crop_bbox)
        if int(np.count_nonzero(crop_mask)) < int(np.count_nonzero(base_mask)):
            self._reject(
                split,
                **common_rejection,
                reason="TEXT_MASK_LOSS_AFTER_INSET",
                base_crop_xyxy_even=list(base_bbox),
                source_crop_xyxy_even=list(crop_bbox),
                text_mask_positive_before=int(np.count_nonzero(base_mask)),
                text_mask_positive_after=int(np.count_nonzero(crop_mask)),
            )
            return
        left, top, right, bottom = crop_bbox
        lq_crop = np.ascontiguousarray(frame["lq"][top:bottom, left:right])
        gt_crop = np.ascontiguousarray(frame["gt"][top:bottom, left:right])
        mask_crop = np.ascontiguousarray(crop_mask, dtype=np.uint8) * 255
        if lq_crop.shape != gt_crop.shape or lq_crop.shape[:2] != mask_crop.shape:
            self._reject(
                split,
                **common_rejection,
                reason="CROP_DIMENSION_MISMATCH",
                lq_hw=list(lq_crop.shape[:2]),
                gt_hw=list(gt_crop.shape[:2]),
                mask_hw=list(mask_crop.shape[:2]),
            )
            return
        source_orientation = _orientation(points)
        rotated_to_horizontal = source_orientation == "VERTICAL"
        native_points = points - np.asarray([left, top], dtype=np.float32)
        if rotated_to_horizontal:
            lq_crop = np.ascontiguousarray(np.rot90(lq_crop, k=1))
            gt_crop = np.ascontiguousarray(np.rot90(gt_crop, k=1))
            mask_crop = np.ascontiguousarray(np.rot90(mask_crop, k=1))
            native_points = _rotate_points_ccw(native_points, crop_bbox[2] - crop_bbox[0])
        native_hw = tuple(int(value) for value in lq_crop.shape[:2])
        if native_hw[0] % 2 or native_hw[1] % 2:
            self._reject(split, **common_rejection, reason="ODD_NATIVE_CROP_SHAPE")
            return
        if not np.any(mask_crop):
            self._reject(split, **common_rejection, reason="EMPTY_NATIVE_TEXT_MASK_CROP")
            return
        if source_line_id in self.source_line_cache:
            prior = self.source_line_cache[source_line_id]
            if prior["native_crop_hw"] != list(native_hw) or prior["source_hashes"] != frame["hashes"]:
                self._reject(split, **common_rejection, reason="DUPLICATE_SOURCE_LINE_CONFLICT")
                return
            self._emit_rows(split, prior, line, line_index)
            return
        token = _source_line_token(source_line_id)
        lq_path = self.stage_root / "crops" / "lq" / f"{token}.png"
        gt_path = self.stage_root / "crops" / "gt" / f"{token}.png"
        mask_path = self.stage_root / "crops" / "text_mask" / f"{token}.png"
        lq_hash = _write_png(lq_path, lq_crop)
        gt_hash = _write_png(gt_path, gt_crop)
        mask_hash = _write_png(mask_path, mask_crop)
        resized_hw = _resize_metadata(native_hw, self.canvas_hw)
        crop_info: dict[str, Any] = {
            "source_line_id": source_line_id,
            "source_line_token": token,
            # Absolute manifest paths name the final root, not its staging
            # sibling.  Relative paths remain naturally valid after rename.
            "lq_crop": _relative_or_absolute(lq_path, self.output_root, relative=self.relative_paths),
            "gt_crop": _relative_or_absolute(gt_path, self.output_root, relative=self.relative_paths),
            "text_mask_crop": _relative_or_absolute(mask_path, self.output_root, relative=self.relative_paths),
            "lq_crop_sha256": lq_hash,
            "gt_crop_sha256": gt_hash,
            "text_mask_crop_sha256": mask_hash,
            "native_crop_hw": list(native_hw),
            "source_crop_hw": [int(bottom - top), int(right - left)],
            "source_crop_xyxy_even": list(crop_bbox),
            "base_crop_xyxy_even": list(base_bbox),
            "crop_bbox_xyxy": list(crop_bbox),
            "inset_requested_px": [self.inset_px] * 4,
            "inset_applied_ltrb": [int(value) for value in actual_inset],
            "inset_detail": inset_detail,
            "source_orientation": source_orientation,
            "orientation": "HORIZONTAL" if rotated_to_horizontal else source_orientation,
            "rotated_to_horizontal": bool(rotated_to_horizontal),
            "rotated_ccw": bool(rotated_to_horizontal),
            "rotation_degrees_ccw": 90 if rotated_to_horizontal else 0,
            "resized_hw": list(resized_hw),
            "resize_applied": False,
            "pixel_scale": "native",
            "polygon_local_native": native_points.astype(float).tolist(),
            "text_size_px": float(text_size),
            "text_mask_positive_pixels": int(np.count_nonzero(mask_crop)),
            "source_border_sides": [],
            "source_border_ambiguous": False,
            "transcription": str(line.get("text", "")),
            "polygon_source": str(line.get("polygon_source", "lines.polygon")),
            "ocr_line_index": int(line_index),
            "ocr_line_id": str(line.get("line_id", line_index)),
            "source_manifest_line": int(frame["source_line_number"]),
            "source_frame_id": _record_identity(source_record, int(frame["source_line_number"])),
            "dataset": str(source_record.get("dataset", "")),
            "stem": _record_identity(source_record, int(frame["source_line_number"])),
            "source_paths": {
                "lq": str(frame["paths"]["lq"]),
                "gt": str(frame["paths"]["gt"]),
                "text_mask": str(frame["paths"]["text_mask"]),
                "ocr": str(frame["paths"]["ocr"]),
            },
            "source_hashes": dict(frame["hashes"]),
            "geometry_version": GEOMETRY_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "manifest_version": MANIFEST_VERSION,
        }
        self.source_line_cache[source_line_id] = crop_info
        self._emit_rows(split, crop_info, line, line_index)

    def _emit_rows(self, split: str, crop_info: Mapping[str, Any], line: Mapping[str, Any], line_index: int) -> None:
        """@shanlan-comment Emit metadata-only tile rows referencing one native crop pair."""
        native_h, native_w = (int(value) for value in crop_info["native_crop_hw"])
        resized_h, resized_w = (int(value) for value in crop_info["resized_hw"])
        native_points = np.asarray(crop_info["polygon_local_native"], dtype=np.float32)
        scale_x = float(resized_w) / float(native_w)
        scale_y = float(resized_h) / float(native_h)
        resized_points = native_points.copy()
        resized_points[:, 0] *= scale_x
        resized_points[:, 1] *= scale_y
        ranges = _split_ranges(resized_w, self.canvas_hw[1], self.split_overlap)
        for tile_index, (tile_start, tile_end) in enumerate(ranges):
            tile_width = int(tile_end - tile_start)
            tile_points = resized_points.copy()
            tile_points[:, 0] -= float(tile_start)
            valid_hw = [resized_h, tile_width]
            cp_eligible = bool(
                np.all(tile_points[:, 0] >= 0.0)
                and np.all(tile_points[:, 0] <= float(tile_width))
                and np.all(tile_points[:, 1] >= 0.0)
                and np.all(tile_points[:, 1] <= float(resized_h))
            )
            row = dict(crop_info)
            row.update(
                {
                    "split": split,
                    "segment_id": f"{crop_info['source_line_id']}::tile{tile_index}",
                    "line_index": int(line_index),
                    "tile_index": int(tile_index),
                    "tile_count": int(len(ranges)),
                    "tile_start_x": int(tile_start),
                    "tile_width": tile_width,
                    "pad_right": max(0, int(self.canvas_hw[1] - tile_width)),
                    "valid_hw": valid_hw,
                    "polygon_local_resized": resized_points.astype(float).tolist(),
                    "polygon_local_resized_tile": tile_points.astype(float).tolist(),
                    "cp_eligible": cp_eligible,
                    "cp_skip_reason": "" if cp_eligible else "SPLIT_TILE_TRUNCATED",
                }
            )
            self.rows[split].append(row)
            self.accepted_line_ids[split].add(str(crop_info["source_line_id"]))
            self.counts["tile_rows"] += 1
            if cp_eligible:
                self.counts["cp_eligible_rows"] += 1
            else:
                self.counts["cp_ineligible_rows"] += 1

    def _process_split(self, split: str) -> None:
        """@shanlan-comment Process one source split while sharing compact unique-frame metadata.

        Input: one immutable ``train.jsonl`` or ``val.jsonl`` source split.
        Output: native crops and tile rows for accepted lines plus rejection
        audit rows.  A frame is decoded only on its first appearance; later
        appearances reuse the accepted line metadata and crop files without
        retaining or re-decoding full-resolution pixel arrays.
        """
        input_path = self.source_root / f"{split}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(str(input_path))
        frame_seen = 0
        for line_number, record in _read_jsonl(input_path):
            if self.max_frames is not None and frame_seen >= self.max_frames:
                break
            frame_seen += 1
            self.counts[f"{split}_frames_input"] += 1
            stem = _record_identity(record, line_number)
            if self.exclude_hx and _is_hx_related(record):
                self.counts[f"{split}_frames_hx_rejected"] += 1
                self._reject(
                    split,
                    stem=stem,
                    source_manifest_line=int(line_number),
                    reason="HX_RELATED_FRAME",
                )
                continue
            key = _frame_key(record)
            cached = self.frame_cache.get(key)
            if cached is None:
                try:
                    frame = self._source_record(record, line_number)
                except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self.counts[f"{split}_frames_invalid"] += 1
                    self._reject(
                        split,
                        stem=stem,
                        source_manifest_line=int(line_number),
                        reason="FRAME_INVALID",
                        detail=str(exc),
                    )
                    continue
                self.counts["unique_frames_decoded"] += 1
                lines = _ocr_lines(frame["ocr_payload"])
                self.counts[f"{split}_ocr_lines_raw"] += len(lines)
                if not lines:
                    self._reject(
                        split,
                        stem=stem,
                        source_manifest_line=int(line_number),
                        reason="NO_VALID_OCR_LINE",
                    )
                    self._cache_frame_metadata(frame, (), key)
                    continue
                accepted_before = set(self.accepted_line_ids[split])
                for line_index, line in enumerate(lines):
                    self._emit_line(split, frame, line, line_index)
                accepted_line_ids = sorted(set(self.accepted_line_ids[split]) - accepted_before)
                self.counts[f"{split}_source_lines_accepted"] += len(accepted_line_ids)
                self.counts[f"{split}_frames_with_ocr"] += 1
                self._cache_frame_metadata(frame, accepted_line_ids, key)
                # Drop all decoded arrays and OCR payload before the next frame.
                del frame
            else:
                self.counts["duplicate_frame_rows_reused"] += 1
                if (key, split) in self.processed_frame_splits:
                    continue
                accepted_before = set(self.accepted_line_ids[split])
                for source_line_id in cached["accepted_line_ids"]:
                    crop_info = self.source_line_cache.get(str(source_line_id))
                    if crop_info is None:
                        self._reject(
                            split,
                            stem=stem,
                            source_manifest_line=int(line_number),
                            reason="CACHED_SOURCE_LINE_MISSING",
                            source_line_id=str(source_line_id),
                        )
                        continue
                    self._emit_rows(split, crop_info, {}, int(crop_info["ocr_line_index"]))
                self.counts[f"{split}_source_lines_accepted"] += len(self.accepted_line_ids[split]) - len(accepted_before)
                self.counts[f"{split}_frames_with_ocr"] += 1
                self.counts[f"{split}_ocr_lines_raw"] += len(cached["accepted_line_ids"])
                self.processed_frame_splits.add((key, split))
                continue
            self.processed_frame_splits.add((key, split))
            if frame_seen == 1 or frame_seen % 250 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "split": split,
                            "frame": frame_seen,
                            "input_frames": self.counts[f"{split}_frames_input"],
                            "unique_frames_decoded": self.counts["unique_frames_decoded"],
                            "source_lines": len(self.source_line_cache),
                            "tile_rows": self.counts["tile_rows"],
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        self.counts[f"{split}_source_lines_reused_or_seen_before"] = sum(
            1
            for key, cached_split in self.processed_frame_splits
            if cached_split == split and key in self.frame_cache
        )
        self.counts[f"{split}_frames_processed"] = frame_seen

    def _write_manifest(self, split: str) -> dict[str, Any]:
        """@shanlan-comment Write one derived split JSONL and return hash/count evidence."""
        path = self.stage_root / "manifests" / f"{split}.jsonl"
        rows = self.rows[split]
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        return {
            "path": str(self.output_root / "manifests" / f"{split}.jsonl"),
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "rows": int(len(rows)),
            "source_lines": int(len({row["source_line_id"] for row in rows})),
            "cp_eligible_rows": int(sum(bool(row.get("cp_eligible")) for row in rows)),
        }

    def _write_rejections(self, split: str) -> dict[str, Any]:
        """@shanlan-comment Write per-split rejection audit rows and reason counts."""
        path = self.stage_root / "audit" / f"{split}.rejected.jsonl"
        entries = self.rejections[split]
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        return {
            "path": str(self.output_root / "audit" / f"{split}.rejected.jsonl"),
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "rows": int(len(entries)),
            "reason_counts": dict(sorted(Counter(str(item.get("reason", "UNKNOWN")) for item in entries).items())),
        }

    def _validate_rows(self, split: str) -> dict[str, Any]:
        """@shanlan-comment Validate manifest paths, dimensions, parity, alignment, and tile metadata."""
        result = {
            "rows_checked": 0,
            "lq_gt_parity_pass": True,
            "mask_parity_pass": True,
            "native_hw_pass": True,
            "even_alignment_pass": True,
            "text_mask_no_loss_pass": True,
            "manifest_paths_pass": True,
            "errors": [],
        }
        def staged_path(value: str) -> Path:
            path = Path(str(value))
            if self.relative_paths:
                return self.stage_root / path
            try:
                return self.stage_root / path.relative_to(self.output_root)
            except ValueError:
                return path

        for row in self.rows[split]:
            result["rows_checked"] += 1
            lq_path = staged_path(str(row["lq_crop"]))
            gt_path = staged_path(str(row["gt_crop"]))
            mask_path = staged_path(str(row["text_mask_crop"]))
            try:
                lq = _read_bgr(lq_path)
                gt = _read_bgr(gt_path)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise ValueError("DERIVED_MASK_READ_FAILED")
                if mask.ndim == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if lq.shape != gt.shape:
                    result["lq_gt_parity_pass"] = False
                    raise ValueError("DERIVED_LQ_GT_MISMATCH")
                if tuple(mask.shape[:2]) != tuple(lq.shape[:2]):
                    result["mask_parity_pass"] = False
                    raise ValueError("DERIVED_MASK_SHAPE_MISMATCH")
                expected = tuple(int(value) for value in row["native_crop_hw"])
                if tuple(lq.shape[:2]) != expected:
                    result["native_hw_pass"] = False
                    raise ValueError("DERIVED_NATIVE_HW_MISMATCH")
                bbox = tuple(int(value) for value in row["source_crop_xyxy_even"])
                if any(value % 2 for value in bbox) or any(value % 2 for value in (bbox[2] - bbox[0], bbox[3] - bbox[1])):
                    result["even_alignment_pass"] = False
                    raise ValueError("DERIVED_ODD_ALIGNMENT")
                if int(np.count_nonzero(mask)) != int(row["text_mask_positive_pixels"]):
                    result["text_mask_no_loss_pass"] = False
                    raise ValueError("DERIVED_MASK_COUNT_MISMATCH")
            except (OSError, ValueError, TypeError) as exc:
                result["manifest_paths_pass"] = False
                result["errors"].append({"segment_id": row.get("segment_id"), "detail": str(exc)})
                if len(result["errors"]) >= 20:
                    break
        result["passed"] = bool(
            result["lq_gt_parity_pass"]
            and result["mask_parity_pass"]
            and result["native_hw_pass"]
            and result["even_alignment_pass"]
            and result["text_mask_no_loss_pass"]
            and result["manifest_paths_pass"]
        )
        return result

    def _write_summary(self, manifest_info: Mapping[str, Any], rejection_info: Mapping[str, Any]) -> dict[str, Any]:
        """@shanlan-comment Write split/global summaries with hashes and validation evidence."""
        validations = {split: self._validate_rows(split) for split in self.splits}
        split_summaries: dict[str, Any] = {}
        for split in self.splits:
            split_summaries[split] = {
                "manifest": manifest_info[split],
                "rejections": rejection_info[split],
                "validation": validations[split],
                "rows": int(len(self.rows[split])),
                "source_lines": int(len({row["source_line_id"] for row in self.rows[split]})),
                "orientations": dict(
                    sorted(Counter(str(row["source_orientation"]) for row in self.rows[split]).items())
                ),
                "inset_applied_ltrb_counts": dict(
                    sorted(
                        (
                            (",".join(str(int(value)) for value in key), count)
                            for key, count in Counter(
                                tuple(int(value) for value in row["inset_applied_ltrb"])
                                for row in self.rows[split]
                            ).items()
                        ),
                        key=lambda item: str(item[0]),
                    )
                ),
            }
            split_path = self.stage_root / "summary" / f"{split}.json"
            split_path.write_text(
                json.dumps(
                    {
                        "materializer_version": MATERIALIZER_VERSION,
                        "geometry_version": GEOMETRY_VERSION,
                        "manifest_version": MANIFEST_VERSION,
                        "split": split,
                        "source_manifest": str(self.source_root / f"{split}.jsonl"),
                        "source_manifest_sha256": _sha256(self.source_root / f"{split}.jsonl"),
                        **split_summaries[split],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        global_summary = {
            "status": "passed" if all(value["validation"]["passed"] for value in split_summaries.values()) else "failed",
            "materializer_version": MATERIALIZER_VERSION,
            "geometry_version": GEOMETRY_VERSION,
            "manifest_version": MANIFEST_VERSION,
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "staging_root": str(self.stage_root),
            "splits": list(self.splits),
            "min_text_size": self.min_text_size,
            "inset_requested_px": [self.inset_px] * 4,
            "canvas_hw": list(self.canvas_hw),
            "split_overlap": self.split_overlap,
            "exclude_hx": self.exclude_hx,
            "max_frames_per_split": self.max_frames,
            "relative_paths": self.relative_paths,
            "counts": dict(sorted(self.counts.items())),
            "split_summaries": split_summaries,
            "stage_storage_bytes": int(
                sum(path.stat().st_size for path in self.stage_root.rglob("*") if path.is_file())
            ),
            "source_frame_cache_count": len(self.frame_cache),
            "native_crop_count": len(self.source_line_cache),
            "created_at_epoch": time.time(),
        }
        global_path = self.stage_root / "summary" / "global.json"
        global_path.write_text(json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return global_summary

    def run(self) -> dict[str, Any]:
        """@shanlan-comment Materialize both splits, validate outputs, and atomically install the final root."""
        self._stage_prepare()
        try:
            config = {
                "materializer_version": MATERIALIZER_VERSION,
                "geometry_version": GEOMETRY_VERSION,
                "manifest_version": MANIFEST_VERSION,
                "source_root": str(self.source_root),
                "splits": list(self.splits),
                "min_text_size": self.min_text_size,
                "inset_requested_px": [self.inset_px] * 4,
                "canvas_hw": list(self.canvas_hw),
                "split_overlap": self.split_overlap,
                "exclude_hx": self.exclude_hx,
                "max_frames_per_split": self.max_frames,
                "relative_paths": self.relative_paths,
            }
            (self.stage_root / "materializer_config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for split in self.splits:
                self._process_split(split)
            manifest_info = {split: self._write_manifest(split) for split in self.splits}
            rejection_info = {split: self._write_rejections(split) for split in self.splits}
            summary = self._write_summary(manifest_info, rejection_info)
            if summary["status"] != "passed":
                raise RuntimeError("OUTPUT_VALIDATION_FAILED")
            # The marker is written before the directory rename so an installed
            # root always contains immutable run identity and validation state.
            (self.stage_root / "COMPLETE").write_text(
                json.dumps(
                    {
                        "materializer_version": MATERIALIZER_VERSION,
                        "geometry_version": GEOMETRY_VERSION,
                        "manifest_version": MANIFEST_VERSION,
                        "summary_sha256": _sha256(self.stage_root / "summary" / "global.json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(self.stage_root, self.output_root)
            summary["output_root_installed"] = True
            summary["output_root"] = str(self.output_root)
            # Keep the final summary self-consistent with the installed path.
            final_summary = self.output_root / "summary" / "global.json"
            final_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            (self.output_root / "COMPLETE").write_text(
                json.dumps(
                    {
                        "materializer_version": MATERIALIZER_VERSION,
                        "geometry_version": GEOMETRY_VERSION,
                        "manifest_version": MANIFEST_VERSION,
                        "summary_sha256": _sha256(final_summary),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return summary
        except Exception:
            if not self.keep_stage:
                print(f"staging retained for audit after failure: {self.stage_root}", file=sys.stderr)
            raise


def _parse_splits(value: str) -> tuple[str, ...]:
    """@shanlan-comment Parse and validate a comma-separated train/val split selection."""
    splits = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not splits or any(item not in {"train", "val"} for item in splits):
        raise ValueError("splits must be a non-empty subset of train,val")
    return splits


def main(argv: Iterable[str] | None = None) -> int:
    """@shanlan-comment Parse CLI paths, run native materialization, and print JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--min-text-size", type=float, default=DEFAULT_MIN_TEXT_SIZE)
    parser.add_argument("--inset-px", type=int, default=DEFAULT_INSET_PX)
    parser.add_argument("--canvas-h", type=int, default=DEFAULT_CANVAS_HW[0])
    parser.add_argument("--canvas-w", type=int, default=DEFAULT_CANVAS_HW[1])
    parser.add_argument("--split-overlap", type=int, default=DEFAULT_SPLIT_OVERLAP)
    parser.add_argument("--exclude-hx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--relative-paths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-stage", action="store_true", help="retain failed staging output and return its path")
    args = parser.parse_args(argv)
    materializer = NativeCropMaterializer(
        source_root=args.source_root,
        output_root=args.output_root,
        splits=_parse_splits(args.splits),
        min_text_size=args.min_text_size,
        inset_px=args.inset_px,
        canvas_hw=(args.canvas_h, args.canvas_w),
        split_overlap=args.split_overlap,
        exclude_hx=args.exclude_hx,
        max_frames=args.max_frames,
        relative_paths=args.relative_paths,
        keep_stage=args.keep_stage,
    )
    summary = materializer.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
