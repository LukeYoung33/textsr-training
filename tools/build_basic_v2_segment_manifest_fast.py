"""Fast IQ560 segment manifest builder using manifest image dimensions.

The builder follows the minimal one-OCR-polygon AABB contract and emits one row
per line tile. It never decodes full images unless image dimensions are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("text_line_crop_v3", _ROOT / "realesrgan/data/text_line_crop_v3.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("crop contract unavailable")
_MOD = importlib.util.module_from_spec(_SPEC); sys.modules[_SPEC.name] = _MOD; _SPEC.loader.exec_module(_MOD)
build_line_crops = _MOD.build_line_crops
assess_crop_size = _MOD.assess_crop_size
SIZE_FILTER_VERSION = _MOD.SIZE_FILTER_VERSION


def _read_hw(path: str) -> tuple[int, int]:
    """@shanlan-comment Read only an image header for geometry expansion."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return int(image.shape[0]), int(image.shape[1])


def _polygons(path: str) -> tuple[str, list[np.ndarray], list[str]]:
    """@shanlan-comment Parse OCR polygons and preserve source field provenance."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("rec_polys", "dt_polys", "rec_boxes"):
        values = data.get(field) or []
        if not values:
            continue
        result = []
        for value in values:
            try:
                points = np.asarray(value, dtype=np.float32).reshape(-1, 2)
            except (TypeError, ValueError):
                continue
            if points.shape[0] >= 4 and np.isfinite(points).all():
                result.append(points)
        texts = [str(item) for item in (data.get("rec_texts") or data.get("text_word") or [])]
        return field, result, texts
    return "", [], []


def expand_record(
    record: dict,
    line_number: int,
    canvas_hw: tuple[int, int],
    overlap: int,
    min_short_side_ratio: float = 0.5,
    min_long_side_ratio: float = 0.5,
):
    """@shanlan-comment Expand one frame record into one manifest row per OCR tile."""
    required = ("lq", "gt", "valid_mask", "text_mask", "ocr")
    missing = [key for key in required if not record.get(key)]
    if missing:
        return [], [{"line_number": line_number, "reason": "MISSING_KEYS", "keys": missing}]
    try:
        image_hw = _read_hw(record["lq"])
        if _read_hw(record["gt"]) != image_hw:
            return [], [{"line_number": line_number, "reason": "LQ_GT_SHAPE_MISMATCH"}]
    except (OSError, ValueError):
        return [], [{"line_number": line_number, "reason": "IMAGE_READ_FAILED"}]
    source_field, polygons, texts = _polygons(record["ocr"])
    rows, filtered = [], []
    if not polygons:
        return [], [{"line_number": line_number, "reason": "NO_VALID_OCR_POLYGON"}]
    for line_index, polygon in enumerate(polygons):
        try:
            crops = build_line_crops(polygon, image_hw, canvas_hw, overlap, line_index=line_index)
        except (TypeError, ValueError) as exc:
            filtered.append({"line_number": line_number, "line_index": line_index, "reason": "CROP_BUILD_FAILED", "detail": str(exc)})
            continue
        source_line_id = f"{record.get('stem', 'frame')}::line{line_index}"
        for crop in crops:
            size = assess_crop_size(
                crop.valid_hw,
                canvas_hw,
                crop.orientation,
                min_short_side_ratio=min_short_side_ratio,
                min_long_side_ratio=min_long_side_ratio,
            )
            if not bool(size["accepted"]):
                filtered.append({
                    "line_number": line_number,
                    "line_index": line_index,
                    "tile_index": crop.tile_index,
                    "reason": str(size["reason"]),
                    "valid_hw": list(crop.valid_hw),
                    "short_side_ratio": float(size["short_side_ratio"]),
                    "long_side_ratio": float(size["long_side_ratio"]),
                })
                continue
            # Direct text-line CP works on the already cropped tile.  Tiling is
            # therefore not a reason to reject a horizontal/oblique sample;
            # vertical samples remain outside the CRNN contract for this run.
            eligible = bool(crop.orientation in ("HORIZONTAL", "OBLIQUE") and crop.text_mask.sum() > 0)
            rows.append({"dataset": record.get("dataset", ""), "stem": record.get("stem", ""), "frame_id": record.get("stem", ""), "source_line_id": source_line_id, "segment_id": f"{source_line_id}::tile{crop.tile_index}", "line_index": line_index, "tile_index": crop.tile_index, "tile_count": crop.tile_count, "orientation": crop.orientation, "polygon_source": source_field, "polygon_local": crop.polygon_local.astype(float).tolist(), "crop_xyxy_even": list(crop.crop_xyxy_even), "valid_hw": list(crop.valid_hw), "canvas_hw": list(canvas_hw), "valid_area": float(crop.valid_mask.sum()), "text_area": float(crop.text_mask.sum()), "size_filter_version": SIZE_FILTER_VERSION, "min_short_side_ratio": float(min_short_side_ratio), "min_long_side_ratio": float(min_long_side_ratio), "short_side_ratio": float(size["short_side_ratio"]), "long_side_ratio": float(size["long_side_ratio"]), "cp_eligible": eligible, "cp_skip_reason": "" if eligible else "VERTICAL_OR_EMPTY", "is_crop_truncated": False, "lq": record["lq"], "gt": record["gt"], "valid_mask": record["valid_mask"], "text_mask": record["text_mask"], "ocr": record["ocr"], "transcription": texts[line_index] if line_index < len(texts) else "", "source_manifest_line": line_number})
    return rows, filtered


def build(input_jsonl: Path, output_jsonl: Path, summary_json: Path, canvas_hw=(96, 320), overlap=64, max_frames=None, min_short_side_ratio=0.5, min_long_side_ratio=0.5):
    """@shanlan-comment Build and hash the complete segment manifest."""
    rows, filtered, frames = [], [], 0
    with input_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if max_frames is not None and frames >= max_frames:
                break
            expanded, rejected = expand_record(json.loads(line), line_number, canvas_hw, overlap, min_short_side_ratio, min_long_side_ratio)
            rows.extend(expanded); filtered.extend(rejected); frames += 1
    output_jsonl.parent.mkdir(parents=True, exist_ok=True); summary_json.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(output_jsonl.read_bytes()).hexdigest()
    summary = {"input": str(input_jsonl), "input_sha256": hashlib.sha256(input_jsonl.read_bytes()).hexdigest(), "output": str(output_jsonl), "output_sha256": digest, "frames": frames, "segments": len(rows), "source_lines": len({row["source_line_id"] for row in rows}), "multi_tile_lines": len({row["source_line_id"] for row in rows if row["tile_count"] > 1}), "cp_eligible": sum(bool(row["cp_eligible"]) for row in rows), "filtered": len(filtered), "filtered_reason_counts": {reason: sum(item.get("reason") == reason for item in filtered) for reason in sorted({item.get("reason") for item in filtered})}, "size_filter_version": SIZE_FILTER_VERSION, "min_short_side_ratio": float(min_short_side_ratio), "min_long_side_ratio": float(min_long_side_ratio), "canvas_hw": list(canvas_hw), "overlap": overlap}
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    """@shanlan-comment Parse CLI arguments and build one segment manifest."""
    parser = argparse.ArgumentParser(); parser.add_argument("--input-jsonl", type=Path, required=True); parser.add_argument("--output-jsonl", type=Path, required=True); parser.add_argument("--summary-json", type=Path, required=True); parser.add_argument("--canvas-h", type=int, default=96); parser.add_argument("--canvas-w", type=int, default=320); parser.add_argument("--overlap", type=int, default=64); parser.add_argument("--min-short-side-ratio", type=float, default=0.5); parser.add_argument("--min-long-side-ratio", type=float, default=0.5); parser.add_argument("--max-frames", type=int, default=None); args = parser.parse_args(argv)
    print(json.dumps(build(args.input_jsonl, args.output_jsonl, args.summary_json, (args.canvas_h, args.canvas_w), args.overlap, args.max_frames, args.min_short_side_ratio, args.min_long_side_ratio), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
