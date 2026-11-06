"""Build filtered, aspect-preserving V4 Y4UV text-line manifests.

The builder consumes the requested frame-level ``training_lists_full_*`` JSONL
files and writes only derived manifests.  Source lists are never modified.
Frames associated with HX are excluded, OCR lines whose minimum-area short side
is below 50 source pixels are excluded, vertical lines are rotated into
horizontal reading order, and every accepted line is resized proportionally to
height 96 before being split or right-padded to 96x480 at dataset time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import struct
from typing import Any, Iterable

import cv2


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from realesrgan.data.basic_text_line_y4uv_v4_geometry import (
    DEFAULT_CANVAS_HW,
    DEFAULT_MIN_TEXT_SIZE,
    DEFAULT_SPLIT_OVERLAP,
    GEOMETRY_VERSION,
    build_line_geometries,
    is_hx_related,
)
from realesrgan.data.text_line_crop_v3 import polygon_to_array


MANIFEST_VERSION = "y4uv_v4_filtered_aspect_resize96_rotate_vertical_v1"
REQUIRED_ASSETS = ("lq", "gt", "ocr")


def _sha256(path: Path) -> str:
    """@shanlan-comment Hash a source or derived manifest for reproducibility."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _image_hw(path: str) -> tuple[int, int]:
    """@shanlan-comment Read one paired image header and return height/width."""
    try:
        with Path(path).open("rb") as handle:
            header = handle.read(24)
        if header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            width, height = struct.unpack(">II", header[16:24])
            if width > 0 and height > 0:
                return int(height), int(width)
    except (OSError, struct.error):
        pass
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return int(image.shape[0]), int(image.shape[1])


def _manifest_image_hw(record: dict[str, Any]) -> tuple[int, int]:
    """@shanlan-comment Prefer manifest dimensions and otherwise inspect the LQ header."""
    image_hw = record.get("image_hw") or record.get("lq_hw")
    if image_hw and len(image_hw) >= 2:
        return int(image_hw[0]), int(image_hw[1])
    return _image_hw(str(record["lq"]))


def _read_polygons(ocr_path: str) -> tuple[str, list[list[list[float]]], list[str]]:
    """@shanlan-comment Parse legacy and V3 OCR JSON polygon schemas with text metadata."""
    payload = json.loads(Path(ocr_path).read_text(encoding="utf-8"))
    for field in ("rec_polys", "dt_polys", "rec_boxes"):
        values = payload.get(field) or []
        if not values:
            continue
        polygons: list[list[list[float]]] = []
        for value in values:
            try:
                polygons.append(polygon_to_array(value).astype(float).tolist())
            except (TypeError, ValueError):
                continue
        texts = payload.get("rec_texts") or payload.get("text_word") or []
        return field, polygons, [str(item) for item in texts]

    lines = payload.get("lines") or []
    if lines:
        polygons = []
        texts = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            value = line.get("polygon") or line.get("original_polygon")
            try:
                polygons.append(polygon_to_array(value).astype(float).tolist())
            except (TypeError, ValueError):
                continue
            texts.append(str(line.get("text", "")))
        return "lines.polygon", polygons, texts
    return "", [], []


def _validate_assets(record: dict[str, Any], line_number: int) -> None:
    """@shanlan-comment Require the paired images and OCR file before deriving geometry."""
    missing = [key for key in REQUIRED_ASSETS if not record.get(key)]
    if missing:
        raise ValueError(f"line {line_number} missing keys: {missing}")
    for key in REQUIRED_ASSETS:
        if not Path(str(record[key])).is_file():
            raise FileNotFoundError(f"line {line_number} {key}: {record[key]}")


def build_v4_manifest(
    input_jsonl: Path,
    output_jsonl: Path,
    summary_json: Path,
    *,
    canvas_hw: tuple[int, int] = DEFAULT_CANVAS_HW,
    min_text_size: float = DEFAULT_MIN_TEXT_SIZE,
    split_overlap: int = DEFAULT_SPLIT_OVERLAP,
    exclude_hx: bool = True,
    max_records: int | None = None,
) -> dict[str, Any]:
    """@shanlan-comment Derive auditable V4 rows from one frame-level source split."""
    rows: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    orientation_counts: Counter[str] = Counter()
    frame_count = 0
    accepted_frame_count = 0
    with input_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if max_records is not None and frame_count >= int(max_records):
                break
            frame_count += 1
            record = json.loads(line)
            stem = str(record.get("stem", record.get("frame_id", f"frame_{line_number}")))
            if exclude_hx and is_hx_related(record):
                filtered.append({"line_number": line_number, "stem": stem, "reason": "HX_RELATED_FRAME"})
                continue
            try:
                _validate_assets(record, line_number)
                lq_hw = _image_hw(str(record["lq"]))
                gt_hw = _image_hw(str(record["gt"]))
                if lq_hw != gt_hw:
                    raise ValueError(f"LQ_GT_SHAPE_MISMATCH:{lq_hw}!={gt_hw}")
                image_hw = lq_hw
                source_field, polygons, texts = _read_polygons(str(record["ocr"]))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                filtered.append(
                    {"line_number": line_number, "stem": stem, "reason": "FRAME_INVALID", "detail": str(exc)}
                )
                continue
            if not polygons:
                filtered.append({"line_number": line_number, "stem": stem, "reason": "NO_VALID_OCR_POLYGON"})
                continue
            accepted_frame_count += 1
            source_line_count = 0
            for line_index, polygon in enumerate(polygons):
                try:
                    geometries, rejection = build_line_geometries(
                        polygon,
                        image_hw,
                        canvas_hw=canvas_hw,
                        min_text_size=min_text_size,
                        split_overlap=split_overlap,
                        line_index=line_index,
                    )
                except (TypeError, ValueError) as exc:
                    filtered.append(
                        {
                            "line_number": line_number,
                            "stem": stem,
                            "line_index": line_index,
                            "reason": "GEOMETRY_FAILED",
                            "detail": str(exc),
                        }
                    )
                    continue
                if rejection is not None:
                    filtered.append(
                        {
                            "line_number": line_number,
                            "stem": stem,
                            "line_index": line_index,
                            **rejection,
                        }
                    )
                    continue
                source_line_id = f"{stem}::line{line_index}"
                transcription = texts[line_index] if line_index < len(texts) else ""
                for geometry in geometries:
                    orientation_counts[geometry.orientation] += 1
                    row = {
                        "manifest_version": MANIFEST_VERSION,
                        "geometry_version": GEOMETRY_VERSION,
                        "dataset": record.get("dataset", ""),
                        "stem": stem,
                        "frame_id": stem,
                        "source_line_id": source_line_id,
                        "segment_id": f"{source_line_id}::tile{geometry.tile_index}",
                        "line_index": line_index,
                        "tile_index": geometry.tile_index,
                        "tile_count": geometry.tile_count,
                        "polygon_source": source_field,
                        "transcription": transcription,
                        "lq": str(record["lq"]),
                        "gt": str(record["gt"]),
                        "source_manifest_line": line_number,
                        **geometry.as_record_fields(),
                    }
                    rows.append(row)
                source_line_count += 1
            if source_line_count == 0:
                accepted_frame_count -= 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "manifest_version": MANIFEST_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "input_manifest": str(input_jsonl),
        "input_sha256": _sha256(input_jsonl),
        "output_manifest": str(output_jsonl),
        "output_sha256": _sha256(output_jsonl),
        "canvas_hw": list(canvas_hw),
        "min_text_size": float(min_text_size),
        "split_overlap": int(split_overlap),
        "exclude_hx": bool(exclude_hx),
        "frame_count": frame_count,
        "accepted_frame_count": accepted_frame_count,
        "segment_count": len(rows),
        "source_line_count": len({row["source_line_id"] for row in rows}),
        "tile_line_count": len({row["source_line_id"] for row in rows if int(row["tile_count"]) > 1}),
        "orientation_counts": dict(sorted(orientation_counts.items())),
        "filtered_count": len(filtered),
        "filtered_reason_counts": dict(sorted(Counter(item.get("reason", "UNKNOWN") for item in filtered).items())),
        "filtered": filtered,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    """@shanlan-comment Build one V4 manifest from CLI paths and print its audit summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--canvas-h", type=int, default=DEFAULT_CANVAS_HW[0])
    parser.add_argument("--canvas-w", type=int, default=DEFAULT_CANVAS_HW[1])
    parser.add_argument("--min-text-size", type=float, default=DEFAULT_MIN_TEXT_SIZE)
    parser.add_argument("--split-overlap", type=int, default=DEFAULT_SPLIT_OVERLAP)
    parser.add_argument("--exclude-hx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)
    summary = build_v4_manifest(
        args.input_jsonl,
        args.output_jsonl,
        args.summary_json,
        canvas_hw=(args.canvas_h, args.canvas_w),
        min_text_size=args.min_text_size,
        split_overlap=args.split_overlap,
        exclude_hx=args.exclude_hx,
        max_records=args.max_records,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
