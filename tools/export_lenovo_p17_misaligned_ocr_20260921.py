"""Export P17 validation OCR annotations whose geometry misses the TextMask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from realesrgan.data.basic_text_line_full_frame import parse_p17_ocr_annotations, rasterize_polygon_mask


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=.85)
    args = parser.parse_args()
    if not 0.0 <= args.coverage_threshold <= 1.0:
        raise ValueError("coverage threshold must be in [0, 1]")
    mismatches = []
    total = 0
    for raw in args.manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw); files = row["files"]
        sample_id = str(row["sample_id"]); total += 1
        lq_path = _path(args.data_root, files["lq"])
        ocr_path = _path(args.data_root, files["ocr"])
        text_mask_path = _path(args.data_root, files["text_mask"])
        lq = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
        text_mask = cv2.imread(str(text_mask_path), cv2.IMREAD_GRAYSCALE)
        if lq is None or text_mask is None or lq.shape[:2] != text_mask.shape:
            raise ValueError(f"{sample_id}: invalid LQ/TextMask pair")
        document = json.loads(ocr_path.read_text(encoding="utf-8"))
        annotations = document.get("ocr", {}).get("annotations", [])
        core = np.zeros(lq.shape[:2], dtype=np.uint8)
        for line in parse_p17_ocr_annotations(document, lq.shape[:2]):
            core |= rasterize_polygon_mask(line.polygon_xy, lq.shape[:2]).astype(np.uint8)
        text = text_mask > 127
        coverage = float((text & (core > 0)).sum() / max(1, text.sum()))
        if coverage >= args.coverage_threshold:
            continue
        compact = []
        for index, item in enumerate(annotations):
            if not isinstance(item, dict):
                continue
            compact.append({key: item.get(key) for key in (
                "instance_id", "text", "transcription", "raw_text", "bbox", "points", "baseline", "visible", "ignore",
            ) if key in item})
        mismatches.append({
            "sample_id": sample_id,
            "text_mask_coverage_by_ocr_polygon": coverage,
            "text_mask_pixels": int(text.sum()),
            "ocr_polygon_pixels": int(core.sum()),
            "image_hw": list(lq.shape[:2]),
            "paths": {"lq": str(lq_path), "gt": str(_path(args.data_root, files["gt"])),
                      "text_mask": str(text_mask_path), "ocr": str(ocr_path)},
            "coordinate_system": document.get("coordinate_system"),
            "geometry_mapping": document.get("geometry_mapping"),
            "crop_provenance": document.get("crop_provenance"),
            "ocr_annotations": compact,
        })
    report = {"schema": "lenovo_p17_misaligned_ocr_v1", "criterion": {
        "metric": "text_mask pixels covered by union of OCR polygons", "coverage_threshold": args.coverage_threshold,
        "selection": "coverage < threshold"}, "manifest": str(args.manifest), "data_root": str(args.data_root),
        "samples_examined": total, "mismatch_count": len(mismatches), "mismatches": mismatches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples_examined": total, "mismatch_count": len(mismatches), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
