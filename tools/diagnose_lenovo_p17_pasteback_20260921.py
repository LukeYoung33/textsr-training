"""Rank P17 pasted-back OCR lines by their actual LQ-to-output change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from realesrgan.data.basic_text_line_full_frame import parse_p17_ocr_annotations, rasterize_polygon_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pasteback-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    coverage_rows = []
    for sample_root in sorted(args.pasteback_root.iterdir()):
        output_path = sample_root / "output.png"
        if not sample_root.is_dir() or not output_path.is_file():
            continue
        sample_id = sample_root.name
        lq = cv2.imread(str(args.data_root / "val" / "LQ" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
        output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        ocr_path = args.data_root / "val" / "OCR" / f"{sample_id}.json"
        if lq is None or output is None or not ocr_path.is_file() or lq.shape != output.shape:
            continue
        ocr_data = json.loads(ocr_path.read_text(encoding="utf-8"))
        annotations = parse_p17_ocr_annotations(ocr_data, lq.shape[:2])
        core_all = np.zeros(lq.shape[:2], dtype=np.uint8)
        for line in annotations:
            mask = rasterize_polygon_mask(line.polygon_xy, lq.shape[:2]) > .5
            core_all |= mask.astype(np.uint8)
            delta = np.abs(output.astype(np.float32) - lq.astype(np.float32))[mask]
            rows.append({"sample_id": sample_id, "line_index": line.line_index, "text": line.text,
                         "pixels": int(mask.sum()), "mean_abs_rgb": float(delta.mean()) if delta.size else 0.0,
                         "p95_abs_rgb": float(np.percentile(delta, 95.0)) if delta.size else 0.0})
        text_mask = cv2.imread(str(args.data_root / "val" / "text_mask" / f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        if text_mask is not None and text_mask.shape == lq.shape[:2]:
            text = text_mask > 127
            coverage_rows.append({"sample_id": sample_id, "text_pixels": int(text.sum()),
                                  "ocr_core_pixels": int(core_all.sum()),
                                  "text_covered_by_ocr": float((text & (core_all > 0)).sum() / max(1, text.sum()))})
    rows.sort(key=lambda item: (item["mean_abs_rgb"], item["p95_abs_rgb"]))
    report = {"lines": len(rows), "low_change_lt_0_5": sum(item["mean_abs_rgb"] < .5 for item in rows),
              "low_change_lt_1": sum(item["mean_abs_rgb"] < 1.0 for item in rows), "ranked": rows,
              "text_mask_coverage": coverage_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("lines", "low_change_lt_0_5", "low_change_lt_1")}, ensure_ascii=False))
    for item in rows[:20]:
        print(json.dumps(item, ensure_ascii=False))
    if coverage_rows:
        print(json.dumps({"text_covered_by_ocr_mean": float(np.mean([item["text_covered_by_ocr"] for item in coverage_rows])),
                          "text_covered_by_ocr_min": float(np.min([item["text_covered_by_ocr"] for item in coverage_rows]))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
