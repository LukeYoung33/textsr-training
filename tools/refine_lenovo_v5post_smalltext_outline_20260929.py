"""Reduce V5-postprocess contour amplification on small OCR text lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SOURCE = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929")
BASE = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_normal_v5post_boardframe_20260929")
OUTPUT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_smalltext_outline_20260929")
ARM = "gtfit_v6_borderrail_h96"


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _line_weight(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    distance = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy)).astype(np.float32)
    return _smoothstep((distance - 6.0) / 32.0).astype(np.float32)


def _preview(normal: np.ndarray, base: np.ndarray, refined: np.ndarray, path: Path) -> None:
    width = min(900, normal.shape[1])
    height = round(normal.shape[0] * width / normal.shape[1])
    views = [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
             for image in (normal, base, refined)]
    title = np.full((34, 3 * width, 3), 245, np.uint8)
    for index, label in enumerate(("NORMAL", "V5 POST", "SMALL TEXT SAFE")):
        cv2.putText(title, label, (index * width + 12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (20, 20, 20), 2)
    cv2.imwrite(str(path), np.vstack((title, np.hstack(views))), [cv2.IMWRITE_JPEG_QUALITY, 92])


def refine(sample_id: str) -> dict:
    normal_path = SOURCE / ARM / "normal" / sample_id / "output.png"
    base_dir = BASE / ARM / sample_id
    normal = cv2.imread(str(normal_path), cv2.IMREAD_COLOR)
    base = cv2.imread(str(base_dir / "output.png"), cv2.IMREAD_COLOR)
    if normal is None or base is None or normal.shape != base.shape:
        raise ValueError(f"missing or mismatched images: {sample_id}")
    report = json.loads((base_dir / "result.json").read_text(encoding="utf-8"))
    height, width = normal.shape[:2]
    strength_sum = np.zeros((height, width), np.float32)
    weight_sum = np.zeros((height, width), np.float32)
    strength_by_line = []
    for line in report["line_reports"]:
        x0, y0, x1, y1 = line["crop_xyxy"]
        crop_height = y1 - y0
        strength = float(np.clip(0.5 + 0.5 * (crop_height - 130.0) / 60.0, 0.5, 1.0))
        weight = _line_weight(crop_height, x1 - x0)
        strength_sum[y0:y1, x0:x1] += strength * weight
        weight_sum[y0:y1, x0:x1] += weight
        strength_by_line.append({"ocr_index": line["ocr_index"], "crop_height": crop_height,
                                 "strength": strength})
    alpha = np.ones((height, width), np.float32)
    np.divide(strength_sum, weight_sum, out=alpha, where=weight_sum > 0)
    refined = np.clip(np.rint(normal.astype(np.float32) +
                              alpha[:, :, None] * (base.astype(np.float32) - normal)), 0, 255).astype(np.uint8)
    directory = OUTPUT / ARM / sample_id
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "output.png"), refined)
    cv2.imwrite(str(directory / "output_full.jpg"), refined, [cv2.IMWRITE_JPEG_QUALITY, 94])
    _preview(normal, base, refined, directory / "preview.jpg")
    record = {"sample_id": sample_id, "arm": ARM, "normal": str(normal_path),
              "base": str(base_dir / "output.png"), "output": str(directory / "output.png"),
              "attenuated_lines": sum(line["strength"] < 1 for line in strength_by_line),
              "line_strengths": strength_by_line,
              "changed_pixels_from_base": int(np.count_nonzero(np.any(refined != base, axis=2)))}
    (directory / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id")
    args = parser.parse_args()
    samples = [args.sample_id] if args.sample_id else sorted(
        path.parent.name for path in (BASE / ARM).glob("*/result.json"))
    for index, sample in enumerate(samples, 1):
        result = refine(sample)
        print(json.dumps({"image": index, "total": len(samples), "sample_id": sample,
                          "attenuated_lines": result["attenuated_lines"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
