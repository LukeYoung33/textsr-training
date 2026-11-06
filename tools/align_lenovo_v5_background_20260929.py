"""Blend V5 line backgrounds to the source while retaining near-text corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from tools.postprocess_lenovo_borderrail_normal_v5_20260929 import (
    _blend_weight, _board_frame_weight, _smoothstep,
)
from tools.text_lq_scurve_mild import _apply_luma_delta, _luma_bgr


SOURCE_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929")
CANDIDATE_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_v5post_rawdelta_20260929")
OUTPUT_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_v5post_bganchored_20260929")
ARM = "gtfit_v6_borderrail_h96"


def align(source: np.ndarray, candidate: np.ndarray, lines: list[dict]) -> tuple[np.ndarray, dict]:
    if source.shape != candidate.shape:
        raise ValueError("source and candidate sizes differ")
    height, width = source.shape[:2]
    source_y = _luma_bgr(source)
    candidate_y = _luma_bgr(candidate)
    support = np.zeros((height, width), np.float32)
    foreground = np.zeros((height, width), np.uint8)
    background = np.zeros((height, width), np.uint8)
    for line in lines:
        x0, y0, x1, y1 = map(int, line["crop_xyxy"])
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid crop: {line['crop_xyxy']}")
        crop = source_y[y0:y1, x0:x1]
        crop_u8 = crop.astype(np.uint8)
        threshold, _ = cv2.threshold(crop_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        light_background = float(np.median(crop)) >= float(threshold)
        if light_background:
            foreground[y0:y1, x0:x1] |= (crop < threshold - 7).astype(np.uint8)
            background[y0:y1, x0:x1] |= (crop > threshold + 15).astype(np.uint8)
        else:
            foreground[y0:y1, x0:x1] |= (crop > threshold + 7).astype(np.uint8)
            background[y0:y1, x0:x1] |= (crop < threshold - 15).astype(np.uint8)
        weight = _blend_weight(source[y0:y1, x0:x1])
        support[y0:y1, x0:x1] += weight
    support = np.minimum(support, 1.0)
    frame_weight, _ = _board_frame_weight(source)
    support *= frame_weight
    distance = cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 3)
    far_background = ((background > 0) & (distance > 25.0) & (support > 0.25)).astype(np.float32)
    denominator = cv2.GaussianBlur(far_background, (0, 0), 60.0)
    numerator = cv2.GaussianBlur((candidate_y - source_y) * far_background, (0, 0), 60.0)
    field = np.divide(numerator, np.maximum(denominator, 1e-3))
    field = np.clip(field, -24.0, 24.0)
    near_text = 1.0 - _smoothstep((distance - 12.0) / 50.0)
    correction = -field * (1.0 - near_text) * support
    correction[denominator < 0.005] = 0.0
    output = _apply_luma_delta(candidate, correction)
    metrics = {"anchor_pixels": int(np.count_nonzero(far_background)),
               "corrected_pixels": int(np.count_nonzero(np.rint(correction))),
               "max_abs_correction": float(np.max(np.abs(correction)))}
    return output, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--arm", default=ARM)
    args = parser.parse_args()
    if args.batch:
        reports = sorted((args.candidate_root / args.arm).glob("*/result.json"))
        if len(reports) != 33:
            raise ValueError(f"expected 33 candidate reports, got {len(reports)}")
        jobs = [(args.source_root / args.arm / "normal" / path.parent.name / "output.png",
                 path.parent / "output.png", path,
                 args.output_root / args.arm / path.parent.name / "output.png") for path in reports]
    else:
        if any(path is None for path in (args.source, args.candidate, args.report, args.output)):
            parser.error("--source, --candidate, --report and --output are required without --batch")
        jobs = [(args.source, args.candidate, args.report, args.output)]
    cv2.setNumThreads(2)
    for index, (source_path, candidate_path, report_path, output_path) in enumerate(jobs, 1):
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        candidate = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
        if source is None or candidate is None:
            raise ValueError(f"missing source or candidate image: {source_path}, {candidate_path}")
        lines = json.loads(report_path.read_text(encoding="utf-8"))["line_reports"]
        result, metrics = align(source, candidate, lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), result):
            raise OSError(f"cannot write {output_path}")
        if args.batch:
            cv2.imwrite(str(output_path.parent / "output_full.jpg"), result,
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
            shutil.copyfile(report_path, output_path.parent / "source_result.json")
            (output_path.parent / "background_anchor.json").write_text(
                json.dumps({"source": str(source_path), "candidate": str(candidate_path), **metrics}, indent=2),
                encoding="utf-8")
        print(json.dumps({"image": index, "total": len(jobs), "sample": output_path.parent.name, **metrics}), flush=True)


if __name__ == "__main__":
    main()
