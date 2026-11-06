"""Run frozen PP-OCRv6 tiny OCR on staged Lenovo Mojito test images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _digest(path: Path) -> str:
    """@luna-comment
    Input: One staged source image path.
    Output: SHA-256 digest of its original bytes.
    Key Steps: Stream data in bounded blocks.
    Key Params: One MiB chunks.
    State/Dependencies: Read-only source access.
    """
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _normalize(result: dict, shape: tuple[int, int]) -> dict:
    """@luna-comment
    Input: One PaddleOCR result JSON and source height/width.
    Output: Validated F008 line annotations with bbox, polygon, text and score.
    Key Steps: Prefer recognized polygons, clip points to source bounds and reject degenerate geometry.
    Key Params: Two-pixel minimum line box; no recognition-score gate.
    State/Dependencies: Pure JSON/NumPy conversion.
    """
    height, width = shape
    payload = result.get("res", result)
    polygons = payload.get("rec_polys")
    if polygons is None:
        polygons = payload.get("dt_polys", [])
    texts = payload.get("rec_texts", [])
    scores = payload.get("rec_scores", [])
    annotations = []
    rejected = 0
    for index, poly in enumerate(polygons):
        points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3 or not np.isfinite(points).all():
            rejected += 1
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        x0, y0 = np.floor(points.min(axis=0)).astype(int)
        x1, y1 = np.ceil(points.max(axis=0)).astype(int) + 1
        if x1 - x0 < 2 or y1 - y0 < 2:
            rejected += 1
            continue
        score = float(scores[index]) if index < len(scores) else None
        if score is not None and not math.isfinite(score):
            score = None
        annotations.append({"bbox": [int(x0), int(y0), int(x1), int(y1)],
                            "points": points.tolist(),
                            "text": str(texts[index]) if index < len(texts) else "",
                            "rec_score": score, "ocr_index": index})
    return {"ocr": {"annotations": annotations, "annotation_count": len(annotations)},
            "rejected_geometry": rejected}


def run(input_root: Path, output_root: Path, detector: Path, recognizer: Path,
        limit: int | None = None) -> dict:
    """@luna-comment
    Input: Staged PPT/blackboard groups, OCR output root and frozen local model directories.
    Output: One raw and normalized OCR JSON plus one overlay per JPG and a batch report.
    Key Steps: Construct the explicit PP-OCRv6 tiny pipeline once, predict every image and validate line geometry.
    Key Params: Source-profile detection limit 1600, DB thresholds 0.3/0.6 and unclip 1.5; CPU runtime.
    State/Dependencies: Reads only staged inputs/models and writes only beneath output_root.
    """
    from paddleocr import PaddleOCR

    images = sorted(path for group in ("ppt", "blackwhite")
                    for path in (input_root / group).glob("*.jpg"))
    if not images or not detector.is_dir() or not recognizer.is_dir():
        raise FileNotFoundError("missing test images or frozen OCR models")
    if limit is not None:
        images = images[:limit]
    engine = PaddleOCR(
        text_detection_model_name="PP-OCRv6_tiny_det",
        text_detection_model_dir=str(detector),
        text_recognition_model_name="PP-OCRv6_tiny_rec",
        text_recognition_model_dir=str(recognizer),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1600,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.6,
        text_det_unclip_ratio=1.5,
        text_rec_score_thresh=0.0,
        text_recognition_batch_size=1,
        return_word_box=True,
        ocr_version="PP-OCRv6",
        lang="ch",
        device="cpu",
    )
    rows = []
    for image_path in images:
        group = image_path.parent.name
        raw_path = output_root / "raw" / group / f"{image_path.stem}.json"
        normalized_path = output_root / "normalized" / group / f"{image_path.stem}.json"
        overlay_path = output_root / "overlay" / group / f"{image_path.stem}.png"
        if normalized_path.is_file() and raw_path.is_file() and overlay_path.is_file():
            normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
        else:
            source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if source is None:
                raise ValueError(f"undecodable input image: {image_path}")
            result_items = list(engine.predict(input=str(image_path)))
            if len(result_items) != 1:
                raise ValueError(f"unexpected OCR result count for {image_path}: {len(result_items)}")
            raw = result_items[0].json
            normalized = _normalize(raw, source.shape[:2])
            normalized.update({"source": str(image_path), "source_sha256": _digest(image_path),
                               "source_hw": list(source.shape[:2]), "ocr_model": "PP-OCRv6_tiny"})
            overlay = source.copy()
            for annotation in normalized["ocr"]["annotations"]:
                polygon = np.rint(np.asarray(annotation["points"])).astype(np.int32)
                cv2.polylines(overlay, [polygon], True, (0, 255, 0), 3, cv2.LINE_AA)
            for path in (raw_path, normalized_path, overlay_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            normalized_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
            if not cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise IOError(overlay_path)
        row = {"group": group, "source": str(image_path), "ocr_raw": str(raw_path),
               "ocr_normalized": str(normalized_path), "overlay": str(overlay_path),
               "lines": normalized["ocr"]["annotation_count"],
               "rejected_geometry": normalized["rejected_geometry"]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {"images": len(rows), "lines": sum(row["lines"] for row in rows), "rows": rows}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    """@luna-comment
    Input: Staged image root, OCR output root, frozen models and optional pilot limit.
    Output: Printed batch counts and process exit status.
    Key Steps: Parse paths, run OCR, publish summary.
    Key Params: Limit processes the first sorted images only.
    State/Dependencies: Writes to one OCR output root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root, args.detector, args.recognizer, args.limit)
    print(json.dumps({"images": result["images"], "lines": result["lines"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
