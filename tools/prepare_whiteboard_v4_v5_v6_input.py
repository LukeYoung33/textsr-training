"""Adapt train-only whiteboard V4 parents and OCR lines for the F007 materializer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2


def prepare(source: Path, output: Path) -> dict:
    """@luna-comment
    Input: Immutable whiteboard V4 root and a new output-local adapter root.
    Output: F007-compatible train/empty-val manifests, normalized OCR JSON, and a count report.
    Key Steps: Check every paired raster against OCR image dimensions and map accepted lines to bbox/points annotations.
    Key Params: Reject OCR lines only when marked rejected/ignored or geometrically invalid; no val samples are invented.
    State/Dependencies: Reads the source tree and writes only beneath the adapter output root.
    """
    source_manifest = source / "manifests" / "train.jsonl"
    if not source_manifest.is_file() or (source / "manifests" / "val.jsonl").exists():
        raise ValueError("expected an existing train-only whiteboard V4 source")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"adapter output is not empty: {output}")
    (output / "manifests").mkdir(parents=True)
    converted = []
    counts = {"train_parents": 0, "train_lines": 0, "rejected_lines": 0,
              "invalid_lines": 0, "val_parents": 0, "val_lines": 0}
    seen_ids = set()
    for source_line in source_manifest.read_text(encoding="utf-8").splitlines():
        if not source_line.strip():
            continue
        parent = json.loads(source_line)
        sample_id = parent["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"duplicate parent ID: {sample_id}")
        seen_ids.add(sample_id)
        source_files = {name: source / rel for name, rel in parent["files"].items() if rel}
        ocr = json.loads(source_files["ocr"].read_text(encoding="utf-8"))
        image_info = ocr["image"]
        width, height = int(image_info["width"]), int(image_info["height"])
        if width < 2 or height < 2:
            raise ValueError(f"invalid source dimensions: {sample_id}")
        for name in ("lq", "gt", "text_mask", "valid_mask"):
            if name not in source_files:
                raise ValueError(f"missing {name}: {sample_id}")
            raster = cv2.imread(str(source_files[name]), cv2.IMREAD_UNCHANGED)
            if raster is None or raster.shape[:2] != (height, width):
                raise ValueError(f"source raster shape mismatch: {sample_id}/{name}")
        annotations = []
        for line in ocr.get("lines", []):
            if line.get("ignore") or line.get("visible") is False or line.get("reject_reasons"):
                counts["rejected_lines"] += 1
                continue
            bbox = line.get("bbox_xyxy")
            polygon = line.get("polygon")
            if (not isinstance(bbox, list) or len(bbox) != 4 or
                    not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bbox) or
                    bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2 or
                    not isinstance(polygon, list) or len(polygon) < 3 or
                    not all(isinstance(p, list) and len(p) >= 2 and
                            all(isinstance(v, (int, float)) and math.isfinite(v) for v in p[:2])
                            for p in polygon)):
                counts["invalid_lines"] += 1
                continue
            annotations.append({"bbox": bbox, "points": [p[:2] for p in polygon],
                                "text": line.get("text", ""),
                                "source_line_id": line.get("line_id"),
                                "rec_score": line.get("rec_score")})
        ocr_rel = Path("train") / "OCR" / f"{sample_id}.json"
        ocr_target = output / ocr_rel
        ocr_target.parent.mkdir(parents=True, exist_ok=True)
        ocr_target.write_text(json.dumps({"ocr": {"image": sample_id,
                                                  "annotations": annotations,
                                                  "annotation_count": len(annotations)},
                                          "source_ocr": str(source_files["ocr"])}, ensure_ascii=False),
                              encoding="utf-8")
        converted.append({"sample_id": sample_id, "cropped_size_wh": [width, height],
                          "files": {**{name: str(path) for name, path in source_files.items() if name != "ocr"},
                                    "ocr": ocr_rel.as_posix()},
                          "source_manifest": str(source_manifest),
                          "source_ocr": str(source_files["ocr"])})
        counts["train_parents"] += 1
        counts["train_lines"] += len(annotations)
    (output / "manifests" / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
    (output / "manifests" / "val.jsonl").write_text("", encoding="utf-8")
    (output / "adapter_report.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


def main() -> int:
    """@luna-comment
    Input: Whiteboard V4 source and output-local adapter paths from CLI.
    Output: Printed line-count report and process exit status.
    Key Steps: Parse explicit paths and run the source-preserving conversion.
    Key Params: One train-only source, no split inference.
    State/Dependencies: Writes only the selected adapter root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
