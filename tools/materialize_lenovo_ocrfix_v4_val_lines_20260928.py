"""Materialize Lenovo 20260920 OCR-fix validation lines with frozen mild S-curve LQ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.materialize_lenovo_mild_scurve_lines import _box, _envelope, _local_annotation
from tools.text_lq_scurve_mild import DEFAULT_CONFIG, preprocess_text_lq_bgr


def _write(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise IOError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-parents", type=int)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    parents = [json.loads(line) for line in (args.source_root / "manifests/val.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if args.max_parents is not None:
        parents = parents[:args.max_parents]
    args.output_root.mkdir(parents=True)
    records = []
    for parent in parents:
        parent_id = parent["sample_id"]
        files = parent["files"]
        lq = cv2.imread(str(args.source_root / files["lq"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(args.source_root / files["gt"]), cv2.IMREAD_COLOR)
        ocr = json.loads((args.source_root / files["ocr"]).read_text(encoding="utf-8"))
        if lq is None or gt is None or lq.shape != gt.shape:
            raise ValueError(f"unaligned source images: {parent_id}")
        annotations = ocr["ocr"]["annotations"]
        for ordinal, annotation in enumerate(annotations):
            if annotation.get("ignore") or annotation.get("visible") is False:
                continue
            box = _box(annotation, lq.shape[:2])
            if box is None:
                continue
            envelope = _envelope(box, lq.shape[:2], "val")
            left, top, right, bottom = envelope
            line_id = f"{parent_id}__line{ordinal:03d}"
            raw = lq[top:bottom, left:right]
            processed = preprocess_text_lq_bgr(raw)
            assets = {
                "lq_raw": args.output_root / "val/LQ_raw" / f"{line_id}.png",
                "lq": args.output_root / "val/LQ_scurve" / f"{line_id}.png",
                "gt": args.output_root / "val/GT" / f"{line_id}.png",
                "ocr": args.output_root / "val/OCR" / f"{line_id}.json",
            }
            _write(assets["lq_raw"], raw)
            _write(assets["lq"], processed)
            _write(assets["gt"], gt[top:bottom, left:right])
            local = {"schema_version": "line_envelope_v1", "ocr": {"image": line_id,
                     "annotations": [_local_annotation(annotation, left, top)], "annotation_count": 1},
                     "source_ocr": str(args.source_root / files["ocr"]), "source_envelope_xyxy": list(envelope),
                     "ocrfix_source": "v4_formula_groups"}
            assets["ocr"].parent.mkdir(parents=True, exist_ok=True)
            assets["ocr"].write_text(json.dumps(local, ensure_ascii=False), encoding="utf-8")
            records.append({"sample_id": line_id, "parent_sample_id": parent_id,
                            "source_release_root": str(args.source_root),
                            "line_ordinal": ordinal, "line_envelope_xyxy_in_parent": list(envelope),
                            "ocr_crop_xyxy": [box[0] - left, box[1] - top, box[2] - left, box[3] - top],
                            "files": {key: str(path) for key, path in assets.items()}})
        print(json.dumps({"parent": parent_id, "lines_total": len(records)}, ensure_ascii=False), flush=True)
    manifest_dir = args.output_root / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "val.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {"source_root": str(args.source_root), "output_root": str(args.output_root),
               "parents": len(parents), "lines": len(records), "preprocess_version": DEFAULT_CONFIG.version,
               "crop": {"vertical_context": .12, "horizontal_context": .03}}
    (manifest_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
