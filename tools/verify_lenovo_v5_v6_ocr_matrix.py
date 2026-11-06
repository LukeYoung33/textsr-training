"""Verify four-arm OCR line datasets after all shard manifests are merged."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.materialize_lenovo_v5_v6_ocr_matrix import ARMS, _load_jsonl


def verify(output: Path, expected: dict[str, int] | None = None,
           arms: tuple[str, ...] = ARMS, report_name: str = "verification.json") -> dict:
    """@luna-comment
    Input: Complete four-arm output root and optional expected split counts.
    Output: Machine-readable verification report or an explicit failure.
    Key Steps: Compare ordered IDs/geometry, resolve and decode all rasters, check alignment and direct-GT identity.
    Key Params: Every referenced asset is checked; no random subset decides completeness.
    State/Dependencies: Reads materialized files and writes one verification JSON report.
    """
    expected = expected or {}
    report = {"schema": "lenovo_v5_v6_ocr_matrix_verify_v1", "counts": {}, "passed": False}
    reference = {}
    checked_shapes: dict[Path, tuple[int, int]] = {}
    checked_ocr: set[Path] = set()
    for arm in arms:
        root = output / arm
        for split in ("train", "val"):
            rows = _load_jsonl(root / "manifests" / f"{split}.jsonl")
            ids = [row["sample_id"] for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate IDs: {arm}/{split}")
            if split in expected and len(rows) != expected[split]:
                raise ValueError(f"unexpected {arm}/{split} count: {len(rows)} != {expected[split]}")
            geometry = [(row["sample_id"], row["ocr_crop_xyxy"], row["line_envelope_xyxy_in_parent"])
                        for row in rows]
            if split in reference and geometry != reference[split]:
                raise ValueError(f"arm geometry mismatch: {arm}/{split}")
            reference.setdefault(split, geometry)
            for row in rows:
                width, height = row["cropped_size_wh"]
                box = row["ocr_crop_xyxy"]
                if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    raise ValueError(f"invalid local OCR: {arm}/{row['sample_id']}")
                for name, rel in row["files"].items():
                    path = Path(os.path.normpath(root / rel))
                    if name == "ocr":
                        if path not in checked_ocr:
                            if not path.is_file():
                                raise FileNotFoundError(path)
                            checked_ocr.add(path)
                        continue
                    previous_shape = checked_shapes.setdefault(path, (height, width))
                    if previous_shape != (height, width):
                        raise ValueError(f"conflicting expected dimensions: {path}")
                if arm.removesuffix("_borderrail") in ("v5", "v6"):
                    local_gt = f"{split}/GT/{row['sample_id']}.png"
                    if row["files"]["gt"] != local_gt:
                        raise ValueError(f"direct GT is not arm-local: {arm}/{row['sample_id']}")
                if row["preprocess"].get("arm") != arm:
                    raise ValueError(f"incorrect arm telemetry: {arm}/{row['sample_id']}")
            report["counts"][f"{arm}_{split}"] = len(rows)
    def check_asset(item: tuple[Path, tuple[int, int]]) -> None:
        """@luna-comment
        Input: One unique image path and the common expected pixel dimensions.
        Output: Successful decode/shape assertion or an explicit error.
        Key Steps: Decode the complete raster once and compare height/width.
        Key Params: The executor bounds concurrent SMB reads to twelve assets.
        State/Dependencies: Read-only OpenCV I/O; no shared image buffer.
        """
        path, shape = item
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != shape:
            raise ValueError(f"missing, undecodable, or misaligned asset: {path}")

    with ThreadPoolExecutor(max_workers=12) as pool:
        for index, _ in enumerate(pool.map(check_asset, checked_shapes.items()), start=1):
            if index % 5000 == 0:
                print(json.dumps({"decoded_unique_assets": index, "total": len(checked_shapes)}), flush=True)
    report["decoded_unique_assets"] = len(checked_shapes)
    report["passed"] = True
    (output / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """@luna-comment
    Input: CLI output root and optional train/val count gates.
    Output: Printed verification report and process exit status.
    Key Steps: Parse expected counts then check every generated pair.
    Key Params: Counts default to unconstrained for a pilot.
    State/Dependencies: Writes verification.json within output root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm names")
    parser.add_argument("--report-name", default="verification.json")
    args = parser.parse_args()
    expected = {k: v for k, v in (("train", args.train_count), ("val", args.val_count)) if v is not None}
    arms = tuple(name.strip() for name in args.arms.split(",") if name.strip())
    if not arms or Path(args.report_name).name != args.report_name:
        parser.error("invalid arms or report name")
    print(json.dumps(verify(args.output, expected, arms, args.report_name), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
