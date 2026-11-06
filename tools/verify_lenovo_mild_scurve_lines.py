"""Verify the materialized Lenovo mild S-curve line dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


RASTER_NAMES = ("lq_raw", "lq", "gt", "text_mask", "valid_mask", "lq_foreground_mask", "fit_support_mask")


def verify(root: Path) -> dict:
    """@luna-comment
    Input: Completed materialized dataset root.
    Output: Verification report; raises on missing or misaligned assets.
    Key Steps: Check manifest counts, pair lists, every asset shape, OCR locality, and fallback identity.
    Key Params: All raster assets decode at the manifest crop size; each OCR file has one line.
    State/Dependencies: Reads materialized files; no file mutation.
    """
    manifest_root = root / "manifests"
    summary = json.loads((manifest_root / "manifest.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("materialization summary is not complete")
    report: dict[str, object] = {"status": "passed", "root": str(root), "counts": {}, "fallback_counts": {}}
    fallback_counts = Counter()
    active_changed = 0
    parent_sets: dict[str, set[str]] = {}
    identity_sets: dict[str, set[str]] = {}
    for split in ("train", "val"):
        rows = [json.loads(line) for line in (manifest_root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != int(summary["counts"][f"{split}_lines"]):
            raise ValueError(f"{split} count differs from summary")
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {split} line IDs")
        parent_sets[split] = {row["parent_sample_id"] for row in rows}
        identity_sets[split] = {str(row.get("source_identity")) for row in rows if row.get("source_identity")}
        expected_pairs = [f"{row['files']['lq']} {row['files']['gt']}" for row in rows]
        expected_raw = [f"{row['files']['lq_raw']} {row['files']['gt']}" for row in rows]
        if (manifest_root / f"{split}_pairs.txt").read_text(encoding="utf-8").splitlines() != expected_pairs:
            raise ValueError(f"{split} processed pair list differs from JSONL")
        if (manifest_root / f"{split}_raw_pairs.txt").read_text(encoding="utf-8").splitlines() != expected_raw:
            raise ValueError(f"{split} raw pair list differs from JSONL")
        for row in rows:
            width, height = row["cropped_size_wh"]
            files = row["files"]
            if not all(name in files for name in ("lq_raw", "lq", "gt", "text_mask", "valid_mask", "ocr")):
                raise ValueError(f"missing required paired asset key: {row['sample_id']}")
            rasters = {}
            for name in (name for name in RASTER_NAMES if name in files):
                path = Path(files[name])
                if not path.is_file():
                    raise FileNotFoundError(str(path))
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.shape[:2] != (height, width):
                    raise ValueError(f"bad raster shape: {path}")
                if name in ("lq_raw", "lq", "gt") and (image.ndim != 3 or image.shape[2] != 3):
                    raise ValueError(f"non-BGR paired image: {path}")
                if name in ("lq_raw", "lq"):
                    rasters[name] = image
            fallback_reason = row["preprocess"]["fallback_reason"]
            fallback_counts[str(fallback_reason)] += 1
            if fallback_reason is not None and not np.array_equal(rasters["lq_raw"], rasters["lq"]):
                raise ValueError(f"fallback is not exact identity: {row['sample_id']}")
            if fallback_reason is None and not np.array_equal(rasters["lq_raw"], rasters["lq"]):
                active_changed += 1
            document = json.loads(Path(files["ocr"]).read_text(encoding="utf-8"))
            annotations = document["ocr"]["annotations"]
            if document["ocr"]["annotation_count"] != 1 or len(annotations) != 1:
                raise ValueError(f"expected one OCR line: {row['sample_id']}")
            left, top, right, bottom = row["ocr_crop_xyxy"]
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError(f"OCR bbox outside crop: {row['sample_id']}")
            if len(annotations[0].get("bbox", [])) != 4:
                raise ValueError(f"missing OCR annotation bbox: {row['sample_id']}")
        report["counts"][split] = len(rows)
    if parent_sets["train"] & parent_sets["val"]:
        raise ValueError("train/val parent sample leakage")
    if identity_sets["train"] & identity_sets["val"]:
        raise ValueError("train/val source identity leakage")
    report["fallback_counts"] = dict(fallback_counts)
    report["active_changed_lines"] = active_changed
    report["excluded_source_images"] = sum(1 for line in (manifest_root / "review_excluded.jsonl").open(encoding="utf-8") if line.strip())
    if report["excluded_source_images"] != summary["excluded_source_images"]:
        raise ValueError("excluded source count differs from summary")
    return report


def smoke_training_reader(root: Path, expected_train_lines: int) -> dict:
    """@luna-comment
    Input: Completed paired dataset root and expected train line count.
    Output: Dataset-reader path and tensor contract evidence.
    Key Steps: Expand crop-local OCR, sample one line, and check prepared LQ reaches the reader.
    Key Params: OCR crop and jitter match the materialized maximum envelope; no kernel triplet is requested.
    State/Dependencies: Reads dataset files through BasicTextLineRemote58Dataset.
    """
    from realesrgan.data.basic_text_line_remote58_dataset import BasicTextLineRemote58Dataset

    dataset = BasicTextLineRemote58Dataset({
        "split": "train", "sources": [{"root": str(root), "manifest": "manifests/train.jsonl",
                                           "split": "train", "weight": 1.0}],
        "canvas_hw": [96, 480], "resize_mode": "height_crop_pad",
        "ocr_line_crop": True, "text_line_crop": False,
        "text_line_vertical_context": 0.12, "text_line_horizontal_context": 0.03,
        "text_line_boundary_jitter": 0.05,
    })
    if len(dataset) != expected_train_lines:
        raise ValueError(f"paired loader expanded {len(dataset)} lines instead of {expected_train_lines}")
    sample = dataset[0]
    if "/LQ_scurve/" not in sample["lq_path"] or "kernel1" in sample:
        raise ValueError("paired loader did not select the prepared LQ route")
    if tuple(sample["lq_rgb"].shape) != (3, 96, 480) or tuple(sample["gt_rgb"].shape) != (3, 96, 480):
        raise ValueError("paired loader did not produce matching 96x480 RGB tensors")
    return {"train_length": len(dataset), "lq_path": sample["lq_path"],
            "lq_rgb_shape": list(sample["lq_rgb"].shape), "kernel1_present": False}


def main() -> int:
    """@luna-comment
    Input: Materialized dataset root on the command line.
    Output: Verification report JSON and exit status.
    Key Steps: Call the read-only verifier and save its result in manifests.
    Key Params: Fail fast on a missing, mismatched, or unreadable line asset.
    State/Dependencies: Writes one verification JSON only after all checks pass.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--smoke-loader", action="store_true")
    args = parser.parse_args()
    result = verify(args.root)
    if args.smoke_loader:
        result["paired_loader_smoke"] = smoke_training_reader(args.root, result["counts"]["train"])
    (args.root / "manifests" / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
