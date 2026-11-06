"""Verify full border-rail OCR revisions against immutable original arms."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.materialize_lenovo_mild_scurve_lines import _load_jsonl
from tools.materialize_lenovo_v5_v6_ocr_matrix import ARMS
from tools.text_lq_scurve_mild import _border_distance


def verify(root: Path, expected_train: int, expected_val: int) -> dict:
    """@luna-comment
    Input: Verified original matrix plus completed four-arm border-rail revision.
    Output: Full pairing, shape and exact-interior verification report.
    Key Steps: Check manifest identity/geometry, decode every revised raster, compare all pixels outside six-pixel border.
    Key Params: Twelve bounded workers cover 5,510 train and 1,028 val rows per arm.
    State/Dependencies: Reads both arm generations; writes only borderrail_verification.json.
    """
    work = []
    counts = {}
    for arm in ARMS:
        revised = f"{arm}_borderrail"
        for split, expected in (("train", expected_train), ("val", expected_val)):
            original_rows = _load_jsonl(root / arm / "manifests" / f"{split}.jsonl")
            revised_rows = _load_jsonl(root / revised / "manifests" / f"{split}.jsonl")
            if len(original_rows) != expected or len(revised_rows) != expected:
                raise ValueError(f"unexpected count in {revised}/{split}")
            for old, new in zip(original_rows, revised_rows):
                if (old["sample_id"] != new["sample_id"] or old["ocr_crop_xyxy"] != new["ocr_crop_xyxy"]
                        or old["line_envelope_xyxy_in_parent"] != new["line_envelope_xyxy_in_parent"]):
                    raise ValueError(f"pair or geometry mismatch in {revised}/{split}")
                if new["preprocess"].get("arm") != revised or new["preprocess"].get("border_rail_revision") is not True:
                    raise ValueError(f"missing revision telemetry in {revised}/{new['sample_id']}")
                for shared in ("ocr", "lq_raw", "text_mask", "valid_mask", "formula_mask", "lq_foreground_mask", "fit_support_mask"):
                    if old["files"].get(shared) != new["files"].get(shared):
                        raise ValueError(f"shared asset changed in {revised}/{new['sample_id']}")
                for kind in (("lq", "gt") if arm.startswith("gtfit") else ("lq",)):
                    work.append((root / arm / old["files"][kind],
                                 root / revised / new["files"][kind], tuple(reversed(new["cropped_size_wh"])),
                                 revised, split, kind))
                if arm in ("v5", "v6"):
                    expected_local = f"{split}/GT/{new['sample_id']}.png"
                    if old["files"]["gt"] != expected_local or new["files"]["gt"] != expected_local:
                        raise ValueError(f"unexpected direct GT path in {revised}/{new['sample_id']}")
                    work.append((root / arm / old["files"]["gt"],
                                 root / revised / new["files"]["gt"], tuple(reversed(new["cropped_size_wh"])),
                                 revised, split, "gt_direct"))
            counts[f"{revised}_{split}"] = len(revised_rows)

    def check_pair(item: tuple) -> tuple[int, str, str, str]:
        """@luna-comment
        Input: One original/revised raster path pair and expected dimensions.
        Output: Changed border-pixel count plus arm, split and asset kind.
        Key Steps: Decode both, assert shape and exact interior identity, count border changes.
        Key Params: Border width is the frozen six-pixel identity guard.
        State/Dependencies: Read-only OpenCV I/O; no shared mutable arrays.
        """
        old_path, new_path, shape, arm, split, kind = item
        old = cv2.imread(str(old_path), cv2.IMREAD_UNCHANGED)
        new = cv2.imread(str(new_path), cv2.IMREAD_UNCHANGED)
        if old is None or new is None or old.shape != new.shape or old.shape[:2] != shape:
            raise ValueError(f"missing or mismatched raster: {new_path}")
        interior = _border_distance(shape) >= 6
        if not np.array_equal(old[interior], new[interior]):
            raise ValueError(f"interior pixels changed: {new_path}")
        changed = int(np.count_nonzero(np.any(old != new, axis=2)))
        if kind == "gt_direct" and changed:
            raise ValueError(f"direct GT pixels changed: {new_path}")
        return changed, arm, split, kind

    changed = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for index, (pixels, arm, split, kind) in enumerate(pool.map(check_pair, work), start=1):
            key = f"{arm}_{split}_{kind}"
            changed[key] = changed.get(key, 0) + pixels
            if index % 5000 == 0:
                print(json.dumps({"verified_raster_pairs": index, "total": len(work)}), flush=True)
    report = {"schema": "lenovo_border_rail_verify_v1", "passed": True,
              "counts": counts, "verified_raster_pairs": len(work), "changed_border_pixels": changed}
    (root / "borderrail_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """@luna-comment
    Input: Matrix root and exact train/val line-count gates.
    Output: Printed verification result and process status.
    Key Steps: Parse command line and run complete original-vs-revision comparison.
    Key Params: Full 260926 gates are 5,510 train and 1,028 val.
    State/Dependencies: Writes one verification report under matrix root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.train_count, args.val_count), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
