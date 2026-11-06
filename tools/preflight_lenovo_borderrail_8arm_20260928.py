"""Check completed border-rail matrix manifests and sampled paired raster assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from make_lenovo_borderrail_8arm_configs_20260928 import ARMS, DATASETS


def check_dataset(name: str, root: Path, train_count: int, val_count: int, sample_count: int) -> dict:
    reports = {}
    for filename in ("verification.json", "borderrail_verification.json", "direct_local_gt_verification.json"):
        report = json.loads((root / filename).read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise ValueError(f"dataset verification did not pass: {root / filename}")
        reports[filename] = report["schema"]
    arm_ids: dict[str, dict[str, set[str]]] = {}
    checked = {}
    for arm in ARMS:
        arm_ids[arm] = {}
        for split, expected in (("train", train_count), ("val", val_count)):
            manifest = root / arm / "manifests" / f"{split}.jsonl"
            ids = set()
            decoded = 0
            stride = max(1, expected // sample_count)
            with manifest.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    line_id = str(row["sample_id"])
                    if line_id in ids:
                        raise ValueError(f"duplicate sample: {manifest}:{line_id}")
                    ids.add(line_id)
                    width, height = row["cropped_size_wh"]
                    x0, y0, x1, y1 = row["ocr_crop_xyxy"]
                    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                        raise ValueError(f"invalid OCR crop: {manifest}:{line_id}")
                    if row["preprocess"]["arm"] != arm or not row["preprocess"].get("border_rail_revision"):
                        raise ValueError(f"wrong arm or missing border-rail revision: {manifest}:{line_id}")
                    if index % stride:
                        continue
                    for key in ("lq", "gt", "text_mask", "valid_mask", "ocr"):
                        path = root / arm / row["files"][key]
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        if key == "ocr":
                            json.loads(path.read_text(encoding="utf-8"))
                        else:
                            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                            if image is None or image.shape[:2] != (height, width):
                                raise ValueError(f"raster shape mismatch: {path}")
                    decoded += 1
            if len(ids) != expected:
                raise ValueError(f"unexpected count: {manifest}: {len(ids)} != {expected}")
            arm_ids[arm][split] = ids
            checked[f"{arm}_{split}"] = {"rows": len(ids), "sampled_pairs_decoded": decoded}
    for split in ("train", "val"):
        reference = arm_ids[ARMS[0]][split]
        if any(arm_ids[arm][split] != reference for arm in ARMS[1:]):
            raise ValueError(f"cross-arm sample IDs differ: {name}/{split}")
    return {"root": str(root), "reports": reports, "arms": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sample-count", type=int, default=32)
    args = parser.parse_args()
    result = {name: check_dataset(name, Path(root), train, val, args.sample_count)
              for name, root, train, val in DATASETS}
    output = {"passed": True, "datasets": result}
    encoded = json.dumps(output, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
