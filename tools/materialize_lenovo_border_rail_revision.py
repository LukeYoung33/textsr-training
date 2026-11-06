"""Build four border-rail revision arms from a verified OCR matrix without refitting interiors."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.materialize_lenovo_mild_scurve_lines import _load_jsonl, _write_image
from tools.materialize_lenovo_v5_v6_ocr_matrix import ARMS, _direct_border_rail, _gtfit_border_curve
from tools.text_gt_fitted_v5_v6_pair import _fit_curve


def _read(path: Path):
    """@luna-comment
    Input: One existing materialized PNG path.
    Output: Decoded unchanged BGR image or explicit failure.
    Key Steps: Load through OpenCV and reject missing/corrupt assets.
    Key Params: All inputs are paired HWC uint8 BGR line rasters.
    State/Dependencies: Read-only filesystem operation.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"missing or undecodable image: {path}")
    return image


def materialize(root: Path, shard_index: int = 0, shard_count: int = 1,
                splits: tuple[str, ...] = ("train", "val"), limit: int | None = None) -> dict:
    """@luna-comment
    Input: Verified matrix root, disjoint shard assignment and optional pilot limit.
    Output: Four sibling border-rail arm shards with original interiors and revised border rasters.
    Key Steps: Reuse raw/processed line assets and telemetry, fit GT lookup once, write only new rasters/manifests.
    Key Params: Six-pixel border repair; source row index modulo shard count determines ownership.
    State/Dependencies: Reads immutable old arms/base; writes only new *_borderrail arm roots.
    """
    if not (root / "verification.json").is_file():
        raise FileNotFoundError("verified original matrix root required")
    counts = {}
    for split in splits:
        input_rows = {arm: _load_jsonl(root / arm / "manifests" / f"{split}.jsonl") for arm in ARMS}
        ids = [row["sample_id"] for row in input_rows["v5"]]
        if any([row["sample_id"] for row in input_rows[arm]] != ids for arm in ARMS):
            raise ValueError(f"source arm IDs differ in {split}")
        output_rows = {arm: [] for arm in ARMS}
        for index, line_id in enumerate(ids):
            if index % shard_count != shard_index:
                continue
            base_row = input_rows["v5"][index]
            raw_lq = _read(root / "v5" / base_row["files"]["lq_raw"])
            raw_gt = _read(root / "base" / split / "GT_raw" / f"{line_id}.png")
            lut = _fit_curve(raw_gt)[0]
            for arm in ARMS:
                source_row = input_rows[arm][index]
                source_root = root / arm
                processed_lq = _read(source_root / source_row["files"]["lq"])
                new_arm = f"{arm}_borderrail"
                target_root = root / new_arm
                target_lq = target_root / split / "LQ" / f"{line_id}.png"
                info = copy.deepcopy(source_row["preprocess"])
                if arm in ("v5", "v6"):
                    revised_lq, changed_lq = _direct_border_rail(raw_lq, processed_lq, info)
                    info["border_rail_lq_changed_pixels"] = changed_lq
                    target_gt = target_root / split / "GT" / f"{line_id}.png"
                    target_gt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / "base" / split / "GT_raw" / f"{line_id}.png", target_gt)
                    info["gt_storage"] = "arm_local_raw_copy"
                else:
                    processed_gt = _read(source_root / source_row["files"]["gt"])
                    revised_lq, changed_lq = _gtfit_border_curve(raw_lq, processed_lq, lut)
                    revised_gt, changed_gt = _gtfit_border_curve(raw_gt, processed_gt, lut)
                    info["border_rail_lq_changed_pixels"] = changed_lq
                    info["border_rail_gt_changed_pixels"] = changed_gt
                    target_gt = target_root / split / "GT" / f"{line_id}.png"
                _write_image(target_lq, revised_lq, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if arm not in ("v5", "v6"):
                    _write_image(target_gt, revised_gt, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                record = copy.deepcopy(source_row)
                record["files"]["lq"] = target_lq.relative_to(target_root).as_posix()
                if target_gt is not None:
                    record["files"]["gt"] = target_gt.relative_to(target_root).as_posix()
                record["preprocess"] = {**info, "arm": new_arm, "border_rail_revision": True}
                output_rows[arm].append(record)
            if len(output_rows["v5"]) % 250 == 0:
                print(json.dumps({"split": split, "shard": shard_index,
                                  "written": len(output_rows["v5"])}), flush=True)
            if limit and len(output_rows["v5"]) >= limit:
                break
        suffix = "" if shard_count == 1 else f".shard{shard_index:02d}of{shard_count:02d}"
        for arm in ARMS:
            manifest_root = root / f"{arm}_borderrail" / "manifests"
            manifest_root.mkdir(parents=True, exist_ok=True)
            (manifest_root / f"{split}{suffix}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows[arm]), encoding="utf-8")
            counts[f"{arm}_{split}"] = len(output_rows[arm])
    summary = {"schema": "lenovo_border_rail_revision_v1", "source_matrix": str(root),
               "shard_index": shard_index, "shard_count": shard_count, "counts": counts}
    summary_path = root / ("borderrail_summary.json" if shard_count == 1 else
                           f"borderrail_summary.shard{shard_index:02d}of{shard_count:02d}.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def merge_shards(root: Path, shard_count: int, splits: tuple[str, ...]) -> dict:
    """@luna-comment
    Input: All completed border-rail shard manifests under one verified matrix root.
    Output: Canonical revision arm train/val manifests and pair lists in source order.
    Key Steps: Require every shard, sort by source line index, reject duplicate IDs and publish counts.
    Key Params: Source index is preserved by original arm manifests.
    State/Dependencies: Writes only *_borderrail canonical manifests and revision summary.
    """
    counts = {}
    order = {split: {row["sample_id"]: i for i, row in enumerate(
        _load_jsonl(root / "v5" / "manifests" / f"{split}.jsonl"))} for split in splits}
    for arm in ARMS:
        manifest_root = root / f"{arm}_borderrail" / "manifests"
        for split in splits:
            rows = []
            for shard in range(shard_count):
                rows.extend(_load_jsonl(manifest_root / f"{split}.shard{shard:02d}of{shard_count:02d}.jsonl"))
            rows.sort(key=lambda row: order[split][row["sample_id"]])
            ids = [row["sample_id"] for row in rows]
            if len(ids) != len(set(ids)) or ids != list(order[split]):
                raise ValueError(f"missing or duplicate revision IDs in {arm}/{split}")
            (manifest_root / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            (manifest_root / f"{split}_pairs.txt").write_text(
                "".join(f"{row['files']['lq']} {row['files']['gt']}\n" for row in rows), encoding="utf-8")
            counts[f"{arm}_borderrail_{split}"] = len(rows)
    (root / "borderrail_summary.json").write_text(json.dumps({"counts": counts, "shards": shard_count}, indent=2), encoding="utf-8")
    return counts


def main() -> int:
    """@luna-comment
    Input: Verified matrix root, shard options, optional split/limit and merge selector.
    Output: JSON revision summary and process status.
    Key Steps: Validate shard arguments and dispatch materialization or canonical merge.
    Key Params: Default covers train and val; eight shards fit the 260926 batch.
    State/Dependencies: Writes only sibling revision arms beneath the matrix root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.shard_count <= 64 and 0 <= args.shard_index < args.shard_count):
        parser.error("invalid shard index/count")
    splits = ("train", "val") if args.split == "both" else (args.split,)
    result = (merge_shards(args.root, args.shard_count, splits) if args.merge else
              materialize(args.root, args.shard_index, args.shard_count, splits, args.limit))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
