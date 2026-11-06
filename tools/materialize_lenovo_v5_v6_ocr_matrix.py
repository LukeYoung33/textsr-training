"""Materialize OCR-expanded direct and GT-fitted V5/V6 text-line datasets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.materialize_lenovo_mild_scurve_lines import (
    MASK_NAMES as BASE_MASK_NAMES, _asset_path, _box, _load_jsonl, _local_annotation, _write_image,
)
from tools.text_gt_fitted_v5_v6_pair import _fit_curve, preprocess_gt_fitted_pair
from tools.text_lq_scurve_mild import (
    V5_CONFIG, V6_CONFIG, _apply_luma_delta, _border_distance, _luma_bgr, preprocess_text_lq_bgr,
)


ARMS = ("v5", "v6", "gtfit_v5", "gtfit_v6")
MASK_NAMES = (*BASE_MASK_NAMES, "formula_mask")


def expanded_envelope(annotation: dict, box: tuple[int, int, int, int],
                      shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], dict]:
    """@luna-comment
    Input: OCR polygon/box, reader-compatible integer box and parent image shape.
    Output: Clipped maximal processing envelope and geometry telemetry.
    Key Steps: Union polygon AABB with bbox, add reader context, inverse-warp/interpolation allowance and six-pixel processing border.
    Key Params: 12%/3% crop context; 1.5% corner jitter; extra ceil(3% maximal short side + 7).
    State/Dependencies: Pure geometry; does not read or write image data.
    """
    height, width = shape
    left, top, right, bottom = box
    line_h, line_w = bottom - top, right - left
    pad_y, pad_x = max(2, round(line_h * .12)), max(2, round(line_w * .03))
    short_side = min(line_h + 2 * pad_y, line_w + 2 * pad_x)
    extra = math.ceil(.03 * short_side + 7)
    points = annotation.get("points")
    if isinstance(points, list) and points:
        valid = [(float(p[0]), float(p[1])) for p in points if isinstance(p, (tuple, list)) and len(p) >= 2
                 and math.isfinite(float(p[0])) and math.isfinite(float(p[1]))]
        if valid:
            left = min(left, math.floor(min(p[0] for p in valid)))
            top = min(top, math.floor(min(p[1] for p in valid)))
            right = max(right, math.ceil(max(p[0] for p in valid)))
            bottom = max(bottom, math.ceil(max(p[1] for p in valid)))
    envelope = (max(0, left - pad_x - extra), max(0, top - pad_y - extra),
                min(width, right + pad_x + extra), min(height, bottom + pad_y + extra))
    return envelope, {"context_xy": [pad_x, pad_y], "warp_border_extra_px": extra,
                      "clipped_to_parent": envelope != (left - pad_x - extra, top - pad_y - extra,
                                                        right + pad_x + extra, bottom + pad_y + extra)}


def _relative(path: Path, arm_root: Path) -> str:
    """@luna-comment
    Input: Sibling dataset asset path and one arm root.
    Output: Portable path relative to the arm root.
    Key Steps: Use one parent reference for shared base assets.
    Key Params: Arms and base are siblings within one output root.
    State/Dependencies: Pure path calculation.
    """
    base_root = arm_root.parent / "base"
    return ((Path("..") / "base" / path.relative_to(base_root)).as_posix()
            if path.is_relative_to(base_root) else path.relative_to(arm_root).as_posix())


def _direct_border_rail(raw: np.ndarray, processed: np.ndarray, info: dict) -> tuple[np.ndarray, int]:
    """@luna-comment
    Input: Raw LQ, frozen V5/V6 output, and its fitted background-side rail telemetry.
    Output: Border-corrected output and changed-pixel count; the interior remains byte-identical.
    Key Steps: Clamp only background-class overshoot in the six-pixel identity border to the fitted rail.
    Key Params: Three-level threshold margin protects uncertain text-side transition pixels.
    State/Dependencies: Pure NumPy/OpenCV transform; no curve refit or filesystem I/O.
    """
    if info.get("fallback_reason") is not None:
        return processed, 0
    polarity = -1 if info.get("polarity") == "light_text_dark_background" else 1
    source_y = _luma_bgr(raw)
    normalized = source_y if polarity == 1 else 255.0 - source_y
    border = _border_distance(source_y.shape) < 6
    background = normalized >= float(info["threshold_stabilized"]) + 3.0
    delta = polarity * np.minimum(float(info["background_target"]) - normalized, 0.0)
    delta[~(border & background)] = 0.0
    corrected = _apply_luma_delta(processed, delta)
    return corrected, int(np.count_nonzero(np.any(corrected != processed, axis=2)))


def _gtfit_border_curve(raw: np.ndarray, processed: np.ndarray, lut: np.ndarray | None) -> tuple[np.ndarray, int]:
    """@luna-comment
    Input: Raw paired image, GT-fitted tone-stage output, and the shared GT lookup table.
    Output: Same lookup table applied to the six-pixel border only, with changed-pixel count.
    Key Steps: Reuse GT-fitted Y mapping at border while leaving all interior pixels byte-identical.
    Key Params: Six-pixel border matches the frozen paired transform.
    State/Dependencies: Pure NumPy/OpenCV transform; lookup fitting is done once per GT pair.
    """
    if lut is None:
        return processed, 0
    source_y = _luma_bgr(raw).astype(np.uint8)
    delta = lut[source_y].astype(np.float32) - source_y.astype(np.float32)
    delta[_border_distance(source_y.shape) >= 6] = 0.0
    corrected = _apply_luma_delta(processed, delta)
    return corrected, int(np.count_nonzero(np.any(corrected != processed, axis=2)))


def _variants(lq, gt, border_rail: bool) -> dict:
    """@luna-comment
    Input: Aligned raw BGR line pair and optional six-pixel border-rail revision.
    Output: Four processed LQ/GT/telemetry triples at original dimensions.
    Key Steps: Call frozen transforms; optionally extend their fitted Y mapping only over identity borders.
    Key Params: False preserves the original materialization exactly.
    State/Dependencies: Pure OpenCV/NumPy plus frozen direct and GT-fitted transform calls.
    """
    result = {}
    for arm, config in (("v5", V5_CONFIG), ("v6", V6_CONFIG)):
        processed, info = preprocess_text_lq_bgr(lq, config, True)
        if border_rail:
            processed, changed = _direct_border_rail(lq, processed, info)
            info["border_rail_changed_pixels"] = changed
        result[arm] = (processed, {**info, "border_rail_revision": border_rail}, gt)
    lut = _fit_curve(gt)[0] if border_rail else None
    for arm in ("gtfit_v5", "gtfit_v6"):
        processed_lq, processed_gt, info = preprocess_gt_fitted_pair(
            lq, gt, chroma=arm.endswith("v6"))
        if border_rail:
            processed_lq, changed_lq = _gtfit_border_curve(lq, processed_lq, lut)
            processed_gt, changed_gt = _gtfit_border_curve(gt, processed_gt, lut)
            info["border_rail_lq_changed_pixels"] = changed_lq
            info["border_rail_gt_changed_pixels"] = changed_gt
        result[arm] = (processed_lq, {**info, "border_rail_revision": border_rail}, processed_gt)
    return result


def materialize(source: Path, output: Path, limit: int | None = None,
                dry_run: bool = False, splits: tuple[str, ...] = ("train", "val"),
                shard_index: int = 0, shard_count: int = 1, border_rail: bool = False) -> dict:
    """@luna-comment
    Input: Exact source manifest directory, new output root and optional pilot limit.
    Output: Four line-dataset arms plus shared raw assets and summary, or read-only counts.
    Key Steps: Load parents once, expand OCR lines, apply direct and GT-fitted transforms, write deterministic manifests.
    Key Params: Frozen V5/V6 APIs, GT-only fitted parameters, six-pixel protected border.
    State/Dependencies: Reads source; writes only beneath output; supports resume of fully written line assets.
    """
    if not (source / "train.jsonl").is_file() and not (source / "val.jsonl").is_file():
        raise FileNotFoundError(f"no source manifest at {source}")
    if "20260920" in str(source) and "ocrfix_v4_formula_groups" not in str(source):
        raise ValueError("260920 requires the designated ocrfix_v4_formula_groups source")
    counts = Counter()
    rows: dict[str, dict[str, list[dict]]] = {arm: {split: [] for split in splits} for arm in ARMS}
    for split in splits:
        manifest = source / f"{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for parent_index, parent in enumerate(_load_jsonl(manifest)):
            if parent_index % shard_count != shard_index:
                continue
            files = {k: _asset_path(source, v) for k, v in parent["files"].items() if v}
            document = json.loads(files["ocr"].read_text(encoding="utf-8"))
            annotations = document.get("ocr", {}).get("annotations", [])
            parent_w, parent_h = parent["cropped_size_wh"]
            planned = []
            for ordinal, ann in enumerate(annotations):
                if not isinstance(ann, dict) or ann.get("ignore") or ann.get("visible") is False:
                    continue
                box = _box(ann, (parent_h, parent_w))
                if box is None:
                    counts["invalid_ocr"] += 1
                    continue
                envelope, geometry = expanded_envelope(ann, box, (parent_h, parent_w))
                planned.append((ordinal, ann, box, envelope, geometry))
            counts[f"{split}_parents"] += 1
            counts[f"{split}_lines"] += len(planned)
            if dry_run:
                if limit and counts[f"{split}_lines"] >= limit:
                    break
                continue
            mask_names = [k for k in MASK_NAMES if k in files]
            decoded = {k: cv2.imread(str(files[k]), cv2.IMREAD_COLOR if k in ("lq", "gt") else cv2.IMREAD_UNCHANGED)
                       for k in ("lq", "gt", *mask_names)}
            if any(value is None or value.shape[:2] != (parent_h, parent_w) for value in decoded.values()):
                raise ValueError(f"missing or misaligned parent assets: {parent['sample_id']}")
            for ordinal, ann, box, envelope, geometry in planned:
                left, top, right, bottom = envelope
                line_id = f"{parent['sample_id']}__line{ordinal:03d}"
                base = output / "base" / split
                raw_paths = {"lq_raw": base / "LQ_raw" / f"{line_id}.png",
                             "gt_raw": base / "GT_raw" / f"{line_id}.png",
                             "ocr": base / "OCR" / f"{line_id}.json"}
                raw_paths.update({name: base / name / f"{line_id}.png" for name in mask_names})
                lq = decoded["lq"][top:bottom, left:right]
                gt = decoded["gt"][top:bottom, left:right]
                if not raw_paths["lq_raw"].is_file():
                    _write_image(raw_paths["lq_raw"], lq, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    _write_image(raw_paths["gt_raw"], gt, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    for name in mask_names:
                        _write_image(raw_paths[name], decoded[name][top:bottom, left:right], [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    raw_paths["ocr"].parent.mkdir(parents=True, exist_ok=True)
                    local = {"schema_version": "line_envelope_v2", "ocr": {"image": line_id,
                             "annotations": [_local_annotation(ann, left, top)], "annotation_count": 1},
                             "source_envelope_xyxy": list(envelope), "geometry": geometry}
                    raw_paths["ocr"].write_text(json.dumps(local, ensure_ascii=False), encoding="utf-8")
                variants = _variants(lq, gt, border_rail)
                for arm, (proc_lq, telemetry, proc_gt) in variants.items():
                    arm_root = output / arm
                    lq_path = arm_root / split / "LQ" / f"{line_id}.png"
                    gt_path = arm_root / split / "GT" / f"{line_id}.png"
                    _write_image(lq_path, proc_lq, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    if arm.startswith("gtfit"):
                        _write_image(gt_path, proc_gt, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    else:
                        gt_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(raw_paths["gt_raw"], gt_path)
                    record = copy.deepcopy(parent)
                    record["parent_sample_id"] = parent["sample_id"]
                    record["source_parent_index"] = parent_index
                    record["line_ordinal"] = ordinal
                    record["sample_id"] = line_id
                    record["cropped_size_wh"] = [right - left, bottom - top]
                    record["ocr_crop_xyxy"] = [box[0] - left, box[1] - top, box[2] - left, box[3] - top]
                    record["line_envelope_xyxy_in_parent"] = list(envelope)
                    record["geometry"] = geometry
                    record["files"] = {"lq": lq_path.relative_to(arm_root).as_posix(),
                                       "gt": _relative(gt_path, arm_root),
                                       "ocr": _relative(raw_paths["ocr"], arm_root),
                                       "lq_raw": _relative(raw_paths["lq_raw"], arm_root),
                                       **{k: _relative(raw_paths[k], arm_root) for k in mask_names}}
                    record["preprocess"] = {"arm": arm, **telemetry}
                    if arm in ("v5", "v6"):
                        record["preprocess"]["gt_storage"] = "arm_local_raw_copy"
                    rows[arm][split].append(record)
                if limit and len(rows["v5"][split]) >= limit:
                    break
            if counts[f"{split}_parents"] % 25 == 0:
                print(json.dumps({"split": split, "parents": counts[f"{split}_parents"],
                                  "lines_written": len(rows["v5"][split])}), flush=True)
            if limit and len(rows["v5"][split]) >= limit:
                break
    summary = {"schema": "lenovo_v5_v6_ocr_matrix_v1", "source": str(source),
               "output": str(output), "counts": dict(counts), "dry_run": dry_run, "limit": limit,
               "shard_index": shard_index, "shard_count": shard_count, "border_rail_revision": border_rail}
    if not dry_run:
        for arm in ARMS:
            manifests = output / arm / "manifests"
            manifests.mkdir(parents=True, exist_ok=True)
            for split, records in rows[arm].items():
                suffix = "" if shard_count == 1 else f".shard{shard_index:02d}of{shard_count:02d}"
                (manifests / f"{split}{suffix}.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
                (manifests / f"{split}{suffix}_pairs.txt").write_text(
                    "".join(f"{row['files']['lq']} {row['files']['gt']}\n" for row in records), encoding="utf-8")
        summary_name = "summary.json" if shard_count == 1 else f"summary.shard{shard_index:02d}of{shard_count:02d}.json"
        (output / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def merge_shards(output: Path, shard_count: int, splits: tuple[str, ...]) -> dict:
    """@luna-comment
    Input: Completed disjoint shard manifests and expected shard count.
    Output: Canonical source-ordered train/val manifests for every arm.
    Key Steps: Require every shard, reject duplicate IDs, sort by parent/line order and publish pair lists.
    Key Params: Shards partition source parent indices modulo shard count.
    State/Dependencies: Writes only canonical manifests beneath the selected output root.
    """
    result = {}
    for arm in ARMS:
        manifests = output / arm / "manifests"
        for split in splits:
            records = []
            for i in range(shard_count):
                records.extend(_load_jsonl(manifests / f"{split}.shard{i:02d}of{shard_count:02d}.jsonl"))
            for row in records:
                row["files"] = {name: path.replace("\\", "/") for name, path in row["files"].items()}
            records.sort(key=lambda row: (row["source_parent_index"], row["line_ordinal"]))
            ids = [row["sample_id"] for row in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate line ID in {arm}/{split}")
            (manifests / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
            (manifests / f"{split}_pairs.txt").write_text(
                "".join(f"{row['files']['lq']} {row['files']['gt']}\n" for row in records), encoding="utf-8")
            result[f"{arm}_{split}"] = len(records)
    (output / "summary.json").write_text(json.dumps({"merged_counts": result, "shards": shard_count}, indent=2), encoding="utf-8")
    return result


def main() -> int:
    """@luna-comment
    Input: Explicit source/output paths, optional split, limit, and preflight flag.
    Output: JSON summary and process exit status.
    Key Steps: Validate CLI arguments and dispatch the deterministic materializer.
    Key Params: Default split set is both train and val.
    State/Dependencies: Writes only beneath selected output unless dry-run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--border-rail", action="store_true")
    args = parser.parse_args()
    splits = ("train", "val") if args.split == "both" else (args.split,)
    if not (1 <= args.shard_count <= 64 and 0 <= args.shard_index < args.shard_count):
        parser.error("invalid shard index/count")
    result = (merge_shards(args.output, args.shard_count, splits) if args.merge else
              materialize(args.source, args.output, args.limit, args.dry_run, splits,
                          args.shard_index, args.shard_count, args.border_rail))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
