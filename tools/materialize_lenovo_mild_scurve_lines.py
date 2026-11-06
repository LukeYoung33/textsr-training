"""Materialize maximum-jitter OCR line envelopes and frozen mild LQ inputs."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import cv2

from tools.text_lq_scurve_mild import DEFAULT_CONFIG, preprocess_text_lq_bgr


VERTICAL_CONTEXT = 0.12
HORIZONTAL_CONTEXT = 0.03
BOUNDARY_JITTER = 0.05
MASK_NAMES = ("text_mask", "valid_mask", "lq_foreground_mask", "fit_support_mask")


def _load_jsonl(path: Path) -> list[dict]:
    """@luna-comment
    Input: One JSONL source manifest path.
    Output: Parsed records in source order.
    Key Steps: Parse every nonblank line and reject duplicate sample IDs.
    Key Params: Source order is preserved for deterministic line identifiers.
    State/Dependencies: Reads only the specified manifest.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample IDs in {path}")
    return rows


def _asset_path(source: Path, value: str) -> Path:
    """@luna-comment
    Input: Manifest directory and an absolute or dataset-root-relative asset path.
    Output: Resolved source asset path without changing the manifest.
    Key Steps: Resolve relative assets against the manifest directory's dataset root.
    Key Params: A `manifests/` directory is one level below the source dataset root.
    State/Dependencies: Pure Path calculation.
    """
    path = Path(value)
    root = source.parent if source.name == "manifests" else source
    return path if path.is_absolute() else root / path


def _box(annotation: dict, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """@luna-comment
    Input: One OCR annotation and source image height/width.
    Output: Clipped integer OCR line box or None for an invalid line.
    Key Steps: Use training-reader floor/ceil and two-pixel minimum semantics.
    Key Params: bbox is xyxy in the already valid-cropped source frame.
    State/Dependencies: Pure coordinate calculation.
    """
    values = annotation.get("bbox")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        left, top, right, bottom = map(float, values)
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (left, top, right, bottom))):
        return None
    height, width = shape
    box = (max(0, math.floor(left)), max(0, math.floor(top)), min(width, math.ceil(right)), min(height, math.ceil(bottom)))
    return box if box[2] - box[0] >= 2 and box[3] - box[1] >= 2 else None


def _envelope(box: tuple[int, int, int, int], shape: tuple[int, int], split: str) -> tuple[int, int, int, int]:
    """@luna-comment
    Input: OCR xyxy box, source shape, and train/val split.
    Output: Largest source crop covering the training reader's context and boundary jitter.
    Key Steps: Expand each edge by context plus train-only maximal jitter, then clip.
    Key Params: Context 0.12/0.03 and jitter 0.05 match the configured training crop.
    State/Dependencies: Pure coordinate calculation matching build_ocr_line_crop_cache._envelope.
    """
    left, top, right, bottom = box
    height, width = shape
    line_height, line_width = bottom - top, right - left
    pad_y = max(2, round(line_height * VERTICAL_CONTEXT))
    pad_x = max(2, round(line_width * HORIZONTAL_CONTEXT))
    jitter_y = round(line_height * BOUNDARY_JITTER) if split == "train" else 0
    jitter_x = round(line_width * BOUNDARY_JITTER) if split == "train" else 0
    return (max(0, left - pad_x - jitter_x), max(0, top - pad_y - jitter_y),
            min(width, right + pad_x + jitter_x), min(height, bottom + pad_y + jitter_y))


def _local_annotation(annotation: dict, left: int, top: int) -> dict:
    """@luna-comment
    Input: Original OCR line annotation and envelope origin.
    Output: Independent copy with its geometry in crop-local coordinates.
    Key Steps: Shift bbox, polygon points, baseline, and per-character font boxes.
    Key Params: Coordinates remain float precision; no clipping discards OCR context.
    State/Dependencies: Uses only a deep copy of the annotation.
    """
    result = copy.deepcopy(annotation)
    if isinstance(result.get("bbox"), list):
        result["bbox"] = [result["bbox"][0] - left, result["bbox"][1] - top,
                          result["bbox"][2] - left, result["bbox"][3] - top]
    if isinstance(result.get("points"), list):
        result["points"] = [[point[0] - left, point[1] - top] for point in result["points"]]
    if isinstance(result.get("baseline"), list) and len(result["baseline"]) == 4:
        x1, y1, x2, y2 = result["baseline"]
        result["baseline"] = [x1 - left, y1 - top, x2 - left, y2 - top]
    for run in result.get("font_runs", []):
        if isinstance(run.get("bbox"), list) and len(run["bbox"]) == 4:
            x1, y1, x2, y2 = run["bbox"]
            run["bbox"] = [x1 - left, y1 - top, x2 - left, y2 - top]
    return result


def _write_image(path: Path, array, params: list[int] | None = None) -> None:
    """@luna-comment
    Input: Output path, image array, and optional encoder parameters.
    Output: A fully encoded image or an explicit write failure.
    Key Steps: Create the parent directory and encode through OpenCV.
    Key Params: All paired image and mask assets use lossless PNG encoding.
    State/Dependencies: Writes only under the chosen dataset output root.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array, params or []):
        raise OSError(f"cannot write {path}")


def materialize(source: Path, output: Path, dry_run: bool = False, limit: int | None = None) -> dict:
    """@luna-comment
    Input: Source manifest directory, new output directory, and optional probe limit.
    Output: Deterministic line-level manifest and summary, or a read-only preflight summary.
    Key Steps: Expand train/val OCR lines, crop paired assets, preprocess only LQ, and verify outputs.
    Key Params: Uses frozen crop constants and text_lq_scurve_mild_v1.
    State/Dependencies: Reads source assets; writes only to a new output root on a full run.
    """
    if output.exists() and not dry_run and not (output.is_symlink() and not any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    fallbacks = Counter()
    estimated_rgb_bytes = 0
    estimated_encoded_bytes = Counter()
    output_rows: dict[str, list[dict]] = {"train": [], "val": []}
    for split in ("train", "val"):
        for source_row in _load_jsonl(source / f"{split}.jsonl"):
            files = {name: _asset_path(source, value) for name, value in source_row["files"].items() if value}
            masks = tuple(name for name in MASK_NAMES if name in files)
            ocr = json.loads(files["ocr"].read_text(encoding="utf-8"))
            annotations = ocr.get("ocr", {}).get("annotations", [])
            if not isinstance(annotations, list):
                raise ValueError(f"invalid OCR annotation list: {files['ocr']}")
            source_width, source_height = source_row["cropped_size_wh"]
            planned = []
            for ordinal, annotation in enumerate(annotations):
                if not isinstance(annotation, dict) or annotation.get("ignore") or annotation.get("visible") is False:
                    counts["ignored_annotations"] += 1
                    continue
                box = _box(annotation, (source_height, source_width))
                if box is None:
                    counts["invalid_annotations"] += 1
                    continue
                envelope = _envelope(box, (source_height, source_width), split)
                planned.append((ordinal, annotation, box, envelope))
                estimated_rgb_bytes += (envelope[2] - envelope[0]) * (envelope[3] - envelope[1]) * 3
            counts[f"{split}_source_images"] += 1
            counts[f"{split}_lines"] += len(planned)
            if counts[f"{split}_source_images"] % 25 == 0:
                print(json.dumps({"progress_split": split, "source_images": counts[f"{split}_source_images"],
                                  "lines": counts[f"{split}_lines"]}), flush=True)
            if dry_run:
                if limit:
                    images = {name: cv2.imread(str(files[name]), cv2.IMREAD_COLOR if name in ("lq", "gt") else cv2.IMREAD_UNCHANGED)
                              for name in ("lq", "gt", *masks)}
                    for name, image in images.items():
                        if image is None or image.shape[:2] != (source_height, source_width):
                            raise ValueError(f"missing or mismatched {name}: {files[name]}")
                    for _, _, _, envelope in planned:
                        left, top, right, bottom = envelope
                        raw = images["lq"][top:bottom, left:right]
                        processed, telemetry = preprocess_text_lq_bgr(raw, return_telemetry=True)
                        fallbacks[str(telemetry.get("fallback_reason"))] += 1
                        for name, image in (("lq_raw", raw), ("lq", processed),
                                            ("gt", images["gt"][top:bottom, left:right]),
                                            *((key, images[key][top:bottom, left:right]) for key in masks)):
                            ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                            if not ok:
                                raise OSError(f"cannot estimate PNG size for {name}")
                            estimated_encoded_bytes[name] += int(encoded.nbytes)
                if limit and counts["train_source_images"] + counts["val_source_images"] >= limit:
                    break
                continue
            images = {name: cv2.imread(str(files[name]), cv2.IMREAD_COLOR if name in ("lq", "gt") else cv2.IMREAD_UNCHANGED)
                      for name in ("lq", "gt", *masks)}
            for name, image in images.items():
                if image is None or image.shape[:2] != (source_height, source_width):
                    raise ValueError(f"missing or mismatched {name}: {files[name]}")
            for ordinal, annotation, box, envelope in planned:
                left, top, right, bottom = envelope
                sample_id = f"{source_row['sample_id']}__line{ordinal:03d}"
                asset_paths = {
                    "lq_raw": output / split / "LQ_raw" / f"{sample_id}.png",
                    "lq": output / split / "LQ_scurve" / f"{sample_id}.png",
                    "gt": output / split / "GT" / f"{sample_id}.png",
                    **{name: output / split / name / f"{sample_id}.png" for name in masks},
                    "ocr": output / split / "OCR" / f"{sample_id}.json",
                }
                raw = images["lq"][top:bottom, left:right]
                processed, telemetry = preprocess_text_lq_bgr(raw, return_telemetry=True)
                fallbacks[str(telemetry.get("fallback_reason"))] += 1
                _write_image(asset_paths["lq_raw"], raw, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                _write_image(asset_paths["lq"], processed, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                _write_image(asset_paths["gt"], images["gt"][top:bottom, left:right], [cv2.IMWRITE_PNG_COMPRESSION, 6])
                for name in masks:
                    _write_image(asset_paths[name], images[name][top:bottom, left:right], [cv2.IMWRITE_PNG_COMPRESSION, 6])
                local_ocr = {"schema_version": "line_envelope_v1", "ocr": {"image": sample_id,
                             "annotations": [_local_annotation(annotation, left, top)], "annotation_count": 1},
                             "source_ocr": str(files["ocr"]), "source_envelope_xyxy": list(envelope)}
                asset_paths["ocr"].parent.mkdir(parents=True, exist_ok=True)
                asset_paths["ocr"].write_text(json.dumps(local_ocr, ensure_ascii=False), encoding="utf-8")
                record = copy.deepcopy(source_row)
                record["source_output_sha256"] = record.pop("output_sha256", None)
                record["source_release_root"] = record.pop("release_root", None)
                record["release_root"] = str(output)
                record["parent_sample_id"] = source_row["sample_id"]
                record["sample_id"] = sample_id
                record["parent_crop_xywh"] = source_row["crop_xywh"]
                record["crop_xywh"] = [left, top, right - left, bottom - top]
                record["cropped_size_wh"] = [right - left, bottom - top]
                record["ocr_crop_xyxy"] = [box[0] - left, box[1] - top, box[2] - left, box[3] - top]
                record["line_envelope_xyxy_in_parent"] = list(envelope)
                record["line_ordinal"] = ordinal
                record["files"] = {name: str(path) for name, path in asset_paths.items()}
                record["preprocess"] = {"version": DEFAULT_CONFIG.version, "fallback_reason": telemetry.get("fallback_reason")}
                output_rows[split].append(record)
                if limit and len(output_rows["train"]) + len(output_rows["val"]) >= limit:
                    break
            if limit and len(output_rows["train"]) + len(output_rows["val"]) >= limit:
                break
        if limit and (counts["train_source_images"] + counts["val_source_images"] >= limit if dry_run
                      else len(output_rows["train"]) + len(output_rows["val"]) >= limit):
            break
    summary = {"schema": "lenovo_line_envelope_scurve_v1", "source_manifests": str(source),
               "output_root": str(output), "crop": {"vertical_context": VERTICAL_CONTEXT,
               "horizontal_context": HORIZONTAL_CONTEXT, "boundary_jitter_train": BOUNDARY_JITTER,
               "boundary_jitter_val": 0.0}, "preprocess_version": DEFAULT_CONFIG.version,
               "counts": dict(counts), "fallback_counts": dict(fallbacks),
               "estimated_encoded_sample_bytes": dict(estimated_encoded_bytes),
               "estimated_uncompressed_one_rgb_copy_bytes": estimated_rgb_bytes,
               "complete": not dry_run and limit is None}
    if not dry_run:
        manifest_dir = output / "manifests"
        manifest_dir.mkdir()
        for split, rows in output_rows.items():
            (manifest_dir / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            (manifest_dir / f"{split}_pairs.txt").write_text(
                "".join(f"{row['files']['lq']} {row['files']['gt']}\n" for row in rows), encoding="utf-8")
            (manifest_dir / f"{split}_raw_pairs.txt").write_text(
                "".join(f"{row['files']['lq_raw']} {row['files']['gt']}\n" for row in rows), encoding="utf-8")
        excluded_path = source / "review_excluded.jsonl"
        excluded = _load_jsonl(excluded_path) if excluded_path.is_file() else []
        (manifest_dir / "review_excluded.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded), encoding="utf-8")
        summary["excluded_source_images"] = len(excluded)
        (manifest_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    """@luna-comment
    Input: Source manifest root, new output root, and optional dry-run or sample limit.
    Output: JSON summary and process exit code.
    Key Steps: Parse arguments, run materialization, and print counts.
    Key Params: Full run writes only to a previously nonexistent destination.
    State/Dependencies: Filesystem effects are limited to the specified output root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifests", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_manifests, args.output_root, args.dry_run, args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
