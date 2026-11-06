"""Resume OCR-line materialization across independent source-image workers."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from tools.materialize_lenovo_mild_scurve_lines import (
    MASK_NAMES, _asset_path, _box, _envelope, _load_jsonl, _local_annotation, _write_image,
)
from tools.text_lq_scurve_mild import DEFAULT_CONFIG, preprocess_text_lq_bgr


def _worker_init() -> None:
    """@luna-comment
    Input: A newly started materialization worker.
    Output: OpenCV configured for one native thread per process.
    Key Steps: Disable nested OpenCV thread oversubscription.
    Key Params: Process count is controlled by the CLI.
    State/Dependencies: Changes only this worker process's OpenCV thread setting.
    """
    cv2.setNumThreads(1)


def _process_source(task: tuple[str, int, dict, str, str]) -> tuple[str, int, list[dict], dict[str, int], int]:
    """@luna-comment
    Input: Split, stable source index, source record, manifest root, and output root.
    Output: Ordered line records, fallback counts, and reused-file count for one source image.
    Key Steps: Reconstruct the exact sequential crop/processing contract and skip complete existing assets.
    Key Params: Every worker owns disjoint sample IDs; existing PNGs are never overwritten when complete.
    State/Dependencies: Reads source images and writes only this source image's output files.
    """
    split, source_index, source_row, source_text, output_text = task
    source, output = Path(source_text), Path(output_text)
    files = {name: _asset_path(source, value) for name, value in source_row["files"].items() if value}
    masks = tuple(name for name in MASK_NAMES if name in files)
    ocr = json.loads(files["ocr"].read_text(encoding="utf-8"))
    annotations = ocr.get("ocr", {}).get("annotations", [])
    source_width, source_height = source_row["cropped_size_wh"]
    planned = []
    for ordinal, annotation in enumerate(annotations):
        if not isinstance(annotation, dict) or annotation.get("ignore") or annotation.get("visible") is False:
            continue
        box = _box(annotation, (source_height, source_width))
        if box is not None:
            planned.append((ordinal, annotation, box, _envelope(box, (source_height, source_width), split)))
    images = {name: cv2.imread(str(files[name]), cv2.IMREAD_COLOR if name in ("lq", "gt") else cv2.IMREAD_UNCHANGED)
              for name in ("lq", "gt", *masks)}
    for name, image in images.items():
        if image is None or image.shape[:2] != (source_height, source_width):
            raise ValueError(f"missing or mismatched {name}: {files[name]}")
    rows = []
    fallbacks = Counter()
    reused = 0
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
        complete_existing = all(path.is_file() and path.stat().st_size > 0 for path in asset_paths.values())
        if complete_existing:
            reused += 1
        else:
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
        rows.append(record)
    return split, source_index, rows, dict(fallbacks), reused


def materialize_parallel(source: Path, output: Path, workers: int) -> dict:
    """@luna-comment
    Input: Source manifest root, existing output root, and worker count.
    Output: Complete deterministic manifests and run summary.
    Key Steps: Dispatch disjoint source images, merge by source order, and atomically publish manifest files.
    Key Params: Existing complete line assets are reused; at least two processes are required.
    State/Dependencies: Writes disjoint image paths and output manifest files only.
    """
    if workers < 2 or not output.is_dir():
        raise ValueError("parallel materialization requires at least two workers and an existing output root")
    if (output / "manifests" / "manifest.json").exists():
        raise FileExistsError("refusing to replace a completed output manifest")
    tasks = []
    source_counts = {}
    for split in ("train", "val"):
        rows = _load_jsonl(source / f"{split}.jsonl")
        source_counts[split] = len(rows)
        tasks.extend((split, index, row, str(source), str(output)) for index, row in enumerate(rows))
    ordered: dict[str, dict[int, list[dict]]] = {"train": {}, "val": {}}
    fallbacks = Counter()
    reused = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = [executor.submit(_process_source, task) for task in tasks]
        for future in as_completed(futures):
            split, index, rows, reasons, reused_count = future.result()
            ordered[split][index] = rows
            fallbacks.update(reasons)
            reused += reused_count
            completed += 1
            if completed % 100 == 0:
                print(json.dumps({"completed_source_images": completed, "total_source_images": len(tasks),
                                  "reused_lines": reused}), flush=True)
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    counts = {}
    for split in ("train", "val"):
        rows = [row for index in range(source_counts[split]) for row in ordered[split][index]]
        counts[f"{split}_source_images"] = source_counts[split]
        counts[f"{split}_lines"] = len(rows)
        targets = {
            f"{split}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            f"{split}_pairs.txt": "".join(f"{row['files']['lq']} {row['files']['gt']}\n" for row in rows),
            f"{split}_raw_pairs.txt": "".join(f"{row['files']['lq_raw']} {row['files']['gt']}\n" for row in rows),
        }
        for name, content in targets.items():
            temporary = manifest_dir / (name + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(manifest_dir / name)
    excluded_path = source / "review_excluded.jsonl"
    excluded = _load_jsonl(excluded_path) if excluded_path.is_file() else []
    (manifest_dir / "review_excluded.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded), encoding="utf-8")
    summary = {"schema": "lenovo_line_envelope_scurve_v1", "source_manifests": str(source),
               "output_root": str(output), "crop": {"vertical_context": 0.12, "horizontal_context": 0.03,
               "boundary_jitter_train": 0.05, "boundary_jitter_val": 0.0},
               "preprocess_version": DEFAULT_CONFIG.version, "counts": counts,
               "fallback_counts": dict(fallbacks), "excluded_source_images": len(excluded),
               "workers": workers, "reused_existing_lines": reused, "complete": True}
    (manifest_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    """@luna-comment
    Input: Source/output roots and requested process count.
    Output: Complete manifest summary and exit code.
    Key Steps: Validate arguments and run resumable parallel materialization.
    Key Params: Workers must be at least two.
    State/Dependencies: File writes are restricted to the chosen output root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifests", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(materialize_parallel(args.source_manifests, args.output_root, args.workers),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
