"""Build a lossless OCR-line envelope cache for fixed-canvas text training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from realesrgan.data.basic_text_line_gtdeg_dataset import BasicTextLineGTDegradeDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _envelope(ocr_crop: tuple[int, int, int, int], image_hw: tuple[int, int],
              vertical_context: float, horizontal_context: float,
              boundary_jitter: float) -> tuple[int, int, int, int]:
    left, top, right, bottom = ocr_crop
    line_h, line_w = bottom - top, right - left
    pad_y = max(2, int(round(line_h * vertical_context)))
    pad_x = max(2, int(round(line_w * horizontal_context)))
    jitter_y = int(round(line_h * boundary_jitter))
    jitter_x = int(round(line_w * boundary_jitter))
    height, width = image_hw
    return (
        max(0, left - pad_x - jitter_x),
        max(0, top - pad_y - jitter_y),
        min(width, right + pad_x + jitter_x),
        min(height, bottom + pad_y + jitter_y),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-lines", required=True, type=int)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite cache root: {args.output_root}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_opt = dict(config["datasets"]["train"])
    train_opt.pop("ocr_line_crop_cache_root", None)
    dataset = BasicTextLineGTDegradeDataset(train_opt)
    if len(dataset) != args.expected_lines:
        raise ValueError(f"expected {args.expected_lines} OCR lines, found {len(dataset)}")

    args.output_root.mkdir(parents=True)
    lines_root = args.output_root / "lines"
    lines_root.mkdir()
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    paths: dict[tuple[str, str, str, str], tuple[Path, Path, Path, Path]] = {}
    for index, entry in enumerate(dataset._entries):
        record = entry["record"]
        values = (
            dataset._path(record, "lq", "lq", entry["root"], entry["split"]),
            dataset._path(record, "gt", "gt", entry["root"], entry["split"]),
            dataset._path(record, "text_mask", "text_mask", entry["root"], entry["split"]),
            dataset._path(record, "valid_mask", "valid_mask", entry["root"], entry["split"]),
        )
        key = tuple(str(value) for value in values)
        grouped[key].append(index)
        paths[key] = values

    records = []
    for key, indices in grouped.items():
        lq_path, gt_path, text_path, valid_path = paths[key]
        lq, gt, text, valid = dataset._load_source_arrays(lq_path, gt_path, text_path, valid_path)
        if lq.shape != gt.shape:
            raise ValueError(f"LQ/GT shape mismatch: {lq_path}")
        full_hw = lq.shape[:2]
        for index in indices:
            record = dataset._entries[index]["record"]
            ocr = tuple(int(value) for value in record["ocr_crop_xyxy"])
            envelope = _envelope(
                ocr, full_hw, dataset.text_line_vertical_context,
                dataset.text_line_horizontal_context, dataset.text_line_boundary_jitter)
            left, top, right, bottom = envelope
            arrays = {
                "lq": np.rint(lq[top:bottom, left:right] * 255.0).astype(np.uint8),
                "gt": np.rint(gt[top:bottom, left:right] * 255.0).astype(np.uint8),
                "text": text[top:bottom, left:right] if text is not None else np.empty((0, 0), dtype=np.uint8),
                "valid": valid[top:bottom, left:right] if valid is not None else np.empty((0, 0), dtype=np.uint8),
                "has_text": np.array([text is not None], dtype=np.uint8),
                "has_valid": np.array([valid is not None], dtype=np.uint8),
                "envelope_xyxy": np.array(envelope, dtype=np.int32),
                "full_hw": np.array(full_hw, dtype=np.int32),
            }
            target = lines_root / f"{index:06d}.npz"
            np.savez_compressed(target, **arrays)
            records.append({
                "index": index,
                "line_id": record["image"],
                "source_lq": str(lq_path),
                "envelope_xyxy": list(envelope),
                "file": f"lines/{target.name}",
                "sha256": _sha256(target),
            })

    manifest = {
        "schema_version": 1,
        "cache_type": "lossless_ocr_line_envelope",
        "config_sha256": _sha256(args.config),
        "source_manifests": dataset.manifest_sha256,
        "line_count": len(records),
        "records": records,
    }
    (args.output_root / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "complete.json").write_text(
        json.dumps({"status": "complete", "line_count": len(records)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

