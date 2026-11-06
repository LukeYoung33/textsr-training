"""Apply direct V5/V6 to exact OCR boxes and render full frames."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import sys
from itertools import repeat
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.materialize_lenovo_mild_scurve_lines import _box
from tools.materialize_lenovo_v5_v6_ocr_matrix import _direct_border_rail
from tools.text_lq_scurve_mild import (
    V5_CONFIG, V6_CONFIG, _apply_chroma_background_return, _luma_bgr,
    _run_normalized, preprocess_text_lq_bgr,
)


ARMS = ("v5", "v6", "v5_borderrail", "v6_borderrail")


def _digest(path: Path) -> str:
    """@luna-comment
    Input: One staged source image.
    Output: SHA-256 digest of original encoded bytes.
    Key Steps: Stream in one-MiB chunks.
    Key Params: Check normalized OCR belongs to this exact staged image.
    State/Dependencies: Read-only source access.
    """
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _write(path: Path, image: np.ndarray) -> None:
    """@luna-comment
    Input: PNG output path and BGR uint8 image.
    Output: Fully encoded file or an explicit failure.
    Key Steps: Create parent directory and use lossless PNG compression.
    Key Params: Compression level three balances test speed and size.
    State/Dependencies: Writes only beneath the selected run output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(path)


def _feather(shape: tuple[int, int], width: int = 8) -> np.ndarray:
    """@luna-comment
    Input: Exact OCR box height and width.
    Output: Float32 alpha with value zero at box edges and one in the interior.
    Key Steps: Apply a smoothstep ramp to the minimum distance from the four box boundaries.
    Key Params: Configurable narrow feather width in pixels.
    State/Dependencies: Pure NumPy calculation.
    """
    height, pixel_width = shape
    width = max(1, min(width, (min(height, pixel_width) - 1) // 2))
    y = np.minimum(np.arange(height), height - 1 - np.arange(height))[:, None]
    x = np.minimum(np.arange(pixel_width), pixel_width - 1 - np.arange(pixel_width))[None, :]
    ramp = np.clip(np.minimum(y, x).astype(np.float32) / width, 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _compose(raw: np.ndarray, regions: list[dict], arm: str, feather_px: int = 8) -> tuple[np.ndarray, int]:
    """@luna-comment
    Input: Full BGR frame, processed OCR regions and selected direct arm.
    Output: Same-size composite and count of changed pixels outside all OCR boxes.
    Key Steps: Accumulate full-box feathered processed-minus-raw deltas and average overlaps.
    Key Params: Divide by at least one to retain the feather when total alpha is below one.
    State/Dependencies: Reads saved line PNGs; does not mutate the source frame.
    """
    height, width = raw.shape[:2]
    delta_sum = np.zeros((height, width, 3), np.float32)
    weight_sum = np.zeros((height, width), np.float32)
    for region in regions:
        left, top, right, bottom = region["ocr_box"]
        processed = cv2.imread(region["files"][arm], cv2.IMREAD_COLOR)
        source_crop = raw[top:bottom, left:right]
        if processed is None or processed.shape != source_crop.shape:
            raise ValueError(f"misaligned processed crop: {region['files'][arm]}")
        alpha = _feather(source_crop.shape[:2], feather_px)
        delta_sum[top:bottom, left:right] += (
            processed.astype(np.float32) - source_crop.astype(np.float32)) * alpha[:, :, None]
        weight_sum[top:bottom, left:right] += alpha
    delta = delta_sum / np.maximum(weight_sum[:, :, None], 1.0)
    result = np.clip(np.rint(raw.astype(np.float32) + delta), 0, 255).astype(np.uint8)
    outside_changed = int(np.count_nonzero(np.any(result[weight_sum == 0] != raw[weight_sum == 0], axis=1)))
    if outside_changed:
        raise ValueError(f"{arm} changed {outside_changed} pixels outside OCR boxes")
    return result, outside_changed


def _force_ambiguous_tile(tile: np.ndarray, config, polarity: int) -> tuple[np.ndarray, dict]:
    """Run the frozen stages with polarity inferred from neighboring line tiles."""
    luma = _luma_bgr(tile)
    normalized = luma if polarity == 1 else 255.0 - luma
    result, telemetry = _run_normalized(tile, normalized, polarity, config)
    if config.chroma_background_return and telemetry.get("fallback_reason") is None:
        result, chroma = _apply_chroma_background_return(result, polarity, config.border_px)
        telemetry.update(chroma)
    telemetry["config_version"] = config.version
    telemetry["polarity_source"] = "neighboring_line_tile"
    return result, telemetry


def _process_tiled(crop: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict]]:
    """@luna-comment
    Input: One exact OCR-box BGR crop.
    Output: Four same-size processed crops and local tile telemetry.
    Key Steps: Divide a wide line into overlapping horizontal tiles, run frozen V5/V6 and border-rail per tile, then blend tile deltas.
    Key Params: Core width is at most twice the line height with a 256-pixel floor; overlap is 25% of height capped at 96 pixels.
    State/Dependencies: Pure image calculation using the frozen preprocessor APIs.
    """
    height, width = crop.shape[:2]
    count = max(1, math.ceil(width / max(256, 2 * height)))
    boundaries = [round(index * width / count) for index in range(count + 1)]
    overlap = min(96, max(24, round(height * .25)))
    delta_sums = {arm: np.zeros(crop.shape, np.float32) for arm in ARMS}
    weight_sum = np.zeros((height, width), np.float32)
    tile_records = []
    pending = []
    for index in range(count):
        left = max(0, boundaries[index] - (overlap if index else 0))
        right = min(width, boundaries[index + 1] + (overlap if index + 1 < count else 0))
        raw_tile = crop[:, left:right]
        x = np.arange(left, right, dtype=np.float32)
        weights = np.ones(right - left, np.float32)
        if index:
            ramp = np.clip((x - left) / (2 * overlap), 0.0, 1.0)
            weights *= ramp * ramp * (3.0 - 2.0 * ramp)
        if index + 1 < count:
            ramp = np.clip((right - 1 - x) / (2 * overlap), 0.0, 1.0)
            weights *= ramp * ramp * (3.0 - 2.0 * ramp)
        telemetry = {}
        images = {}
        for arm, config in (("v5", V5_CONFIG), ("v6", V6_CONFIG)):
            processed, info = preprocess_text_lq_bgr(raw_tile, config, True)
            images[arm] = processed
            telemetry[arm] = info
        pending.append((left, right, weights, raw_tile, images, telemetry))
    reliable = [(index, 1 if info["polarity"] == "dark_text_light_background" else -1)
                for index, (_, _, _, _, _, records) in enumerate(pending)
                if (info := records["v5"]).get("fallback_reason") is None and "polarity" in info]
    if not reliable:
        luma = _luma_bgr(crop)
        threshold, _ = cv2.threshold(luma.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        line_polarity = 1 if np.mean(luma > threshold) >= .5 else -1
    for index, (left, right, weights, raw_tile, images, telemetry) in enumerate(pending):
        weight = weights[None, :, None]
        weight_sum[:, left:right] += weights[None, :]
        for arm, config in (("v5", V5_CONFIG), ("v6", V6_CONFIG)):
            info = telemetry[arm]
            if info.get("fallback_reason") == "ambiguous_border_polarity":
                neighbor = min(reliable, key=lambda item: abs(item[0] - index)) if reliable else None
                polarity = neighbor[1] if neighbor else line_polarity
                images[arm], info = _force_ambiguous_tile(raw_tile, config, polarity)
                info["polarity_source"] = f"tile_{neighbor[0]}" if neighbor else "line_otsu_majority"
                telemetry[arm] = info
            processed = images[arm]
            border, changed = _direct_border_rail(raw_tile, processed, info)
            delta_sums[arm][:, left:right] += (processed.astype(np.float32) - raw_tile.astype(np.float32)) * weight
            delta_sums[f"{arm}_borderrail"][:, left:right] += (border.astype(np.float32) - raw_tile.astype(np.float32)) * weight
            telemetry[f"{arm}_borderrail"] = {"border_rail_changed_pixels": changed}
        tile_records.append({"index": index, "box_xyxy_in_line": [left, 0, right, height], "telemetry": telemetry})
    result = {
        arm: np.clip(np.rint(crop.astype(np.float32) + delta_sums[arm] / np.maximum(weight_sum[:, :, None], 1e-6)), 0, 255).astype(np.uint8)
        for arm in ARMS
    }
    return result, tile_records


def process_image(image_path: Path, ocr_path: Path, output_root: Path, feather_px: int = 8) -> dict:
    """@luna-comment
    Input: One staged JPG, its normalized OCR JSON and run output root.
    Output: Four full-frame PNGs, line crops and per-image processing metadata.
    Key Steps: Validate OCR source hash, split each original OCR box into local tiles, then feather full-box composites.
    Key Params: No crop extension or GT-fitted processing; adaptive smooth box-edge feather.
    State/Dependencies: Reads staged source/OCR and writes only to output_root.
    """
    raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"undecodable source image: {image_path}")
    document = json.loads(ocr_path.read_text(encoding="utf-8"))
    if document["source_sha256"] != _digest(image_path) or document["source_hw"] != list(raw.shape[:2]):
        raise ValueError(f"OCR does not match source image: {image_path}")
    group = image_path.parent.name
    regions = []
    for ordinal, annotation in enumerate(document["ocr"]["annotations"]):
        box = _box(annotation, raw.shape[:2])
        if box is None:
            continue
        left, top, right, bottom = box
        raw_crop = raw[top:bottom, left:right]
        line_id = f"{image_path.stem}__line{ordinal:03d}"
        files = {"raw": output_root / "lines" / "raw" / group / f"{line_id}.png"}
        _write(files["raw"], raw_crop)
        processed, tiles = _process_tiled(raw_crop)
        for arm in ARMS:
            files[arm] = output_root / "lines" / arm / group / f"{line_id}.png"
            _write(files[arm], processed[arm])
        regions.append({"line_id": line_id, "ocr_index": ordinal, "ocr_box": list(box),
                        "envelope": list(box), "geometry": {"mode": "original_ocr_box", "padding_xy": [0, 0]},
                        "files": {key: str(path) for key, path in files.items()},
                        "tiles": tiles})
    fullframe = {}
    for arm in ARMS:
        composite, outside_changed = _compose(raw, regions, arm, feather_px)
        path = output_root / "fullframe" / arm / group / f"{image_path.stem}.png"
        _write(path, composite)
        fullframe[arm] = {"path": str(path), "outside_changed_pixels": outside_changed,
                          "changed_pixels": int(np.count_nonzero(np.any(composite != raw, axis=2)))}
    result = {"source": str(image_path), "ocr": str(ocr_path), "group": group,
              "height": raw.shape[0], "width": raw.shape[1], "ocr_lines": len(document["ocr"]["annotations"]),
              "processed_lines": len(regions), "feather_px": feather_px, "regions": regions, "fullframe": fullframe}
    metadata_path = output_root / "metadata" / group / f"{image_path.stem}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"group": group, "image": image_path.name, "ocr_lines": result["ocr_lines"],
            "processed_lines": len(regions), "fullframe": {arm: value["path"] for arm, value in fullframe.items()},
            "metadata": str(metadata_path)}


def main() -> int:
    """@luna-comment
    Input: Staged image root, completed OCR root, output root, worker count and optional pilot limit.
    Output: Printed per-image rows and one batch summary.
    Key Steps: Require OCR completion, process source images in bounded workers and verify four outputs per image.
    Key Params: Sorted group/image order; limit is for a bounded pilot only.
    State/Dependencies: Writes only to the requested preprocess output root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ocr-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--feather-px", type=int, default=8)
    parser.add_argument("--image", type=str, help="Process only one staged image name for a pilot")
    args = parser.parse_args()
    paths = sorted(path for group in ("ppt", "blackwhite")
                   for path in (args.input_root / group).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError("no staged JPG test images")
    if args.image:
        paths = [path for path in paths if path.name == args.image or path.stem == args.image]
        if not paths:
            raise FileNotFoundError(f"staged image not found: {args.image}")
    if args.limit is not None:
        paths = paths[:args.limit]
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.feather_px < 1:
        raise ValueError("feather-px must be positive")
    ocr_paths = []
    for image_path in paths:
        ocr_path = args.ocr_root / "normalized" / image_path.parent.name / f"{image_path.stem}.json"
        if not ocr_path.is_file():
            raise FileNotFoundError(f"OCR must run first: {ocr_path}")
        ocr_paths.append(ocr_path)
    if args.workers == 1:
        result_rows = map(process_image, paths, ocr_paths, repeat(args.output_root), repeat(args.feather_px))
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        result_rows = executor.map(process_image, paths, ocr_paths, repeat(args.output_root), repeat(args.feather_px))
    rows = []
    for row in result_rows:
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.workers > 1:
        executor.shutdown()
    summary = {"images": len(rows), "processed_lines": sum(row["processed_lines"] for row in rows),
               "arms": list(ARMS), "rows": rows}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": summary["images"], "processed_lines": summary["processed_lines"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
