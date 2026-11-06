"""Experimental curved text-line inference at a glyph-normalized scale.

This diagnostic path is intentionally line-selective.  It estimates a curved
baseline and effective glyph height from the DB probability map plus local RGB
contrast, rectifies one OCR candidate to a straight strip, restores it in
96x480 tiles, and maps the result back through the inverse mesh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer_basic_text_line_full_image import (
    NativeY4UVTileRunner,
    _load_network,
    _read_rgb,
    _resize_hwc,
    _write_gray,
    _write_rgb,
    load_lq_preprocess_config,
    load_network_config,
    rgb_to_yuv444,
    standard_pasteback,
    standard_polygon_paste_alpha,
    yuv444_to_rgb,
)


def _fit_curve_and_glyph_height(rgb: np.ndarray, probability: np.ndarray, polygon: np.ndarray, core_to_glyph: float = 2.5) -> tuple[np.ndarray, float, dict]:
    """Fit a quadratic centerline and estimate visible glyph height in source pixels."""
    height, width = rgb.shape[:2]
    minimum = np.floor(polygon.min(axis=0)).astype(np.int32)
    maximum = np.ceil(polygon.max(axis=0)).astype(np.int32)
    x0, y0 = np.clip(minimum, [0, 0], [width - 1, height - 1])
    x1, y1 = np.clip(maximum, [x0 + 2, y0 + 2], [width, height])
    crop = rgb[y0:y1, x0:x1]
    probability_full = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    seed = probability_full[y0:y1, x0:x1] >= .20
    gray = cv2.cvtColor(np.clip(crop * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray, (0, 0), 5.0)
    contrast = np.abs(gray - local_mean) >= .035
    centers_x, centers_y = [], []
    for col in range(seed.shape[1]):
        ys = np.flatnonzero(seed[:, col])
        if ys.size:
            centers_x.append(float(col)); centers_y.append(float(np.mean(ys)))
    if len(centers_x) < 8:
        # DB evidence is too sparse: retain a straight line and a conservative
        # source-scale estimate rather than inventing a curve.
        center = (y1 - y0) * .5
        return np.full(x1 - x0, center, np.float32), max(4.0, .45 * (y1 - y0)), {
            "status": "probability_sparse", "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)]
        }
    degree = 2 if len(centers_x) >= 16 else 1
    coefficients = np.polyfit(np.asarray(centers_x), np.asarray(centers_y), degree)
    xs = np.arange(x1 - x0, dtype=np.float32)
    centerline = np.polyval(coefficients, xs).astype(np.float32)
    core_height = max(3.0, float(np.median([np.ptp(np.flatnonzero(seed[:, col])) + 1 for col in range(seed.shape[1]) if seed[:, col].any()])))
    spans = []
    radius = int(np.ceil(max(2.5 * core_height, 12.0)))
    for col, center in enumerate(centerline):
        lo, hi = max(0, int(np.floor(center - radius))), min(seed.shape[0], int(np.ceil(center + radius + 1)))
        ys = np.flatnonzero((contrast[lo:hi, col] | seed[lo:hi, col])) + lo
        if ys.size >= 2:
            spans.append(float(ys.max() - ys.min() + 1))
    contrast_height = float(np.median(spans)) if len(spans) >= 8 else core_height
    # DB represents a shrunken text core.  Local contrast can bridge adjacent
    # rows or background edges on curved packaging, so use it only when it is
    # consistent with a calibrated core-to-glyph expansion.
    calibrated_height = float(core_height * core_to_glyph)
    glyph_height = contrast_height if .70 * calibrated_height <= contrast_height <= 1.35 * calibrated_height else calibrated_height
    glyph_height = float(np.clip(glyph_height, max(3.0, core_height), max(4.0, y1 - y0)))
    curvature = float(np.max(np.abs(centerline - np.polyval(np.polyfit(xs, centerline, 1), xs))))
    return centerline, glyph_height, {
        "status": "ok", "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)], "core_height_px": core_height,
        "glyph_height_px": glyph_height, "contrast_height_px": contrast_height, "core_to_glyph": core_to_glyph,
        "glyph_samples": len(spans), "curvature_px": curvature,
        "curve_coefficients": [float(value) for value in coefficients],
    }


def _tile_restore(strip: np.ndarray, runner: NativeY4UVTileRunner, tile_width: int = 480, overlap: int = 96) -> np.ndarray:
    """Restore a 96-pixel-high strip with complementary horizontal overlap."""
    height, width = strip.shape[:2]
    if height != 96:
        raise ValueError("rectified strip must be 96 pixels high")
    starts = list(range(0, max(1, width - tile_width + 1), max(1, tile_width - overlap)))
    if not starts or starts[-1] != max(0, width - tile_width):
        starts.append(max(0, width - tile_width))
    total = np.zeros((3, height, width), np.float32); weights = np.zeros((height, width), np.float32)
    for start in sorted(set(starts)):
        end = min(width, start + tile_width)
        tile = strip[:, start:end]
        canvas = np.pad(tile, ((0, 0), (0, tile_width - tile.shape[1]), (0, 0)), mode="edge")
        prediction = yuv444_to_rgb(runner.run_rgb(canvas))[:, :tile.shape[1]].transpose(2, 0, 1)
        weight = np.ones((height, end - start), np.float32)
        ramp = min(overlap, max(1, (end - start) // 3))
        if start > 0:
            weight[:, :ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)[None]
        if end < width:
            weight[:, -ramp:] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)[None]
        total[:, :, start:end] += prediction * weight[None]
        weights[:, start:end] += weight
    return total / np.maximum(weights[None], 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument("--probability-map", type=Path, required=True)
    parser.add_argument("--line-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--network-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--target-glyph-height", type=float, default=64.0)
    parser.add_argument("--core-to-glyph", type=float, default=2.5,
                        help="Calibrated DB probability-core to visible-glyph height ratio.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-rectify", action="store_true",
                        help="Keep the measured glyph scale but use an ordinary horizontal strip for A/B comparison.")
    args = parser.parse_args()
    rgb = _read_rgb(args.input); data = json.loads(args.ocr_json.read_text(encoding="utf-8")); probability = np.load(args.probability_map)
    polygon = np.asarray(data["dt_polys"][args.line_index], np.float32)
    curve, glyph_height, meta = _fit_curve_and_glyph_height(rgb, probability, polygon, args.core_to_glyph)
    x0, y0, x1, y1 = meta["crop_xyxy"]; scale = float(args.target_glyph_height / max(glyph_height, 1e-6))
    source_height = 96.0 / scale; center_ref = 48.0
    source_width = x1 - x0; out_width = max(8, int(round(source_width * scale)))
    uu, vv = np.meshgrid(np.arange(out_width, dtype=np.float32), np.arange(96, dtype=np.float32))
    source_x = (x0 + uu / scale).astype(np.float32)
    curve_source = curve + y0
    line_y = np.full(out_width, float(np.median(curve_source)), np.float32) if args.disable_rectify else np.interp(source_x[0], np.arange(source_width, dtype=np.float32) + x0, curve_source)
    source_y = (line_y[None] + (vv - center_ref) / scale).astype(np.float32)
    strip = cv2.remap(rgb, source_x, source_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    network, _ = _load_network(args.checkpoint, device, args.network_config)
    runner = NativeY4UVTileRunner(network, device, load_lq_preprocess_config(args.network_config))
    prediction = _tile_restore(strip, runner)
    # Inverse mesh: sample rectified output at each original crop pixel.
    xx, yy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    local_curve = np.full(x1 - x0, float(np.median(curve_source)), np.float32) if args.disable_rectify else np.interp(xx[0], np.arange(source_width, dtype=np.float32) + x0, curve_source)
    map_x = ((xx - x0) * scale).astype(np.float32); map_y = (center_ref + (yy - local_curve[None]) * scale).astype(np.float32)
    restored_crop = cv2.remap(prediction.transpose(1, 2, 0), map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Invalid inverse-map samples are restored to LQ before the one canonical
    # compositor runs.  They are not a second boundary-alpha implementation.
    valid_strip = (map_x >= 0.0) & (map_x <= out_width - 1.0) & (map_y >= 0.0) & (map_y <= 95.0)
    restored_crop[~valid_strip] = rgb[y0:y1, x0:x1][~valid_strip]
    frame_yuv = rgb_to_yuv444(rgb); output_yuv = frame_yuv.copy(); owner = np.zeros(rgb.shape[:2], np.float32)
    alpha = standard_polygon_paste_alpha(rgb, polygon, (x0, y0, x1, y1), 4)
    standard_pasteback(frame_yuv, output_yuv, owner, (x0, y0, x1, y1), rgb_to_yuv444(restored_crop), alpha, 1.0)
    output = yuv444_to_rgb(output_yuv)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(args.output, output); _write_rgb(args.debug_dir / "rectified_input.png", strip); _write_rgb(args.debug_dir / "rectified_prediction.png", prediction.transpose(1, 2, 0)); _write_gray(args.debug_dir / "paste_alpha.png", alpha); _write_gray(args.debug_dir / "paste_valid_strip.png", valid_strip.astype(np.float32))
    meta.update({"line_index": args.line_index, "target_glyph_height_px": args.target_glyph_height, "scale": scale, "rectification": not args.disable_rectify, "rectified_hw": [96, out_width]})
    (args.debug_dir / "measurement.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
