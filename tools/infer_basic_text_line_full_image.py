"""Full-image inference for the BASIC text-line restoration model.

Pipeline contract:

1. Read an image and PPOCR-style OCR JSON.
2. Use each OCR polygon's AABB as the native-scale core region and expand it
   when the fixed 96x320 model canvas has capacity.
3. Split only oversized long axes into overlapping native-scale windows.
4. Restore every segment in packed NV21/YUV420 input -> YUV444 output space.
5. Fuse segments of the same OCR line first, then fuse neighboring lines with
   confidence-aware ownership.  A text-core is opaque and an outward ring is
   feathered into the original image, so unowned background pixels are kept
   unchanged.

The default route intentionally does not resize or perspective-warp a text
line.  An explicit ``resize_bypass`` route is available for controlled
experiments: only native short-axis bypasses whose polygon AABB ratio is in a
configured range are uniformly resized, split, restored, and mapped back.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.data.basic_text_line_full_frame import (  # noqa: E402
    CANVAS_HW,
    OCRLine,
    FullFrameSegment,
    attach_line_index,
    build_full_frame_segments,
    build_height96_tiled_full_frame_segments,
    build_native_tiled_full_frame_segments,
    build_p17_height96_tiled_full_frame_segments,
    build_resized_full_frame_segments,
    feather_alpha,
    pad_even_rgb,
    parse_ocr_json,
    parse_p17_ocr_annotations,
    rasterize_polygon_mask,
    tile_axis_weight,
)
from tools.basic_v2_eval_network import build_eval_network, load_network_config, resolve_canvas_hw  # noqa: E402
from realesrgan.data.basic_text_line_y4uv_v3 import numpy_rgb_to_y4uv, rgb_to_y4uv, y4uv_to_yuv444  # noqa: E402
from realesrgan.lenovo_p17_preprocess import normalize_lq_preprocess_config, preprocess_lq_rgb  # noqa: E402
from realesrgan.lenovo_p17_torch_gpu import preprocess_p17_torch_gpu_with_debug  # noqa: E402
from realesrgan.utils import YUV2RGB  # noqa: E402
try:
    from tools.lenovo_stage_a_preprocess_screen import (  # noqa: E402
        _p10_measure_profiles,
        _p10_reconstruct_rgb_from_y,
        _p10_segment_text,
        _p11_scatter_weighted,
        _rgb_to_luma,
        _p11_kernel_wiener_scurve,
        candidate_parameters,
        load_config,
    )
except ModuleNotFoundError:  # batch entry imports this file as a direct module
    from lenovo_stage_a_preprocess_screen import (  # type: ignore[no-redef]
        _p10_measure_profiles,
        _p10_reconstruct_rgb_from_y,
        _p10_segment_text,
        _p11_scatter_weighted,
        _rgb_to_luma,
        _p11_kernel_wiener_scurve,
        candidate_parameters,
        load_config,
    )


_P11_FROZEN_CONFIG = load_config(ROOT / "options" / "lenovo_stage_a_preprocess_screen_p11_kernel_wiener_scurve.json")
_P11_FROZEN_PARAMS = next(
    candidate for candidate in candidate_parameters(_P11_FROZEN_CONFIG, "P11")
    if candidate["name"] == "lambda_0p03"
)


def _axis_aligned_polygon(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """@luna-comment Build a clockwise float32 rectangle from already-clipped bounds.

    Input: finite full-frame x/y bounds with ``x1>x0`` and ``y1>y0``. Output:
    a four-point clockwise ``float32`` polygon. Key Steps: preserve only the
    supplied axis-aligned envelope. Key Params: none. State/Dependencies:
    stateless NumPy construction used by opt-in P17 geometry recovery.
    """
    return np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=np.float32)


def _load_mask_guided_geometry_stats(path: Path) -> dict[str, float | str]:
    """@luna-comment Load and validate the versioned training geometry quantiles.

    Input: JSON produced by the companion training-manifest statistics tool.
    Output: flattened required quantiles plus its recorded version/hash.
    Key Steps: parse JSON, require finite ordered quantiles, and reject an
    incompatible artifact before any OCR geometry can change. Key Params: the
    q05/q50/q95 ratio fields define y-only recovery bounds. State/Dependencies:
    filesystem-only; no model or preprocessing state is changed.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    quantiles = payload.get("quantiles", {})
    required = (
        "top_margin_over_hink_q50", "bottom_margin_over_hink_q50",
        "hbox_over_hink_q05", "hbox_over_hink_q95",
    )
    result: dict[str, float | str] = {
        "stats_version": str(payload.get("stats_version", "")),
        "stats_sha256": str(payload.get("stats_sha256", "")),
    }
    if not result["stats_version"] or not result["stats_sha256"]:
        raise ValueError("mask-guided geometry stats require version and sha256")
    for name in required:
        value = float(quantiles.get(name, float("nan")))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid mask-guided geometry quantile: {name}")
        result[name] = value
    if float(result["hbox_over_hink_q05"]) > float(result["hbox_over_hink_q95"]):
        raise ValueError("mask-guided geometry stats have reversed hbox bounds")
    return result


def _line_is_mask_recovery_eligible(line: OCRLine) -> bool:
    """@luna-comment Decide whether a detected OCR polygon is a horizontal line.

    Input: one full-frame OCR polygon. Output: true only for a non-degenerate
    horizontal envelope whose leading edge is within 15 degrees. Key Steps:
    compare AABB aspect and leading-edge angle. Key Params: the fixed 15-degree
    limit prevents y-only correction from distorting slanted/vertical text.
    State/Dependencies: stateless OCR geometry inspection.
    """
    polygon = np.asarray(line.polygon_xy, dtype=np.float32)
    width = float(np.ptp(polygon[:, 0]))
    height = float(np.ptp(polygon[:, 1]))
    if width <= height or min(width, height) < 1.0:
        return False
    edge = polygon[1] - polygon[0] if polygon.shape[0] >= 2 else np.asarray((width, 0.0))
    angle = abs(float(np.degrees(np.arctan2(float(edge[1]), float(edge[0])))))
    return angle <= 15.0


def _write_geometry_overlay(path: Path, frame_rgb: np.ndarray, original: OCRLine, corrected: OCRLine | None, band: tuple[int, int] | None) -> None:
    """@luna-comment Export an unambiguous original/corrected OCR-box overlay.

    Input: normalized RGB frame, original line, optional corrected line, and
    optional selected y band. Output: PNG with yellow original, green corrected,
    and red support-band boundaries. Key Steps: draw only measured geometry;
    no pixels are used for restoration. Key Params: none. State/Dependencies:
    OpenCV drawing and the caller-owned debug path.
    """
    overlay = np.rint(np.clip(frame_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.polylines(overlay, [np.rint(original.polygon_xy).astype(np.int32)], True, (255, 220, 20), 2)
    if corrected is not None:
        cv2.polylines(overlay, [np.rint(corrected.polygon_xy).astype(np.int32)], True, (20, 255, 40), 2)
    if band is not None:
        for y in band:
            cv2.line(overlay, (0, int(y)), (overlay.shape[1] - 1, int(y)), (255, 40, 40), 1)
    _write_rgb(path, overlay.astype(np.float32) / 255.0)


def _recover_mask_guided_ocr_lines(
    lines: list[OCRLine], mask_raw: np.ndarray, frame_rgb: np.ndarray, stats: dict[str, float | str], debug_dir: Path | None,
) -> tuple[list[OCRLine], dict[int, dict[str, Any]]]:
    """@luna-comment Recover horizontal OCR y-bounds from P17 raw-mask row bands.

    Input: parsed OCR lines, full-frame binary P17 raw mask, source RGB, and
    validated training quantiles. Output: replacement lines plus a decision for
    every input line. Key Steps: limit each search to half an OCR height plus
    adjacent-line midpoints, threshold row mass at 1% of OCR width, reject
    merged/truncated or horizontally unowned bands, then map trusted ink height
    through training ratios; all other lines retain their exact polygon. Key
    Params: 3-pixel vertical closing only connects row support; correction is
    y-only. State/Dependencies: raw mask is probe-only and debug files are
    evidence, never model input.
    """
    height, width = mask_raw.shape
    binary = np.asarray(mask_raw, dtype=bool)
    ordered = sorted(lines, key=lambda item: float(np.mean(item.polygon_xy[:, 1])))
    centers = [float(np.mean(item.polygon_xy[:, 1])) for item in ordered]
    recovered: dict[int, OCRLine] = {}
    decisions: dict[int, dict[str, Any]] = {}
    for order_index, line in enumerate(ordered):
        polygon = np.asarray(line.polygon_xy, dtype=np.float32)
        x0, y0 = polygon.min(axis=0)
        x1, y1 = polygon.max(axis=0)
        record: dict[str, Any] = {
            "line_index": int(line.line_index), "original_polygon": polygon.tolist(),
            "status": "fallback", "reason": "UNSET", "mask_band_y": None,
            "search_y": None, "corrected_polygon": polygon.tolist(),
            "stats_version": stats["stats_version"], "stats_sha256": stats["stats_sha256"],
        }
        corrected: OCRLine | None = None
        band: tuple[int, int] | None = None
        profile = np.zeros((0,), dtype=np.int32)
        if not _line_is_mask_recovery_eligible(line):
            record["reason"] = "SLANTED_OR_VERTICAL_OR_DEGENERATE"
        else:
            left, right = max(0, int(np.floor(x0))), min(width, int(np.ceil(x1)))
            original_height = float(y1 - y0)
            lower = max(0, int(np.floor(y0 - .5 * original_height)))
            upper = min(height, int(np.ceil(y1 + .5 * original_height)))
            if order_index:
                lower = max(lower, int(np.floor((centers[order_index - 1] + centers[order_index]) * .5)))
            if order_index + 1 < len(ordered):
                upper = min(upper, int(np.ceil((centers[order_index] + centers[order_index + 1]) * .5)))
            lower, upper = max(0, lower), min(height, upper)
            record["search_y"] = [lower, upper]
            if right - left < 2 or upper - lower < 2:
                record["reason"] = "EMPTY_X_OR_NEIGHBOR_ENVELOPE"
            else:
                profile = binary[lower:upper, left:right].sum(axis=1).astype(np.int32)
                active = profile >= max(2, int(np.ceil(.01 * (right - left))))
                support = cv2.morphologyEx(active.astype(np.uint8)[:, None], cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))[:, 0] > 0
                starts = np.flatnonzero(support & np.r_[True, ~support[:-1]])
                ends = np.flatnonzero(support & np.r_[~support[1:], True]) + 1
                candidates = [(int(start), int(end)) for start, end in zip(starts, ends) if int(end - start) >= 2]
                overlap = []
                for start, end in candidates:
                    interval = max(0.0, min(float(lower + end), float(y1)) - max(float(lower + start), float(y0)))
                    union = max(float(lower + end), float(y1)) - min(float(lower + start), float(y0))
                    overlap.append((interval / max(union, 1.0e-6), int(profile[start:end].sum()), start, end))
                if not overlap or max(item[0] for item in overlap) <= 0.0:
                    record["reason"] = "NO_OVERLAPPING_MASK_BAND"
                else:
                    _, _, band_start, band_end = max(overlap, key=lambda item: (item[0], item[1]))
                    raw_rows = np.flatnonzero(active[band_start:band_end]) + band_start
                    if raw_rows.size < 2:
                        record["reason"] = "INSUFFICIENT_RAW_MASK_SUPPORT"
                    else:
                        raw_start, raw_end = int(raw_rows[0]), int(raw_rows[-1] + 1)
                        band_y0, band_y1 = lower + raw_start, lower + raw_end
                        band = (band_y0, band_y1)
                        ink_height = float(band_y1 - band_y0)
                        full_band_pixels = int(binary[band_y0:band_y1].sum())
                        local_band_pixels = int(binary[band_y0:band_y1, left:right].sum())
                        raw_run_count = sum(bool(active[start:end].any()) for start, end in candidates if start >= band_start and end <= band_end)
                        touches_probe = raw_start == 0 or raw_end == upper - lower
                        probe_edge_is_valid = (raw_start == 0 and lower == 0 and y0 <= 0.0) or (raw_end == upper - lower and upper == height and y1 >= height - 1.0)
                        center_offset = abs((band_y0 + band_y1) * .5 - (float(y0) + float(y1)) * .5)
                        record.update({"mask_band_y": list(band), "ink_height": ink_height, "mask_pixels_in_band": local_band_pixels, "full_mask_pixels_in_band": full_band_pixels, "horizontal_ownership": float(local_band_pixels / max(full_band_pixels, 1)), "center_offset": center_offset, "raw_run_count": raw_run_count})
                        if touches_probe and not probe_edge_is_valid:
                            record["reason"] = "MASK_MERGED_OR_TRUNCATED"
                        elif raw_run_count > 1:
                            record["reason"] = "MASK_MERGED_OR_TRUNCATED"
                        elif local_band_pixels < 8 or float(local_band_pixels / max(full_band_pixels, 1)) < .70:
                            record["reason"] = "MASK_HORIZONTAL_OWNERSHIP_LOW"
                        elif center_offset > max(.75 * original_height, 1.5 * ink_height):
                            record["reason"] = "MASK_BAND_NOT_OWNED_BY_OCR_LINE"
                        else:
                            target_y0 = float(band_y0) - float(stats["top_margin_over_hink_q50"]) * ink_height
                            target_y1 = float(band_y1) + float(stats["bottom_margin_over_hink_q50"]) * ink_height
                            target_height = float(np.clip(
                                target_y1 - target_y0,
                                ink_height * float(stats["hbox_over_hink_q05"]),
                                ink_height * float(stats["hbox_over_hink_q95"]),
                            ))
                            center = (target_y0 + target_y1) * .5
                            target_y0, target_y1 = center - target_height * .5, center + target_height * .5
                            target_y0, target_y1 = max(float(lower), target_y0), min(float(upper), target_y1)
                            if target_y1 - target_y0 < target_height - 1e-4:
                                shift = target_height - (target_y1 - target_y0)
                                target_y0, target_y1 = max(float(lower), target_y0 - shift), min(float(upper), target_y1 + shift)
                            if target_y1 - target_y0 < 2.0:
                                record["reason"] = "CANNOT_FIT_NEIGHBOR_ENVELOPE"
                            else:
                                corrected = replace(line, polygon_xy=_axis_aligned_polygon(float(x0), target_y0, float(x1), target_y1))
                                recovered[line.line_index] = corrected
                                record.update({"status": "accepted", "reason": "RAW_MASK_BAND_RECOVERED", "corrected_polygon": corrected.polygon_xy.tolist()})
        decisions[line.line_index] = record
        if debug_dir is not None:
            line_dir = debug_dir / "geometry_recovery" / f"line_{line.line_index:04d}"
            line_dir.mkdir(parents=True, exist_ok=True)
            _write_geometry_overlay(line_dir / "overlay.png", frame_rgb, line, corrected, band)
            source_bounds = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))), min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1))))
            _write_rgb(line_dir / "original_box.png", frame_rgb[source_bounds[1]:source_bounds[3], source_bounds[0]:source_bounds[2]])
            chosen = corrected or line
            cp = chosen.polygon_xy
            cx0, cy0 = cp.min(axis=0); cx1, cy1 = cp.max(axis=0)
            corrected_bounds = (max(0, int(np.floor(cx0))), max(0, int(np.floor(cy0))), min(width, int(np.ceil(cx1))), min(height, int(np.ceil(cy1))))
            _write_rgb(line_dir / "corrected_box.png", frame_rgb[corrected_bounds[1]:corrected_bounds[3], corrected_bounds[0]:corrected_bounds[2]])
            _write_gray(line_dir / "mask_support.png", binary[corrected_bounds[1]:corrected_bounds[3], corrected_bounds[0]:corrected_bounds[2]])
            with (line_dir / "row_profile.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle); writer.writerow(("y", "mask_pixels"))
                start_y = int(record["search_y"][0]) if record["search_y"] else 0
                writer.writerows((start_y + index, int(value)) for index, value in enumerate(profile))
            (line_dir / "geometry.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    result = [recovered.get(line.line_index, line) for line in lines]
    return result, decisions


def _apply_p11_edge_narrow_v2(candidate_yuv: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Astra Light V2's single-pass directional-mask P10 S-curve adapter."""
    candidate_rgb = yuv444_to_rgb(candidate_yuv)
    y = _rgb_to_luma(candidate_rgb)
    raw_mask, segmentation = _p10_segment_text(y, _P11_FROZEN_PARAMS)
    if raw_mask is None:
        return candidate_yuv, {"postprocess": "v2_skipped_no_accepted_p10_mask", "_v2_raw_mask": np.zeros_like(y, dtype=bool)}
    raw_mask = np.asarray(raw_mask, dtype=bool)
    boundary = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    gx = cv2.Sobel(raw_mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(raw_mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    clean_mask = raw_mask.copy()
    removed = 0
    filled = 0
    for row, column in np.argwhere(boundary):
        magnitude = float(np.hypot(gx[row, column], gy[row, column]))
        if magnitude <= 1.0e-6:
            continue
        tangent_x, tangent_y = -float(gy[row, column]) / magnitude, float(gx[row, column]) / magnitude
        offsets = np.asarray((-2.0, -1.0, 1.0, 2.0), dtype=np.float32)
        samples = cv2.remap(
            raw_mask.astype(np.float32),
            (float(column) + offsets * tangent_x).reshape(1, -1),
            (float(row) + offsets * tangent_y).reshape(1, -1),
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1) >= 0.5
        if raw_mask[row, column] and not np.any(samples):
            clean_mask[row, column] = False
            removed += 1
        elif not raw_mask[row, column] and bool(np.all(samples)):
            clean_mask[row, column] = True
            filled += 1
    profile_params = dict(_P11_FROZEN_PARAMS)
    profile_params["cleanup"] = "none"
    measured = _p10_measure_profiles(y, clean_mask, profile_params)
    profiles = measured["profiles"]
    delta_sum = np.zeros_like(y, dtype=np.float32)
    weight_sum = np.zeros_like(y, dtype=np.float32)
    accepted = 0
    for profile in profiles:
        background = float(profile["background_y"])
        foreground = float(profile["foreground_y"])
        difference = foreground - background
        if abs(difference) < 0.12:
            continue
        values = np.asarray(profile["values"], dtype=np.float32)
        q = np.clip((values - background) / difference, 0.0, 1.0)
        power = np.power(q, 2.20)
        inverse = np.power(1.0 - q, 2.20)
        shaped = power / np.maximum(power + inverse, 1.0e-12)
        target = background + difference * shaped
        point_weight = ((q > 0.02) & (q < 0.98)).astype(np.float32)
        _p11_scatter_weighted(
            delta_sum, weight_sum,
            np.asarray(profile["_map_x"], dtype=np.float32),
            np.asarray(profile["_map_y"], dtype=np.float32),
            target - values, point_weight,
        )
        accepted += 1
    write_support = weight_sum > 1.0e-6
    correction = np.zeros_like(y, dtype=np.float32)
    correction[write_support] = delta_sum[write_support] / weight_sum[write_support]
    out_y = np.clip(y + correction, 0.0, 1.0)
    output_rgb, _ = _p10_reconstruct_rgb_from_y(candidate_rgb, y, out_y)
    output_yuv = rgb_to_yuv444(output_rgb)
    result = candidate_yuv.copy()
    result[0, write_support] = output_yuv[0, write_support]
    return result, {
        "postprocess": "p11_edge_narrow_v2",
        "p10_segmentation_accepted": bool(segmentation.get("segmentation_accepted", False)),
        "p10_polarity": segmentation.get("polarity"),
        "raw_mask_pixels": int(raw_mask.sum()),
        "directional_spike_removed": removed,
        "directional_notch_filled": filled,
        "profile_total": int(measured["profile_count"]),
        "profile_accepted": accepted,
        "gamma": 2.20,
        "write_pixels": int(write_support.sum()),
        "write_fraction": float(write_support.mean()),
        "mean_abs_y_delta": float(np.mean(np.abs(correction[write_support]))) if np.any(write_support) else 0.0,
        "p95_abs_y_delta": float(np.percentile(np.abs(correction[write_support]), 95)) if np.any(write_support) else 0.0,
        "_v2_raw_mask": raw_mask,
        "_v2_clean_mask": clean_mask,
        "_v2_write_support": write_support,
        "_v2_delta_y": correction,
    }


def _apply_p11_edge_narrow_v3(candidate_yuv: np.ndarray, valid_support: np.ndarray | None = None, use_directional: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply the V3 Y S-curve adapter with raw or directional mask geometry.

    Input: a CHW YUV444 prediction, optional valid-domain mask, and a directional-mask switch.
    Output: corrected YUV444 plus raw, directional, and write-support telemetry at line resolution.
    Key Steps: segment text, optionally clean tangent-direction defects, measure edge anchors, and write only accepted Y transitions.
    Key Params: ``use_directional`` chooses raw versus cleaned mask geometry; ``valid_support`` excludes padded prediction pixels.
    State/Dependencies: reads frozen P10/P11 helpers and changes only supported Y samples in a new output array.
    """
    candidate_rgb = yuv444_to_rgb(candidate_yuv)
    y = _rgb_to_luma(candidate_rgb)
    valid_support = np.ones_like(y, dtype=bool) if valid_support is None else np.asarray(valid_support, dtype=bool)
    raw_mask, segmentation = _p10_segment_text(y, _P11_FROZEN_PARAMS, valid_support)
    if raw_mask is None:
        raw_mask = np.asarray(segmentation.get("_p10_low_class", np.zeros_like(y, dtype=bool)), dtype=bool) & valid_support
    if not raw_mask.any():
        empty = np.zeros_like(y, dtype=bool)
        return candidate_yuv, {
            "postprocess": "v3_skipped_no_accepted_p10_mask",
            "_v3_raw_mask": empty,
            "_v3_clean_mask": empty,
            "_v3_edge_candidate": empty,
            "_v3_write_support": empty,
            "_v3_delta_y": np.zeros_like(y, dtype=np.float32),
        }
    raw_mask = np.asarray(raw_mask, dtype=bool) & valid_support
    raw_u8 = raw_mask.astype(np.uint8)
    boundary = cv2.morphologyEx(raw_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    tangent_gx = cv2.Sobel(raw_u8.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    tangent_gy = cv2.Sobel(raw_u8.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    clean_mask = raw_mask.copy()
    removed = 0
    filled = 0
    tangent_offsets = np.asarray((-3.0, -2.0, -1.0, 1.0, 2.0, 3.0), dtype=np.float32)
    for row, column in np.argwhere(boundary) if use_directional else ():
        magnitude = float(np.hypot(tangent_gx[row, column], tangent_gy[row, column]))
        if magnitude <= 1.0e-6:
            continue
        tangent_x = -float(tangent_gy[row, column]) / magnitude
        tangent_y = float(tangent_gx[row, column]) / magnitude
        samples = cv2.remap(
            raw_u8.astype(np.float32),
            (float(column) + tangent_offsets * tangent_x).reshape(1, -1),
            (float(row) + tangent_offsets * tangent_y).reshape(1, -1),
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1) >= 0.5
        if raw_mask[row, column] and not np.any(samples):
            clean_mask[row, column] = False
            removed += 1
        elif not raw_mask[row, column] and bool(np.all(samples)):
            clean_mask[row, column] = True
            filled += 1

    clean_u8 = clean_mask.astype(np.uint8)
    inside_distance = cv2.distanceTransform(clean_u8, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~clean_mask).astype(np.uint8), cv2.DIST_L2, 5)
    signed = inside_distance - outside_distance
    edge_candidate = (np.abs(signed) <= 12.0) & valid_support
    normal_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    normal_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    normal_length = np.hypot(normal_x, normal_y)
    normal_valid = edge_candidate & (normal_length > 1.0e-6)
    normal_x = np.divide(normal_x, normal_length, out=np.zeros_like(normal_x), where=normal_length > 1.0e-6)
    normal_y = np.divide(normal_y, normal_length, out=np.zeros_like(normal_y), where=normal_length > 1.0e-6)

    out_y = y.copy()
    write_support = np.zeros_like(clean_mask, dtype=bool)
    anchor_class_valid = np.zeros_like(clean_mask, dtype=bool)
    contrast_valid = np.zeros_like(clean_mask, dtype=bool)
    q_valid = np.zeros_like(clean_mask, dtype=bool)
    local_q = np.full_like(y, np.nan, dtype=np.float32)
    local_background = np.zeros_like(y, dtype=np.float32)
    local_foreground = np.zeros_like(y, dtype=np.float32)
    sample_background_distances = np.asarray((12.0, 16.0, 20.0, 24.0), dtype=np.float32)
    sample_foreground_distances = np.asarray((12.0, 16.0, 20.0, 24.0), dtype=np.float32)
    for row, column in np.argwhere(normal_valid):
        nx = float(normal_x[row, column])
        ny = float(normal_y[row, column])
        background_x = float(column) - nx * sample_background_distances
        background_y = float(row) - ny * sample_background_distances
        foreground_x = float(column) + nx * sample_foreground_distances
        foreground_y = float(row) + ny * sample_foreground_distances
        sampled_valid = cv2.remap(valid_support.astype(np.float32), np.concatenate((background_x, foreground_x)).reshape(1, -1), np.concatenate((background_y, foreground_y)).reshape(1, -1), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1) >= .5
        if not np.all(sampled_valid):
            continue
        background_class = cv2.remap(
            clean_u8.astype(np.float32), background_x.reshape(1, -1), background_y.reshape(1, -1),
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
        ).reshape(-1) >= 0.5
        foreground_class = cv2.remap(
            clean_u8.astype(np.float32), foreground_x.reshape(1, -1), foreground_y.reshape(1, -1),
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
        ).reshape(-1) >= 0.5
        if np.any(background_class) or not np.all(foreground_class):
            continue
        anchor_class_valid[row, column] = True
        background_samples = cv2.remap(
            y, background_x.reshape(1, -1), background_y.reshape(1, -1),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        ).reshape(-1)
        foreground_samples = cv2.remap(
            y, foreground_x.reshape(1, -1), foreground_y.reshape(1, -1),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        ).reshape(-1)
        background_median = float(np.median(background_samples))
        foreground_median = float(np.median(foreground_samples))
        if foreground_median > background_median:
            background_value = float(np.min(background_samples))
            foreground_value = float(np.max(foreground_samples))
        else:
            background_value = float(np.max(background_samples))
            foreground_value = float(np.min(foreground_samples))
        difference = foreground_value - background_value
        if abs(difference) < 0.12:
            continue
        contrast_valid[row, column] = True
        q = float(np.clip((y[row, column] - background_value) / difference, 0.0, 1.0))
        local_q[row, column] = q
        if not 0.01 < q < 0.99:
            continue
        q_valid[row, column] = True
        numerator = q ** 2.40
        shaped = numerator / max(numerator + (1.0 - q) ** 2.40, 1.0e-12)
        out_y[row, column] = np.clip(background_value + difference * shaped, 0.0, 1.0)
        write_support[row, column] = True
        local_background[row, column] = background_value
        local_foreground[row, column] = foreground_value

    output_rgb, _ = _p10_reconstruct_rgb_from_y(candidate_rgb, y, out_y)
    output_yuv = rgb_to_yuv444(output_rgb)
    result = candidate_yuv.copy()
    result[0, write_support] = output_yuv[0, write_support]
    delta_y = out_y - y
    widths_before: list[float] = []
    widths_after: list[float] = []
    evaluation_offsets = np.arange(-12.0, 12.25, 0.25, dtype=np.float32)
    for row, column in np.argwhere(q_valid):
        nx = float(normal_x[row, column])
        ny = float(normal_y[row, column])
        background_value = float(local_background[row, column])
        difference = float(local_foreground[row, column] - background_value)
        if abs(difference) < 1.0e-6:
            continue
        map_x = (float(column) + nx * evaluation_offsets).reshape(1, -1)
        map_y = (float(row) + ny * evaluation_offsets).reshape(1, -1)
        for sampled, collector in ((y, widths_before), (out_y, widths_after)):
            curve = cv2.remap(sampled, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1)
            normalized = (curve - background_value) / difference
            crossings: list[float] = []
            for target in (0.10, 0.90):
                delta = normalized - target
                indices = np.flatnonzero(delta[:-1] * delta[1:] <= 0.0)
                if indices.size == 0:
                    crossings = []
                    break
                index = int(indices[np.argmin(np.abs(evaluation_offsets[indices]))])
                denominator = float(normalized[index + 1] - normalized[index])
                fraction = 0.0 if abs(denominator) <= 1.0e-6 else float((target - normalized[index]) / denominator)
                crossings.append(float(evaluation_offsets[index] + 0.25 * fraction))
            if len(crossings) == 2:
                collector.append(abs(crossings[1] - crossings[0]))
    pre_width = float(np.median(widths_before)) if widths_before else 0.0
    post_width = float(np.median(widths_after)) if widths_after else 0.0
    return result, {
        "postprocess": "p11_edge_narrow_v3_1",
        "p10_segmentation_accepted": bool(segmentation.get("segmentation_accepted", False)),
        "p10_polarity": segmentation.get("polarity"),
        "raw_mask_pixels": int(raw_mask.sum()),
        "directional_spike_removed": removed,
        "directional_notch_filled": filled,
        "clean_mask_pixels": int(clean_mask.sum()),
        "edge_candidate_pixels": int(edge_candidate.sum()),
        "normal_valid_pixels": int(normal_valid.sum()),
        "anchor_class_valid_pixels": int(anchor_class_valid.sum()),
        "contrast_valid_pixels": int(contrast_valid.sum()),
        "q_transition_pixels": int(q_valid.sum()),
        "write_pixels": int(write_support.sum()),
        "write_fraction_of_edge_candidate": float(write_support.sum() / max(int(edge_candidate.sum()), 1)),
        "write_inside_clean_pixels": int(np.count_nonzero(write_support & clean_mask)),
        "write_outside_clean_pixels": int(np.count_nonzero(write_support & ~clean_mask)),
        "mean_abs_y_delta": float(np.mean(np.abs(delta_y[write_support]))) if np.any(write_support) else 0.0,
        "p95_abs_y_delta": float(np.percentile(np.abs(delta_y[write_support]), 95)) if np.any(write_support) else 0.0,
        "median_pre_10_90_width": pre_width,
        "median_post_10_90_width": post_width,
        "median_10_90_width_reduction": pre_width - post_width,
        "background_platform_pixels": int(np.count_nonzero(contrast_valid & (local_q <= 0.01))),
        "background_platform_y_changed_pixels": int(np.count_nonzero(
            (result[0] != candidate_yuv[0]) & contrast_valid & (local_q <= 0.01)
        )),
        "foreground_platform_pixels": int(np.count_nonzero(contrast_valid & (local_q >= 0.99))),
        "foreground_platform_y_changed_pixels": int(np.count_nonzero(
            (result[0] != candidate_yuv[0]) & contrast_valid & (local_q >= 0.99)
        )),
        "outside_write_y_changed_pixels": int(np.count_nonzero(result[0, ~write_support] != candidate_yuv[0, ~write_support])),
        "uv_changed_pixels": int(np.count_nonzero(result[1:] != candidate_yuv[1:])),
        "_v3_raw_mask": raw_mask,
        "_v3_clean_mask": clean_mask,
        "_v3_edge_candidate": edge_candidate,
        "_v3_anchor_class_valid": anchor_class_valid,
        "_v3_q_transition": q_valid,
        "_v3_write_support": write_support,
        "_v3_delta_y": delta_y,
        "_v3_local_q": local_q,
    }


def _select_pasteback_alpha(
    postprocess: str,
    line_alpha: np.ndarray,
    valid: np.ndarray,
    postprocess_record: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Select legacy or mask-only alpha for one OCR line pasteback.

    Input: postprocess mode, legacy floating alpha, valid prediction support, and optional postprocess telemetry.
    Output: a float32 alpha map plus telemetry that identifies the selected support domain.
    Key Steps: keep legacy feather alpha by default; for V5 comparison modes intersect the selected binary P10 mask with valid support.
    Key Params: the two V5 mode names map only to `_v3_raw_mask` or `_v3_clean_mask`.
    State/Dependencies: pure NumPy selection; it does not read source pixels or alter ownership behavior.
    """
    valid = np.asarray(valid, dtype=bool)
    legacy_alpha = np.asarray(line_alpha, dtype=np.float32) * valid.astype(np.float32)
    key_by_mode = {
        "p11_valid_maskraw_compare_v5": ("_v3_raw_mask", "raw"),
        "p11_valid_maskdirectional_compare_v5": ("_v3_clean_mask", "directional"),
    }
    selection = key_by_mode.get(postprocess)
    if selection is None:
        return legacy_alpha, {
            "selected_mask_kind": "legacy_feather",
            "selected_mask_pixels": int(np.count_nonzero(legacy_alpha)),
            "pasteback_alpha_pixels": int(np.count_nonzero(legacy_alpha)),
            "merged_alpha_used": True,
        }
    if postprocess_record is None or selection[0] not in postprocess_record:
        raise ValueError(f"{postprocess} did not provide its selected mask")
    selected_mask = np.asarray(postprocess_record[selection[0]], dtype=bool)
    if selected_mask.shape != valid.shape:
        raise ValueError("selected postprocess mask shape mismatch")
    mask_only_alpha = (selected_mask & valid).astype(np.float32)
    return mask_only_alpha, {
        "selected_mask_kind": selection[1],
        "selected_mask_pixels": int(selected_mask.sum()),
        "pasteback_alpha_pixels": int(mask_only_alpha.sum()),
        "merged_alpha_used": False,
    }


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write image: {path}")


def _write_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.rint(np.asarray(value, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"failed to write mask: {path}")


def _p17_textmask_fallback(mask_path: Path, image_hw: tuple[int, int]) -> list[tuple[OCRLine, np.ndarray]]:
    """Extract text-line supports from a validation TextMask without OCR detection."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image_hw:
        raise ValueError(f"invalid P17 text mask: {mask_path}")
    binary = (mask > 127).astype(np.uint8)
    # Join characters within a line, retaining the original mask as alpha.
    joined = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    lines: list[tuple[OCRLine, np.ndarray]] = []
    for label in range(1, count):
        left, top, width, height, _ = stats[label]
        component = (labels == label) & (binary > 0)
        if component.sum() < 12 or width < 2 or height < 2:
            continue
        right, bottom = left + width, top + height
        polygon = np.asarray([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)
        lines.append((OCRLine(len(lines), polygon, 1.0, "", "text_mask"), component.astype(np.float32)))
    return lines


def rgb_to_nv21(rgb: np.ndarray) -> np.ndarray:
    """Convert an even RGB image to normalized packed NV21 ``[1,H+H/2,W]``."""
    image = np.asarray(rgb, dtype=np.float32)
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("NV21 input requires even height and width")
    yuv = cv2.cvtColor(np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    u = cv2.resize(yuv[..., 1], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(yuv[..., 2], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    vu = np.empty((height // 2, width), dtype=np.uint8)
    vu[:, 0::2] = v
    vu[:, 1::2] = u
    packed = np.concatenate([yuv[..., 0], vu], axis=0).astype(np.float32) / 255.0
    return packed[None]


def rgb_to_yuv444(rgb: np.ndarray) -> np.ndarray:
    image = np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV).astype(np.float32).transpose(2, 0, 1) / 255.0


def yuv444_to_rgb(yuv_chw: np.ndarray) -> np.ndarray:
    image = np.asarray(yuv_chw, dtype=np.float32).transpose(1, 2, 0)
    rgb = cv2.cvtColor(np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_YUV2RGB)
    return rgb.astype(np.float32) / 255.0


def load_lq_preprocess_config(config_path: Path | None) -> dict[str, Any]:
    """Load the training-time LQ adapter settings for checkpoint-consistent inference."""
    if config_path is None:
        return normalize_lq_preprocess_config(None)
    with Path(config_path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"training config must be a mapping: {config_path}")
    return normalize_lq_preprocess_config(document.get("lq_preprocess"))


class NativeY4UVTileRunner:
    """Run a V4/V5 native six-channel checkpoint within the OCR compositor."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        lq_preprocess_config: dict[str, Any] | None = None,
    ) -> None:
        """@luna-comment
        Input: Evaluation network, torch device and optional LQ preprocessing settings.
        Output: Initialized runner with normalized settings and zero call counter.
        Key Steps: Retain network/device and normalize the preprocessing contract.
        Key Params: Disabled preprocessing lets explicit external line processing pass through.
        State/Dependencies: Stores network/configuration references and mutable call count.
        """
        self.network = network
        self.device = device
        self.lq_preprocess_config = normalize_lq_preprocess_config(lq_preprocess_config)
        self.calls = 0

    def run_rgb(self, canvas_rgb: np.ndarray) -> np.ndarray:
        """@luna-comment Run one RGB canvas through the V5 adapter and return its YUV444 prediction.

        Input: a normalized HWC RGB canvas and the runner's fixed network/configuration.
        Output: a CHW float YUV444 prediction for the existing OCR-line merger.
        Key Steps: create a device tensor, apply LQ preprocessing only when enabled, pack Y4UV, run inference, and convert back to YUV444.
        Key Params: ``lq_preprocess_config[\"enabled\"]`` preserves exact input identity when disabled.
        State/Dependencies: increments the runner call counter and uses the configured torch network/device.
        """
        original_rgb = torch.from_numpy(np.ascontiguousarray(canvas_rgb)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        preprocessed_rgb = (
            preprocess_lq_rgb(original_rgb, self.lq_preprocess_config)
            if self.lq_preprocess_config["enabled"] else original_rgb
        )
        native = rgb_to_y4uv(preprocessed_rgb)
        with torch.inference_mode():
            if self.lq_preprocess_config["enabled"] and self.lq_preprocess_config["residual_input_source"] == "original":
                original = rgb_to_y4uv(original_rgb)
                output = self.network(original, skip_input_y4uv=native)
            else:
                output = self.network(native)
        yuv444 = y4uv_to_yuv444(output.detach().cpu())
        rgb = YUV2RGB(yuv444, format="RGB", is_train=False, uv_normalized=True)[0]
        output_rgb = np.transpose(rgb.numpy(), (1, 2, 0)).astype(np.float32)
        self.calls += 1
        return rgb_to_yuv444(output_rgb)


def _pad_canvas(rgb_crop: np.ndarray, canvas_hw: tuple[int, int], edge_replicate: bool = False) -> np.ndarray:
    """@luna-comment
    Input: Float HWC RGB crop, target height/width and padding mode.
    Output: Target-size float canvas; rejects a crop exceeding the target.
    Key Steps: Replicate bottom/right edges or place the crop in a zero canvas.
    Key Params: edge_replicate selects border padding for height-normalized tiles.
    State/Dependencies: OpenCV/NumPy only; source pixels are not mutated.
    """
    canvas_h, canvas_w = map(int, canvas_hw)
    height, width = rgb_crop.shape[:2]
    if height > canvas_h or width > canvas_w:
        raise ValueError(f"crop {rgb_crop.shape[:2]} exceeds model canvas {canvas_hw}")
    if edge_replicate:
        return cv2.copyMakeBorder(rgb_crop, 0, canvas_h - height, 0, canvas_w - width, cv2.BORDER_REPLICATE)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas[:height, :width] = rgb_crop
    return canvas


def _resize_hwc(image: np.ndarray, target_hw: tuple[int, int], *, interpolation: int | None = None) -> np.ndarray:
    """@luna-comment Resize one float HWC RGB/YUV tile to an explicit ``(H, W)``.

    Input: an HWC float tile and positive target height/width; an optional OpenCV
    interpolation constant overrides the normal area-downsample/linear-upsample rule.
    Output: a float32 HWC tile of the requested shape without shared-state changes.
    Key Steps: retain exact-shape copies, select the requested/default kernel, then resize.
    Key Params: ``interpolation`` is used by experimental prediction restoration only.
    State/Dependencies: OpenCV resize semantics.
    """
    source = np.asarray(image, dtype=np.float32)
    target_h, target_w = map(int, target_hw)
    if source.shape[:2] == (target_h, target_w):
        return source.copy()
    selected = interpolation if interpolation is not None else (
        cv2.INTER_AREA if target_h <= source.shape[0] and target_w <= source.shape[1] else cv2.INTER_LINEAR
    )
    return cv2.resize(source, (target_w, target_h), interpolation=selected).astype(np.float32, copy=False)


def _resize_chw(image: np.ndarray, target_hw: tuple[int, int], *, interpolation: int | None = None) -> np.ndarray:
    """@luna-comment Restore a float CHW model prediction to a target spatial shape.

    Input: three-dimensional channel-first prediction and source-frame target size.
    Output: channel-first float32 prediction with unchanged channel count.
    Key Steps: validate CHW rank, transpose to HWC, delegate interpolation, transpose back.
    Key Params: ``interpolation`` selects the prediction restoration kernel.
    State/Dependencies: delegates to ``_resize_hwc`` only.
    """
    chw = np.asarray(image, dtype=np.float32)
    if chw.ndim != 3:
        raise ValueError("CHW image expected")
    return _resize_hwc(chw.transpose(1, 2, 0), target_hw, interpolation=interpolation).transpose(2, 0, 1)


def _load_network(
    checkpoint: Path,
    device: torch.device,
    network_config: Path | None = None,
) -> tuple[torch.nn.Module, str]:
    """@guang-yang-comment Build baseline or YAML-selected registry network and strict-load it.

    Input: checkpoint, target device, and optional training YAML. Output: eval-mode
    network and checkpoint digest. Key Steps: resolve ``network_g`` through the
    shared registry builder, then enforce strict state compatibility. Key Params:
    absent config retains baseline behavior. State/Dependencies: torch registry.
    """

    return build_eval_network(checkpoint, device, network_config)


def _tile_debug(
    debug_dir: Path,
    segment: FullFrameSegment,
    input_canvas: np.ndarray,
    output_canvas: np.ndarray,
) -> None:
    """@luna-comment
    Input: Debug directory, segment identity, RGB input canvas and CHW YUV prediction.
    Output: Input/output and paired PNGs under the segment's line directory.
    Key Steps: Decode prediction color and publish same-coordinate panels.
    Key Params: Line and tile indices bind stable diagnostic filenames.
    State/Dependencies: Uses existing color/write helpers and writes debug artifacts.
    """
    line_dir = debug_dir / f"line_{segment.line_index:04d}"
    stem = f"tile_{segment.tile_index:02d}"
    output_rgb = yuv444_to_rgb(output_canvas)
    _write_rgb(line_dir / f"{stem}_input.png", input_canvas)
    _write_rgb(line_dir / f"{stem}_output.png", output_rgb)
    _write_rgb(line_dir / f"{stem}_comparison.png", np.concatenate([input_canvas, output_rgb], axis=1))


def _native_grid_weight(segment: FullFrameSegment, segments: list[FullFrameSegment]) -> np.ndarray:
    """@luna-comment
    Input: One native segment and its neighboring segments in the same image.
    Output: Float32 two-dimensional overlap weight for the segment's valid crop.
    Key Steps: Measure four side overlaps and multiply complementary linear ramps.
    Key Params: Half-pixel sampling supplies nonzero overlap-boundary weights.
    State/Dependencies: Reads segment geometry; NumPy only, no input mutation.
    """
    height, width = segment.valid_hw
    left, top, right, bottom = segment.crop_xyxy
    x_weight = np.ones(width, dtype=np.float32)
    y_weight = np.ones(height, dtype=np.float32)

    def _overlap_on_side(axis: str, before: bool) -> int:
        """@luna-comment
        Input: Horizontal/vertical axis and preceding/following side selector.
        Output: Maximum geometrically intersecting overlap width, or zero.
        Key Steps: Reject nonintersecting neighbors and measure the selected side.
        Key Params: Axis and side select one of four crop-boundary ramps.
        State/Dependencies: Reads the enclosing segment coordinates and neighbor list.
        """
        values: list[int] = []
        for other in segments:
            if other is segment:
                continue
            o_left, o_top, o_right, o_bottom = other.crop_xyxy
            if axis == "x":
                if min(bottom, o_bottom) <= max(top, o_top):
                    continue
                if before and o_left < left < o_right:
                    values.append(o_right - left)
                if not before and o_left < right < o_right:
                    values.append(right - o_left)
            else:
                if min(right, o_right) <= max(left, o_left):
                    continue
                if before and o_top < top < o_bottom:
                    values.append(o_bottom - top)
                if not before and o_top < bottom < o_bottom:
                    values.append(bottom - o_top)
        return max(values, default=0)

    left_overlap, right_overlap = _overlap_on_side("x", True), _overlap_on_side("x", False)
    top_overlap, bottom_overlap = _overlap_on_side("y", True), _overlap_on_side("y", False)
    if left_overlap:
        x_weight *= np.clip((np.arange(width, dtype=np.float32) + .5) / float(left_overlap), 0.0, 1.0)
    if right_overlap:
        x_weight *= np.clip((width - np.arange(width, dtype=np.float32) - .5) / float(right_overlap), 0.0, 1.0)
    if top_overlap:
        y_weight *= np.clip((np.arange(height, dtype=np.float32) + .5) / float(top_overlap), 0.0, 1.0)
    if bottom_overlap:
        y_weight *= np.clip((height - np.arange(height, dtype=np.float32) - .5) / float(bottom_overlap), 0.0, 1.0)
    return y_weight[:, None] * x_weight[None, :]


def _refine_ocr_core_to_glyph_support(crop_rgb: np.ndarray, ocr_core: np.ndarray) -> np.ndarray:
    """@luna-comment
    Input: Float HWC RGB crop and aligned OCR-core support.
    Output: Float local-contrast support confined to the original OCR core.
    Key Steps: Estimate local contrast, threshold, dilate and check occupancy.
    Key Params: Crop-scaled Gaussian sigma and bounded occupancy preserve weak support.
    State/Dependencies: OpenCV/NumPy only; unsupported cases retain the OCR core.

    Restrict a rectangular OCR polygon to locally contrasted glyph pixels.

    OCR polygons describe a line envelope, not its actual ink.  Treating that
    envelope as opaque lets a model's low-frequency background change be pasted
    back as a visible rectangle.  A local-luma contrast mask keeps the opaque
    region on glyphs; the existing feather still supplies the narrow context
    ring that restoration needs around blurred strokes.
    """
    core = (np.asarray(ocr_core, dtype=np.float32) > 0.5).astype(np.uint8)
    if not np.any(core):
        return core.astype(np.float32)
    image = np.clip(np.asarray(crop_rgb, dtype=np.float32), 0.0, 1.0)
    luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sigma = float(np.clip(min(luma.shape) / 12.0, 2.0, 8.0))
    local_contrast = np.abs(luma - cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma))
    values = local_contrast[core > 0]
    if values.size < 8:
        return core.astype(np.float32)
    otsu, _ = cv2.threshold(
        np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    threshold = max(2.0 / 255.0, float(otsu) / 255.0)
    support = ((local_contrast >= threshold) & (core > 0)).astype(np.uint8)
    # Preserve antialiased contours and avoid pinhole alpha changes, while
    # never extending the opaque ownership beyond the OCR polygon.
    support = cv2.dilate(support, np.ones((3, 3), dtype=np.uint8), iterations=1) & core
    occupancy = float(support.sum()) / float(core.sum())
    if occupancy < 0.002 or occupancy > 0.72:
        return core.astype(np.float32)
    return support.astype(np.float32)


def _aabb_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return AABB IoU for two OCR polygons."""
    first_min, first_max = first.min(axis=0), first.max(axis=0)
    second_min, second_max = second.min(axis=0), second.max(axis=0)
    inter_w = max(0.0, min(float(first_max[0]), float(second_max[0])) - max(float(first_min[0]), float(second_min[0])))
    inter_h = max(0.0, min(float(first_max[1]), float(second_max[1])) - max(float(first_min[1]), float(second_min[1])))
    intersection = inter_w * inter_h
    first_area = max(0.0, float(first_max[0] - first_min[0])) * max(0.0, float(first_max[1] - first_min[1]))
    second_area = max(0.0, float(second_max[0] - second_min[0])) * max(0.0, float(second_max[1] - second_min[1]))
    return intersection / max(first_area + second_area - intersection, 1e-12)


def _polygon_angle_degrees(polygon: np.ndarray) -> float:
    """Return the principal quadrilateral direction, folded into [-90, 90]."""
    points = np.asarray(polygon, dtype=np.float32)
    edges = np.roll(points, -1, axis=0) - points
    vector = edges[np.argmax(np.sum(edges * edges, axis=1))]
    angle = float(np.degrees(np.arctan2(float(vector[1]), float(vector[0]))))
    return ((angle + 90.0) % 180.0) - 90.0


def _clip_ocr_polygon(raw: Any, frame_hw: tuple[int, int]) -> np.ndarray | None:
    """Normalize one detector polygon while retaining the detector record index."""
    try:
        from realesrgan.data.basic_text_line_full_frame import normalize_ocr_polygon
        polygon = normalize_ocr_polygon(raw).astype(np.float32, copy=True)
    except (TypeError, ValueError):
        return None
    height, width = frame_hw
    polygon[:, 0] = np.clip(polygon[:, 0], 0.0, max(0.0, width - 1.0))
    polygon[:, 1] = np.clip(polygon[:, 1], 0.0, max(0.0, height - 1.0))
    if np.any(np.ptp(polygon, axis=0) < 1.0):
        return None
    return polygon


def _dense_group_detector_contract(data: Any, frame_hw: tuple[int, int]) -> tuple[dict[int, dict[str, Any]], str]:
    """Read the optional one-to-one DB detector geometry contract."""
    if not isinstance(data, dict):
        return {}, "OCR_NOT_MAPPING"
    fields = ("core_polys", "unclip_polys", "dt_polys")
    if any(not isinstance(data.get(name), list) for name in fields):
        return {}, "DETECTOR_CONTRACT_UNAVAILABLE"
    if len({len(data[name]) for name in fields}) != 1:
        return {}, "DETECTOR_CONTRACT_LENGTH_MISMATCH"
    scores = data.get("dt_scores", [])
    contract: dict[int, dict[str, Any]] = {}
    for index, (core_raw, unclip_raw, final_raw) in enumerate(zip(data["core_polys"], data["unclip_polys"], data["dt_polys"])):
        core = _clip_ocr_polygon(core_raw, frame_hw)
        unclip = _clip_ocr_polygon(unclip_raw, frame_hw)
        final = _clip_ocr_polygon(final_raw, frame_hw)
        if core is None or unclip is None or final is None:
            continue
        try:
            score = float(scores[index]) if index < len(scores) else 1.0
        except (TypeError, ValueError):
            score = 1.0
        contract[index] = {"core": core, "unclip": unclip, "final": final, "score": max(0.0, score) if np.isfinite(score) else 1.0}
    return contract, "AVAILABLE"


def _aabb(poly: np.ndarray) -> tuple[float, float, float, float]:
    low, high = np.min(poly, axis=0), np.max(poly, axis=0)
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


def _interval_overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _plan_dense_small_text_groups(data: Any, frame_hw: tuple[int, int], min_score: float = 0.40) -> tuple[list[dict[str, Any]], str]:
    """Group horizontal cross-row unclip collisions without suppressing others.

    Component membership is decided here.  Probability analysis subsequently
    estimates patch scale, and cannot veto sparse or columnar group members.
    """
    contract, status = _dense_group_detector_contract(data, frame_hw)
    if not contract:
        return [], status
    entries = []
    for index, item in contract.items():
        core_box, unclip_box, final_box = _aabb(item["core"]), _aabb(item["unclip"]), _aabb(item["final"])
        core_h, unclip_h = core_box[3] - core_box[1], unclip_box[3] - unclip_box[1]
        angle = _polygon_angle_degrees(item["core"])
        if item["score"] < min_score or abs(angle) > 10.0 or core_h <= 1.0 or unclip_h <= 1.0:
            continue
        entries.append({"index": index, **item, "core_box": core_box, "unclip_box": unclip_box, "final_box": final_box,
                        "core_h": core_h, "angle": angle, "center_y": (core_box[1] + core_box[3]) * .5})
    entries.sort(key=lambda item: item["center_y"])
    groups: list[dict[str, Any]] = []
    # Cross-row unclip collisions form the routing graph.  Do not require a
    # common horizontal centre: parameter tables have short, offset columns.
    adjacency = {i: set() for i in range(len(entries))}
    for i, previous in enumerate(entries):
        for j in range(i + 1, len(entries)):
            item = entries[j]
            first, second = previous["unclip_box"], item["unclip_box"]
            overlap_y = _interval_overlap((first[1], first[3]), (second[1], second[3]))
            overlap_x = _interval_overlap((first[0], first[2]), (second[0], second[2]))
            min_h = min(first[3] - first[1], second[3] - second[1])
            min_w = min(first[2] - first[0], second[2] - second[0])
            delta = item["center_y"] - previous["center_y"]
            ratio = item["core_h"] / previous["core_h"]
            if (delta >= .25 * min(previous["core_h"], item["core_h"])
                    and .67 <= ratio <= 1.50 and overlap_y >= .30 * min_h
                    and overlap_x >= .35 * min_w and abs(item["angle"] - previous["angle"]) <= 5.0):
                adjacency[i].add(j); adjacency[j].add(i)
    seen: set[int] = set()
    for root in adjacency:
        if root in seen:
            continue
        stack, component = [root], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node); component.append(entries[node]); stack.extend(adjacency[node] - seen)
        component.sort(key=lambda item: item["center_y"])
        heights = np.asarray([item["core_h"] for item in component], np.float32)
        median_h = float(np.median(heights))
        row_centers: list[float] = []
        for item in component:
            if not row_centers or item["center_y"] - row_centers[-1] > .35 * median_h:
                row_centers.append(item["center_y"])
        if len(row_centers) < 3:
            continue
        groups.append({"member_indices": [item["index"] for item in component], "members": component,
                       "core_thickness": median_h, "row_centers": row_centers,
                       "center_spacing": float(np.median(np.diff(row_centers))),
                       "confidence_min": float(min(item["score"] for item in component)),
                       "confidence_mean": float(np.mean([item["score"] for item in component])),
                       "group_trigger": "cross_row_unclip_overlap"})
    return groups, status

def _suppress_generic_ocr_duplicates(lines: list[OCRLine], iou_threshold: float) -> tuple[list[OCRLine], int]:
    """Keep one high-confidence OCR line from duplicate detector contours."""
    if iou_threshold <= 0.0:
        return lines, 0
    ranked = sorted(
        lines,
        key=lambda line: (-float(line.score), -float(np.prod(np.ptp(line.polygon_xy, axis=0))), int(line.line_index)),
    )
    kept: list[OCRLine] = []
    for line in ranked:
        if any(_aabb_iou(line.polygon_xy, existing.polygon_xy) >= iou_threshold for existing in kept):
            continue
        kept.append(line)
    return sorted(kept, key=lambda line: line.line_index), len(lines) - len(kept)


def _line_laplacian_variance(frame_rgb: np.ndarray, line: OCRLine) -> float:
    """Measure local high-frequency energy inside an OCR line envelope."""
    height, width = frame_rgb.shape[:2]
    minimum = np.floor(line.polygon_xy.min(axis=0)).astype(np.int32)
    maximum = np.ceil(line.polygon_xy.max(axis=0)).astype(np.int32)
    x0, y0 = np.clip(minimum, [0, 0], [width - 1, height - 1])
    x1, y1 = np.clip(maximum, [x0 + 1, y0 + 1], [width, height])
    crop = cv2.cvtColor(frame_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if crop.size < 9:
        return 0.0
    return float(cv2.Laplacian(crop, cv2.CV_32F).var())


def standard_polygon_paste_alpha(frame_rgb: np.ndarray, polygon_xy: np.ndarray, crop_xyxy: tuple[int, int, int, int], blend_radius: int) -> np.ndarray:
    """Build the canonical glyph-support alpha used by every inference route."""
    left, top, right, bottom = crop_xyxy
    crop = frame_rgb[top:bottom, left:right]
    local_polygon = np.asarray(polygon_xy, np.float32) - np.asarray([left, top], np.float32)
    core = rasterize_polygon_mask(local_polygon, crop.shape[:2])
    return feather_alpha(_refine_ocr_core_to_glyph_support(crop, core), int(blend_radius))


def standard_pasteback(frame_yuv: np.ndarray, output_yuv: np.ndarray, owner_score: np.ndarray, crop_xyxy: tuple[int, int, int, int], prediction_yuv: np.ndarray, alpha: np.ndarray, score: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """@luna-comment
    Input: Base/output YUV canvases, ownership scores, crop box, candidate patch, alpha, and priority score.
    Output: Candidate crop, update mask, and alpha; mutates output and owner arrays at eligible pixels.
    Key Steps: Blend candidate with the base, then write where alpha-weighted priority wins ownership.
    Key Params: The 1e-6 eligibility margin and score floor make ownership deterministic.
    State/Dependencies: NumPy-only composition; output_yuv and owner_score are shared mutable state.
    """
    left, top, right, bottom = crop_xyxy
    base = frame_yuv[:, top:bottom, left:right]
    candidate = alpha[None] * prediction_yuv + (1.0 - alpha[None]) * base
    owner = owner_score[top:bottom, left:right]
    update = (alpha > 1e-6) & (alpha * max(.01, float(score)) > owner + 1e-6)
    target = output_yuv[:, top:bottom, left:right]
    target[:, update] = candidate[:, update]
    owner[update] = alpha[update] * max(.01, float(score))
    return candidate, update, alpha


def _apply_pasteback_y_scurve(candidate_yuv: np.ndarray, source_yuv: np.ndarray, update: np.ndarray, alpha: np.ndarray, *, mode: str, gamma: float, strength: float, max_delta: float) -> tuple[np.ndarray, dict[str, float | str]]:
    """@luna-comment
    Input: Candidate/source YUV crops, write mask, alpha, and selected postprocess parameters.
    Output: Corrected YUV crop with mode and magnitude diagnostics.
    Key Steps: Route the requested foreground-only edge or subpixel contrast correction through support guards.
    Key Params: Mode selects behavior; gamma, strength, and max_delta bound the selected tone curve.
    State/Dependencies: Uses OpenCV and NumPy; does not mutate the supplied crops.
    """
    if mode == "none":
        return candidate_yuv, {"postprocess": "none", "postprocess_mean_abs_delta": 0.0}
    y = candidate_yuv[0].astype(np.float32, copy=False)
    if mode == "source_edge":
        source_y = source_yuv[0].astype(np.float32, copy=False)
        core = alpha >= 0.75
        _, binary = cv2.threshold(np.rint(np.clip(source_y, 0, 1) * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = binary > 0
        foreground = bright if np.count_nonzero(bright & core) >= np.count_nonzero((~bright) & core) else ~bright
        inner = cv2.erode(foreground.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        edge = foreground & ~inner
        result = candidate_yuv.copy()
        result[0, update & edge] = source_y[update & edge]
        return result, {"postprocess": "source_edge", "postprocess_mean_abs_delta": float(np.mean(np.abs(y[update & edge] - source_y[update & edge]))) if np.any(update & edge) else 0.0}
    if mode == "subpixel_transition":
        # Preserve the candidate's existing edge coverage.  This is a colour
        # repair, not an edge generator: no distance field, dilation, or new
        # transition is allowed to move or broaden a glyph contour.
        candidate_rgb_hwc = yuv444_to_rgb(candidate_yuv)
        gray = cv2.cvtColor(np.clip(np.rint(candidate_rgb_hwc * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        candidate_rgb = candidate_rgb_hwc.transpose(2, 0, 1).copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = binary > 0
        strip_h = max(2, min(16, gray.shape[0] // 5))
        top_bright = float(np.mean(bright[:strip_h]))
        bottom_bright = float(np.mean(bright[-strip_h:]))
        if min(top_bright, bottom_bright) >= 0.75:
            foreground = ~bright
        elif max(top_bright, bottom_bright) <= 0.25:
            foreground = bright
        else:
            return candidate_yuv, {"postprocess": "subpixel_skipped_ambiguous_background", "postprocess_mean_abs_delta": 0.0}
        # Remove only provable one-pixel spikes for platform sampling.  Their
        # original pixels are intentionally never written, so topology cleanup
        # cannot change a thin stroke, a hole, or a punctuation mark.
        neighbours = cv2.filter2D(foreground.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), np.uint8)) - foreground.astype(np.int16)
        spike = foreground & (neighbours == 1)
        cleaned = foreground.copy()
        cleaned[spike] = False
        changed_class = foreground != cleaned
        core_fg = cv2.erode(cleaned.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        core_bg = cv2.erode((~cleaned).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        labels_count, labels, _, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), 8)
        proposed_rgb = candidate_rgb.copy()
        writable = np.zeros_like(update, dtype=bool)
        accepted = 0
        for label in range(1, labels_count):
            component = labels == label
            kernel = np.ones((15, 15), np.uint8)
            local = cv2.dilate(component.astype(np.uint8), kernel, iterations=1) > 0
            fg_samples = core_fg & local
            bg_samples = core_bg & local
            if np.count_nonzero(fg_samples) < 9 or np.count_nonzero(bg_samples) < 9:
                continue
            foreground_level = np.median(candidate_rgb[:, fg_samples], axis=1)
            background_level = np.median(candidate_rgb[:, bg_samples], axis=1)
            fg_mad = float(np.median(np.abs(candidate_rgb[0, fg_samples] - foreground_level[0])))
            bg_mad = float(np.median(np.abs(candidate_rgb[0, bg_samples] - background_level[0])))
            if fg_mad > 3.0 / 255.0 or bg_mad > 3.0 / 255.0 or abs(float(foreground_level[0] - background_level[0])) < 0.10:
                continue
            influence = local & ~changed_class
            color = candidate_rgb[:, influence]
            direction = foreground_level - background_level
            denom = float(np.dot(direction, direction))
            if denom <= 1e-8:
                continue
            coverage = np.clip(np.sum((color - background_level[:, None]) * direction[:, None], axis=0) / denom, 0.0, 1.0)
            fitted = background_level[:, None] + direction[:, None] * coverage[None]
            compatible = np.max(np.abs(fitted - color), axis=0) <= 3.0 / 255.0
            coords = np.flatnonzero(influence)
            selected = coords[compatible]
            if selected.size == 0:
                continue
            proposed_rgb.reshape(3, -1)[:, selected] = fitted[:, compatible]
            writable.reshape(-1)[selected] = True
            accepted += 1
        active = update & writable
        if not np.any(active):
            return candidate_yuv, {"postprocess": "subpixel_skipped", "postprocess_mean_abs_delta": 0.0, "postprocess_components": 0}
        repaired_yuv = rgb_to_yuv444(proposed_rgb.transpose(1, 2, 0))
        result = candidate_yuv.copy()
        result[:, active] = repaired_yuv[:, active]
        return result, {"postprocess": "subpixel_transition", "postprocess_mean_abs_delta": float(np.mean(np.abs(result[0, active] - y[active]))), "postprocess_components": accepted}
    core = alpha >= 0.75
    _, otsu = cv2.threshold(np.rint(np.clip(y, 0, 1) * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = otsu > 0
    foreground = bright if np.count_nonzero(bright & core) >= np.count_nonzero((~bright) & core) else ~bright
    # Keep Otsu's side of the threshold, but retain the existing glyph alpha as
    # the spatial safety boundary rather than clipping away anti-aliased edges.
    foreground &= alpha > 1.0e-6
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    cleaned = np.zeros_like(foreground, dtype=np.uint8)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 4:
            cleaned[labels == label] = 1
    foreground = cleaned.astype(bool)
    # The blurred mask supplies a one-sided feather at the text boundary; no
    # background pixel can receive a delta because foreground gates the write.
    inner = cv2.erode(foreground.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    foreground_values = y[inner] if inner.any() else y[foreground]
    if foreground_values.size == 0:
        return candidate_yuv, {"postprocess": "otsu_foreground_skipped", "postprocess_mean_abs_delta": 0.0}
    foreground_level = float(np.median(foreground_values))
    dilated_near = cv2.dilate(foreground.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    dilated_far = cv2.dilate(foreground.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1) > 0
    background_ring = dilated_far & ~dilated_near
    background_values = y[background_ring]
    if background_values.size == 0:
        return candidate_yuv, {"postprocess": "otsu_foreground_skipped_no_background", "postprocess_mean_abs_delta": 0.0}
    background_level = float(np.median(background_values))
    distance_in = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 3)
    distance_out = cv2.distanceTransform((~foreground).astype(np.uint8), cv2.DIST_L2, 3)
    signed_distance = distance_in - distance_out
    # A 0.75-px logistic transition contracts the measured blurred edge while
    # retaining a continuous 1--2 px antialias ramp on both sides.
    coverage = 1.0 / (1.0 + np.exp(-signed_distance / 0.75))
    target = background_level + (foreground_level - background_level) * coverage
    band_weight = np.clip((3.5 - np.abs(signed_distance)) / 3.5, 0.0, 1.0)
    delta = np.clip((target - y) * band_weight * strength * alpha, -max_delta, max_delta)
    active = update & (band_weight > 0.0)
    result = candidate_yuv.copy()
    result[0, active] = np.clip(y[active] + delta[active], 0.0, 1.0)
    return result, {"postprocess": "otsu_two_sided_transition", "postprocess_mean_abs_delta": float(np.mean(np.abs(delta[active]))) if active.any() else 0.0, "postprocess_background_level": background_level, "postprocess_foreground_level": foreground_level}


def _process_line(
    frame_rgb: np.ndarray,
    frame_yuv: np.ndarray,
    owner_score: np.ndarray,
    output_yuv: np.ndarray,
    line: OCRLine,
    segments: list[FullFrameSegment],
    network: torch.nn.Module,
    device: torch.device,
    *,
    canvas_hw: tuple[int, int],
    blend_radius: int,
    blend_radius_model_space: bool,
    restore_interpolation: str,
    input_interpolation: str,
    postprocess: str,
    postprocess_gamma: float,
    postprocess_strength: float,
    postprocess_max_delta: float,
    debug_dir: Path | None,
    tile_runner: Any | None = None,
    quality_guard: bool = False,
    line_prediction_callback: Callable[[tuple[int, int, int, int], np.ndarray, np.ndarray, OCRLine], None] | None = None,
    line_prediction_support_callback: Callable[[tuple[int, int, int, int], np.ndarray, np.ndarray, OCRLine], None] | None = None,
    line_prediction_full_box_callback: Callable[[tuple[int, int, int, int], np.ndarray, np.ndarray, OCRLine], None] | None = None,
) -> dict[str, Any]:
    """@luna-comment Restore one OCR line and composite its source-frame prediction.

    Input: source frame tensors, OCR geometry, model segments, network/device, and pasteback controls.
    Output: per-line telemetry after confidence-aware ownership update of ``output_yuv``.
    Key Steps: resize/pad range-valid model crops, infer tiles, restore predictions, fuse overlaps, then use legacy or V5 mask-only pasteback.
    Key Params: bicubic input crops are clipped only at the model boundary to preserve P17's [0,1] precondition; ``restore_interpolation`` affects only non-native prediction-to-source restoration; two V5 modes select binary mask-only alpha.
    State/Dependencies: mutates the shared output and owner-score canvases through ``standard_pasteback``.
    """
    if not segments:
        return {
            "line_index": line.line_index,
            "status": "bypass",
            "reason": "NO_FIT_SEGMENTS",
            "score": line.score,
            "text": line.text,
            "segments": 0,
            "route": "bypass",
        }

    min_x = min(segment.crop_xyxy[0] for segment in segments)
    min_y = min(segment.crop_xyxy[1] for segment in segments)
    max_x = max(segment.crop_xyxy[2] for segment in segments)
    max_y = max(segment.crop_xyxy[3] for segment in segments)
    local_height, local_width = max_y - min_y, max_x - min_x
    line_sum = np.zeros((3, local_height, local_width), dtype=np.float32)
    line_weight = np.zeros((local_height, local_width), dtype=np.float32)
    line_alpha = np.zeros((local_height, local_width), dtype=np.float32)
    full_sum = np.zeros_like(line_sum) if line_prediction_full_box_callback is not None else None
    full_weight = np.zeros_like(line_weight) if line_prediction_full_box_callback is not None else None

    for segment in segments:
        left, top, right, bottom = segment.crop_xyxy
        crop_rgb = frame_rgb[top:bottom, left:right]
        model_valid_hw = segment.model_valid_hw or segment.valid_hw
        model_crop_rgb = crop_rgb
        if segment.resize_mode == "stretch":
            model_crop_rgb = _resize_hwc(crop_rgb, canvas_hw)
        elif segment.resize_mode != "native":
            model_crop_rgb = _resize_hwc(crop_rgb, model_valid_hw, interpolation=(cv2.INTER_CUBIC if input_interpolation == "bicubic" else None))
        else:
            model_crop_rgb = crop_rgb
        # Bicubic interpolation can overshoot a valid RGB input by a few
        # ulps.  P17 deliberately rejects out-of-range model inputs, so keep
        # this boundary clip local to the resampled model crop rather than
        # changing source-frame pixels, recovered geometry, or prediction data.
        model_crop_rgb = np.clip(model_crop_rgb, 0.0, 1.0)
        segment_canvas_hw = segment.model_canvas_hw or canvas_hw
        canvas_rgb = _pad_canvas(
            model_crop_rgb, segment_canvas_hw,
            edge_replicate=segment.resize_mode == "height96_tile_pad",
        )
        if tile_runner is None:
            packed = torch.from_numpy(rgb_to_nv21(canvas_rgb)).unsqueeze(0).to(device)
            with torch.inference_mode():
                prediction_canvas = network(packed)[0].detach().cpu().numpy()
        else:
            # V4/V5 consume their native six-channel Y4UV representation.
            # OCR geometry and the output compositor remain unchanged.
            prediction_canvas = tile_runner.run_rgb(canvas_rgb)
        prediction_canvas = np.clip(prediction_canvas, 0.0, 1.0)

        valid_h, valid_w = segment.valid_hw
        prediction = prediction_canvas[:, :model_valid_hw[0], :model_valid_hw[1]]
        if segment.resize_mode != "native":
            kernel = cv2.INTER_CUBIC if restore_interpolation == "bicubic" else cv2.INTER_LINEAR
            prediction = _resize_chw(prediction, segment.valid_hw, interpolation=kernel)
        core = segment.core_mask_local
        if core is None:
            core = rasterize_polygon_mask(segment.polygon_local, (valid_h, valid_w))
            core = _refine_ocr_core_to_glyph_support(crop_rgb, core)
        else:
            core = np.asarray(core, dtype=np.float32)
            if core.shape != (valid_h, valid_w):
                raise ValueError("segment core mask shape mismatch")
        feather_radius = int(blend_radius)
        if blend_radius_model_space and segment.resize_mode == "height96_tile_pad":
            feather_radius = max(1, int(round(float(blend_radius) / max(float(segment.resize_scale), 1e-6))))
        alpha = feather_alpha(core, feather_radius)
        tile_weight = (
            _native_grid_weight(segment, segments)
            if segment.model_canvas_hw is not None and segment.tile_count == 1
            else tile_axis_weight(segment.valid_hw, segment.axis, segment.tile_count)
        )
        coefficient = alpha * tile_weight

        local_y0, local_x0 = top - min_y, left - min_x
        local_y1, local_x1 = local_y0 + valid_h, local_x0 + valid_w
        line_sum[:, local_y0:local_y1, local_x0:local_x1] += (
            prediction[:, :valid_h, :valid_w] * coefficient[None]
        )
        line_weight[local_y0:local_y1, local_x0:local_x1] += coefficient
        if full_sum is not None and full_weight is not None:
            full_coefficient = np.maximum(tile_weight, 1e-3)
            full_sum[:, local_y0:local_y1, local_x0:local_x1] += (
                prediction[:, :valid_h, :valid_w] * full_coefficient[None]
            )
            full_weight[local_y0:local_y1, local_x0:local_x1] += full_coefficient
        line_alpha[local_y0:local_y1, local_x0:local_x1] = np.maximum(
            line_alpha[local_y0:local_y1, local_x0:local_x1], alpha
        )

        if debug_dir is not None:
            _tile_debug(debug_dir, segment, canvas_rgb, prediction_canvas)

    valid = line_weight > 1e-6
    line_prediction = np.zeros_like(line_sum)
    line_prediction[:, valid] = line_sum[:, valid] / line_weight[valid][None]
    if postprocess == "p11_valid_maskraw_compare_v5":
        line_prediction, postprocess_record = _apply_p11_edge_narrow_v3(line_prediction, valid, use_directional=False)
        postprocess_record["postprocess"] = "p11_valid_maskraw_compare_v5"
    elif postprocess == "p11_valid_maskdirectional_compare_v5":
        line_prediction, postprocess_record = _apply_p11_edge_narrow_v3(line_prediction, valid, use_directional=True)
        postprocess_record["postprocess"] = "p11_valid_maskdirectional_compare_v5"
    elif postprocess == "p11_valid_support_polarity_guard_edge_narrow_v4_2":
        line_prediction, postprocess_record = _apply_p11_edge_narrow_v3(line_prediction, valid)
        postprocess_record["postprocess"] = "p11_valid_support_polarity_guard_edge_narrow_v4_2"
        postprocess_record["analysis_domain"] = "line_weight_valid_only"
        postprocess_record["valid_pixels"] = int(valid.sum())
        postprocess_record["invalid_pixels_ignored"] = int((~valid).sum())
        postprocess_record["invalid_pixels_read"] = 0
        postprocess_record["p10_reported_polarity"] = "not_evaluated_full_border"
        postprocess_record["invalid_y_changed_pixels"] = int(np.count_nonzero(line_prediction[0, ~valid]))
    elif postprocess == "p11_edge_narrow_v3_1":
        line_prediction, postprocess_record = _apply_p11_edge_narrow_v3(line_prediction)
    elif postprocess == "p11_edge_narrow_v2":
        line_prediction, postprocess_record = _apply_p11_edge_narrow_v2(line_prediction)
    elif postprocess == "y_scurve":
        p11_rgb, p11_meta = _p11_kernel_wiener_scurve(yuv444_to_rgb(line_prediction), _P11_FROZEN_PARAMS)
        line_prediction = rgb_to_yuv444(p11_rgb)
        postprocess_record = {
            "postprocess": "p11_kernel_wiener_scurve_lambda_0p03",
            "p11_status": p11_meta.get("p11_status"),
            "p11_kernel_status": p11_meta.get("p11_kernel_status"),
            "p11_scurve_status": p11_meta.get("p11_scurve_status"),
            "p11_write_coverage": float(p11_meta.get("p11_write_coverage", 0.0)),
        }
    else:
        postprocess_record = None
    pasteback_alpha, alpha_record = _select_pasteback_alpha(postprocess, line_alpha, valid, postprocess_record)
    if postprocess_record is not None:
        postprocess_record.update(alpha_record)
    crop_xyxy = (min_x, min_y, max_x, max_y)
    if line_prediction_callback is not None:
        line_prediction_callback(crop_xyxy, line_prediction, pasteback_alpha, line)
    if line_prediction_support_callback is not None:
        line_prediction_support_callback(crop_xyxy, line_prediction, valid, line)
    if line_prediction_full_box_callback is not None:
        full_valid = full_weight > 1e-6
        full_prediction = np.zeros_like(full_sum)
        full_prediction[:, full_valid] = full_sum[:, full_valid] / full_weight[full_valid][None]
        line_prediction_full_box_callback(crop_xyxy, full_prediction, full_valid, line)
    base_patch = frame_yuv[:, min_y:max_y, min_x:max_x]
    candidate, update, _ = standard_pasteback(
        frame_yuv, output_yuv, owner_score, crop_xyxy, line_prediction,
        pasteback_alpha, line.score,
    )
    if postprocess_record is None:
        candidate, postprocess_record = _apply_pasteback_y_scurve(
            candidate, base_patch, update, line_alpha * valid.astype(np.float32), mode=postprocess,
            gamma=postprocess_gamma, strength=postprocess_strength, max_delta=postprocess_max_delta,
        )
    output_patch = output_yuv[:, min_y:max_y, min_x:max_x]
    output_patch[:, update] = candidate[:, update]
    output_patch = output_yuv[:, min_y:max_y, min_x:max_x]
    active = line_alpha > 1e-6
    candidate_delta = np.abs(candidate - base_patch).mean(axis=0)
    written_delta = np.abs(output_patch - base_patch).mean(axis=0)

    if debug_dir is not None:
        line_dir = debug_dir / f"line_{line.line_index:04d}"
        _write_rgb(line_dir / "merged_output.png", yuv444_to_rgb(line_prediction))
        _write_rgb(line_dir / "source_patch.png", frame_rgb[min_y:max_y, min_x:max_x])
        _write_rgb(line_dir / "candidate_pasteback.png", yuv444_to_rgb(candidate))
        _write_rgb(line_dir / "pasted_after_line.png", yuv444_to_rgb(output_patch))
        _write_gray(line_dir / "merged_alpha.png", line_alpha)
        _write_gray(line_dir / "write_mask.png", update.astype(np.float32))
        _write_gray(line_dir / "ownership_blocked_mask.png", (valid & ~update).astype(np.float32))
        if postprocess_record.get("postprocess") == "p11_edge_narrow_v2":
            _write_gray(line_dir / "mask_raw.png", postprocess_record["_v2_raw_mask"].astype(np.float32))
            _write_gray(line_dir / "mask_directional.png", postprocess_record["_v2_clean_mask"].astype(np.float32))
            _write_gray(line_dir / "write_support.png", postprocess_record["_v2_write_support"].astype(np.float32))
            _write_gray(line_dir / "delta_y.png", np.abs(postprocess_record["_v2_delta_y"]))
        if postprocess_record.get("postprocess") in ("p11_edge_narrow_v3_1", "p11_valid_support_polarity_guard_edge_narrow_v4_2", "p11_valid_maskraw_compare_v5", "p11_valid_maskdirectional_compare_v5"):
            _write_gray(line_dir / "mask_raw.png", postprocess_record["_v3_raw_mask"].astype(np.float32))
            _write_gray(line_dir / "mask_directional.png", postprocess_record["_v3_clean_mask"].astype(np.float32))
            _write_gray(line_dir / "edge_candidate_two_sided.png", postprocess_record["_v3_edge_candidate"].astype(np.float32))
            _write_gray(line_dir / "anchor_class_valid.png", postprocess_record["_v3_anchor_class_valid"].astype(np.float32))
            _write_gray(line_dir / "q_transition.png", postprocess_record["_v3_q_transition"].astype(np.float32))
            _write_gray(line_dir / "write_support.png", postprocess_record["_v3_write_support"].astype(np.float32))
            _write_gray(line_dir / "delta_y_signed.png", (postprocess_record["_v3_delta_y"] + 1.0) * 0.5)
        (line_dir / "metadata.json").write_text(
            json.dumps({
                "line_index": line.line_index,
                "text": line.text,
                "source": line.source,
                "score": line.score,
                "segments": [
                    {
                        "tile_index": item.tile_index,
                        "tile_count": item.tile_count,
                        "orientation": item.orientation,
                        "axis": item.axis,
                        "crop_xyxy": list(item.crop_xyxy),
                        "valid_hw": list(item.valid_hw),
                        "resize_mode": item.resize_mode,
                        "resize_scale": item.resize_scale,
                        "model_valid_hw": list(item.model_valid_hw or item.valid_hw),
                    }
                    for item in segments
                ],
                "local_frame_xyxy": [min_x, min_y, max_x, max_y],
                "updated_pixels": int(update.sum()),
                "candidate_lq_delta_mean": float(candidate_delta[active].mean()) if active.any() else 0.0,
                "written_lq_delta_mean": float(written_delta[active].mean()) if active.any() else 0.0,
                "ownership_blocked_pixels": int((valid & ~update).sum()),
                "pasteback_alpha_pixels": int(np.count_nonzero(pasteback_alpha)),
                "selected_mask_kind": alpha_record["selected_mask_kind"],
                "merged_alpha_used": alpha_record["merged_alpha_used"],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    postprocess_record = {
        key: value for key, value in postprocess_record.items()
        if not key.startswith("_v2_") and not key.startswith("_v3_")
    }
    return {
        "line_index": line.line_index,
        "status": "processed",
        "reason": "",
        "score": line.score,
        "text": line.text,
        "source": line.source,
        "orientation": segments[0].orientation,
        "segments": len(segments),
        "route": segments[0].resize_mode,
        "resize_scale": segments[0].resize_scale,
        "restore_interpolation": restore_interpolation if segments[0].resize_mode != "native" else "not_applicable",
        "updated_pixels": int(update.sum()),
        "candidate_lq_delta_mean": float(candidate_delta[active].mean()) if active.any() else 0.0,
        "written_lq_delta_mean": float(written_delta[active].mean()) if active.any() else 0.0,
        "ownership_blocked_pixels": int((valid & ~update).sum()),
        "crop_xyxy_union": [min_x, min_y, max_x, max_y],
        "input_interpolation": input_interpolation if segments[0].resize_mode != "native" else "not_applicable",
        **postprocess_record,
    }


def _load_dense_probability_map(path: Path | None, source_hw: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    """Load an exported DB probability map with an explicit source transform."""
    if path is None:
        return None, "PROBABILITY_MAP_UNAVAILABLE"
    try:
        probability = np.asarray(np.load(path), dtype=np.float32)
    except (OSError, ValueError):
        return None, "PROBABILITY_MAP_LOAD_FAILED"
    if probability.ndim != 2 or probability.shape[0] < 8 or probability.shape[1] < 8 or not np.isfinite(probability).all():
        return None, "PROBABILITY_MAP_INVALID"
    return np.clip(probability, 0.0, 1.0), "AVAILABLE"


def _extract_dense_probability_bands(group: dict[str, Any], probability: np.ndarray, source_hw: tuple[int, int]) -> tuple[list[dict[str, Any]], str]:
    """Estimate spacing from probability evidence without changing group membership.

    Sparse table cells are allowed to occupy only one horizontal subregion.
    Core-row hypotheses constrain the search; neither five-window continuity
    nor a second acceptance gate may discard an already routed dense group.
    """
    source_h, source_w = source_hw; map_h, map_w = probability.shape
    boxes = [member["final_box"] for member in group["members"]]
    x0 = max(0, int(np.floor(min(box[0] for box in boxes) * map_w / source_w))); x1 = min(map_w, int(np.ceil(max(box[2] for box in boxes) * map_w / source_w)))
    y0 = max(0, int(np.floor(min(box[1] for box in boxes) * map_h / source_h))); y1 = min(map_h, int(np.ceil(max(box[3] for box in boxes) * map_h / source_h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("dense probability region is empty")
    member_centers = np.asarray(group.get("row_centers", sorted(item["center_y"] for item in group["members"])), np.float32)
    nominal_pitch = float(group["center_spacing"])
    # Sum only high-probability pixels.  A short field is evidence even when
    # most of the big-box width is blank; no horizontal-coverage requirement.
    region = probability[y0:y1, x0:x1]
    profile = np.where(region >= .20, region, 0.0).sum(axis=1)
    bands = []
    for ordinal, expected in enumerate(member_centers):
        lower = .5 * (member_centers[ordinal - 1] + expected) if ordinal else expected - .5 * nominal_pitch
        upper = .5 * (expected + member_centers[ordinal + 1]) if ordinal + 1 < len(member_centers) else expected + .5 * nominal_pitch
        lo = max(0, int(np.floor(lower * map_h / source_h)) - y0)
        hi = min(len(profile), int(np.ceil(upper * map_h / source_h)) - y0)
        local = profile[lo:hi]
        if local.size and float(local.max()) > 0.0:
            support = np.flatnonzero(local >= .15 * float(local.max()))
            mass = local[support]
            center = float((np.sum((support + lo + y0 + .5) * mass) / mass.sum()) * source_h / map_h)
            width = float((support[-1] - support[0] + 1) * source_h / map_h)
            source = "probability_profile"
        else:
            # A missing band estimate changes scale provenance, never routing.
            center, width, source = float(expected), float(group["core_thickness"]), "pre_unclip_core"
        bands.append({"center": center, "width": width, "scale_source": source})
    thickness = float(np.median([band["width"] for band in bands]))
    centers = np.asarray([band["center"] for band in bands], np.float32)
    pitch = float(np.median(np.diff(centers)))
    if not np.isfinite(pitch) or pitch <= 1.0:
        raise ValueError("dense group has no positive pre-unclip/probability row spacing")
    group["probability_bands"] = bands
    group["band_thickness"] = thickness
    group["band_pitch"] = pitch
    group["band_pitch_mad_ratio"] = float(np.median(np.abs(np.diff(centers) - pitch)) / pitch)
    group["probability_supported_rows"] = sum(band["scale_source"] == "probability_profile" for band in bands)
    return bands, "AVAILABLE"

def _dense_group_axis_starts(length: int, patch: int, stride: int | None = None) -> list[int]:
    """Cover one axis with 25-percent-overlapped fixed source patches."""
    if length <= patch:
        return [0]
    stride = max(1, int(stride) if stride is not None else int(round(patch * .75)))
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return sorted(set(starts))


def _dense_group_patch_weight(rect: tuple[int, int, int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Return a complementary 2-D overlap window for a grouped patch."""
    left, top, right, bottom = rect
    h, w = bottom - top, right - left
    x, y = np.ones(w, np.float32), np.ones(h, np.float32)
    for before in (True, False):
        xo: list[int] = []; yo: list[int] = []
        for ol, ot, or_, ob in rects:
            if (ol, ot, or_, ob) == rect:
                continue
            if min(bottom, ob) > max(top, ot):
                if before and ol < left < or_: xo.append(or_ - left)
                if not before and ol < right < or_: xo.append(right - ol)
            if min(right, or_) > max(left, ol):
                if before and ot < top < ob: yo.append(ob - top)
                if not before and ot < bottom < ob: yo.append(bottom - ot)
        if xo:
            o = max(xo); x *= np.clip((np.arange(w, dtype=np.float32) + .5) / o, 0, 1) if before else np.clip((w - np.arange(w, dtype=np.float32) - .5) / o, 0, 1)
        if yo:
            o = max(yo); y *= np.clip((np.arange(h, dtype=np.float32) + .5) / o, 0, 1) if before else np.clip((h - np.arange(h, dtype=np.float32) - .5) / o, 0, 1)
    return y[:, None] * x[None, :]


def _process_dense_text_group(frame_rgb: np.ndarray, frame_yuv: np.ndarray, owner_score: np.ndarray,
                              output_yuv: np.ndarray, group: dict[str, Any], network: torch.nn.Module,
                              device: torch.device, *, blend_radius: int, debug_dir: Path | None,
                              tile_runner: Any | None, core_target_px: float, vertical_margin_ratio: float = .50) -> dict[str, Any]:
    """Restore a DB collision group once using fixed 5:1 source patches."""
    image_h, image_w = frame_rgb.shape[:2]
    band_thickness = float(group["band_thickness"])
    pitch = float(group["band_pitch"])
    # The complete candidate union is covered; probability pitch sets only
    # patch dimensions, never the extent or membership of the group.
    boxes = [member["final_box"] for member in group["members"]]
    left = max(0, int(np.floor(min(box[0] for box in boxes))))
    top = max(0, int(np.floor(min(box[1] for box in boxes))))
    right = min(image_w, int(np.ceil(max(box[2] for box in boxes))))
    bottom = min(image_h, int(np.ceil(max(box[3] for box in boxes))))
    region_h, region_w = bottom - top, right - left
    patch_h = max(8, int(round(pitch))); patch_w = max(40, patch_h * 5)
    rects = [(x, y, min(region_w, x + patch_w), min(region_h, y + patch_h))
             for y in _dense_group_axis_starts(region_h, patch_h) for x in _dense_group_axis_starts(region_w, patch_w)]
    total, weights = np.zeros((3, region_h, region_w), np.float32), np.zeros((region_h, region_w), np.float32)
    for patch_index, (x0, y0, x1, y1) in enumerate(rects):
        source = frame_rgb[top + y0:top + y1, left + x0:left + x1]
        padded = cv2.copyMakeBorder(source, 0, patch_h - source.shape[0], 0, patch_w - source.shape[1], cv2.BORDER_REPLICATE)
        canvas = _resize_hwc(padded, (96, 480))
        if tile_runner is None:
            packed = torch.from_numpy(rgb_to_nv21(canvas)).unsqueeze(0).to(device)
            with torch.inference_mode(): prediction_canvas = network(packed)[0].detach().cpu().numpy()
        else:
            prediction_canvas = tile_runner.run_rgb(canvas)
        prediction = _resize_chw(np.clip(prediction_canvas, 0, 1), (patch_h, patch_w))[:, :source.shape[0], :source.shape[1]]
        weight = _dense_group_patch_weight((x0, y0, x1, y1), rects)
        total[:, y0:y1, x0:x1] += prediction * weight[None]; weights[y0:y1, x0:x1] += weight
        if debug_dir is not None:
            patch_dir = debug_dir / f"dense_group_{group['group_index']:02d}" / "patches"
            _write_rgb(patch_dir / f"{patch_index:03d}_input.png", canvas); _write_rgb(patch_dir / f"{patch_index:03d}_output.png", yuv444_to_rgb(prediction_canvas)); _write_gray(patch_dir / f"{patch_index:03d}_weight.png", weight)
    valid = weights > 1e-6; prediction = np.zeros_like(total); prediction[:, valid] = total[:, valid] / weights[valid][None]
    core = np.zeros((region_h, region_w), np.uint8)
    for member in group["members"]:
        polygon = member["core"].copy(); polygon[:, 0] -= left; polygon[:, 1] -= top
        core |= rasterize_polygon_mask(polygon, (region_h, region_w)).astype(np.uint8)
    # Composite the reconstructed big box once; never gate predictions by
    # sharpness, contrast or per-glyph output appearance.
    border = np.minimum(np.arange(region_h) + .5, region_h - np.arange(region_h) - .5)[:, None]
    border = np.minimum(border, np.minimum(np.arange(region_w) + .5, region_w - np.arange(region_w) - .5)[None, :])
    alpha = np.minimum(1.0, border / max(1, int(blend_radius))).astype(np.float32)
    base = frame_yuv[:, top:bottom, left:right]; candidate = alpha[None] * prediction + (1 - alpha[None]) * base
    score = max(.01, max(float(member["score"]) for member in group["members"])); ownership = owner_score[top:bottom, left:right]
    update = valid & ((alpha * score) > ownership + 1e-6); output_yuv[:, top:bottom, left:right][:, update] = candidate[:, update]; ownership[update] = (alpha * score)[update]
    if debug_dir is not None:
        group_dir = debug_dir / f"dense_group_{group['group_index']:02d}"
        _write_rgb(group_dir / "source_region.png", frame_rgb[top:bottom, left:right]); _write_rgb(group_dir / "fused_output.png", yuv444_to_rgb(prediction)); _write_gray(group_dir / "weight_sum.png", weights); _write_gray(group_dir / "core_mask.png", core); _write_gray(group_dir / "alpha.png", alpha)
        (group_dir / "patch_grid.json").write_text(json.dumps({"patch_source_hw": [patch_h, patch_w], "crop_xyxy": [left, top, right, bottom], "rects": rects, "weight_min": float(weights.min()), "weight_max": float(weights.max())}, indent=2), encoding="utf-8")
    return {"group_index": group["group_index"], "status": "processed", "route": "dense_small_text_group_5to1", "member_line_indices": group["member_indices"], "band_thickness": band_thickness, "band_pitch": pitch, "vertical_margin": 0.0, "rows_per_patch": 1, "patch_source_hw": [patch_h, patch_w], "patch_count": len(rects), "crop_xyxy": [left, top, right, bottom], "updated_pixels": int(update.sum())}
def infer_full_image(
    input_path: Path,
    ocr_json_path: Path,
    checkpoint: Path,
    output_path: Path,
    *,
    device_name: str = "cuda",
    context_ratio: float = 0.15,
    overlap: int = 64,
    blend_radius: int = 8,
    min_score: float = 0.0,
    resize_bypass: bool = False,
    resize_min_ratio: float = 0.5,
    resize_max_ratio: float = 1.5,
    resize_target_short_ratio: float = 0.75,
    debug_dir: Path | None = None,
    network: torch.nn.Module | None = None,
    checkpoint_sha256: str | None = None,
    network_config: Path | None = None,
    resize_mode: str = "native",
    ocr_format: str = "generic",
    p17_vertical_context: float = 0.12,
    p17_horizontal_context: float = 0.03,
    mask_guided_ocr_recovery: bool = False,
    mask_guided_geometry_stats: Path | None = None,
    native_tile_height_threshold: int = 128,
    force_native_tile_line_indices: set[int] | None = None,
    large_tile_canvas_hw: tuple[int, int] = (256, 256),
    large_tile_overlap: int = 96,
    large_tile_overlap_x: int | None = None,
    large_tile_overlap_y: int | None = None,
    text_mask_path: Path | None = None,
    p17_textmask_coverage_threshold: float = 0.85,
    tile_runner: Any | None = None,
    generic_ocr_nms_iou: float = 0.0,
    sharp_text_bypass_laplacian: float = 0.0,
    sharp_text_bypass_max_height: int = 128,
    quality_guard: bool = False,
    dense_group_core_target_px: float | None = None,
    dense_group_probability_map: Path | None = None,
    dense_group_vertical_margin_ratio: float = 0.50,
    restore_interpolation: str = "linear",
    input_interpolation: str = "default",
    postprocess: str = "none",
    postprocess_gamma: float = 1.3,
    postprocess_strength: float = 0.5,
    postprocess_max_delta: float = 8.0 / 255.0,
    line_prediction_callback: Callable[[tuple[int, int, int, int], np.ndarray, np.ndarray, OCRLine], None] | None = None,
    line_prediction_support_callback: Callable[[tuple[int, int, int, int], np.ndarray, np.ndarray, OCRLine], None] | None = None,
    line_prediction_full_box_callback: Callable[[tuple[int, int, int, int], np.ndarray, np.ndarray, OCRLine], None] | None = None,
) -> dict[str, Any]:
    """@luna-comment Run one OCR-driven restoration, optionally recovering P17 line y-geometry.

    Input: an RGB frame, its OCR JSON, strict checkpoint/network route, and
    per-run options.  Output: the restored PNG plus route JSON and, when
    requested, per-line geometry evidence under ``debug_dir``.  Key Steps:
    parse OCR, optionally use the frozen P17 *raw mask only* to correct
    reliable horizontal line y-bounds, construct segments, restore/fuse, and
    write machine-readable telemetry.  Key Params: the recovery switch is
    opt-in and consumes versioned training geometry statistics; P17 contexts
    remain the existing 12% vertical / 3% horizontal values.  State/
    Dependencies: the supplied shared network/tile runner owns model input
    preprocessing; a geometry probe must never replace that input.
    """
    resize_mode = str(resize_mode).strip().lower()
    if resize_mode not in ("native", "native_tile", "native_tile_height192x960", "stretch", "height96_tile_pad", "p17_height96_tile_pad"):
        raise ValueError("unsupported resize_mode")
    force_native_tile_line_indices = set(force_native_tile_line_indices or ())
    if restore_interpolation not in ("linear", "bicubic"):
        raise ValueError("restore_interpolation must be 'linear' or 'bicubic'")
    if input_interpolation not in ("default", "bicubic") or postprocess not in ("none", "y_scurve", "otsu_foreground", "source_edge", "subpixel_transition", "p11_edge_narrow_v2", "p11_edge_narrow_v3", "p11_edge_narrow_v3_1", "p11_valid_support_polarity_guard_edge_narrow_v4_2", "p11_valid_maskraw_compare_v5", "p11_valid_maskdirectional_compare_v5"):
        raise ValueError("unsupported interpolation or postprocess mode")
    if ocr_format not in ("generic", "p17_annotations"):
        raise ValueError("unsupported ocr_format")
    if mask_guided_ocr_recovery and resize_mode != "p17_height96_tile_pad":
        raise ValueError("mask-guided OCR recovery requires p17_height96_tile_pad")
    if mask_guided_ocr_recovery and mask_guided_geometry_stats is None:
        raise ValueError("mask-guided OCR recovery requires --mask-guided-geometry-stats")
    if int(native_tile_height_threshold) < 2:
        raise ValueError("native_tile_height_threshold must be at least 2")
    if (len(large_tile_canvas_hw) != 2 or any(int(value) <= 0 or int(value) % 8 for value in large_tile_canvas_hw)):
        raise ValueError("large_tile_canvas_hw must contain positive multiples of eight")
    large_tile_overlap_x = int(large_tile_overlap if large_tile_overlap_x is None else large_tile_overlap_x)
    large_tile_overlap_y = int(large_tile_overlap if large_tile_overlap_y is None else large_tile_overlap_y)
    if not (0 <= large_tile_overlap_x < int(large_tile_canvas_hw[1]) and
            0 <= large_tile_overlap_y < int(large_tile_canvas_hw[0])):
        raise ValueError("large tile overlap must be non-negative and smaller than its axis")
    canvas_hw = resolve_canvas_hw(network_config)
    original_rgb = _read_rgb(input_path)
    frame_rgb, original_hw = pad_even_rgb(original_rgb)
    frame_hw = frame_rgb.shape[:2]
    ocr_data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    lines = (
        parse_p17_ocr_annotations(ocr_data, frame_hw)
        if ocr_format == "p17_annotations"
        else parse_ocr_json(ocr_data, frame_hw)
    )
    ocr_lines_input = len(lines)
    dense_groups: list[dict[str, Any]] = []
    detector_contract_status = "NOT_APPLICABLE"
    grouped_member_indices: set[int] = set()
    if ocr_format == "generic":
        dense_groups, detector_contract_status = _plan_dense_small_text_groups(ocr_data, frame_hw, float(min_score))
        probability, probability_status = _load_dense_probability_map(dense_group_probability_map, frame_hw)
        if probability is None:
            if dense_group_probability_map is not None:
                raise ValueError(probability_status)
            dense_groups = []
            detector_contract_status += "_" + probability_status
        else:
            for group in dense_groups:
                _extract_dense_probability_bands(group, probability, frame_hw)
            detector_contract_status += "_" + probability_status
        for group_index, group in enumerate(dense_groups):
            group["group_index"] = group_index
            grouped_member_indices.update(group["member_indices"])
    ocr_lines_suppressed = 0
    # Dense routing never suppresses ordinary score-qualified candidates.
    # Legacy NMS/quality/sharpness arguments remain accepted for old launchers
    # but do not remove a candidate or reject a network prediction.
    line_masks: dict[int, np.ndarray] = {}
    textmask_fallback_used = False
    textmask_status = "not_requested"
    if resize_mode == "p17_height96_tile_pad" and text_mask_path is not None:
        text_mask = cv2.imread(str(text_mask_path), cv2.IMREAD_GRAYSCALE)
        if text_mask is None or text_mask.shape != frame_hw:
            # OCR annotations are authoritative for OCR-fixed validation sets.
            # An older TextMask may use a different crop geometry, so it cannot
            # safely be used as a fallback or invalidate the render.
            textmask_status = "unavailable_or_shape_mismatch"
        else:
            textmask_status = "compatible"
            text_binary = text_mask > 127
            ocr_core = np.zeros(frame_hw, dtype=np.uint8)
            for line in lines:
                ocr_core |= rasterize_polygon_mask(line.polygon_xy, frame_hw).astype(np.uint8)
            coverage = float((text_binary & (ocr_core > 0)).sum() / max(1, text_binary.sum()))
            if coverage < float(p17_textmask_coverage_threshold):
                fallback = _p17_textmask_fallback(text_mask_path, frame_hw)
                if fallback:
                    lines = [item[0] for item in fallback]
                    line_masks = {line.line_index: mask for line, mask in fallback}
                    textmask_fallback_used = True
                    textmask_status = "fallback_used"
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    if network is None:
        if network_config is None:
            network, checkpoint_sha256 = _load_network(checkpoint, device)
        else:
            network, checkpoint_sha256 = _load_network(checkpoint, device, network_config)
    elif checkpoint_sha256 is None:
        checkpoint_sha256 = "provided_network"

    geometry_decisions: dict[int, dict[str, Any]] = {}
    geometry_stats: dict[str, float | str] | None = None
    if mask_guided_ocr_recovery:
        geometry_stats = _load_mask_guided_geometry_stats(mask_guided_geometry_stats)
        lq_config = getattr(tile_runner, "lq_preprocess_config", load_lq_preprocess_config(network_config))
        if not bool(lq_config.get("enabled")) or str(lq_config.get("implementation")) != "torch_gpu":
            raise ValueError("mask-guided OCR recovery requires enabled torch_gpu P17 preprocessing")
        if device.type != "cuda":
            raise ValueError("mask-guided OCR recovery requires CUDA for the frozen raw-mask probe")
        probe = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device=device, dtype=torch.float32)
        with torch.no_grad():
            _, probe_debug = preprocess_p17_torch_gpu_with_debug(probe, lq_config)
        mask_raw = probe_debug["text_mask"][0, 0].detach().to("cpu").numpy() > 0.5
        if mask_raw.shape != frame_hw:
            raise RuntimeError("P17 raw text-mask shape does not match full-frame geometry")
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            _write_gray(debug_dir / "p17_geometry_mask_raw.png", mask_raw)
        lines, geometry_decisions = _recover_mask_guided_ocr_lines(lines, mask_raw, frame_rgb, geometry_stats, debug_dir)
        if debug_dir is not None:
            recovery_report = {
                "enabled": True,
                "stats": geometry_stats,
                "lines": [geometry_decisions[index] for index in sorted(geometry_decisions)],
            }
            (debug_dir / "geometry_recovery.json").write_text(json.dumps(recovery_report, ensure_ascii=False, indent=2), encoding="utf-8")

    frame_yuv = rgb_to_yuv444(frame_rgb)
    output_yuv = frame_yuv.copy()
    owner_score = np.zeros(frame_hw, dtype=np.float32)
    line_records: list[dict[str, Any]] = []
    segments_total = 0
    segments_native = 0
    segments_resized = 0
    t0 = time.perf_counter()

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for group in dense_groups:
        record = _process_dense_text_group(frame_rgb, frame_yuv, owner_score, output_yuv, group, network, device,
                                           blend_radius=blend_radius, debug_dir=debug_dir, tile_runner=tile_runner,
                                           core_target_px=float(dense_group_core_target_px or 72.0), vertical_margin_ratio=float(dense_group_vertical_margin_ratio))
        line_records.append(record)
        segments_total += int(record["patch_count"])
        segments_resized += int(record["patch_count"])

    for line in lines:
        if line.line_index in grouped_member_indices:
            continue
        if line.score < float(min_score):
            line_records.append({
                "line_index": line.line_index,
                "status": "bypass",
                "reason": "SCORE_BELOW_THRESHOLD",
                "score": line.score,
                "text": line.text,
                "segments": 0,
            })
            continue
        line_height = float(np.ptp(line.polygon_xy[:, 1]))
        route = "native"
        if resize_mode == "p17_height96_tile_pad":
            raw_segments, reason = build_p17_height96_tiled_full_frame_segments(
                line.polygon_xy, frame_hw, canvas_hw, overlap=overlap,
                vertical_context=p17_vertical_context, horizontal_context=p17_horizontal_context,
            )
        elif resize_mode == "height96_tile_pad":
            raw_segments, reason = build_height96_tiled_full_frame_segments(line.polygon_xy, frame_hw, canvas_hw, overlap=overlap)
            route = "height96_resize"
        elif resize_mode == "native_tile":
            if line_height <= float(native_tile_height_threshold) and line.line_index not in force_native_tile_line_indices:
                if ocr_format == "p17_annotations":
                    raw_segments, reason = build_p17_height96_tiled_full_frame_segments(
                        line.polygon_xy, frame_hw, canvas_hw, overlap=overlap,
                        vertical_context=p17_vertical_context, horizontal_context=p17_horizontal_context,
                    )
                else:
                    raw_segments, reason = build_height96_tiled_full_frame_segments(
                        line.polygon_xy, frame_hw, canvas_hw, overlap=overlap,
                    )
                route = "height96_resize"
            else:
                raw_segments, reason = build_native_tiled_full_frame_segments(
                    line.polygon_xy, frame_hw, tuple(map(int, large_tile_canvas_hw)),
                    overlap_x=large_tile_overlap_x, overlap_y=large_tile_overlap_y,
                )
                route = "native_2d_tile"
        elif resize_mode == "native_tile_height192x960":
            if line_height <= float(native_tile_height_threshold):
                raw_segments, reason = build_height96_tiled_full_frame_segments(
                    line.polygon_xy, frame_hw, canvas_hw, overlap=overlap,
                )
                route = "height96_resize"
            else:
                height192_canvas = (192, 960)
                raw_segments, reason = build_height96_tiled_full_frame_segments(
                    line.polygon_xy, frame_hw, height192_canvas, overlap=96,
                )
                raw_segments = [replace(item, model_canvas_hw=height192_canvas) for item in raw_segments]
                route = "height192_resize"
        else:
            raw_segments, reason = build_full_frame_segments(line.polygon_xy, frame_hw, canvas_hw, context_ratio=context_ratio, overlap=overlap)
        if resize_mode == "stretch":
            raw_segments = [replace(
                segment,
                resize_mode="stretch",
                resize_scale=(canvas_hw[0] / float(segment.valid_hw[0]), canvas_hw[1] / float(segment.valid_hw[1])),
                model_valid_hw=canvas_hw,
            ) for segment in raw_segments]
        if not raw_segments and resize_bypass and resize_mode not in ("height96_tile_pad", "p17_height96_tile_pad", "native_tile"):
            raw_segments, resize_reason = build_resized_full_frame_segments(
                line.polygon_xy,
                frame_hw,
                CANVAS_HW,
                context_ratio=context_ratio,
                overlap=overlap,
                min_ratio=resize_min_ratio,
                max_ratio=resize_max_ratio,
                target_short_ratio=resize_target_short_ratio,
            )
            if raw_segments:
                route = "uniform_short_axis"
            else:
                reason = resize_reason or reason
        segments = attach_line_index(raw_segments, line.line_index)
        if line.line_index in line_masks:
            segments = [replace(item, core_mask_local=line_masks[line.line_index][item.crop_xyxy[1]:item.crop_xyxy[3], item.crop_xyxy[0]:item.crop_xyxy[2]]) for item in segments]
        segments_total += len(segments)
        if route in ("uniform_short_axis", "height96_resize", "height192_resize"):
            segments_resized += len(segments)
        else:
            segments_native += len(segments)
        if not segments:
            line_records.append({
                "line_index": line.line_index,
                "status": "bypass",
                "reason": reason or "NO_FIT_SEGMENTS",
                "score": line.score,
                "text": line.text,
                "segments": 0,
                "route": "bypass",
            })
            continue
        line_record = _process_line(
            frame_rgb,
            frame_yuv,
            owner_score,
            output_yuv,
            line,
            segments,
            network,
            device,
            canvas_hw=canvas_hw,
            blend_radius=blend_radius,
            blend_radius_model_space=(resize_mode == "p17_height96_tile_pad" or
                                      (resize_mode == "native_tile" and route == "height96_resize" and ocr_format == "p17_annotations")),
            restore_interpolation=restore_interpolation,
            input_interpolation=input_interpolation,
            postprocess=postprocess,
            postprocess_gamma=postprocess_gamma,
            postprocess_strength=postprocess_strength,
            postprocess_max_delta=postprocess_max_delta,
            debug_dir=debug_dir,
            tile_runner=tile_runner,
            quality_guard=False,
            line_prediction_callback=line_prediction_callback,
            line_prediction_support_callback=line_prediction_support_callback,
            line_prediction_full_box_callback=line_prediction_full_box_callback,
        )
        line_record["route"] = route
        if geometry_decisions:
            line_record["geometry_recovery"] = geometry_decisions[line.line_index]
        line_records.append(line_record)

    output_rgb_even = yuv444_to_rgb(output_yuv)
    processed = owner_score > 0.0
    # Preserve unprocessed pixels exactly in RGB.  This avoids introducing a
    # whole-frame RGB->YUV->RGB round-trip on background-only regions.
    output_rgb_even[~processed] = frame_rgb[~processed]
    output_rgb = output_rgb_even[: original_hw[0], : original_hw[1]]
    _write_rgb(output_path, output_rgb)

    group_for_line = {index: group["group_index"] for group in dense_groups for index in group["member_indices"]}
    ordinary_records = {item["line_index"]: item for item in line_records if "line_index" in item}
    candidate_routes = []
    overlay = np.rint(original_rgb * 255).clip(0, 255).astype(np.uint8)
    for line in lines:
        if line.score < float(min_score):
            continue
        group_index = group_for_line.get(line.line_index)
        dense = group_index is not None
        item = {"line_index": line.line_index, "score": float(line.score),
                "route": "dense" if dense else "ordinary", "dense_group_index": group_index,
                "status": "processed" if dense else ordinary_records[line.line_index]["status"],
                "polygon": line.polygon_xy.tolist()}
        candidate_routes.append(item)
        color = (255, 70 + 50 * (int(group_index) % 3), 30) if dense else (30, 180, 255)
        polygon = np.rint(line.polygon_xy).astype(np.int32)
        cv2.polylines(overlay, [polygon], True, color, 2)
        position = tuple(np.maximum(0, polygon.min(axis=0)).tolist())
        cv2.putText(overlay, f"{line.line_index}:D{group_index}" if dense else f"{line.line_index}:O", position,
                    cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
    route_path = output_path.with_suffix(".routes.json")
    overlay_path = output_path.with_suffix(".routes.png")
    route_report = {"score_qualified_candidates": len(candidate_routes),
                    "dense_members": len(group_for_line),
                    "ordinary_candidates": sum(row["route"] == "ordinary" for row in candidate_routes),
                    "bypass_count": sum(row["status"] == "bypass" for row in candidate_routes),
                    "candidate_routes": candidate_routes}
    if route_report["dense_members"] + route_report["ordinary_candidates"] != len(candidate_routes):
        raise RuntimeError("candidate routing is not an exhaustive disjoint partition")
    route_path.write_text(json.dumps(route_report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rgb(overlay_path, overlay.astype(np.float32) / 255.0)

    summary = {
        "status": "passed",
        "input": str(input_path),
        "ocr_json": str(ocr_json_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "canvas_hw": list(canvas_hw),
        "resize_mode": resize_mode,
        "restore_interpolation": restore_interpolation,
        "ocr_format": ocr_format,
        "ocr_lines_input": ocr_lines_input,
        "ocr_lines_suppressed": ocr_lines_suppressed,
        "generic_ocr_nms_iou": 0.0,
        "route_report": str(route_path),
        "route_overlay": str(overlay_path),
        "candidate_routing": route_report,
        "dense_group_detector_contract": detector_contract_status,
        "dense_groups_candidate_count": len(dense_groups),
        "dense_groups": [{key: value for key, value in group.items() if key != "members"} for group in dense_groups],
        "dense_group_member_lines_suppressed": len(grouped_member_indices),
        "dense_group_core_target_px": dense_group_core_target_px,
        "sharp_text_bypass_laplacian": 0.0,
        "quality_guard": False,
        "p17_context": (
            {"vertical_ratio": float(p17_vertical_context), "horizontal_ratio": float(p17_horizontal_context)}
            if resize_mode == "p17_height96_tile_pad" else None
        ),
        "mask_guided_ocr_recovery": {
            "enabled": bool(mask_guided_ocr_recovery),
            "stats": geometry_stats,
            "accepted_lines": sum(item["status"] == "accepted" for item in geometry_decisions.values()),
            "fallback_lines": sum(item["status"] == "fallback" for item in geometry_decisions.values()),
            "debug_report": str(debug_dir / "geometry_recovery.json") if mask_guided_ocr_recovery and debug_dir else None,
        },
        "textmask_fallback_used": textmask_fallback_used,
        "textmask_status": textmask_status,
        "original_hw": list(original_hw),
        "working_hw": list(frame_hw),
        "lines_total": len(lines),
        "lines_processed": sum(item.get("status") == "processed" for item in line_records),
        "lines_bypassed": sum(item.get("status") == "bypass" for item in line_records),
        "segments_total": segments_total,
        "segments_native": segments_native,
        "segments_resized": segments_resized,
        "lines_resized": sum(item.get("route") in ("uniform_short_axis", "height96_resize", "height192_resize") for item in line_records),
        "processed_pixels": int(processed.sum()),
        "processed_fraction": float(processed.mean()),
        "context_ratio": float(context_ratio),
        "overlap": int(overlap),
        "blend_radius": int(blend_radius),
        "native_tile_height_threshold": int(native_tile_height_threshold),
        "force_native_tile_line_indices": sorted(force_native_tile_line_indices),
        "large_tile_canvas_hw": list(map(int, large_tile_canvas_hw)),
        "large_tile_overlap": int(large_tile_overlap),
        "large_tile_overlap_x": large_tile_overlap_x,
        "large_tile_overlap_y": large_tile_overlap_y,
        "pasteback_policy": {
            "tile_fusion": "two_dimensional_complementary_overlap",
            "ocr_envelope_alpha": "local_luma_contrast_glyph_support",
            "glyph_support_dilation_px": 1,
            "glyph_support_max_occupancy": 0.72,
            "context_feather_radius_px": int(blend_radius),
        },
        "lq_preprocess": getattr(tile_runner, "lq_preprocess_config", None),
        "min_score": float(min_score),
        "resize_bypass": bool(resize_bypass),
        "resize_min_ratio": float(resize_min_ratio),
        "resize_max_ratio": float(resize_max_ratio),
        "resize_target_short_ratio": float(resize_target_short_ratio),
        "elapsed_sec": float(time.perf_counter() - t0),
        "output": str(output_path),
        "debug_dir": str(debug_dir) if debug_dir else None,
        "lines": line_records,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """@luna-comment Parse the single-image evaluator CLI and write its summary.

    Input: image, OCR JSON, checkpoint, output path, and optional network config.
    Output: output image, JSON summary, and process exit code. Key Steps: parse
    options and invoke ``infer_full_image``. Key Params: ``--network-config``
    selects a registry graph; the optional mask-guided switch requires a
    versioned training-geometry JSON and preserves baseline geometry by default.
    State/Dependencies: local filesystem and inference helper.
    """

    parser = argparse.ArgumentParser(description="OCR JSON driven full-image BASIC text-line restoration")
    parser.add_argument("--input", type=Path, required=True, help="Input RGB/BGR image")
    parser.add_argument("--ocr-json", type=Path, required=True, help="PPOCR-style OCR JSON")
    parser.add_argument("--checkpoint", type=Path, required=True, help="BASIC V2 checkpoint")
    parser.add_argument("--network-config", type=Path, default=None,
                        help="Optional training YAML; network_g is built via the registry")
    parser.add_argument("--resize-mode", choices=("native", "native_tile", "native_tile_height192x960", "stretch", "height96_tile_pad", "p17_height96_tile_pad"), default="native",
                        help="native preserves crop geometry; stretch resizes every model tile to the full canvas")
    parser.add_argument("--output", type=Path, required=True, help="Output image at the original resolution")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-ratio", type=float, default=0.15,
                        help="AABB outward expansion as a fraction of the short side")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Native-pixel overlap between long-line tiles")
    parser.add_argument("--blend-radius", type=int, default=8,
                        help="Outward feather radius in native pixels")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--ocr-format", choices=("generic", "p17_annotations"), default="generic")
    parser.add_argument("--p17-vertical-context", type=float, default=.12)
    parser.add_argument("--p17-horizontal-context", type=float, default=.03)
    parser.add_argument("--mask-guided-ocr-recovery", action="store_true")
    parser.add_argument("--mask-guided-geometry-stats", type=Path, default=None)
    parser.add_argument("--native-tile-height-threshold", type=int, default=128,
                        help="Resize OCR regions up to this height; preserve scale and 2-D tile taller regions.")
    parser.add_argument("--large-tile-canvas", type=int, default=256,
                        help="Square V5 canvas for taller OCR regions; must be a multiple of eight.")
    parser.add_argument("--large-tile-overlap", type=int, default=96,
                        help="Two-dimensional overlap for the large native tile route.")
    parser.add_argument("--generic-ocr-nms-iou", type=float, default=0.0,
                        help="Deprecated compatibility argument; every ordinary candidate is retained.")
    parser.add_argument("--sharp-text-bypass-laplacian", type=float, default=0.0,
                        help="Deprecated compatibility argument; sharpness bypass is disabled.")
    parser.add_argument("--sharp-text-bypass-max-height", type=int, default=128,
                        help="Maximum OCR-envelope height eligible for the sharp-text bypass.")
    parser.add_argument("--disable-quality-guard", action="store_true",
                        help="Compatibility flag: output quality rejection has been removed.")
    parser.add_argument("--dense-group-core-target-px", type=float, default=None,
                        help="Compatibility field; dense patch scale is estimated from probability-band spacing.")
    parser.add_argument("--dense-group-probability-map", type=Path, default=None, help="DB probability map for grouped text-band recovery.")
    parser.add_argument("--text-mask", type=Path, default=None,
                        help="Optional P17 validation TextMask fallback for misaligned OCR annotations")
    parser.add_argument("--resize-bypass", action="store_true",
                        help="Uniformly resize eligible native short-axis bypasses before model inference")
    parser.add_argument("--resize-min-ratio", type=float, default=0.5)
    parser.add_argument("--resize-max-ratio", type=float, default=1.5)
    parser.add_argument("--resize-target-short-ratio", type=float, default=0.75,
                        help="Target core short-axis ratio inside the model short capacity")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="Optional per-line/per-tile debug output directory")
    args = parser.parse_args(argv)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    network, checkpoint_sha256 = _load_network(args.checkpoint, device, args.network_config)
    network_type = str(load_network_config(args.network_config).get("type", "")) if args.network_config else ""
    tile_runner = (
        NativeY4UVTileRunner(network, device, load_lq_preprocess_config(args.network_config))
        if "Y4UV" in network_type.upper() else None
    )
    summary = infer_full_image(
        args.input, args.ocr_json, args.checkpoint, args.output,
        device_name=args.device,
        context_ratio=args.context_ratio,
        overlap=args.overlap,
        blend_radius=args.blend_radius,
        min_score=args.min_score,
        resize_bypass=args.resize_bypass,
        resize_min_ratio=args.resize_min_ratio,
        resize_max_ratio=args.resize_max_ratio,
        resize_target_short_ratio=args.resize_target_short_ratio,
        debug_dir=args.debug_dir,
        network_config=args.network_config,
        network=network,
        checkpoint_sha256=checkpoint_sha256,
        resize_mode=args.resize_mode,
        ocr_format=args.ocr_format,
        p17_vertical_context=args.p17_vertical_context,
        p17_horizontal_context=args.p17_horizontal_context,
        mask_guided_ocr_recovery=args.mask_guided_ocr_recovery,
        mask_guided_geometry_stats=args.mask_guided_geometry_stats,
        native_tile_height_threshold=args.native_tile_height_threshold,
        large_tile_canvas_hw=(args.large_tile_canvas, args.large_tile_canvas),
        large_tile_overlap=args.large_tile_overlap,
        generic_ocr_nms_iou=args.generic_ocr_nms_iou,
        sharp_text_bypass_laplacian=args.sharp_text_bypass_laplacian,
        sharp_text_bypass_max_height=args.sharp_text_bypass_max_height,
        quality_guard=False,
        dense_group_core_target_px=args.dense_group_core_target_px,
        dense_group_probability_map=args.dense_group_probability_map,
        text_mask_path=args.text_mask,
        tile_runner=tile_runner,
    )
    print(json.dumps({key: summary[key] for key in (
        "status", "device", "lines_total", "lines_processed", "lines_bypassed",
        "segments_total", "segments_native", "segments_resized", "lines_resized",
        "processed_pixels", "processed_fraction", "elapsed_sec", "output",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
























