"""Continuous polygon-crossing low-frequency photometric harmonisation (V5.4).

V5.3 treated every OCR line polygon as a hard Dirichlet boundary.  That is
visually safe for the text pixels, but it also creates a visible ring when the
line polygon contains ordinary paper/background.  V5.4 uses the polygon only
to exclude unreliable residual samples (with a configurable halo), estimates
the LQ-minus-GT low-frequency residual from the remaining background, and
evaluates one continuous field over the complete image.  The field is added to
the V5 baseline in YUV, so no LQ high-frequency detail is copied.

This module is diagnostic-only.  It never publishes a dataset version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .background_photometric_v5 import (
    _dilate,
    _normalize_binary_mask,
    _percentile,
    _residual_metrics,
)
from .background_photometric_v5_1 import _normal_jump, _poly_mask, load_ocr_polygons
from .background_photometric_v5_3 import _masked_lowpass, _polygon_masks, _write_image
from .y_alignment import rgb2yuv_numpy, yuv2rgb_numpy


SCHEMA_VERSION = "background_photometric_v5_4_audit_v1"


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _robust_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, np.float32)
    if values.size == 0:
        return np.zeros(3, np.float32), np.ones(3, np.float32), np.ones(3, np.float32)
    center = np.median(values, axis=0).astype(np.float32)
    mad = np.median(np.abs(values - center), axis=0).astype(np.float32)
    # Y, U, V are deliberately bounded separately.  The bounds prevent one
    # overexposed area from turning the continuous field into clipping.
    clip = np.clip(3.0 * 1.4826 * mad, [2.0, 1.5, 1.5], [32.0, 18.0, 18.0]).astype(np.float32)
    return center, mad, clip


def _continuous_field(
    delta: np.ndarray,
    support: np.ndarray,
    *,
    sigma: float,
    center: np.ndarray,
    clip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth field and its support confidence.

    Two normalized Gaussian scales are blended by their support confidence.
    This matters inside an OCR exclusion halo: the narrow estimate naturally
    loses confidence there, and the broad estimate supplies a continuous
    interpolation instead of a mask-shaped hole.
    """
    clipped = np.clip(delta, center - clip, center + clip).astype(np.float32)
    narrow, narrow_conf = _masked_lowpass(clipped, support, max(1.0, float(sigma)))
    broad, broad_conf = _masked_lowpass(clipped, support, max(2.0, float(sigma) * 2.5))
    # Gaussian support confidence is in [0, 1] for an interior support pixel.
    # The smoothstep avoids a hard denominator threshold around polygons.
    cn = _smoothstep(narrow_conf / 0.35)
    cb = _smoothstep(broad_conf / 0.18)
    anchor = center.reshape(1, 1, 3).astype(np.float32)
    broad_or_anchor = cb[..., None] * broad + (1.0 - cb[..., None]) * anchor
    field = cn[..., None] * narrow + (1.0 - cn[..., None]) * broad_or_anchor
    # A final low-frequency pass removes tiny confidence-grid curvature and
    # makes the field C0/C1-like across both sides of the polygon halo.
    field = cv2.GaussianBlur(field.astype(np.float32), (0, 0), max(0.8, float(sigma) * 0.35))
    confidence = np.maximum(cn, cb).astype(np.float32)
    return field.astype(np.float32), confidence, np.maximum(narrow_conf, broad_conf).astype(np.float32)


def _field_jump(field_yuv: np.ndarray, region: np.ndarray, valid: np.ndarray, step: int = 2) -> dict[str, float]:
    """Measure the correction jump itself across an OCR polygon boundary."""
    region_bool = region > 0
    boundary = region_bool & (cv2.erode(region_bool.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    ys, xs = np.nonzero(boundary & (valid > 0))
    height, width = region_bool.shape
    values: list[np.ndarray] = []
    signed = cv2.GaussianBlur(region_bool.astype(np.float32), (0, 0), 2.0)
    gx = -cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    gy = -cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(gx * gx + gy * gy) + 1e-6
    gx, gy = gx / norm, gy / norm
    for y, x in zip(ys.tolist(), xs.tolist()):
        xi = int(round(x - float(gx[y, x]) * step)); yi = int(round(y - float(gy[y, x]) * step))
        xo = int(round(x + float(gx[y, x]) * step)); yo = int(round(y + float(gy[y, x]) * step))
        if not (0 <= xi < width and 0 <= xo < width and 0 <= yi < height and 0 <= yo < height):
            continue
        if not region_bool[yi, xi] or region_bool[yo, xo] or not (valid[yi, xi] and valid[yo, xo]):
            continue
        values.append(np.abs(field_yuv[yi, xi] - field_yuv[yo, xo]))
    if not values:
        return {"count": 0, "Y_p50": 0.0, "Y_p95": 0.0, "U_p95": 0.0, "V_p95": 0.0}
    array = np.asarray(values, np.float32)
    return {"count": int(len(values)), "Y_p50": _percentile(array[:, 0], 50),
            "Y_p95": _percentile(array[:, 0], 95), "U_p95": _percentile(array[:, 1], 95),
            "V_p95": _percentile(array[:, 2], 95)}


def _score(metrics: dict[str, Any], channel: str) -> float:
    item = next((x for x in metrics.get("channels", []) if x.get("channel") == channel), {})
    return float(np.mean([float(item.get(key, 0.0)) for key in ("mean_abs", "median_abs", "p90_abs")]))


def align_background_v5_4(
    lq_bgr: np.ndarray,
    gt_bgr: np.ndarray,
    valid_mask: np.ndarray,
    text_mask: np.ndarray,
    *,
    polygons: Sequence[np.ndarray] | None,
    v5_base_bgr: np.ndarray,
    exclusion_halo: int = 12,
    sigma: float = 18.0,
    min_support_pixels: int = 256,
    seam_tolerance: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Build a V5.4 candidate with one continuous field over all valid pixels."""
    if lq_bgr.shape != gt_bgr.shape or v5_base_bgr.shape != gt_bgr.shape:
        raise ValueError("LQ/GT/V5 shape mismatch")
    h, w = gt_bgr.shape[:2]
    valid = _normalize_binary_mask(valid_mask, (h, w), "valid_mask") > 0
    text = _normalize_binary_mask(text_mask, (h, w), "text_mask") > 0
    v5 = np.asarray(v5_base_bgr, np.uint8).copy()
    masks = _polygon_masks(polygons, (h, w)) if polygons else []
    polygon_union = np.zeros((h, w), np.uint8)
    for mask in masks:
        polygon_union[mask > 0] = 255
    empty = {"v5": v5, "candidate": v5.copy(), "selected": v5.copy(),
             "polygon_mask": polygon_union, "exclusion_halo": np.zeros((h, w), np.uint8),
             "support": np.zeros((h, w), np.uint8), "fitted_delta_y": np.zeros((h, w), np.float32),
             "fitted_delta_u": np.zeros((h, w), np.float32), "fitted_delta_v": np.zeros((h, w), np.float32),
             "field_confidence": np.zeros((h, w), np.float32),
             # Keep fallback diagnostics serializable: callers always render
             # correction planes, even when no field passed the safety gates.
             "correction_y": np.zeros((h, w), np.float32),
             "correction_u": np.zeros((h, w), np.float32),
             "correction_v": np.zeros((h, w), np.float32)}
    if not masks:
        audit = {"schema_version": SCHEMA_VERSION, "status": "fallback", "accepted": False,
                 "fallback_reason": "NO_TEXT_POLYGON", "pixels": {"polygon_changed": 0, "invalid_changed": 0}}
        return v5.copy(), audit, empty

    exclusion = _dilate((polygon_union > 0) | text, int(max(0, exclusion_halo))) > 0
    support = valid & ~exclusion
    lq_yuv = rgb2yuv_numpy(lq_bgr.astype(np.float32)[..., ::-1])
    gt_yuv = rgb2yuv_numpy(gt_bgr.astype(np.float32)[..., ::-1])
    v5_yuv = rgb2yuv_numpy(v5.astype(np.float32)[..., ::-1])
    desired = lq_yuv - gt_yuv
    current = v5_yuv - gt_yuv
    delta = desired - current
    values = delta[support]
    if values.shape[0] < int(min_support_pixels):
        audit = {"schema_version": SCHEMA_VERSION, "status": "fallback", "accepted": False,
                 "fallback_reason": "INSUFFICIENT_BACKGROUND_SUPPORT", "pixels": {"support": int(values.shape[0]), "polygon_changed": 0, "invalid_changed": 0}}
        empty["exclusion_halo"] = exclusion.astype(np.uint8) * 255
        empty["support"] = support.astype(np.uint8) * 255
        return v5.copy(), audit, empty
    center, mad, clip = _robust_stats(values)
    if center[0] > 80 or np.max(np.abs(center[1:])) > 50 or mad[0] > 24 or np.max(mad[1:]) > 14:
        audit = {"schema_version": SCHEMA_VERSION, "status": "fallback", "accepted": False,
                 "fallback_reason": "UNSTABLE_BACKGROUND_RESIDUAL", "residual_median_yuv": center.tolist(), "residual_mad_yuv": mad.tolist(), "pixels": {"support": int(values.shape[0]), "polygon_changed": 0, "invalid_changed": 0}}
        return v5.copy(), audit, empty

    field, confidence, raw_confidence = _continuous_field(delta, support, sigma=sigma, center=center, clip=clip)
    candidate_yuv = v5_yuv + field
    candidate_rgb_float = yuv2rgb_numpy(candidate_yuv)
    clipping_pixels = int(np.count_nonzero(np.any((candidate_rgb_float < 0.0) | (candidate_rgb_float > 255.0), axis=2) & valid))
    candidate_bgr = np.rint(np.clip(candidate_rgb_float[..., ::-1], 0, 255)).astype(np.uint8)
    candidate = v5.copy()
    candidate[valid] = candidate_bgr[valid]
    candidate[~valid] = v5[~valid]

    # V5.4's only hard write constraint is valid-mask preservation.  A small
    # lambda line search controls clipping/over-correction without restoring
    # the obsolete polygon hard boundary.
    eval_mask = valid & ~exclusion
    if not np.any(eval_mask):
        eval_mask = valid
    before = _residual_metrics(lq_bgr, v5, eval_mask)
    before_all = _residual_metrics(lq_bgr, v5, valid)
    baseline_jump = [_normal_jump(v5, m, valid.astype(np.uint8), step=2) for m in masks]
    baseline_field_jump = [_field_jump(np.zeros_like(v5_yuv), m, valid) for m in masks]
    evaluations: list[dict[str, Any]] = []
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        image = v5.copy() if lam == 0 else np.rint(np.clip(v5.astype(np.float32) + lam * (candidate.astype(np.float32) - v5.astype(np.float32)), 0, 255)).astype(np.uint8)
        image[~valid] = v5[~valid]
        metrics = _residual_metrics(lq_bgr, image, eval_mask)
        metrics_all = _residual_metrics(lq_bgr, image, valid)
        jumps = [_normal_jump(image, m, valid.astype(np.uint8), step=2) for m in masks]
        jump = _percentile(np.asarray([x["Y_p95"] for x in jumps], np.float32), 95)
        nonregressed = all(_score(metrics, c) <= _score(before, c) + (0.35 if c == "Y" else 0.25) for c in ("Y", "U", "V"))
        improvement = _score(before, "Y") - _score(metrics, "Y")
        evaluations.append({"lambda": float(lam), "image": image, "residual": metrics, "residual_all": metrics_all,
                            "normal_jump_Y_p95": float(jump), "local_Y_improvement": float(improvement),
                            "residual_nonregressed": bool(nonregressed),
                            # A small but repeatable local improvement is
                            # sufficient here: V5.4 is specifically a
                            # continuity experiment, and demanding the old
                            # 0.05 image-residual threshold can incorrectly
                            # select byte-identical V5 on real samples whose
                            # seam is the dominant visual defect.
                            "useful": bool(improvement >= 0.01 or jump < _percentile(np.asarray([x["Y_p95"] for x in baseline_jump], np.float32), 95) - 0.05)})
    positive = [item for item in evaluations if item["lambda"] > 0 and item["residual_nonregressed"] and item["useful"]]
    chosen = positive[-1] if positive else evaluations[0]
    selected = chosen["image"].copy()
    selected[~valid] = v5[~valid]
    polygon_changed = int(np.any(selected[polygon_union > 0] != v5[polygon_union > 0], axis=1).sum()) if np.any(polygon_union) else 0
    invalid_changed = int(np.any(selected[~valid] != v5[~valid], axis=1).sum()) if np.any(~valid) else 0
    selected_yuv = rgb2yuv_numpy(selected.astype(np.float32)[..., ::-1])
    correction = selected_yuv - v5_yuv
    correction_jump = [_field_jump(correction, m, valid) for m in masks]
    selected_jumps = [_normal_jump(selected, m, valid.astype(np.uint8), step=2) for m in masks]
    baseline_y = _percentile(np.asarray([x["Y_p95"] for x in baseline_jump], np.float32), 95)
    selected_y = _percentile(np.asarray([x["Y_p95"] for x in selected_jumps], np.float32), 95)
    highpass = correction - cv2.GaussianBlur(correction.astype(np.float32), (0, 0), max(1.0, float(sigma) * 0.5))
    highpass_rms = float(np.sqrt(np.mean(highpass[valid] ** 2))) if np.any(valid) else 0.0
    correction_jump_p95 = _percentile(np.asarray([x["Y_p95"] for x in correction_jump], np.float32), 95)
    continuity_ok = bool(correction_jump_p95 <= float(seam_tolerance))
    accepted = bool(positive and polygon_changed >= 0 and invalid_changed == 0 and clipping_pixels <= max(1, int(valid.sum() * 0.001)) and continuity_ok and np.isfinite(selected).all())
    if not accepted:
        # A diagnostic candidate must never leak through a failed continuity
        # or clipping gate. Keep the fallback byte-identical to V5.
        selected = v5.copy()
    audit = {"schema_version": SCHEMA_VERSION, "status": "accepted" if accepted else "fallback", "accepted": accepted,
             "fallback_reason": None if accepted else "NO_ACCEPTABLE_LAMBDA", "parameters": {"exclusion_halo": int(exclusion_halo), "sigma": float(sigma), "min_support_pixels": int(min_support_pixels), "seam_tolerance": float(seam_tolerance), "lambda_grid": [0.0, 0.25, 0.5, 0.75, 1.0]},
             "shape": [int(w), int(h)], "region_count": len(masks), "pixels": {"polygon": int(np.count_nonzero(polygon_union)), "exclusion_halo": int(exclusion.sum()), "support": int(support.sum()), "polygon_changed": polygon_changed, "invalid_changed": invalid_changed, "rgb_clipping": clipping_pixels},
             "residual_median_yuv": center.tolist(), "residual_mad_yuv": mad.tolist(), "residual_clip_yuv": clip.tolist(),
             "residual_v5": before, "residual_selected": chosen["residual"], "residual_all_v5": before_all, "residual_all_selected": chosen["residual_all"],
             "continuity": {"normal_jump_v5": baseline_jump, "normal_jump_selected": selected_jumps, "normal_jump_Y_p95_v5": float(baseline_y), "normal_jump_Y_p95_selected": float(selected_y), "correction_jump_selected": correction_jump, "correction_jump_Y_p95": correction_jump_p95, "correction_jump_gate_passed": continuity_ok, "improved": bool(selected_y <= baseline_y + float(seam_tolerance))},
             "field": {"Y_p01": float(np.percentile(field[..., 0], 1)), "Y_p99": float(np.percentile(field[..., 0], 99)), "highpass_rms_yuv": highpass_rms, "confidence_p50": float(np.percentile(confidence, 50)), "confidence_p05": float(np.percentile(confidence, 5))},
             "line_search": [{k: v for k, v in item.items() if k != "image"} for item in evaluations]}
    arrays = {"v5": v5, "candidate": candidate, "selected": selected, "polygon_mask": polygon_union,
              "exclusion_halo": exclusion.astype(np.uint8) * 255, "support": support.astype(np.uint8) * 255,
              "fitted_delta_y": field[..., 0], "fitted_delta_u": field[..., 1], "fitted_delta_v": field[..., 2],
              "field_confidence": confidence, "correction_y": correction[..., 0], "correction_u": correction[..., 1], "correction_v": correction[..., 2]}
    return selected, audit, arrays


def generate_real_diagnostic(output: Path, *, lq_path: Path, gt_path: Path, valid_path: Path, text_path: Path,
                             ocr_path: Path, v5_base_path: Path, sample_id: str = "S24U_195_image_07",
                             exclusion_halo: int = 12, sigma: float = 18.0, force: bool = False) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"diagnostic directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    read_color = lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR)
    read_gray = lambda p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    lq, gt, valid, text, v5 = read_color(lq_path), read_color(gt_path), read_gray(valid_path), read_gray(text_path), read_color(v5_base_path)
    if any(image is None for image in (lq, gt, valid, text, v5)):
        raise FileNotFoundError("one or more diagnostic inputs cannot be read")
    polygons = load_ocr_polygons(ocr_path, gt.shape[:2])
    selected, audit, arrays = align_background_v5_4(lq, gt, valid, text, polygons=polygons, v5_base_bgr=v5, exclusion_halo=exclusion_halo, sigma=sigma)
    diff = np.max(np.abs(selected.astype(np.int16) - v5.astype(np.int16)), axis=2).astype(np.uint8)
    correction_yuv = np.stack([arrays["correction_y"], arrays["correction_u"], arrays["correction_v"]], axis=-1).astype(np.float32)
    boundary_jump_map = np.zeros(gt.shape[:2], np.float32)
    for region in _polygon_masks(polygons, gt.shape[:2]):
        region_bool = region > 0
        boundary = region_bool & (cv2.erode(region_bool.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
        ys, xs = np.nonzero(boundary & (valid > 0))
        signed = cv2.GaussianBlur(region_bool.astype(np.float32), (0, 0), 2.0)
        gx = -cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3); gy = -cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.sqrt(gx * gx + gy * gy) + 1e-6; gx, gy = gx / norm, gy / norm
        for y, x in zip(ys.tolist(), xs.tolist()):
            xi = int(round(x - float(gx[y, x]) * 2)); yi = int(round(y - float(gy[y, x]) * 2)); xo = int(round(x + float(gx[y, x]) * 2)); yo = int(round(y + float(gy[y, x]) * 2))
            if 0 <= xi < gt.shape[1] and 0 <= xo < gt.shape[1] and 0 <= yi < gt.shape[0] and 0 <= yo < gt.shape[0] and region_bool[yi, xi] and not region_bool[yo, xo] and valid[yi, xi] and valid[yo, xo]:
                boundary_jump_map[y, x] = abs(float(correction_yuv[yi, xi, 0] - correction_yuv[yo, xo, 0]))
    files = {"LQ.png": lq, "GT_V3.png": gt, "GT_V5_baseline.png": v5, "GT_V5_4_candidate.png": arrays["candidate"], "GT_V5_4_selected.png": selected,
             "V5_vs_V5_4_abs_diff_x8.png": cv2.applyColorMap(np.clip(diff * 8, 0, 255), cv2.COLORMAP_TURBO), "polygon_mask.png": arrays["polygon_mask"],
             "exclusion_halo.png": arrays["exclusion_halo"], "support.png": arrays["support"], "field_confidence.png": np.rint(np.clip(arrays["field_confidence"], 0, 1) * 255).astype(np.uint8),
             "correction_Y.png": arrays["correction_y"], "correction_U.png": arrays["correction_u"], "correction_V.png": arrays["correction_v"], "boundary_jump_map.png": cv2.applyColorMap(np.clip(boundary_jump_map * 32, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO), "boundary_jump_map_Y.png": np.clip(boundary_jump_map * 32, 0, 255).astype(np.uint8), "valid_mask.png": valid}
    for name, image in files.items():
        _write_image(output / name, image)
    def label(image: np.ndarray, title: str) -> np.ndarray:
        image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (640, 484)); cv2.putText(image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA); return image
    _write_image(output / "comparison_v5_vs_v5_4.png", np.concatenate([label(gt, "GT_V3"), label(v5, "GT_V5 baseline"), label(arrays["candidate"], "V5.4 continuous candidate"), label(cv2.applyColorMap(np.clip(diff * 8, 0, 255), cv2.COLORMAP_TURBO), "V5 -> V5.4 diff x8")], axis=1))
    # Enlarged visual is intentionally a crop around all OCR polygons, not a
    # glyph visualization.  It shows polygon, halo, field and output together.
    masks = _polygon_masks(polygons, gt.shape[:2])
    crops = []
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask > 0)
        if not ys.size: continue
        x0, x1 = max(0, int(xs.min()) - 80), min(gt.shape[1], int(xs.max()) + 81); y0, y1 = max(0, int(ys.min()) - 80), min(gt.shape[0], int(ys.max()) + 81)
        strip = np.concatenate([label(v5[y0:y1, x0:x1], f"r{index} V5"), label(selected[y0:y1, x0:x1], f"r{index} V5.4"), label(arrays["polygon_mask"][y0:y1, x0:x1], f"r{index} polygon"), label(arrays["field_confidence"][y0:y1, x0:x1], f"r{index} confidence")], axis=1)
        crops.append(cv2.resize(strip, (strip.shape[1] * 2, strip.shape[0] * 2), interpolation=cv2.INTER_NEAREST))
    if crops: _write_image(output / "boundary_zoom_8x.png", np.concatenate(crops, axis=0))
    audit["diagnostic"] = {"sample_id": sample_id, "output_directory": str(output), "source_inputs": {"lq": str(lq_path), "gt": str(gt_path), "valid_mask": str(valid_path), "text_mask": str(text_path), "ocr": str(ocr_path), "v5_base": str(v5_base_path)}, "files": sorted(path.name for path in output.glob("*.png")) + ["audit.json"], "note": "V5.4 continuous field; OCR polygons are excluded from residual statistics only, never used as hard write boundaries or glyph estimates."}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "accepted": bool(audit["accepted"]), "audit": audit}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "lq", "gt", "valid-mask", "text-mask", "ocr", "v5-base"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample-id", default="S24U_195_image_07"); parser.add_argument("--exclusion-halo", type=int, default=12); parser.add_argument("--sigma", type=float, default=18.0); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = generate_real_diagnostic(args.output, lq_path=args.lq, gt_path=args.gt, valid_path=args.valid_mask, text_path=args.text_mask, ocr_path=args.ocr, v5_base_path=args.v5_base, sample_id=args.sample_id, exclusion_halo=args.exclusion_halo, sigma=args.sigma, force=args.force)
    print(json.dumps({"status": "generated", "output": result["output"], "accepted": result["accepted"], "fallback_reason": result["audit"].get("fallback_reason")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
