"""Offline GT-fitted tone and chroma maps for aligned text-line pairs."""

from __future__ import annotations

import cv2
import numpy as np

from tools.text_lq_scurve_mild import _apply_luma_delta, _border_distance, _luma_bgr, _smoothstep


def _fit_curve(gt: np.ndarray, border_px: int = 6) -> tuple[np.ndarray | None, dict]:
    """@luna-comment
    Input: Aligned uint8 BGR GT line and protected border width.
    Output: GT-derived 256-entry luma LUT or a confidence fallback with fit metadata.
    Key Steps: Detect border polarity, Otsu classes and robust core/background anchors; build a bounded S curve with endpoint rails.
    Key Params: Class share 0.03, contrast 20, gamma 1.8, pivot 0.5.
    State/Dependencies: Pure NumPy/OpenCV computation; source is unchanged.
    """
    y = _luma_bgr(gt)
    border = _border_distance(y.shape) < border_px
    if min(y.shape) < 2 * border_px + 20:
        return None, {"fallback_reason": "crop_too_small"}
    threshold, _ = cv2.threshold(y.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_share = float(np.mean(y[border] > threshold))
    if high_share >= 0.8:
        polarity = 1
    elif high_share <= 0.2:
        polarity = -1
    else:
        return None, {"fallback_reason": "ambiguous_gt_border_polarity"}
    normalized = y if polarity == 1 else 255.0 - y
    otsu, _ = cv2.threshold(normalized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = normalized <= otsu
    share = float(np.mean(foreground))
    if min(share, 1.0 - share) < 0.03:
        return None, {"fallback_reason": "insufficient_gt_class_share"}
    core = (cv2.erode(foreground.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~border
    background = border & ~foreground
    if np.count_nonzero(core) < 32 or np.count_nonzero(background) < 64:
        return None, {"fallback_reason": "insufficient_gt_support"}
    low = float(np.median(normalized[core]))
    high = float(np.median(normalized[background]))
    if high - low < 20:
        return None, {"fallback_reason": "insufficient_gt_contrast"}
    levels = np.arange(256, dtype=np.float32)
    t = np.clip((levels - low) / (high - low), 0.0, 1.0)
    power = 1.8
    curved = t ** power / np.maximum(t ** power + (1.0 - t) ** power, 1e-8)
    mapped = low + (high - low) * curved
    # The two constant rails remove out-of-range dark and bright overshoot.
    lut_normalized = np.clip(np.rint(mapped), 0, 255).astype(np.uint8)
    lut = lut_normalized if polarity == 1 else (255 - lut_normalized[::-1]).astype(np.uint8)
    return lut, {"fallback_reason": None, "polarity": polarity, "gt_otsu": int(otsu),
                 "gt_foreground_anchor": low, "gt_background_rail": high,
                 "gt_foreground_share": share}


def _map_y(image: np.ndarray, lut: np.ndarray, border_px: int) -> np.ndarray:
    """@luna-comment
    Input: BGR line, shared GT-derived luma LUT and protected border width.
    Output: A new BGR image after common-channel luminance displacement.
    Key Steps: Look up source Y, form signed delta and hold the crop border unchanged.
    Key Params: Six-pixel border is reserved for later training geometry.
    State/Dependencies: Pure array transform with no mutation.
    """
    y = _luma_bgr(image).astype(np.uint8)
    delta = lut[y].astype(np.float32) - y.astype(np.float32)
    delta[_border_distance(y.shape) < border_px] = 0
    return _apply_luma_delta(image, delta)


def _gradient(channel: np.ndarray) -> np.ndarray:
    """@luna-comment
    Input: Half-resolution float32 channel.
    Output: Maximum three-tap Sobel slope on an eight-bit scale.
    Key Steps: Compute horizontal and vertical slopes and take their maximum.
    Key Params: Sobel gain is divided by eight like frozen V6.
    State/Dependencies: Pure OpenCV computation.
    """
    dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    return np.maximum(np.abs(dx), np.abs(dy)) / 8.0


def _fit_chroma(gt: np.ndarray, polarity: int, border_px: int) -> tuple[dict | None, dict]:
    """@luna-comment
    Input: Tone-mapped GT, GT-derived text polarity, and protected border width.
    Output: GT-only chroma reference field and normal tolerance, or fallback.
    Key Steps: Select smooth distant background, robustly smooth U/V, then fit V6 normal projection limit.
    Key Params: Half-resolution distance 9, 41-pixel box, 95th percentile.
    State/Dependencies: Pure NumPy/OpenCV; no LQ statistics enter the fit.
    """
    height, width = gt.shape[:2]
    if min(height // 2, width // 2) < 16:
        return None, {"chroma_fallback_reason": "small_chroma_canvas"}
    yuv = cv2.cvtColor(gt, cv2.COLOR_BGR2YUV).astype(np.float32)
    sy, su, sv = cv2.split(cv2.resize(yuv, (width // 2, height // 2), interpolation=cv2.INTER_AREA))
    threshold, _ = cv2.threshold(sy.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = sy > threshold if polarity == -1 else sy < threshold
    distance = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 3)
    reliable = ((distance >= 9) & (_gradient(cv2.GaussianBlur(sy, (0, 0), 0.7)) < 9)
                & (_gradient(su) < 4) & (_gradient(sv) < 4))
    b = max(1, border_px // 2)
    reliable[:b] = False; reliable[-b:] = False; reliable[:, :b] = False; reliable[:, -b:] = False
    count = int(np.count_nonzero(reliable))
    if count < 64:
        return None, {"chroma_fallback_reason": "insufficient_gt_chroma_reference", "gt_chroma_reference_pixels": count}
    weights = reliable.astype(np.float32)
    median_u, median_v = float(np.median(su[reliable])), float(np.median(sv[reliable]))

    def reference(weight: np.ndarray, channel: np.ndarray, median: float) -> np.ndarray:
        """@luna-comment
        Input: GT reliability weights, half-resolution chroma and robust prior.
        Output: Smooth GT background chroma reference.
        Key Steps: Box-filter weighted values and normalize with a small prior.
        Key Params: 41-pixel window and 0.005 prior match frozen V6.
        State/Dependencies: Pure OpenCV computation.
        """
        return (cv2.boxFilter(weight * channel, -1, (41, 41), normalize=True) + .005 * median) / (
            cv2.boxFilter(weight, -1, (41, 41), normalize=True) + .005)

    ru, rv = reference(weights, su, median_u), reference(weights, sv, median_v)
    residual = np.hypot(su - ru, sv - rv)
    weights *= np.clip(6 / (residual + 1e-5), 0, 1)
    ru, rv = reference(weights, su, median_u), reference(weights, sv, median_v)
    length = np.hypot(128 - ru, 128 - rv) + 1e-6
    au, av = (128 - ru) / length, (128 - rv) / length
    limit = max(1.5, float(np.percentile(((su - ru) * au + (sv - rv) * av)[reliable], 95)))
    size = (width, height)
    field = {"u": cv2.resize(ru, size), "v": cv2.resize(rv, size),
             "support": cv2.resize(cv2.boxFilter(weights, -1, (41, 41), normalize=True), size),
             "limit": limit, "polarity": polarity}
    return field, {"chroma_fallback_reason": None, "gt_chroma_reference_pixels": count,
                   "gt_chroma_threshold": limit, "gt_chroma_reference_uv": [median_u, median_v]}


def _map_chroma(image: np.ndarray, field: dict, border_px: int) -> np.ndarray:
    """@luna-comment
    Input: Tone-mapped paired image and GT-only UV field/tolerance.
    Output: Background-corrected BGR image with exact protected Y and text.
    Key Steps: Project excess toward neutral, protect text and border, restore any RGB conversion Y drift.
    Key Params: V6 14-level cap, 0.9 strength, first antialias pixel guard.
    State/Dependencies: Pure NumPy/OpenCV transform; only image residual varies between the pair.
    """
    height, width = image.shape[:2]
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32)
    y, u, v = cv2.split(yuv)
    ru, rv, support = field["u"], field["v"], field["support"]
    length = np.hypot(128 - ru, 128 - rv) + 1e-6
    au, av = (128 - ru) / length, (128 - rv) / length
    projection = (u - ru) * au + (v - rv) * av
    perpendicular = -(u - ru) * av + (v - rv) * au
    excess = np.maximum(projection - field["limit"], 0)
    confidence = (np.clip((support - .025) / .10, 0, 1) * np.clip((length - 3) / 6, 0, 1)
                  * np.clip((8 - np.abs(perpendicular)) / 4, 0, 1))
    threshold, _ = cv2.threshold(y.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = y > threshold if field["polarity"] == -1 else y < threshold
    distance = cv2.distanceTransform((~foreground).astype(np.uint8), cv2.DIST_L2, 3)
    spatial = np.clip((distance - 2.4) / 3.6, 0, 1) * np.clip((70 - distance) / 20, 0, 1)
    amount = .9 * np.minimum(excess, 14) * _smoothstep(excess / 3) * confidence * spatial
    protected = cv2.dilate(foreground.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    border = _border_distance((height, width)) < border_px
    amount[protected | border] = 0
    changed = amount > .01
    if not np.any(changed):
        return image.copy()
    yuv[:, :, 1] = u - amount * au
    yuv[:, :, 2] = v - amount * av
    output = cv2.cvtColor(np.clip(np.rint(yuv), 0, 255).astype(np.uint8), cv2.COLOR_YUV2BGR)
    output[~changed] = image[~changed]
    output[_luma_bgr(output) != _luma_bgr(image)] = image[_luma_bgr(output) != _luma_bgr(image)]
    return output


def preprocess_gt_fitted_pair(lq: np.ndarray, gt: np.ndarray, chroma: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """@luna-comment
    Input: Aligned uint8 BGR LQ/GT line crops and optional V6-style chroma stage.
    Output: Independently mapped LQ/GT using only GT-fitted parameters and telemetry.
    Key Steps: Fit GT tone, apply one LUT to both images, optionally fit GT chroma and apply one field to both.
    Key Params: Six-pixel protected border; failed fits return exact identity or tone-only pair.
    State/Dependencies: Pure pair transform; no filesystem or mutable global state.
    """
    if (lq.shape != gt.shape or lq.ndim != 3 or lq.shape[2] != 3
            or lq.dtype != np.uint8 or gt.dtype != np.uint8):
        raise ValueError("LQ and GT must be aligned uint8 BGR images")
    lut, info = _fit_curve(gt)
    info["fit_source"] = "gt"
    info["version"] = "gt_fitted_v6_pair_v1" if chroma else "gt_fitted_v5_pair_v1"
    if lut is None:
        return lq.copy(), gt.copy(), info
    out_lq, out_gt = _map_y(lq, lut, 6), _map_y(gt, lut, 6)
    info["lut_sha256"] = __import__("hashlib").sha256(lut.tobytes()).hexdigest()
    if chroma:
        field, chroma_info = _fit_chroma(out_gt, info["polarity"], 6)
        info.update(chroma_info)
        if field is not None:
            out_lq, out_gt = _map_chroma(out_lq, field, 6), _map_chroma(out_gt, field, 6)
    info["lq_changed_pixels"] = int(np.count_nonzero(np.any(out_lq != lq, axis=2)))
    info["gt_changed_pixels"] = int(np.count_nonzero(np.any(out_gt != gt, axis=2)))
    return out_lq, out_gt, info
