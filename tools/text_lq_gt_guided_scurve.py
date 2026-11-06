"""GT-derived tone curve applied to aligned raw LQ text-line crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from tools.text_lq_scurve_mild import _apply_luma_delta, _border_distance, _luma_bgr


@dataclass(frozen=True)
class GTGuidedCurveConfig:
    version: str = "text_lq_gt_guided_scurve_v1"
    border_px: int = 6
    min_class_share: float = 0.03
    min_contrast: float = 20.0
    gamma: float = 1.8
    pivot: float = 0.42
    blend: float = 0.70
    max_delta: float = 18.0


DEFAULT_CONFIG = GTGuidedCurveConfig()


def preprocess_text_lq_from_gt(
    lq: np.ndarray, gt: np.ndarray, config: GTGuidedCurveConfig = DEFAULT_CONFIG
) -> tuple[np.ndarray, dict[str, object]]:
    """@luna-comment
    Input: Spatially aligned HWC uint8 BGR raw LQ and GT line crops.
    Output: New BGR LQ and JSON-serializable GT-curve telemetry.
    Key Steps: Fit GT polarity/anchors, form a bounded monotone S-curve, then apply it to LQ luma.
    Key Params: Frozen v1 pivot, gamma, strength, border protection, and confidence gates.
    State/Dependencies: Pure OpenCV/NumPy operation; never modifies either input.
    """
    if (not isinstance(lq, np.ndarray) or not isinstance(gt, np.ndarray) or lq.shape != gt.shape
            or lq.ndim != 3 or lq.shape[2] != 3 or lq.dtype != np.uint8 or gt.dtype != np.uint8):
        raise ValueError("LQ and GT must be aligned HWC uint8 BGR arrays")
    if config != DEFAULT_CONFIG:
        raise ValueError("GT-guided v1 configuration is frozen")
    info: dict[str, object] = {"config_version": config.version, "curve_source": "gt"}

    def fallback(reason: str) -> tuple[np.ndarray, dict[str, object]]:
        """@luna-comment
        Input: A low-confidence reason code.
        Output: Exact copied LQ and reason-bearing telemetry.
        Key Steps: Mark identity behavior without changing source arrays.
        Key Params: The reason is persisted in each dataset manifest row.
        State/Dependencies: No filesystem or shared state.
        """
        info.update({"fallback_reason": reason, "changed_pixels": 0})
        return lq.copy(), info

    if min(lq.shape[:2]) < 2 * config.border_px + 20:
        return fallback("crop_too_small")
    gt_luma = _luma_bgr(gt)
    gt_threshold, _ = cv2.threshold(gt_luma.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = _border_distance(gt_luma.shape) < config.border_px
    high_share = float(np.mean(gt_luma[border] > gt_threshold))
    if high_share >= 0.80:
        polarity = 1
    elif high_share <= 0.20:
        polarity = -1
    else:
        return fallback("ambiguous_gt_border_polarity")
    normalized_gt = gt_luma if polarity == 1 else 255.0 - gt_luma
    normalized_threshold, _ = cv2.threshold(
        normalized_gt.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    foreground = normalized_gt < normalized_threshold + 0.5
    foreground_share = float(np.mean(foreground))
    if min(foreground_share, 1.0 - foreground_share) < config.min_class_share:
        return fallback("insufficient_gt_class_share")
    core = cv2.erode(foreground.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    core &= ~border
    background = border & ~foreground
    if np.count_nonzero(core) < 32 or np.count_nonzero(background) < 64:
        return fallback("insufficient_gt_support")
    foreground_anchor = float(np.median(normalized_gt[core]))
    background_anchor = float(np.median(normalized_gt[background]))
    span = background_anchor - foreground_anchor
    if span < config.min_contrast:
        return fallback("insufficient_gt_contrast")

    lq_luma = _luma_bgr(lq)
    normalized_lq = lq_luma if polarity == 1 else 255.0 - lq_luma
    t = np.clip((normalized_lq - foreground_anchor) / span, 0.0, 1.0)
    left = np.power(t, config.gamma)
    right = np.power(1.0 - t, config.gamma)
    k = (config.pivot / (1.0 - config.pivot)) ** config.gamma
    curve = left / np.maximum(left + k * right, 1e-8)
    delta = np.clip(config.blend * span * (curve - t), -config.max_delta, config.max_delta)
    delta[(normalized_lq <= foreground_anchor) | (normalized_lq >= background_anchor) | border] = 0.0
    output = _apply_luma_delta(lq, polarity * delta)
    info.update({
        "fallback_reason": None,
        "polarity": "dark_text_light_background" if polarity == 1 else "light_text_dark_background",
        "gt_otsu_threshold": float(normalized_threshold),
        "gt_foreground_anchor": foreground_anchor,
        "gt_background_anchor": background_anchor,
        "gt_foreground_share": foreground_share,
        "changed_pixels": int(np.count_nonzero(np.any(output != lq, axis=2))),
        "border_max_change": int(np.max(np.abs(output[border].astype(np.int16) - lq[border].astype(np.int16)))),
        "curve": asdict(config),
    })
    return output, info
