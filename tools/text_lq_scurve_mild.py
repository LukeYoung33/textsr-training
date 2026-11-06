"""Versioned CPU preprocessing for single-line text LQ images."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MildSCurveConfig:
    version: str = "text_lq_scurve_mild_v1"
    border_px: int = 6
    min_class_share: float = 0.03
    min_contrast: float = 20.0
    local_box_px: int = 33
    local_prior: float = 0.02
    field_smooth_px: int = 21
    edge_gamma: float = 1.8
    core_gamma: float = 1.8
    core_tau: float = 0.52
    background_strength: float = 0.85
    interior_strength: float = 0.50
    interior_cap: float = 15.0
    exterior_strength: float = 0.60
    exterior_cap: float = 18.0
    border_target_offset: float = 2.0
    overshoot_strength: float = 0.0
    overshoot_cap: float = 0.0
    overshoot_margin: float = 5.0
    tail_background_strength: float = 0.0
    tail_background_cap: float = 0.0
    tail_foreground_strength: float = 0.0
    tail_foreground_cap: float = 0.0
    global_platform_tails: bool = False
    chroma_background_return: bool = False


DEFAULT_CONFIG = MildSCurveConfig()
V2_CONFIG = MildSCurveConfig(
    version="text_lq_scurve_mild_v2",
    overshoot_strength=0.65,
    overshoot_cap=28.0,
)
V3_CONFIG = MildSCurveConfig(
    version="text_lq_scurve_mild_v3",
    overshoot_strength=1.0,
    overshoot_cap=50.0,
    overshoot_margin=2.0,
)
V4_CONFIG = MildSCurveConfig(
    version="text_lq_scurve_mild_v4",
    tail_background_strength=0.80,
    tail_background_cap=48.0,
    tail_foreground_strength=0.55,
    tail_foreground_cap=12.0,
)
V5_CONFIG = MildSCurveConfig(
    version="text_lq_scurve_mild_v5",
    global_platform_tails=True,
)
V6_CONFIG = MildSCurveConfig(
    version="text_lq_scurve_mild_v6",
    global_platform_tails=True,
    chroma_background_return=True,
)


def _luma_bgr(image: np.ndarray) -> np.ndarray:
    """@luna-comment
    Input: An HWC uint8 BGR image with three channels.
    Output: An HxW float32 8-bit-scale luma image; no side effects.
    Key Steps: Use OpenCV's frozen BGR-to-YCrCb conversion and select Y.
    Key Params: The OpenCV conversion fixes rounding and channel order.
    State/Dependencies: Depends only on OpenCV and the input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def _apply_luma_delta(image: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """@luna-comment
    Input: HWC uint8 BGR and an HxW signed 8-bit-scale luma delta.
    Output: New uint8 BGR with one shared per-pixel channel offset.
    Key Steps: Round once, limit offset by RGB headroom, and copy untouched bytes.
    Key Params: Channel headroom prevents channel-specific gamut clipping.
    State/Dependencies: Pure NumPy computation; input is never modified.
    """
    rounded = np.rint(delta).astype(np.int16)
    low = -np.min(image, axis=2).astype(np.int16)
    high = 255 - np.max(image, axis=2).astype(np.int16)
    applied = np.minimum(np.maximum(rounded, low), high)
    return (image.astype(np.int16) + applied[:, :, None]).astype(np.uint8)


def _box(image: np.ndarray, width: int) -> np.ndarray:
    """@luna-comment
    Input: A float32 image and positive odd box width in pixels.
    Output: Same-size reflected-border box mean; no side effects.
    Key Steps: Apply one normalized OpenCV box filter.
    Key Params: Width controls local reference scale.
    State/Dependencies: Uses OpenCV with reflected image boundaries.
    """
    return cv2.boxFilter(image, -1, (width, width), normalize=True, borderType=cv2.BORDER_REFLECT)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    """@luna-comment
    Input: A float32 image of normalized ramp coordinates.
    Output: Same-shape cubic ramp, clipped to zero and one outside the unit interval.
    Key Steps: Clip then apply a cubic polynomial with zero endpoint derivatives.
    Key Params: Unit input endpoints delimit spatial blend windows.
    State/Dependencies: Pure NumPy computation with no shared state.
    """
    limited = np.clip(value, 0.0, 1.0)
    return limited * limited * (3.0 - 2.0 * limited)


def _border_distance(shape: tuple[int, int]) -> np.ndarray:
    """@luna-comment
    Input: Height and width of the valid text-line crop.
    Output: Integer distance of each pixel center to the nearest crop boundary.
    Key Steps: Take the minimum horizontal and vertical edge distances.
    Key Params: The caller compares this map with the configured border width.
    State/Dependencies: Pure NumPy computation with no shared state.
    """
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    return np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy))


def _estimate_background_field(
    luminance: np.ndarray, samples: np.ndarray, reference: float, config: MildSCurveConfig
) -> np.ndarray:
    """@luna-comment
    Input: Normalized luma, reliable background mask, scalar background reference, and config.
    Output: Smooth float32 local background field on the same 8-bit scale.
    Key Steps: Two regularized masked-box passes, robust reweighting, and two smoothing boxes.
    Key Params: Local prior protects zero support; box widths bound cost and field scale.
    State/Dependencies: Uses OpenCV box filters; no shared or persistent state.
    """
    weight = samples.astype(np.float32)
    denominator = _box(weight, config.local_box_px)
    numerator = _box(weight * luminance, config.local_box_px)
    field = (numerator + config.local_prior * reference) / (denominator + config.local_prior)
    residual = luminance[samples] - field[samples]
    sigma = max(1.0, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
    weight *= np.clip(3.0 * sigma / (np.abs(luminance - field) + 1e-6), 0.0, 1.0)
    denominator = _box(weight, config.local_box_px)
    numerator = _box(weight * luminance, config.local_box_px)
    field = (numerator + config.local_prior * reference) / (denominator + config.local_prior)
    field = _box(_box(field, config.field_smooth_px), config.field_smooth_px)
    return np.clip(field, reference - 25.0, reference + 25.0).astype(np.float32)


def _run_normalized(
    original: np.ndarray, normalized_luma: np.ndarray, polarity: int, config: MildSCurveConfig
) -> tuple[np.ndarray, dict[str, object]]:
    """@luna-comment
    Input: Valid BGR line, dark-text normalized luma, polarity sign, and frozen config.
    Output: Same-shape BGR image and per-stage diagnostic values.
    Key Steps: Estimate masks/references, stabilize luma, sharpen edges, flatten supported ink, and reduce eligible tails.
    Key Params: V1 fixes the mild experiment; V2/V3 add spatial exterior correction; V4 adds local tails; V5/V6 share global luma rails.
    State/Dependencies: Uses OpenCV and NumPy; no filesystem or shared state.
    """
    height, width = normalized_luma.shape
    border_distance = _border_distance((height, width))
    border = border_distance < config.border_px
    threshold0, _ = cv2.threshold(normalized_luma.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold0 += 0.5
    foreground0 = normalized_luma < threshold0
    class_share = float(np.mean(foreground0))
    if min(class_share, 1.0 - class_share) < config.min_class_share:
        return original.copy(), {"fallback_reason": "insufficient_otsu_class_share"}

    eroded = cv2.erode(foreground0.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    grad_x = cv2.Sobel(normalized_luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized_luma, cv2.CV_32F, 0, 1, ksize=3)
    flat = np.maximum(np.abs(grad_x), np.abs(grad_y)) < 30.0
    core = eroded & flat & ~border
    neighbor_count = cv2.filter2D(foreground0.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    thin = foreground0 & ~eroded & (normalized_luma >= threshold0 - 7) & (neighbor_count <= 4)
    distance_background = cv2.distanceTransform((~foreground0).astype(np.uint8), cv2.DIST_L2, 3)
    background = (~foreground0) & (distance_background > 3) & flat & ~border
    if np.count_nonzero(core) < 32 or np.count_nonzero(background) < 64:
        return original.copy(), {"fallback_reason": "insufficient_reliable_samples"}

    foreground_median = float(np.median(normalized_luma[core]))
    background_line = float(np.median(normalized_luma[background]))
    if background_line - foreground_median < config.min_contrast:
        return original.copy(), {"fallback_reason": "insufficient_contrast"}
    border_samples = border & (normalized_luma > threshold0 + 15)
    if np.count_nonzero(border_samples) < 64:
        return original.copy(), {"fallback_reason": "insufficient_border_background"}
    background_target = float(np.median(normalized_luma[border_samples]) + config.border_target_offset)
    foreground_anchor = float(np.clip(np.percentile(normalized_luma[core], 10) - 1, 0, threshold0 - 20))
    if not foreground_anchor + 5 < threshold0 < background_target - 5:
        return original.copy(), {"fallback_reason": "invalid_curve_anchors"}
    background_field = _estimate_background_field(normalized_luma, background, background_line, config)

    # P2: reproduce the bounded dehalo and continuous-background candidate.
    guide = cv2.bilateralFilter(normalized_luma, 7, 30, 2)
    normalized = (guide - foreground_anchor) / np.maximum(background_field - foreground_anchor, config.min_contrast)
    bounded = np.clip(normalized, 1e-6, 1.0 - 1e-6)
    left = (bounded / config.core_tau) ** config.core_gamma
    right = ((1.0 - bounded) / (1.0 - config.core_tau)) ** config.core_gamma
    curve = left / (left + right)
    core_tone = np.where(
        (normalized > 0) & (normalized < config.core_tau),
        np.clip((background_field - foreground_anchor) * (curve - normalized), -28.0, 0.0),
        0.0,
    )
    soft_core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 1.0)
    soft_edge = cv2.GaussianBlur((foreground0 & ~eroded).astype(np.float32), (0, 0), 1.0)
    core_weight = (0.7 + 0.3 * soft_core) * core
    edge_mask = foreground0 & ~eroded & ~thin
    edge_weight = (0.7 + 0.3 * soft_edge) * edge_mask
    delta_core = core_weight * (core_tone + guide - normalized_luma)
    delta_edge = edge_weight * np.clip((background_target - guide) * 0.2, 0.0, 14.0)
    background_taper = np.clip((border_distance - 5) / 18.0, 0.0, 1.0)
    background_weight = np.clip((normalized_luma - threshold0) / 40.0, 0.0, 1.0)
    delta_background = np.where(
        ~foreground0,
        config.background_strength * np.clip(background_target - normalized_luma, -28.0, 28.0)
        * background_weight * background_taper,
        0.0,
    )
    dehalo_delta = delta_core + delta_edge + delta_background
    dehalo_delta[border | thin] = 0.0
    stage2 = _apply_luma_delta(original, polarity * dehalo_delta)
    luminance2 = _luma_bgr(stage2)
    working2 = luminance2 if polarity == 1 else 255.0 - luminance2

    # P3a: a continuous curve preserves the stabilized Otsu threshold side.
    threshold1, _ = cv2.threshold(working2.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold1 = float(threshold1)
    foreground1 = working2 <= threshold1
    if not foreground_anchor + 5 < threshold1 < background_target - 5:
        return original.copy(), {"fallback_reason": "invalid_stabilized_threshold"}
    lower = foreground_anchor + (threshold1 - foreground_anchor) * np.clip(
        (working2 - foreground_anchor) / (threshold1 - foreground_anchor), 0.0, 1.0
    ) ** config.edge_gamma
    upper = background_target - (background_target - threshold1) * np.clip(
        (background_target - working2) / (background_target - threshold1), 0.0, 1.0
    ) ** config.edge_gamma
    mapped = np.where(working2 <= threshold1, lower, upper)
    intensity = np.clip((working2 - foreground_anchor) / (background_target - foreground_anchor), 0.0, 1.0)
    edge_blend = (4.0 * intensity * (1.0 - intensity)) ** 2
    edge_delta = np.where(
        (working2 >= foreground_anchor) & (working2 <= background_target),
        np.clip((mapped - working2) * edge_blend, -20.0, 20.0),
        0.0,
    )
    if config.global_platform_tails:
        edge_delta = np.where(
            working2 < foreground_anchor, foreground_anchor - working2,
            np.where(working2 > background_target, background_target - working2, edge_delta),
        )
    edge_delta[border | thin] = 0.0
    stage3 = _apply_luma_delta(stage2, polarity * edge_delta)
    luminance3 = _luma_bgr(stage3)
    working3 = luminance3 if polarity == 1 else 255.0 - luminance3

    # P3b: only bright, supported foreground interiors approach a local dark reference.
    foreground_distance = cv2.distanceTransform(foreground1.astype(np.uint8), cv2.DIST_L2, 3)
    component_count, components, component_stats, _ = cv2.connectedComponentsWithStats(foreground1.astype(np.uint8), 8)
    interior_reference = np.full_like(working3, foreground_anchor + 10.0)
    for component_id in range(1, component_count):
        if component_stats[component_id, cv2.CC_STAT_AREA] < 20:
            continue
        supported = (components == component_id) & (foreground_distance >= 2.5)
        if np.count_nonzero(supported) < 20:
            continue
        target = np.clip(np.percentile(working3[supported], 25), foreground_anchor, threshold1 - 15)
        interior_reference[components == component_id] = target
    interior_weight = np.clip((foreground_distance - 1.5) / 2.5, 0.0, 1.0)
    interior_delta = -interior_weight * np.minimum(
        config.interior_cap, config.interior_strength * np.maximum(working3 - interior_reference - 3.0, 0.0)
    )
    interior_delta[~foreground1 | border | thin] = 0.0
    stage4 = _apply_luma_delta(stage3, polarity * interior_delta)
    luminance4 = _luma_bgr(stage4)
    working4 = luminance4 if polarity == 1 else 255.0 - luminance4

    # P4: return the background-side blur tail, retaining a partial first AA pixel.
    exterior_distance = cv2.distanceTransform((~foreground1).astype(np.uint8), cv2.DIST_L2, 3)
    near_weight = np.clip(0.5 + 0.25 * (exterior_distance - 1.0), 0.0, 1.0)
    far_weight = np.cos(0.5 * np.pi * np.clip((exterior_distance - 7.0) / 4.0, 0.0, 1.0)) ** 2
    far_weight[exterior_distance >= 11.0] = 0.0
    intensity_weight = np.clip((working4 - (threshold1 + 2.0)) / 15.0, 0.0, 1.0)
    exterior_delta = np.where(
        (~foreground1) & (working4 < background_target),
        np.clip(
            config.exterior_strength * (background_target - working4) * near_weight * far_weight * intensity_weight,
            0.0,
            config.exterior_cap,
        ),
        0.0,
    )
    exterior_delta[border | thin] = 0.0
    stage5 = _apply_luma_delta(stage4, polarity * exterior_delta)
    stage5_luma = _luma_bgr(stage5)
    stage5_working = stage5_luma if polarity == 1 else 255.0 - stage5_luma

    # V2: smooth the opposite-sign outer halo against the local background field.
    if config.overshoot_strength > 0:
        near_weight = _smoothstep((exterior_distance - 2.0) / 3.0)
        far_weight = 1.0 - _smoothstep((exterior_distance - 10.0) / 14.0)
        overshoot = np.maximum(stage5_working - background_field - config.overshoot_margin, 0.0)
        overshoot_delta = -np.minimum(config.overshoot_cap, config.overshoot_strength * overshoot)
        overshoot_delta *= near_weight * far_weight
        overshoot_delta[foreground1 | border] = 0.0
        output = _apply_luma_delta(stage5, polarity * overshoot_delta)
    elif config.tail_background_strength > 0:
        # V4: the mask selects the reference, while residual magnitude supplies
        # the weight; no fixed distance from the glyph limits the background tail.
        background_excess = np.maximum(stage5_working - background_field - 2.0, 0.0)
        background_return = -np.minimum(
            config.tail_background_cap,
            config.tail_background_strength * background_excess * background_excess / (background_excess + 4.0),
        )
        background_return = np.maximum(background_return, threshold1 + 1.0 - stage5_working)
        background_return[foreground1 | border] = 0.0
        foreground_excess = np.maximum(interior_reference - 2.0 - stage5_working, 0.0)
        foreground_return = np.minimum(
            config.tail_foreground_cap,
            config.tail_foreground_strength * foreground_excess * foreground_excess / (foreground_excess + 4.0),
        )
        foreground_return *= np.clip((foreground_distance - 0.5) / 2.0, 0.0, 1.0)
        foreground_return[~foreground1 | border | thin] = 0.0
        output = _apply_luma_delta(stage5, polarity * (background_return + foreground_return))
    else:
        output = stage5
    output_luma = _luma_bgr(output)
    output_working = output_luma if polarity == 1 else 255.0 - output_luma
    if np.any((output_working <= threshold1) != foreground1):
        return original.copy(), {"fallback_reason": "fixed_mask_crossing"}
    if np.any(output[border] != original[border]):
        return original.copy(), {"fallback_reason": "border_changed"}
    exterior_band = (~foreground1) & (exterior_distance > 0) & (exterior_distance < 7) & ~border
    exterior_deficit_before = (
        float(np.mean(np.maximum(background_target - working4[exterior_band], 0.0)))
        if np.any(exterior_band) else 0.0
    )
    exterior_deficit_after = (
        float(np.mean(np.maximum(background_target - stage5_working[exterior_band], 0.0)))
        if np.any(exterior_band) else 0.0
    )
    telemetry: dict[str, object] = {
        "fallback_reason": None,
        "polarity": "dark_text_light_background" if polarity == 1 else "light_text_dark_background",
        "threshold_input": threshold0,
        "threshold_stabilized": threshold1,
        "foreground_median": foreground_median,
        "foreground_anchor": foreground_anchor,
        "background_line": background_line,
        "background_target": background_target,
        "core_count": int(np.count_nonzero(core)),
        "background_sample_count": int(np.count_nonzero(background)),
        "thin_protected_count": int(np.count_nonzero(thin)),
        "dehalo_changed_pixels": int(np.count_nonzero(np.any(stage2 != original, axis=2))),
        "edge_changed_pixels": int(np.count_nonzero(np.any(stage3 != stage2, axis=2))),
        "interior_changed_pixels": int(np.count_nonzero(np.any(stage4 != stage3, axis=2))),
        "exterior_changed_pixels": int(np.count_nonzero(np.any(stage5 != stage4, axis=2))),
        "exterior_band_deficit_before": exterior_deficit_before,
        "exterior_band_deficit_after": exterior_deficit_after,
        "exterior_foreground_max_change": int(np.max(np.abs(stage5[foreground1].astype(np.int16) - stage4[foreground1].astype(np.int16)))),
        "border_max_change": int(np.max(np.abs(output[border].astype(np.int16) - original[border].astype(np.int16)))),
        "fixed_mask_crossings": int(np.count_nonzero((output_working <= threshold1) != foreground1)),
    }
    if config.overshoot_strength > 0:
        overshoot_band = (~foreground1) & (exterior_distance >= 6) & (exterior_distance < 15) & ~border
        telemetry["overshoot_changed_pixels"] = int(np.count_nonzero(np.any(output != stage5, axis=2)))
        telemetry["overshoot_foreground_max_change"] = int(
            np.max(np.abs(output[foreground1].astype(np.int16) - stage5[foreground1].astype(np.int16)))
        )
        telemetry["overshoot_band_before"] = float(
            np.mean(np.maximum(stage5_working[overshoot_band] - background_field[overshoot_band], 0.0))
        ) if np.any(overshoot_band) else 0.0
        telemetry["overshoot_band_after"] = float(
            np.mean(np.maximum(output_working[overshoot_band] - background_field[overshoot_band], 0.0))
        ) if np.any(overshoot_band) else 0.0
    if config.tail_background_strength > 0:
        telemetry["tail_background_changed_pixels"] = int(np.count_nonzero(background_return))
        telemetry["tail_foreground_changed_pixels"] = int(np.count_nonzero(foreground_return))
        telemetry["tail_background_mean_abs_delta"] = float(np.mean(np.abs(background_return[~foreground1 & ~border])))
        tail_band = (~foreground1) & (exterior_distance >= 6) & (exterior_distance < 15) & ~border
        telemetry["tail_band_excess_before"] = float(np.mean(
            np.maximum(stage5_working[tail_band] - background_field[tail_band] - 2.0, 0.0)
        )) if np.any(tail_band) else 0.0
        telemetry["tail_band_excess_after"] = float(np.mean(
            np.maximum(output_working[tail_band] - background_field[tail_band] - 2.0, 0.0)
        )) if np.any(tail_band) else 0.0
    if config.global_platform_tails:
        background_tail = (working2 > background_target) & ~border & ~thin
        telemetry["global_background_tail_pixels"] = int(np.count_nonzero(background_tail))
        telemetry["global_background_rail_misses_gt1"] = int(np.count_nonzero(
            np.abs(output_working[background_tail] - background_target) > 1.0
        ))
    return output, telemetry


def _apply_chroma_background_return(
    image: np.ndarray, polarity: int, border_px: int
) -> tuple[np.ndarray, dict[str, object]]:
    """@luna-comment
    Input: Successful V5-stage uint8 BGR line, text polarity, and protected crop width.
    Output: BGR with background-only U/V return and telemetry; source image is unchanged.
    Key Steps: Fit a robust half-size background chroma field, project excess toward neutral, then enforce full-size text and Y guards.
    Key Params: Distant smooth samples, a one-sided 95th-percentile threshold, and bounded soft correction freeze the V6 candidate.
    State/Dependencies: Uses OpenCV/NumPy only; no persistent state or filesystem I/O.
    """
    height, width = image.shape[:2]
    small_height, small_width = height // 2, width // 2
    if min(small_height, small_width) < 16:
        return image.copy(), {"chroma_fallback_reason": "small_chroma_canvas", "chroma_changed_pixels": 0}

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32)
    y, u, v = cv2.split(yuv)
    small_y, small_u, small_v = cv2.split(
        cv2.resize(yuv, (small_width, small_height), interpolation=cv2.INTER_AREA)
    )
    threshold, _ = cv2.threshold(small_y.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = small_y > threshold if polarity == -1 else small_y < threshold
    distance = cv2.distanceTransform((~foreground).astype(np.uint8), cv2.DIST_L2, 3)

    def gradient(channel: np.ndarray) -> np.ndarray:
        """@luna-comment
        Input: One half-resolution float32 Y/U/V channel.
        Output: Maximum absolute three-tap Sobel slope on an eight-bit scale.
        Key Steps: Evaluate both axes and divide the kernel gain by eight.
        Key Params: The caller's 9/4-level gates select flat reference pixels.
        State/Dependencies: Pure OpenCV operations with no shared state.
        """
        dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        return np.maximum(np.abs(dx), np.abs(dy)) / 8.0

    reliable = (
        (distance >= 9.0)
        & (gradient(cv2.GaussianBlur(small_y, (0, 0), 0.7)) < 9.0)
        & (gradient(small_u) < 4.0)
        & (gradient(small_v) < 4.0)
    )
    small_border = max(1, border_px // 2)
    reliable[:small_border, :] = False
    reliable[-small_border:, :] = False
    reliable[:, :small_border] = False
    reliable[:, -small_border:] = False
    reference_count = int(np.count_nonzero(reliable))
    if reference_count < 64:
        return image.copy(), {
            "chroma_fallback_reason": "insufficient_chroma_reference",
            "chroma_reference_pixels": reference_count,
            "chroma_changed_pixels": 0,
        }

    prior = 0.005
    weights = reliable.astype(np.float32)
    support = _box(weights, 41)
    median_u = float(np.median(small_u[reliable]))
    median_v = float(np.median(small_v[reliable]))
    reference_u = (_box(weights * small_u, 41) + prior * median_u) / (support + prior)
    reference_v = (_box(weights * small_v, 41) + prior * median_v) / (support + prior)
    residual = np.sqrt((small_u - reference_u) ** 2 + (small_v - reference_v) ** 2)
    weights *= np.clip(6.0 / (residual + 1e-5), 0.0, 1.0)
    support = _box(weights, 41)
    reference_u = (_box(weights * small_u, 41) + prior * median_u) / (support + prior)
    reference_v = (_box(weights * small_v, 41) + prior * median_v) / (support + prior)

    neutral_u = 128.0 - reference_u
    neutral_v = 128.0 - reference_v
    chroma_length = np.sqrt(neutral_u ** 2 + neutral_v ** 2) + 1e-6
    axis_u, axis_v = neutral_u / chroma_length, neutral_v / chroma_length
    reference_projection = (small_u - reference_u) * axis_u + (small_v - reference_v) * axis_v
    normal_limit = max(1.5, float(np.percentile(reference_projection[reliable], 95)))

    def full_size(channel: np.ndarray) -> np.ndarray:
        """@luna-comment
        Input: Half-resolution float32 chroma reference, support, or distance map.
        Output: Bilinear full-resolution float32 map.
        Key Steps: Resize to the exact original text-line shape.
        Key Params: The small image has half the original integer dimensions.
        State/Dependencies: Pure OpenCV interpolation with no shared state.
        """
        return cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)

    reference_u, reference_v, support = map(full_size, (reference_u, reference_v, support))
    distance = 2.0 * full_size(distance)
    neutral_u, neutral_v = 128.0 - reference_u, 128.0 - reference_v
    chroma_length = np.sqrt(neutral_u ** 2 + neutral_v ** 2) + 1e-6
    axis_u, axis_v = neutral_u / chroma_length, neutral_v / chroma_length
    projection = (u - reference_u) * axis_u + (v - reference_v) * axis_v
    perpendicular = -(u - reference_u) * axis_v + (v - reference_v) * axis_u
    excess = np.maximum(projection - normal_limit, 0.0)
    confidence = (
        np.clip((support - 0.025) / 0.10, 0.0, 1.0)
        * np.clip((chroma_length - 3.0) / 6.0, 0.0, 1.0)
        * np.clip((8.0 - np.abs(perpendicular)) / 4.0, 0.0, 1.0)
    )
    spatial = np.clip((distance - 1.2) / 1.8, 0.0, 1.0) * np.clip((35.0 - distance) / 10.0, 0.0, 1.0)
    amount = 0.9 * np.minimum(excess, 14.0) * _smoothstep(excess / 3.0) * confidence * spatial

    full_threshold, _ = cv2.threshold(y.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    full_foreground = y > full_threshold if polarity == -1 else y < full_threshold
    protected = cv2.dilate(full_foreground.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    amount[protected] = 0.0
    border = _border_distance((height, width)) < border_px
    amount[border] = 0.0
    changed = amount > 0.01
    if not np.any(changed):
        return image.copy(), {
            "chroma_fallback_reason": None,
            "chroma_reference_pixels": reference_count,
            "chroma_threshold": normal_limit,
            "chroma_changed_pixels": 0,
            "chroma_y_drift_max": 0,
            "chroma_foreground_changed_pixels": 0,
            "chroma_border_changed_pixels": 0,
        }
    yuv[:, :, 1] = u - amount * axis_u
    yuv[:, :, 2] = v - amount * axis_v
    output = cv2.cvtColor(np.clip(np.rint(yuv), 0, 255).astype(np.uint8), cv2.COLOR_YUV2BGR)
    output[~changed] = image[~changed]
    y_drift = _luma_bgr(output) != _luma_bgr(image)
    output[y_drift] = image[y_drift]
    actual_change = np.any(output != image, axis=2)
    return output, {
        "chroma_fallback_reason": None,
        "chroma_reference_pixels": reference_count,
        "chroma_threshold": normal_limit,
        "chroma_changed_pixels": int(np.count_nonzero(actual_change)),
        "chroma_y_drift_max": int(np.max(np.abs(_luma_bgr(output) - _luma_bgr(image)))),
        "chroma_foreground_changed_pixels": int(np.count_nonzero(actual_change & full_foreground)),
        "chroma_border_changed_pixels": int(np.count_nonzero(actual_change & border)),
    }


def preprocess_text_lq_bgr(
    image: np.ndarray, config: MildSCurveConfig = DEFAULT_CONFIG, return_telemetry: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """@luna-comment
    Input: Full-line uint8 HWC BGR image and an immutable versioned config.
    Output: Same-shape uint8 BGR, optionally with stage telemetry; input remains untouched.
    Key Steps: Validate, detect border-backed polarity, run the selected luma flow, and optionally return background chroma under V6.
    Key Params: The config freezes mild strengths, protection widths, and reference support gates.
    State/Dependencies: Uses OpenCV/NumPy only; no global mutable state or file I/O.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an HWC uint8 BGR array")
    if not isinstance(config, MildSCurveConfig) or config not in (DEFAULT_CONFIG, V2_CONFIG, V3_CONFIG, V4_CONFIG, V5_CONFIG, V6_CONFIG):
        raise ValueError("frozen mild configuration must match a named V1/V2/V3/V4/V5/V6 configuration exactly")
    if min(image.shape[:2]) < 2 * config.border_px + 20:
        result = image.copy()
        telemetry: dict[str, object] = {"fallback_reason": "crop_too_small"}
        return (result, telemetry) if return_telemetry else result
    luminance = _luma_bgr(image)
    threshold, _ = cv2.threshold(luminance.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = _border_distance(luminance.shape) < config.border_px
    high = luminance[border] > threshold
    high_share = float(np.mean(high))
    if high_share >= 0.80:
        polarity = 1
    elif high_share <= 0.20:
        polarity = -1
    else:
        result = image.copy()
        telemetry = {"fallback_reason": "ambiguous_border_polarity"}
        return (result, telemetry) if return_telemetry else result
    normalized_luma = luminance if polarity == 1 else 255.0 - luminance
    result, telemetry = _run_normalized(image, normalized_luma, polarity, config)
    if config.chroma_background_return and telemetry.get("fallback_reason") is None:
        result, chroma_telemetry = _apply_chroma_background_return(result, polarity, config.border_px)
        telemetry.update(chroma_telemetry)
    telemetry["config_version"] = config.version
    return (result, telemetry) if return_telemetry else result


def main() -> int:
    """@luna-comment
    Input: CLI input/output image paths and optional telemetry JSON path.
    Output: Process exit code and written image/telemetry artifacts.
    Key Steps: Parse paths, read BGR, call the reusable API once, and save outputs.
    Key Params: V1 remains the default; --version selects a separately named V1-V6 candidate.
    State/Dependencies: Reads and writes only requested paths through OpenCV/JSON.
    """
    parser = argparse.ArgumentParser(description="Apply the frozen mild S-curve text LQ preprocessor")
    parser.add_argument("input", type=Path, help="input BGR-compatible image")
    parser.add_argument("output", type=Path, help="output image path")
    parser.add_argument("--telemetry", type=Path, help="optional JSON telemetry path")
    parser.add_argument("--version", choices=("v1", "v2", "v3", "v4", "v5", "v6"), default="v1", help="frozen processing version")
    args = parser.parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"cannot read input image: {args.input}")
    config = {"v1": DEFAULT_CONFIG, "v2": V2_CONFIG, "v3": V3_CONFIG, "v4": V4_CONFIG, "v5": V5_CONFIG, "v6": V6_CONFIG}[args.version]
    output, telemetry = preprocess_text_lq_bgr(image, config=config, return_telemetry=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        parser.error(f"cannot write output image: {args.output}")
    if args.telemetry:
        args.telemetry.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry.write_text(json.dumps({"config": asdict(config), "run": telemetry}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
