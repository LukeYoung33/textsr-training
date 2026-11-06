"""Background photometric alignment for a published V3 training release.

The V4 release already contains the trusted text-region photometric result.
This module adds a low-frequency GT-to-LQ correction to the non-text region
without changing V4 in place.  It is intentionally independent from the V2.1
photometric runner and can be run from a published ``dataset_V4_manifest``::

    python -m AlignPipeline.background_photometric_v5 \
        --manifest /data/training/dataset_V4_manifest.json \
        --output-root /data/training

The output is a new, versioned projection (``*_V5``), plus one JSON audit per
sample.  The implementation uses a masked-normalized Gaussian residual field
and a smooth distance-transform blend.  In particular, pixels in
``dilate(text_mask)`` and ``valid_mask == 0`` never contribute to the
background estimate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:  # Keep the numerical color definition identical to the existing stages.
    from .y_alignment import rgb2yuv_numpy, yuv2rgb_numpy
except ImportError:  # pragma: no cover - direct script execution
    from y_alignment import rgb2yuv_numpy, yuv2rgb_numpy


SCHEMA_VERSION = "background_photometric_v5_audit_v1"
RELEASE_SCHEMA_VERSION = "training_publish_v5"
DEFAULT_EXCLUDE_RADIUS = 12
DEFAULT_BLEND_RADIUS = 16
DEFAULT_Y_SIGMA = 64.0
DEFAULT_UV_SIGMA = 5.0
DEFAULT_SEAM_GRADIENT_LIMIT = 32.0
DEFAULT_MIN_Y_IMPROVEMENT = 0.5
DEFAULT_Y_NONREGRESSION_TOLERANCE = 0.25
DEFAULT_CHROMA_NONREGRESSION_TOLERANCE = 0.25


class V5InputError(ValueError):
    """Raised when a V4 manifest violates the input contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        temporary.chmod(0o644)
    except OSError:
        pass
    os.replace(temporary, path)


def _normalize_binary_mask(mask: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
    if mask is None:
        raise V5InputError(f"missing {label}")
    if mask.ndim == 3:
        if mask.shape[2] != 1 and not (
            np.array_equal(mask[..., 0], mask[..., 1])
            and np.array_equal(mask[..., 0], mask[..., 2])
        ):
            raise V5InputError(f"{label} has different channels")
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.shape != shape:
        actual = None if mask is None else tuple(mask.shape)
        raise V5InputError(f"{label} shape {actual} != {shape}")
    if mask.dtype != np.uint8:
        mask = np.asarray(mask)
    # Published masks are normally 0/255, while compatible manifests in
    # experiments sometimes use 0/1.  Accept both encodings explicitly.
    threshold = 0 if float(np.max(mask, initial=0)) <= 1.0 else 127
    return (mask > threshold).astype(np.uint8)


def _read_color(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise V5InputError(f"failed to read {label}: {path}")
    return image


def _read_mask(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise V5InputError(f"failed to read {label}: {path}")
    return image


def _masked_gaussian_normalized(
    value: np.ndarray,
    support: np.ndarray,
    sigma: float,
    *,
    fallback: float,
    min_support_weight: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth ``value`` using only ``support`` pixels.

    This is deliberately *not* ``GaussianBlur(value)``.  The denominator
    normalizes away holes produced by the dilated text mask, so zeros in the
    excluded region cannot bias the residual towards black.
    """

    sigma = float(sigma)
    if sigma <= 0:
        reliability = support.astype(np.float32)
        output = np.where(reliability >= min_support_weight, value, float(fallback))
        return output.astype(np.float32), reliability.astype(np.float32)
    support_f = (support > 0).astype(np.float32)
    weighted = np.where(support_f > 0, value, 0.0).astype(np.float32)
    numerator = cv2.GaussianBlur(weighted, (0, 0), sigmaX=sigma, sigmaY=sigma)
    denominator = cv2.GaussianBlur(support_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    reliable = denominator >= float(min_support_weight)
    output = np.full_like(value, float(fallback), dtype=np.float32)
    output[reliable] = numerator[reliable] / np.maximum(denominator[reliable], eps)
    return output, denominator.astype(np.float32)


def _robust_channel_residual(
    residual: np.ndarray,
    support: np.ndarray,
    *,
    max_delta: float,
    min_clip: float,
) -> tuple[np.ndarray, dict[str, float]]:
    values = residual[support > 0]
    if values.size == 0:
        return residual.astype(np.float32), {
            "median": 0.0,
            "mad": 0.0,
            "p01": 0.0,
            "p99": 0.0,
            "p99_abs": 0.0,
            "robust_clip": 0.0,
        }
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    robust_clip = float(np.clip(max(6.0 * scale, min_clip), min_clip, max_delta))
    clipped = np.clip(residual, center - robust_clip, center + robust_clip).astype(np.float32)
    stats = {
        "median": center,
        "mad": mad,
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
        "p99_abs": float(np.percentile(np.abs(values), 99)),
        "robust_clip": robust_clip,
    }
    return clipped, stats


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel)


def _soft_text_alpha(text_mask: np.ndarray, blend_radius: int) -> np.ndarray:
    """Return alpha=1 in text and a monotone smooth falloff outside it."""

    text = (text_mask > 0).astype(np.uint8)
    if not np.any(text):
        return np.zeros(text.shape, dtype=np.float32)
    distance = cv2.distanceTransform((1 - text).astype(np.uint8), cv2.DIST_L2, 5)
    radius = max(1.0, float(blend_radius))
    t = np.clip(1.0 - distance / radius, 0.0, 1.0)
    # Smoothstep is C1 continuous at both ends and avoids a hard mask seam.
    alpha = t * t * (3.0 - 2.0 * t)
    alpha[text > 0] = 1.0
    return alpha.astype(np.float32)


def _luma(image_bgr: np.ndarray) -> np.ndarray:
    image = image_bgr.astype(np.float32)
    return 0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2]


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _residual_metrics(lq: np.ndarray, image: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    lq_yuv = rgb2yuv_numpy(lq.astype(np.float32)[..., ::-1])
    image_yuv = rgb2yuv_numpy(image.astype(np.float32)[..., ::-1])
    mask = support > 0
    channels = []
    for index, name in enumerate(("Y", "U", "V")):
        values = (lq_yuv[..., index] - image_yuv[..., index])[mask]
        channels.append({
            "channel": name,
            "mean_abs": float(np.mean(np.abs(values))) if values.size else 0.0,
            "median_abs": _percentile(np.abs(values), 50),
            "p90_abs": _percentile(np.abs(values), 90),
            "p95_abs": _percentile(np.abs(values), 95),
        })
    return {"support_pixels": int(mask.sum()), "channels": channels}


def _seam_metrics(
    v4: np.ndarray,
    v5: np.ndarray,
    text_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    ring_radius: int,
    gradient_limit: float,
) -> dict[str, Any]:
    text = (text_mask > 0).astype(np.uint8)
    valid = valid_mask > 0
    ring = (_dilate(text, ring_radius) > 0) & valid
    outside_ring = ring & ~(text > 0)
    v4_luma = _luma(v4)
    v5_luma = _luma(v5)
    change_luma = v5_luma - v4_luma
    v4_gradient = _gradient_magnitude(v4_luma)
    v5_gradient = _gradient_magnitude(v5_luma)
    change_gradient = _gradient_magnitude(change_luma)
    v4_yuv = rgb2yuv_numpy(v4.astype(np.float32)[..., ::-1])
    v5_yuv = rgb2yuv_numpy(v5.astype(np.float32)[..., ::-1])
    change_yuv = v5_yuv - v4_yuv
    correction_gradient_by_channel: dict[str, dict[str, float]] = {}
    correction_gradient_values: list[float] = []
    for index, channel in enumerate(("Y", "U", "V")):
        channel_gradient = _gradient_magnitude(change_yuv[..., index])
        p95 = _percentile(channel_gradient[outside_ring], 95)
        maximum = float(np.max(channel_gradient[outside_ring])) if np.any(outside_ring) else 0.0
        correction_gradient_by_channel[channel] = {"p95": p95, "max": maximum}
        correction_gradient_values.append(p95)
    metrics = {
        "ring_pixels": int(outside_ring.sum()),
        "v4_ring_gradient_luma_p95": _percentile(v4_gradient[outside_ring], 95),
        "v5_ring_gradient_luma_p95": _percentile(v5_gradient[outside_ring], 95),
        "v5_minus_v4_ring_gradient_luma_p95": _percentile(v5_gradient[outside_ring], 95)
        - _percentile(v4_gradient[outside_ring], 95),
        "v5_change_gradient_luma_p95": _percentile(change_gradient[outside_ring], 95),
        "v5_change_gradient_luma_max": float(np.max(change_gradient[outside_ring])) if np.any(outside_ring) else 0.0,
        "correction_gradient_yuv_p95_by_channel": correction_gradient_by_channel,
        "correction_gradient_yuv_p95_max": max(correction_gradient_values, default=0.0),
    }
    # This metric isolates the new correction field, so a natural text edge in
    # V4 is not mistaken for a V5 seam.  It is intentionally conservative;
    # smoothstep with the default 16 px band normally stays well below it.
    metrics["passed"] = bool(
        np.isfinite(metrics["v5_change_gradient_luma_p95"])
        and metrics["v5_change_gradient_luma_p95"] <= float(gradient_limit)
        and np.isfinite(metrics["correction_gradient_yuv_p95_max"])
        and metrics["correction_gradient_yuv_p95_max"] <= float(gradient_limit)
    )
    return metrics


def _residual_gate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    min_improvement: float,
    nonregression_tolerance: float,
    chroma_nonregression_tolerance: float,
) -> dict[str, Any]:
    """Require a measurable, non-regressing Y residual improvement.

    The gate is based only on the background support region.  All three Y
    summaries are required not to worsen and their arithmetic mean supplies
    the objective improvement score.  U/V have no minimum-improvement target,
    but their mean/median/P90 must also not worsen beyond an independent small
    tolerance; this prevents a useful luma update from hiding a colour
    regression.
    """

    before_channels = before.get("channels", []) if isinstance(before, Mapping) else []
    after_channels = after.get("channels", []) if isinstance(after, Mapping) else []
    before_by_channel = {
        str(item.get("channel")): item for item in before_channels if isinstance(item, Mapping)
    }
    after_by_channel = {
        str(item.get("channel")): item for item in after_channels if isinstance(item, Mapping)
    }
    before_y = before_by_channel.get("Y")
    after_y = after_by_channel.get("Y")
    keys = ("mean_abs", "median_abs", "p90_abs")
    if not isinstance(before_y, Mapping) or not isinstance(after_y, Mapping):
        return {
            "passed": False,
            "reason": "NO_BACKGROUND_RESIDUAL_IMPROVEMENT",
            "before_score": 0.0,
            "after_score": 0.0,
            "improvement": 0.0,
            "minimum_required": float(min_improvement),
            "y_before": {},
            "y_after": {},
            "uv_non_regressed": False,
            "uv_before": {},
            "uv_after": {},
        }
    before_values = {key: float(before_y.get(key, 0.0)) for key in keys}
    after_values = {key: float(after_y.get(key, 0.0)) for key in keys}
    non_regressed = all(
        after_values[key] <= before_values[key] + float(nonregression_tolerance)
        for key in keys
    )
    before_score = float(np.mean(list(before_values.values())))
    after_score = float(np.mean(list(after_values.values())))
    improvement = before_score - after_score
    required = max(float(min_improvement), before_score * 0.02)
    uv_before: dict[str, dict[str, float]] = {}
    uv_after: dict[str, dict[str, float]] = {}
    uv_non_regressed = True
    for channel in ("U", "V"):
        before_channel = before_by_channel.get(channel)
        after_channel = after_by_channel.get(channel)
        if not isinstance(before_channel, Mapping) or not isinstance(after_channel, Mapping):
            uv_non_regressed = False
            continue
        before_values_uv = {key: float(before_channel.get(key, 0.0)) for key in keys}
        after_values_uv = {key: float(after_channel.get(key, 0.0)) for key in keys}
        uv_before[channel] = before_values_uv
        uv_after[channel] = after_values_uv
        uv_non_regressed = uv_non_regressed and all(
            after_values_uv[key] <= before_values_uv[key] + float(chroma_nonregression_tolerance)
            for key in keys
        )
    if not non_regressed or not uv_non_regressed:
        reason = "RESIDUAL_REGRESSION"
        passed = False
    elif improvement < required:
        reason = "NO_BACKGROUND_RESIDUAL_IMPROVEMENT"
        passed = False
    else:
        reason = None
        passed = True
    return {
        "passed": passed,
        "reason": reason,
        "y_non_regressed": non_regressed,
        "before_score": before_score,
        "after_score": after_score,
        "improvement": improvement,
        "minimum_required": required,
        "nonregression_tolerance": float(nonregression_tolerance),
        "chroma_nonregression_tolerance": float(chroma_nonregression_tolerance),
        "y_before": before_values,
        "y_after": after_values,
        "uv_non_regressed": uv_non_regressed,
        "uv_before": uv_before,
        "uv_after": uv_after,
    }


def align_background_v5(
    lq_bgr: np.ndarray,
    gt_v4_bgr: np.ndarray,
    valid_mask: np.ndarray,
    text_mask: np.ndarray,
    *,
    exclude_radius: int = DEFAULT_EXCLUDE_RADIUS,
    blend_radius: int = DEFAULT_BLEND_RADIUS,
    y_sigma: float = DEFAULT_Y_SIGMA,
    uv_sigma: float = DEFAULT_UV_SIGMA,
    min_support_pixels: int = 256,
    min_support_fraction: float = 0.02,
    min_support_weight: float = 0.05,
    max_delta: float = 96.0,
    max_residual_p99: float = 160.0,
    clip_fraction_limit: float = 0.05,
    seam_gradient_limit: float = DEFAULT_SEAM_GRADIENT_LIMIT,
    seam_ring_radius: int = 8,
    min_y_improvement: float = DEFAULT_MIN_Y_IMPROVEMENT,
    y_nonregression_tolerance: float = DEFAULT_Y_NONREGRESSION_TOLERANCE,
    chroma_nonregression_tolerance: float = DEFAULT_CHROMA_NONREGRESSION_TOLERANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align non-text GT pixels and return ``(image, audit_payload)``.

    A rejected sample always returns a byte-equivalent pixel array to V4.  The
    caller may then copy the V4 PNG byte-for-byte, which also makes fallback
    publication auditable.
    """

    if lq_bgr.shape != gt_v4_bgr.shape or lq_bgr.ndim != 3 or lq_bgr.shape[2] != 3:
        raise V5InputError(f"LQ/GT shape mismatch: {lq_bgr.shape} vs {gt_v4_bgr.shape}")
    height, width = gt_v4_bgr.shape[:2]
    valid = _normalize_binary_mask(valid_mask, (height, width), "valid_mask")
    text = _normalize_binary_mask(text_mask, (height, width), "text_mask")
    valid_bool = valid > 0
    text_bool = text > 0
    params = {
        "exclude_radius": int(exclude_radius),
        "blend_radius": int(blend_radius),
        "y_sigma": float(y_sigma),
        "uv_sigma": float(uv_sigma),
        "min_support_pixels": int(min_support_pixels),
        "min_support_fraction": float(min_support_fraction),
        "min_support_weight": float(min_support_weight),
        "max_delta": float(max_delta),
        "max_residual_p99": float(max_residual_p99),
        "clip_fraction_limit": float(clip_fraction_limit),
        "seam_gradient_limit": float(seam_gradient_limit),
        "seam_ring_radius": int(seam_ring_radius),
        "min_y_improvement": float(min_y_improvement),
        "y_nonregression_tolerance": float(y_nonregression_tolerance),
        "chroma_nonregression_tolerance": float(chroma_nonregression_tolerance),
    }
    base_audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        # The V4 producer fills OCR line/polygon support, not a glyph-level
        # segmentation.  Keep this explicit so downstream reviewers do not
        # over-interpret the protected core or its feather ring.
        "text_mask_semantics": "published_ocr_polygon_or_line_region",
        "parameters": params,
        "shape": [int(width), int(height)],
        "pixels": {
            "valid": int(valid_bool.sum()),
            "text": int(text_bool.sum()),
            "text_core_changed": 0,
            "invalid_changed": 0,
        },
    }

    # The V4 text result is only protected if there is an actual text mask.
    # This safety rule also avoids treating a no-text image as a normal
    # background calibration sample with a different semantic contract.
    if not np.any(text_bool):
        base_audit.update({
            "status": "fallback",
            "accepted": False,
            "fallback_reason": "NO_TEXT_MASK",
            "background": {"support_pixels": 0, "support_fraction": 0.0, "low_support_fraction": 1.0},
            "residual_before": _residual_metrics(lq_bgr, gt_v4_bgr, valid_bool),
            "residual_after": _residual_metrics(lq_bgr, gt_v4_bgr, valid_bool),
            "residual_gate": {"passed": False, "reason": "NO_BACKGROUND_RESIDUAL_IMPROVEMENT"},
            "seam": {"passed": True, "ring_pixels": 0},
        })
        return gt_v4_bgr.copy(), base_audit

    excluded_text = _dilate(text, int(exclude_radius)) > 0
    support = valid_bool & ~excluded_text
    support_pixels = int(support.sum())
    total_valid = max(1, int(valid_bool.sum()))
    support_fraction = float(support_pixels / total_valid)
    base_audit["background"] = {
        "support_pixels": support_pixels,
        "support_fraction": support_fraction,
        "excluded_text_pixels": int(excluded_text.sum()),
    }
    if support_pixels < int(min_support_pixels) or support_fraction < float(min_support_fraction):
        base_audit.update({
            "status": "fallback",
            "accepted": False,
            "fallback_reason": "INSUFFICIENT_BACKGROUND_SUPPORT",
            "background": {**base_audit["background"], "low_support_fraction": 1.0},
            "residual_before": _residual_metrics(lq_bgr, gt_v4_bgr, support),
            "residual_after": _residual_metrics(lq_bgr, gt_v4_bgr, support),
            "residual_gate": {"passed": False, "reason": "NO_BACKGROUND_RESIDUAL_IMPROVEMENT"},
            "seam": {"passed": True, "ring_pixels": 0},
        })
        return gt_v4_bgr.copy(), base_audit

    lq_yuv = rgb2yuv_numpy(lq_bgr.astype(np.float32)[..., ::-1])
    gt_yuv = rgb2yuv_numpy(gt_v4_bgr.astype(np.float32)[..., ::-1])
    residual = lq_yuv - gt_yuv
    delta_channels: list[np.ndarray] = []
    channel_audit: list[dict[str, Any]] = []
    low_support = np.zeros((height, width), dtype=bool)
    for index, channel_name in enumerate(("Y", "U", "V")):
        clipped_residual, stats = _robust_channel_residual(
            residual[..., index], support,
            max_delta=float(max_delta), min_clip=8.0,
        )
        if stats["p99_abs"] > float(max_residual_p99):
            # Record the condition here; the common rejection below gives the
            # sample one deterministic reason while retaining diagnostics.
            stats["unstable"] = True
        else:
            stats["unstable"] = False
        delta, reliability = _masked_gaussian_normalized(
            clipped_residual,
            support,
            float(y_sigma if index == 0 else uv_sigma),
            fallback=stats["median"],
            min_support_weight=float(min_support_weight),
        )
        delta_channels.append(delta)
        low_support |= reliability < float(min_support_weight)
        channel_audit.append({"channel": channel_name, **stats})
    delta_yuv = np.stack(delta_channels, axis=-1)
    low_support_fraction = float(np.mean(low_support[valid_bool])) if np.any(valid_bool) else 1.0
    base_audit["background"].update({
        "low_support_pixels": int(low_support[valid_bool].sum()),
        "low_support_fraction": low_support_fraction,
        "channels": channel_audit,
    })

    # Apply the continuous residual field to all GT pixels.  The candidate is
    # intentionally computed beneath text as well: both sides of the blend
    # then originate from one field, preventing two-transform edge seams.
    candidate_yuv = gt_yuv + delta_yuv
    candidate_rgb = yuv2rgb_numpy(candidate_yuv)[..., ::-1]
    clip_mask = (candidate_rgb < 0.0) | (candidate_rgb > 255.0)
    clip_pixels = int(np.any(clip_mask, axis=2)[valid_bool].sum())
    valid_channels = max(1, int(valid_bool.sum()) * 3)
    clip_fraction = float(clip_mask[valid_bool].sum() / valid_channels)
    candidate_rgb = np.clip(candidate_rgb, 0.0, 255.0)

    alpha = _soft_text_alpha(text, int(blend_radius))
    output_float = alpha[..., None] * gt_v4_bgr.astype(np.float32) + (1.0 - alpha[..., None]) * candidate_rgb
    output_float = np.where(valid_bool[..., None], output_float, gt_v4_bgr.astype(np.float32))
    # Explicit assignment makes the text-core and invalid-region invariants
    # independent of floating point arithmetic and PNG round trips.
    output_float[text_bool] = gt_v4_bgr[text_bool]
    output_float[~valid_bool] = gt_v4_bgr[~valid_bool]
    output = np.rint(np.clip(output_float, 0.0, 255.0)).astype(np.uint8)
    text_changed = int(np.any(output[text_bool] != gt_v4_bgr[text_bool], axis=1).sum()) if np.any(text_bool) else 0
    invalid_changed = int(np.any(output[~valid_bool] != gt_v4_bgr[~valid_bool], axis=1).sum()) if np.any(~valid_bool) else 0
    # The seam gate must cover the complete feather band, even if the caller
    # selected a smaller diagnostic ring by mistake.
    effective_seam_ring_radius = max(int(seam_ring_radius), int(blend_radius))
    seam = _seam_metrics(
        gt_v4_bgr, output, text, valid,
        ring_radius=effective_seam_ring_radius, gradient_limit=float(seam_gradient_limit),
    )
    seam["ring_radius"] = effective_seam_ring_radius
    residual_before = _residual_metrics(lq_bgr, gt_v4_bgr, support)
    residual_after_candidate = _residual_metrics(lq_bgr, output, support)
    residual_gate = _residual_gate(
        residual_before,
        residual_after_candidate,
        min_improvement=float(min_y_improvement),
        nonregression_tolerance=float(y_nonregression_tolerance),
        chroma_nonregression_tolerance=float(chroma_nonregression_tolerance),
    )
    unstable = any(bool(item.get("unstable")) for item in channel_audit)
    reasons: list[str] = []
    if unstable:
        reasons.append("UNSTABLE_RESIDUAL")
    if low_support_fraction > 0.85:
        reasons.append("LOW_SUPPORT_FIELD")
    if clip_fraction > float(clip_fraction_limit):
        reasons.append("RGB_CLIPPING")
    if text_changed:
        reasons.append("TEXT_CORE_CHANGED")
    if invalid_changed:
        reasons.append("INVALID_REGION_CHANGED")
    if not seam.get("passed", False):
        reasons.append("SEAM_METRIC_FAILED")
    if not residual_gate.get("passed", False):
        reasons.append(str(residual_gate.get("reason") or "NO_BACKGROUND_RESIDUAL_IMPROVEMENT"))
    accepted = not reasons
    if not accepted:
        output = gt_v4_bgr.copy()
        residual_after = residual_before
    else:
        residual_after = residual_after_candidate
    base_audit.update({
        "status": "accepted" if accepted else "fallback",
        "accepted": accepted,
        "fallback_reason": None if accepted else reasons[0],
        "fallback_reasons": reasons,
        "pixels": {
            **base_audit["pixels"],
            "candidate_clipped": clip_pixels,
            "candidate_clip_fraction": clip_fraction,
            "blend_pixels": int(np.count_nonzero((alpha > 0.0) & (alpha < 1.0) & valid_bool)),
            "background_pixels_changed": int(np.any(output[valid_bool & ~text_bool] != gt_v4_bgr[valid_bool & ~text_bool], axis=1).sum()),
            "text_core_changed": text_changed,
            "invalid_changed": invalid_changed,
        },
        "residual_before": residual_before,
        "residual_after": residual_after,
        "residual_after_candidate": residual_after_candidate,
        "residual_gate": residual_gate,
        "seam": seam,
    })
    return output, base_audit


def _resolve_manifest_path(manifest_path: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _safe_id(value: Any) -> str:
    sample_id = str(value)
    if not sample_id or sample_id in {".", ".."} or Path(sample_id).name != sample_id:
        raise V5InputError(f"unsafe sample_id: {sample_id!r}")
    return sample_id


def _find_path(directory: Path, sample_id: str, suffixes: Sequence[str]) -> Path:
    for suffix in suffixes:
        candidate = directory / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise V5InputError(f"missing {sample_id} in {directory}")


def _field_directory(directories: Mapping[str, Any], field: str, root: Path, version: str) -> Path:
    aliases = {
        "lq": ("lq", "LQ", "LQ_V3"),
        "gt": ("gt", "GT", "GT_V3"),
        "valid_mask": ("valid_mask", "valid", "valid_mask_V3"),
        "text_mask": ("text_mask", "text", "text_mask_V3"),
        "ocr": ("ocr", "OCR", "OCR_V3"),
    }
    for alias in aliases[field]:
        if alias in directories and directories[alias]:
            return _resolve_manifest_path(root / "__manifest__.json", directories[alias])
    return root / (f"{field if field not in {'lq', 'gt'} else field.upper()}_{version}" if field in {"lq", "gt"} else f"{field}_{version}")


def _sample_file(
    sample: Mapping[str, Any], field: str, directory: Path, sample_id: str, manifest_path: Path,
) -> Path:
    explicit_keys = {
        "lq": ("lq_path", "LQ_path", "lq"),
        "gt": ("gt_path", "GT_path", "gt"),
        "valid_mask": ("valid_mask_path", "valid_path", "valid_mask"),
        "text_mask": ("text_mask_path", "text_path", "text_mask"),
        "ocr": ("ocr_path", "OCR_path", "ocr"),
    }
    for key in explicit_keys[field]:
        value = sample.get(key)
        if isinstance(value, (str, os.PathLike)) and str(value):
            candidate = _resolve_manifest_path(manifest_path, value)
            if candidate.is_file():
                return candidate
    suffixes = (".json",) if field == "ocr" else (".png", ".jpg", ".jpeg", ".webp")
    return _find_path(directory, sample_id, suffixes)


def load_v3_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise V5InputError(f"cannot parse manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise V5InputError("V3 manifest must contain a samples array")
    version = str(payload.get("version", "V3")).upper()
    if version not in {"V3", ""} and "V3" not in manifest_path.name.upper():
        raise V5InputError(f"expected a V3 manifest, got version={version!r}")
    directories = payload.get("directories") if isinstance(payload.get("directories"), Mapping) else {}
    root = manifest_path.parent
    field_dirs = {
        field: _field_directory(directories, field, root, "V3")
        for field in ("lq", "gt", "valid_mask", "text_mask", "ocr")
    }
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload["samples"]:
        if not isinstance(item, Mapping) or not item.get("sample_id"):
            raise V5InputError("each V3 sample must contain sample_id")
        sample = dict(item)
        sample_id = _safe_id(sample["sample_id"])
        if sample_id in seen:
            raise V5InputError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        sample["sample_id"] = sample_id
        samples.append(sample)
    # The directories may be provided under arbitrary absolute paths, but the
    # files are resolved per sample below so explicit sample paths remain valid.
    return payload, samples, field_dirs


def _copy_verified(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    if source_hash != sha256_file(destination):
        raise IOError(f"copy hash mismatch: {source} -> {destination}")
    try:
        destination.chmod(0o644)
    except OSError:
        pass
    return source_hash


def _write_gt(output: np.ndarray, source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), output, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(f"failed to write GT_V5: {destination}")
    try:
        destination.chmod(0o644)
    except OSError:
        pass
    return sha256_file(destination)


def publish_v5(
    manifest_path: Path,
    output_root: Path,
    *,
    exclude_radius: int = DEFAULT_EXCLUDE_RADIUS,
    blend_radius: int = DEFAULT_BLEND_RADIUS,
    y_sigma: float = DEFAULT_Y_SIGMA,
    uv_sigma: float = DEFAULT_UV_SIGMA,
    min_support_pixels: int = 256,
    min_support_fraction: float = 0.02,
    min_support_weight: float = 0.05,
    max_delta: float = 96.0,
    max_residual_p99: float = 160.0,
    clip_fraction_limit: float = 0.05,
    seam_gradient_limit: float = DEFAULT_SEAM_GRADIENT_LIMIT,
    seam_ring_radius: int = 8,
    min_y_improvement: float = DEFAULT_MIN_Y_IMPROVEMENT,
    y_nonregression_tolerance: float = DEFAULT_Y_NONREGRESSION_TOLERANCE,
    chroma_nonregression_tolerance: float = DEFAULT_CHROMA_NONREGRESSION_TOLERANCE,
) -> dict[str, Any]:
    """Publish a new V5 projection from ``dataset_V4_manifest.json``.

    Existing V5 directories/manifests are refused so an accidental rerun can
    never silently replace a prior release.  A temporary staging directory is
    atomically promoted only after every selected sample succeeds.
    """

    manifest_path = manifest_path.resolve()
    payload, samples, field_dirs = load_v3_manifest(manifest_path)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    names = {
        "lq": "LQ_V5",
        "gt": "GT_V5",
        "valid_mask": "valid_mask_V5",
        "text_mask": "text_mask_V5",
        "ocr": "OCR_V5",
        "audit": "audit_V5",
    }
    targets = {key: output_root / value for key, value in names.items()}
    manifest_target = output_root / "dataset_V5_manifest.json"
    conflicts = [str(path) for path in [*targets.values(), manifest_target] if path.exists()]
    if conflicts:
        raise FileExistsError("refusing to overwrite existing V5 outputs: " + ", ".join(conflicts))

    staging = Path(tempfile.mkdtemp(prefix=".stage-background-v5-", dir=str(output_root)))
    staged = {key: staging / value for key, value in names.items()}
    for directory in staged.values():
        directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    try:
        for sample in samples:
            sample_id = str(sample["sample_id"])
            paths = {
                field: _sample_file(sample, field, field_dirs[field], sample_id, manifest_path)
                for field in ("lq", "gt", "valid_mask", "text_mask", "ocr")
            }
            lq = _read_color(paths["lq"], "LQ")
            gt = _read_color(paths["gt"], "GT_V3")
            valid = _read_mask(paths["valid_mask"], "valid_mask_V3")
            text = _read_mask(paths["text_mask"], "text_mask_V3")
            if lq.shape != gt.shape:
                raise V5InputError(f"{sample_id}: LQ/GT shape mismatch {lq.shape} vs {gt.shape}")
            if valid.shape != gt.shape[:2] or text.shape != gt.shape[:2]:
                raise V5InputError(f"{sample_id}: mask shape mismatch")
            output, audit = align_background_v5(
                lq, gt, valid, text,
                exclude_radius=exclude_radius,
                blend_radius=blend_radius,
                y_sigma=y_sigma,
                uv_sigma=uv_sigma,
                min_support_pixels=min_support_pixels,
                min_support_fraction=min_support_fraction,
                min_support_weight=min_support_weight,
                max_delta=max_delta,
                max_residual_p99=max_residual_p99,
                clip_fraction_limit=clip_fraction_limit,
                seam_gradient_limit=seam_gradient_limit,
                seam_ring_radius=seam_ring_radius,
                min_y_improvement=min_y_improvement,
                y_nonregression_tolerance=y_nonregression_tolerance,
                chroma_nonregression_tolerance=chroma_nonregression_tolerance,
            )
            # Every versioned companion is a byte-for-byte copy.  OCR is not
            # reinterpreted by this stage; the V4 annotation remains the source
            # of truth for the protected text mask.
            output_gt = staged["gt"] / f"{sample_id}.png"
            if audit.get("accepted"):
                output_gt_hash = _write_gt(output, paths["gt"], output_gt)
            else:
                output_gt_hash = _copy_verified(paths["gt"], output_gt)
            hashes_in = {field: sha256_file(path) for field, path in paths.items()}
            hashes_out = {
                "gt": output_gt_hash,
                "lq": _copy_verified(paths["lq"], staged["lq"] / f"{sample_id}{paths['lq'].suffix}"),
                "valid_mask": _copy_verified(paths["valid_mask"], staged["valid_mask"] / f"{sample_id}{paths['valid_mask'].suffix}"),
                "text_mask": _copy_verified(paths["text_mask"], staged["text_mask"] / f"{sample_id}{paths['text_mask'].suffix}"),
                "ocr": _copy_verified(paths["ocr"], staged["ocr"] / f"{sample_id}{paths['ocr'].suffix}"),
            }
            audit["sample_id"] = sample_id
            audit["inputs"] = {
                field: {"path": str(path), "sha256": hashes_in[field]}
                for field, path in paths.items()
            }
            audit["outputs"] = {
                field: {
                    "path": str(targets[field] / f"{sample_id}{Path(value).suffix}"),
                    "sha256": hashes_out[field],
                }
                for field, value in {
                    "gt": "x.png",
                    "lq": paths["lq"].name,
                    "valid_mask": paths["valid_mask"].name,
                    "text_mask": paths["text_mask"].name,
                    "ocr": paths["ocr"].name,
                }.items()
            }
            audit["output_gt_byte_identical_to_v4"] = bool(hashes_in["gt"] == output_gt_hash)
            audit_path = staged["audit"] / f"{sample_id}.json"
            _write_json(audit_path, audit)
            audit_records.append(audit)
            records.append({
                "sample_id": sample_id,
                "status": "accepted" if audit.get("accepted") else "fallback",
                "fallback_reason": audit.get("fallback_reason"),
                "shape": [int(gt.shape[1]), int(gt.shape[0])],
                "files": {
                    "lq": str(targets["lq"] / f"{sample_id}{paths['lq'].suffix}"),
                    "gt": str(targets["gt"] / f"{sample_id}.png"),
                    "valid_mask": str(targets["valid_mask"] / f"{sample_id}{paths['valid_mask'].suffix}"),
                    "text_mask": str(targets["text_mask"] / f"{sample_id}{paths['text_mask'].suffix}"),
                    "ocr": str(targets["ocr"] / f"{sample_id}{paths['ocr'].suffix}"),
                    "audit": str(targets["audit"] / f"{sample_id}.json"),
                },
                "input_hashes": hashes_in,
                "output_hashes": hashes_out,
            })
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "version": "V5",
            "source_manifest": str(manifest_path),
            "sample_count": len(audit_records),
            "accepted_count": sum(bool(item.get("accepted")) for item in audit_records),
            "fallback_count": sum(not bool(item.get("accepted")) for item in audit_records),
            "samples": audit_records,
        }
        _write_json(staging / "audit_manifest.json", aggregate)
        release = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "version": "V5",
            "source_version": "V3",
            "text_mask_semantics": "published_ocr_polygon_or_line_region",
            "source_manifest": str(manifest_path),
            "sample_count": len(records),
            "accepted_count": sum(item["status"] == "accepted" for item in records),
            "fallback_count": sum(item["status"] == "fallback" for item in records),
            "directories": {key: str(targets[key]) for key in names},
            "audit_manifest": str(output_root / "audit_V5" / "audit_manifest.json"),
            "parameters": {
                "exclude_radius": int(exclude_radius),
                "blend_radius": int(blend_radius),
                "y_sigma": float(y_sigma),
                "uv_sigma": float(uv_sigma),
                "min_support_pixels": int(min_support_pixels),
                "min_support_fraction": float(min_support_fraction),
                "min_support_weight": float(min_support_weight),
                "max_delta": float(max_delta),
                "max_residual_p99": float(max_residual_p99),
                "clip_fraction_limit": float(clip_fraction_limit),
                "seam_gradient_limit": float(seam_gradient_limit),
                "seam_ring_radius": int(seam_ring_radius),
                "min_y_improvement": float(min_y_improvement),
                "y_nonregression_tolerance": float(y_nonregression_tolerance),
                "chroma_nonregression_tolerance": float(chroma_nonregression_tolerance),
            },
            "samples": records,
        }
        # Keep useful selection/cleaning provenance from V4, without copying
        # its version or directory pointers into the new release.
        for key in ("excluded_cleaning_count", "excluded_cleaning", "cleaning_report", "cleaning_report_status"):
            if key in payload:
                release[key] = copy.deepcopy(payload[key])
        _write_json(staging / "dataset_V5_manifest.json", release)
        for key in names:
            os.replace(staged[key], targets[key])
        # The aggregate audit is inside audit_V5 so its path is stable.
        os.replace(staging / "audit_manifest.json", targets["audit"] / "audit_manifest.json")
        os.replace(staging / "dataset_V5_manifest.json", manifest_target)
        return release
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="published dataset_V3_manifest.json")
    parser.add_argument("--output-root", type=Path, required=True, help="training root for new *_V5 directories")
    parser.add_argument("--exclude-radius", type=int, default=DEFAULT_EXCLUDE_RADIUS)
    parser.add_argument("--blend-radius", type=int, default=DEFAULT_BLEND_RADIUS)
    parser.add_argument("--y-sigma", type=float, default=DEFAULT_Y_SIGMA)
    parser.add_argument("--uv-sigma", type=float, default=DEFAULT_UV_SIGMA)
    parser.add_argument("--min-support-pixels", type=int, default=256)
    parser.add_argument("--min-support-fraction", type=float, default=0.02)
    parser.add_argument("--min-support-weight", type=float, default=0.05)
    parser.add_argument("--max-delta", type=float, default=96.0)
    parser.add_argument("--max-residual-p99", type=float, default=160.0)
    parser.add_argument("--clip-fraction-limit", type=float, default=0.05)
    parser.add_argument("--seam-gradient-limit", type=float, default=DEFAULT_SEAM_GRADIENT_LIMIT)
    parser.add_argument("--seam-ring-radius", type=int, default=8)
    parser.add_argument("--min-y-improvement", type=float, default=DEFAULT_MIN_Y_IMPROVEMENT)
    parser.add_argument("--y-nonregression-tolerance", type=float, default=DEFAULT_Y_NONREGRESSION_TOLERANCE)
    parser.add_argument("--chroma-nonregression-tolerance", type=float, default=DEFAULT_CHROMA_NONREGRESSION_TOLERANCE)
    args = parser.parse_args(argv)
    result = publish_v5(
        args.manifest,
        args.output_root,
        exclude_radius=args.exclude_radius,
        blend_radius=args.blend_radius,
        y_sigma=args.y_sigma,
        uv_sigma=args.uv_sigma,
        min_support_pixels=args.min_support_pixels,
        min_support_fraction=args.min_support_fraction,
        min_support_weight=args.min_support_weight,
        max_delta=args.max_delta,
        max_residual_p99=args.max_residual_p99,
        clip_fraction_limit=args.clip_fraction_limit,
        seam_gradient_limit=args.seam_gradient_limit,
        seam_ring_radius=args.seam_ring_radius,
        min_y_improvement=args.min_y_improvement,
        y_nonregression_tolerance=args.y_nonregression_tolerance,
        chroma_nonregression_tolerance=args.chroma_nonregression_tolerance,
    )
    print(json.dumps({
        "status": "published",
        "version": "V5",
        "sample_count": result["sample_count"],
        "accepted_count": result["accepted_count"],
        "fallback_count": result["fallback_count"],
        "manifest": str(args.output_root / "dataset_V5_manifest.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
