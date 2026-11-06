from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    """@luna-comment Convert an HWC RGB tensor to BT.601 luma without color decoding.

    Input: HxWx3 float32 RGB tensor.
    Output: HxW float32 luma in the same nominal range.
    Key Steps: Apply fixed RGB luminance weights whose sum is one.
    Key Params: 0.299/0.587/0.114 match existing evaluator metric convention.
    State/Dependencies: None.
    """
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """@luna-comment Build a normalised odd-sized Gaussian point-spread kernel.

    Input: Positive odd kernel size and positive sigma.
    Output: Float32 2-D kernel whose sum is one.
    Key Steps: Evaluate the separable Gaussian and normalise safely.
    Key Params: Kernel size controls support; sigma controls blur width.
    State/Dependencies: Numpy only.
    """
    size = int(size)
    sigma = float(sigma)
    if size < 3 or size % 2 == 0 or sigma <= 0.0:
        raise ValueError("deblur kernel size must be odd and sigma must be positive")
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    one = np.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel = np.outer(one, one)
    total = float(kernel.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("deblur kernel normalisation failed")
    return (kernel / total).astype(np.float32)


def _p10_segment_text(y: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    """@luna-comment Build a conservative text mask for P10 boundary sampling.

    Input: HxW luma plane and optional mask-confidence parameters.
    Output: Binary foreground mask or None plus polarity, confidence, and
      rejection telemetry; the mask is never inferred from GT.
    Key Steps: Blur and quantise Y for Otsu, infer polarity from the ROI border,
      remove tiny connected components, and enforce separation/coverage rules.
    Key Params: Segmentation sigma .6, border-majority threshold .60, median
      gap .12, Otsu separation .45, and foreground coverage [.01,.65].
    State/Dependencies: OpenCV thresholding/components only; no file writes.
    """
    try:
        if y.ndim != 2 or not np.isfinite(y).all() or y.size == 0:
            return None, {"segmentation_accepted": False, "segmentation_reason": "invalid_luma"}
        sigma = float(params.get("mask_sigma", 0.6))
        border_share = float(params.get("mask_border_share", 0.60))
        if not sigma > 0.0 or not 0.5 <= border_share <= 1.0:
            raise ValueError("invalid P10 mask parameters")
        smooth = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        encoded = np.clip(np.rint(smooth * 255.0), 0, 255).astype(np.uint8)
        threshold, binary = cv2.threshold(encoded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_mask = binary.astype(bool)
        border_width = max(1, round(min(y.shape) * 0.03))
        border = np.zeros_like(high_mask, dtype=bool)
        border[:border_width] = True
        border[-border_width:] = True
        border[:, :border_width] = True
        border[:, -border_width:] = True
        high_border = float(high_mask[border].mean())
        high_roi = float(high_mask.mean())
        if high_border >= border_share:
            foreground = ~high_mask
            polarity = "dark_text_on_light"
        elif 1.0 - high_border >= border_share:
            foreground = high_mask
            polarity = "light_text_on_dark"
        elif high_roi >= border_share:
            foreground = ~high_mask
            polarity = "dark_text_on_light_roi_majority"
        elif 1.0 - high_roi >= border_share:
            foreground = high_mask
            polarity = "light_text_on_dark_roi_majority"
        else:
            return None, {"segmentation_accepted": False, "segmentation_reason": "ambiguous_polarity", "otsu_threshold": float(threshold) / 255.0}
        original_area = int(foreground.sum())
        min_area = max(2, min(8, round(y.shape[0] * y.shape[1] * 0.00002)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(foreground, dtype=bool)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                keep |= labels == label
        removed_fraction = 1.0 - float(keep.sum()) / max(original_area, 1)
        low_values = y[~high_mask]
        high_values = y[high_mask]
        total_variance = float(np.var(y))
        separation = 0.0 if total_variance <= 1.0e-12 or not low_values.size or not high_values.size else float((low_values.size / y.size) * (high_values.size / y.size) * (float(low_values.mean()) - float(high_values.mean())) ** 2 / total_variance)
        median_gap = abs(float(np.median(low_values)) - float(np.median(high_values))) if low_values.size and high_values.size else 0.0
        foreground_fraction = float(keep.mean())
        accepted = bool(0.01 <= foreground_fraction <= 0.65 and median_gap >= 0.12 and separation >= 0.45 and removed_fraction <= 0.15)
        meta = {
            "segmentation_accepted": accepted,
            "segmentation_reason": None if accepted else "low_confidence_text_mask",
            "polarity": polarity,
            "otsu_threshold": float(threshold) / 255.0,
            "foreground_fraction": foreground_fraction,
            "median_gap": median_gap,
            "otsu_separation": separation,
            "removed_foreground_fraction": removed_fraction,
        }
        return (keep if accepted else None), meta
    except (TypeError, ValueError, FloatingPointError, cv2.error) as error:
        return None, {"segmentation_accepted": False, "segmentation_reason": f"invalid_p10_text_mask:{error}"}


def _p10_isotonic_nondecreasing(values: np.ndarray) -> np.ndarray:
    """@luna-comment Project a sampled transition onto a nondecreasing profile.

    Input: One-dimensional finite profile values ordered from background to
      foreground and interpreted as normalised transition coordinates.
    Output: Same-length float32 profile with the least-squares nondecreasing
      isotonic projection.
    Key Steps: Pool adjacent violating blocks with their weighted means, then
      expand the block levels in the original sample order.
    Key Params: Equal unit weights make the projection deterministic and avoid
      letting oversampled platform points dominate the cleanup.
    State/Dependencies: Numpy only; no global state or image mutation.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("isotonic input must be finite one-dimensional values")
    levels: list[float] = []
    counts: list[int] = []
    for value in values.astype(np.float64):
        levels.append(float(value))
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total = counts[-2] + counts[-1]
            merged = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / total
            levels[-2:] = [merged]
            counts[-2:] = [total]
    projected = np.empty_like(values, dtype=np.float32)
    cursor = 0
    for level, count in zip(levels, counts):
        projected[cursor : cursor + count] = float(level)
        cursor += count
    return projected


def _p10_crossing_offset(values: np.ndarray, offsets: np.ndarray, level: float) -> float | None:
    """@luna-comment Locate the first subpixel crossing of a profile level.

    Input: Finite sampled transition values, matching ordered offsets, and a
      level in [0,1].
    Output: Linear-interpolated offset or None when the level is not crossed.
    Key Steps: Find the first adjacent bracket and interpolate with a safe
      zero-denominator midpoint fallback.
    Key Params: First crossing follows the background-to-foreground normal and
      keeps width/crossing telemetry reproducible under ringing.
    State/Dependencies: Numpy only; no image or global state.
    """
    for index in range(max(0, values.size - 1)):
        left = float(values[index])
        right = float(values[index + 1])
        if (left - level) == 0.0:
            return float(offsets[index])
        if (left - level) * (right - level) <= 0.0:
            denominator = right - left
            fraction = 0.5 if abs(denominator) <= 1.0e-12 else (level - left) / denominator
            return float(offsets[index] + np.clip(fraction, 0.0, 1.0) * (offsets[index + 1] - offsets[index]))
    if values.size and float(values[-1]) == level:
        return float(offsets[-1])
    return None


def _p10_stable_platform(
    values: np.ndarray,
    offsets: np.ndarray,
    side: str,
    params: dict[str, Any],
) -> tuple[float, float, tuple[float, float]] | None:
    """@luna-comment Find the nearest low-variation platform on one edge side.

    Input: Normal-sampled Y profile, ordered offsets, side ``outside`` or
      ``inside``, and platform tolerance parameters.
    Output: Platform median, P90-P10 spread, and the accepted offset interval,
      or None if no stable run of at least two pixels exists.
    Key Steps: Scan contiguous windows outward from the boundary and accept the
      first window whose spread is below absolute/relative contrast tolerance.
    Key Params: Minimum run is 2 px, scan limit is 10 px, absolute tolerance is
      2/255, relative tolerance is .03, and contrast is estimated robustly.
    State/Dependencies: Numpy quantiles only; no writes or external state.
    """
    if side not in {"outside", "inside"} or values.ndim != 1 or values.size != offsets.size:
        raise ValueError("invalid P10 platform profile")
    step = float(params.get("sample_step", 0.5))
    min_run = float(params.get("min_platform_run", 2.0))
    scan_limit = float(params.get("platform_scan_limit", 10.0))
    abs_tolerance = float(params.get("platform_tolerance_abs", 2.0 / 255.0))
    rel_tolerance = float(params.get("platform_tolerance_rel", 0.03))
    robust_contrast = float(np.quantile(values, 0.90) - np.quantile(values, 0.10))
    tolerance = max(abs_tolerance, rel_tolerance * robust_contrast)
    min_samples = max(2, int(math.ceil(min_run / max(step, 1.0e-6))))
    eligible = np.flatnonzero((offsets <= -min_run) & (offsets >= -scan_limit)) if side == "outside" else np.flatnonzero((offsets >= min_run) & (offsets <= scan_limit))
    if eligible.size < min_samples:
        return None
    ordered = eligible[::-1] if side == "outside" else eligible
    for start in range(0, ordered.size - min_samples + 1):
        indices = ordered[start : start + min_samples]
        window = values[indices]
        spread = float(np.quantile(window, 0.90) - np.quantile(window, 0.10))
        if spread <= tolerance:
            return float(np.median(window)), spread, (float(offsets[indices].min()), float(offsets[indices].max()))
    return None


def _p10_measure_profiles(y: np.ndarray, mask: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """@luna-comment Measure reliable per-boundary normal profiles for P10 cleanup.

    Input: Deblurred Y plane, accepted binary text mask, and frozen profile
      geometry/platform parameters.
    Output: Profile records with foreground/background anchors, monotone target
      coordinates, widths/crossings, and aggregate reliability telemetry.
    Key Steps: Derive signed-distance normals, sample each normal with
      ``cv2.remap`` from -16 to +16 px at .5 px, reject missing/returning
      platforms, and prepare monotone or contract-.90 targets.
    Key Params: Half width 16 px, step .5 px, seed stride 3 px, platform run
      >=2 px, contrast >=.12, and cleanup modes none/monotone/contract_090.
    State/Dependencies: OpenCV distance transforms/remapping and numpy only;
      no GT, colour planes, or persistent state.
    """
    try:
        if y.ndim != 2 or mask.shape != y.shape or not np.isfinite(y).all():
            raise ValueError("P10 profile inputs have incompatible shapes")
        half_width = float(params.get("profile_half_width", 16.0))
        step = float(params.get("sample_step", 0.5))
        seed_stride = max(1, int(params.get("boundary_stride", 3)))
        min_contrast = float(params.get("min_profile_contrast", 0.12))
        cleanup = str(params.get("cleanup", "monotone")).lower()
        contract = float(params.get("contract_factor", 0.90))
        max_profiles = max(1, int(params.get("max_profiles", 128)))
        if not (half_width >= 4.0 and step > 0.0 and min_contrast > 0.0 and cleanup in {"none", "monotone", "contract_090"}):
            raise ValueError("invalid P10 profile parameters")
        if cleanup == "contract_090" and not 0.0 < contract < 1.0:
            raise ValueError("P10 contract factor must be in (0,1)")
        offsets = np.arange(-half_width, half_width + step * 0.5, step, dtype=np.float32)
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        signed = inside - outside
        grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
        coordinates = np.argwhere(boundary)
        coordinates = coordinates[::seed_stride]
        if coordinates.shape[0] > max_profiles:
            coordinates = coordinates[:max_profiles]
        profiles: list[dict[str, Any]] = []
        rejected = 0
        for row, column in coordinates:
            magnitude = float(norm[row, column])
            if magnitude <= 1.0e-5:
                rejected += 1
                continue
            normal_x = float(grad_x[row, column] / magnitude)
            normal_y = float(grad_y[row, column] / magnitude)
            map_x = (float(column) + offsets * normal_x).astype(np.float32).reshape(1, -1)
            map_y = (float(row) + offsets * normal_y).astype(np.float32).reshape(1, -1)
            if float(map_x.min()) < 1.0 or float(map_x.max()) > y.shape[1] - 2.0 or float(map_y.min()) < 1.0 or float(map_y.max()) > y.shape[0] - 2.0:
                rejected += 1
                continue
            values = cv2.remap(y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1).astype(np.float32)
            mask_values = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101).reshape(-1) >= 0.5
            outside_idx = offsets <= -2.0
            inside_idx = offsets >= 2.0
            if outside_idx.sum() < 2 or inside_idx.sum() < 2 or float((~mask_values[outside_idx]).mean()) < 0.75 or float(mask_values[inside_idx].mean()) < 0.75:
                rejected += 1
                continue
            background = _p10_stable_platform(values, offsets, "outside", params)
            foreground = _p10_stable_platform(values, offsets, "inside", params)
            if background is None or foreground is None:
                rejected += 1
                continue
            background_level, background_spread, background_interval = background
            foreground_level, foreground_spread, foreground_interval = foreground
            contrast = abs(foreground_level - background_level)
            if not math.isfinite(contrast) or contrast < min_contrast:
                rejected += 1
                continue
            oriented = np.clip((values - background_level) / (foreground_level - background_level), 0.0, 1.0).astype(np.float32)
            projected = np.clip(_p10_isotonic_nondecreasing(oriented), 0.0, 1.0)
            target_t = projected.copy()
            if cleanup == "contract_090":
                target_t = np.clip(0.5 + (projected - 0.5) / contract, 0.0, 1.0).astype(np.float32)
            target_values = (background_level + (foreground_level - background_level) * target_t).astype(np.float32)
            raw_reverse_steps = int(np.sum(np.diff(oriented) < -1.0 / 255.0))
            projected_reverse_steps = int(np.sum(np.diff(projected) < -1.0 / 255.0))
            before_10 = _p10_crossing_offset(oriented, offsets, 0.10)
            before_90 = _p10_crossing_offset(oriented, offsets, 0.90)
            after_10 = _p10_crossing_offset(target_t, offsets, 0.10)
            after_90 = _p10_crossing_offset(target_t, offsets, 0.90)
            before_01 = _p10_crossing_offset(oriented, offsets, 0.01)
            before_99 = _p10_crossing_offset(oriented, offsets, 0.99)
            after_01 = _p10_crossing_offset(target_t, offsets, 0.01)
            after_99 = _p10_crossing_offset(target_t, offsets, 0.99)
            profile = {
                "row": int(row),
                "column": int(column),
                "normal_x": normal_x,
                "normal_y": normal_y,
                "background_y": background_level,
                "foreground_y": foreground_level,
                "contrast": contrast,
                "background_spread": background_spread,
                "foreground_spread": foreground_spread,
                "background_interval": list(background_interval),
                "foreground_interval": list(foreground_interval),
                "raw_reverse_steps": raw_reverse_steps,
                "projected_reverse_steps": projected_reverse_steps,
                "before_10_90_width": None if before_10 is None or before_90 is None else float(before_90 - before_10),
                "after_10_90_width": None if after_10 is None or after_90 is None else float(after_90 - after_10),
                "before_1_99_width": None if before_01 is None or before_99 is None else float(before_99 - before_01),
                "after_1_99_width": None if after_01 is None or after_99 is None else float(after_99 - after_01),
                "before_half_crossing": _p10_crossing_offset(oriented, offsets, 0.5),
                "after_half_crossing": _p10_crossing_offset(target_t, offsets, 0.5),
                "offsets": [float(value) for value in offsets],
                "values": [float(value) for value in values],
                "target_values": [float(value) for value in target_values],
                "_map_x": map_x.reshape(-1),
                "_map_y": map_y.reshape(-1),
                "_delta": target_values - values if cleanup != "none" else np.zeros_like(values),
            }
            profiles.append(profile)
        widths_before = [float(profile["before_10_90_width"]) for profile in profiles if profile["before_10_90_width"] is not None]
        widths_after = [float(profile["after_10_90_width"]) for profile in profiles if profile["after_10_90_width"] is not None]
        widths_1_99_before = [float(profile["before_1_99_width"]) for profile in profiles if profile["before_1_99_width"] is not None]
        widths_1_99_after = [float(profile["after_1_99_width"]) for profile in profiles if profile["after_1_99_width"] is not None]
        crossing_shifts = [abs(float(profile["after_half_crossing"]) - float(profile["before_half_crossing"])) for profile in profiles if profile["before_half_crossing"] is not None and profile["after_half_crossing"] is not None]
        return {
            "profiles": profiles,
            "profile_count": len(profiles),
            "profile_rejected_count": rejected,
            "profile_accept_fraction": float(len(profiles) / max(coordinates.shape[0], 1)),
            "profile_10_90_width_before_median": float(np.median(widths_before)) if widths_before else None,
            "profile_10_90_width_after_median": float(np.median(widths_after)) if widths_after else None,
            "profile_1_99_width_before_median": float(np.median(widths_1_99_before)) if widths_1_99_before else None,
            "profile_1_99_width_after_median": float(np.median(widths_1_99_after)) if widths_1_99_after else None,
            "profile_half_crossing_shift_median": float(np.median(crossing_shifts)) if crossing_shifts else None,
            "profile_reverse_steps_before": int(sum(int(profile["raw_reverse_steps"]) for profile in profiles)),
            "profile_reverse_steps_after": int(sum(int(profile["projected_reverse_steps"]) for profile in profiles)),
            "profile_cleanup_mode": cleanup,
            "profile_contract_factor": contract if cleanup == "contract_090" else None,
        }
    except (TypeError, ValueError, FloatingPointError, cv2.error) as error:
        return {"profiles": [], "profile_count": 0, "profile_rejected_count": 0, "profile_accept_fraction": 0.0, "profile_measurement_error": str(error), "profile_cleanup_mode": str(params.get("cleanup", "monotone"))}


def _p10_reconstruct_rgb_from_y(
    rgb: np.ndarray, y_input: np.ndarray, y_target: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """@luna-comment Reconstruct RGB from a target Y while preserving Cb/Cr.

    Input: Finite HxWx3 RGB tensor, its BT.601 Y plane, and a finite target Y
      plane with the same shape.
    Output: Gamut-constrained RGB tensor plus requested/applied delta and
      chroma telemetry; no quality decision or identity substitution occurs.
    Key Steps: Form one shared luma delta, clip it to each pixel's available
      RGB interval, reconstruct all channels equally, and measure Cb/Cr error.
    Key Params: BT.601 coefficients and the full-range Cb/Cr denominators are
      fixed by the Stage-A direct-RGB contract.
    State/Dependencies: Numpy only; no GT, model, colour-plane, or global state.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or y_input.shape != rgb.shape[:2] or y_target.shape != y_input.shape:
        raise ValueError("P10 RGB/Y reconstruction shapes are incompatible")
    if not np.isfinite(rgb).all() or not np.isfinite(y_input).all() or not np.isfinite(y_target).all():
        raise ValueError("P10 RGB/Y reconstruction inputs must be finite")
    requested = y_target - y_input
    lower = -np.min(rgb, axis=2)
    upper = 1.0 - np.max(rgb, axis=2)
    applied = np.minimum(np.maximum(requested, lower), upper).astype(np.float32)
    output = (rgb + applied[..., None]).astype(np.float32)
    if not np.isfinite(output).all() or float(output.min()) < -2.0e-6 or float(output.max()) > 1.0 + 2.0e-6:
        raise ValueError("P10 RGB reconstruction is non-finite or outside gamut")
    y_out = _rgb_to_luma(output)
    cb_in = (rgb[..., 2] - y_input) / 1.772
    cr_in = (rgb[..., 0] - y_input) / 1.402
    chroma_error = max(
        float(np.max(np.abs((output[..., 2] - y_out) / 1.772 - cb_in))),
        float(np.max(np.abs((output[..., 0] - y_out) / 1.402 - cr_in))),
    )
    return output, {
        "clipping_fraction": 0.0,
        "mean_abs_requested_delta": float(np.mean(np.abs(requested))),
        "mean_abs_applied_delta": float(np.mean(np.abs(applied))),
        "p95_abs_applied_delta": float(np.percentile(np.abs(applied), 95.0)) if applied.size else 0.0,
        "fraction_abs_delta_ge_4_255": float(np.mean(np.abs(applied) >= 4.0 / 255.0)),
        "gamut_limited_fraction": float(np.mean(np.abs(requested - applied) > 1.0e-7)),
        "mean_abs_requested_minus_applied": float(np.mean(np.abs(requested - applied))),
        "chroma_max_abs_error": chroma_error,
    }


def _p11_gaussian_cdf(offsets: np.ndarray, sigma: float, center: float) -> np.ndarray:
    """@luna-comment Evaluate a finite Gaussian step response on normal offsets.

    Input: One-dimensional finite offsets in pixels, positive sigma, and edge
      centre in the same pixel coordinate system.
    Output: Float32 values in [0,1] for Phi((offset-center)/sigma).
    Key Steps: Scale offsets by sqrt(2) and evaluate the standard-library
      error function without adding a SciPy dependency.
    Key Params: Sigma is the equivalent isotropic blur width; center accounts
      for subpixel placement of the sampled boundary.
    State/Dependencies: Python math and numpy only.
    """
    if sigma <= 0.0 or not math.isfinite(sigma) or not math.isfinite(center):
        raise ValueError("Gaussian CDF requires finite positive sigma and center")
    values = np.asarray(offsets, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Gaussian CDF offsets must be finite and one-dimensional")
    scale = math.sqrt(2.0) * sigma
    return np.fromiter(
        (0.5 * (1.0 + math.erf((float(value) - center) / scale)) for value in values),
        dtype=np.float32,
        count=values.size,
    )


def _p11_fit_gaussian_edge_profile(profile: dict[str, Any], params: dict[str, Any]) -> dict[str, Any] | None:
    """@luna-comment Fit one text-edge normal to an equivalent Gaussian step.

    Input: P10-compatible normal profile with offsets, sampled Y values, stable
      foreground/background platform levels, and normal direction metadata.
    Output: JSON-safe fit record containing sigma, centre, robust residuals,
      bound-hit state, and straight-profile diagnostics, or None if rejected.
    Key Steps: Orient the profile background-to-foreground, reject weak or
      strongly reversing edges, then grid-search sigma and subpixel centre with
      a Huber loss for a+b*Phi.
    Key Params: Sigma bounds default to [.3,4.0] px, centre search is +/-4 px,
      contrast floor is .12, and the Huber transition defaults to .05 luma.
    State/Dependencies: Numpy/math only; the profile and source image remain
      read-only.
    """
    try:
        offsets = np.asarray(profile["offsets"], dtype=np.float32)
        values = np.asarray(profile["values"], dtype=np.float32)
        background = float(profile["background_y"])
        foreground = float(profile["foreground_y"])
        if offsets.ndim != 1 or values.shape != offsets.shape or not np.isfinite(offsets).all() or not np.isfinite(values).all():
            return None
        contrast = abs(foreground - background)
        min_contrast = float(params.get("kernel_min_contrast", 0.12))
        if not math.isfinite(contrast) or contrast < min_contrast:
            return None
        oriented = np.clip((values - background) / (foreground - background), 0.0, 1.0)
        reverse_steps = int(np.sum(np.diff(oriented) < -1.0 / 255.0))
        max_reverse = int(params.get("kernel_max_reverse_steps", 8))
        if reverse_steps > max_reverse:
            return None
        sigma_min = float(params.get("kernel_sigma_min", 0.3))
        sigma_max = float(params.get("kernel_sigma_max", 4.0))
        centre_limit = float(params.get("kernel_center_limit", 4.0))
        centre_step = float(params.get("kernel_center_step", 0.25))
        sigma_step = float(params.get("kernel_sigma_step", 0.05))
        huber_delta = float(params.get("kernel_huber_delta", 0.05))
        if not (0.0 < sigma_min <= sigma_max and centre_limit > 0.0 and centre_step > 0.0 and sigma_step > 0.0 and huber_delta > 0.0):
            raise ValueError("invalid P11 Gaussian-fit bounds")
        sigma_grid = np.arange(sigma_min, sigma_max + sigma_step * 0.5, sigma_step, dtype=np.float32)
        centre_grid = np.arange(-centre_limit, centre_limit + centre_step * 0.5, centre_step, dtype=np.float32)
        if sigma_grid.size == 0 or centre_grid.size == 0:
            raise ValueError("empty P11 Gaussian-fit grid")
        best_loss = float("inf")
        best_sigma = None
        best_centre = None
        best_prediction = None
        for sigma in sigma_grid:
            for centre in centre_grid:
                cdf = _p11_gaussian_cdf(offsets, float(sigma), float(centre))
                prediction = background + (foreground - background) * cdf
                residual = prediction.astype(np.float32) - values
                absolute = np.abs(residual)
                loss = np.where(
                    absolute <= huber_delta,
                    0.5 * np.square(absolute),
                    huber_delta * (absolute - 0.5 * huber_delta),
                )
                mean_loss = float(np.mean(loss))
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_sigma = float(sigma)
                    best_centre = float(centre)
                    best_prediction = prediction
        if best_sigma is None or best_prediction is None:
            return None
        residual = best_prediction.astype(np.float32) - values
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        median_abs = float(np.median(np.abs(residual)))
        max_fit_rmse = float(params.get("kernel_max_fit_rmse", 0.08))
        if not math.isfinite(rmse) or rmse > max_fit_rmse:
            return None
        normal_x = float(profile.get("normal_x", 0.0))
        normal_y = float(profile.get("normal_y", 0.0))
        return {
            "row": int(profile.get("row", -1)),
            "column": int(profile.get("column", -1)),
            "normal_x": normal_x,
            "normal_y": normal_y,
            "contrast": contrast,
            "sigma": best_sigma,
            "center": best_centre,
            "fit_rmse": rmse,
            "fit_median_abs_error": median_abs,
            "fit_huber_loss": best_loss,
            "reverse_steps": reverse_steps,
            "sigma_at_lower_bound": bool(abs(best_sigma - sigma_min) <= sigma_step * 0.51),
            "sigma_at_upper_bound": bool(abs(best_sigma - sigma_max) <= sigma_step * 0.51),
            "background_y": background,
            "foreground_y": foreground,
            "offsets": [float(value) for value in offsets],
            "values": [float(value) for value in values],
            "fit_values": [float(value) for value in best_prediction],
        }
    except (KeyError, TypeError, ValueError, FloatingPointError):
        return None


def _p11_gaussian_psf(sigma: float) -> np.ndarray:
    """@luna-comment Build the normalized isotropic Gaussian PSF for P11.

    Input: Finite positive equivalent blur sigma in pixels.
    Output: Odd-sized float32 PSF covering approximately three sigma per side.
    Key Steps: Select a bounded odd support and evaluate the existing Gaussian
      kernel builder with a unit-sum normalization.
    Key Params: Support radius is ceil(3*sigma), balancing FFT cost and tail
      coverage.
    State/Dependencies: Reuses the pure numpy Gaussian kernel helper.
    """
    sigma = float(sigma)
    if not math.isfinite(sigma) or not 0.3 <= sigma <= 4.0:
        raise ValueError("P11 PSF sigma must be finite and within [0.3,4.0]")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    return _gaussian_kernel(2 * radius + 1, sigma)


def _p11_estimate_equivalent_kernel(
    y: np.ndarray, mask: np.ndarray, params: dict[str, Any]
) -> dict[str, Any]:
    """@luna-comment Estimate one isotropic Gaussian PSF from reliable text edges.

    Input: Finite HxW luma plane, accepted P10 text mask, and P11 fit bounds.
    Output: Kernel status, median sigma/MAD, direction and residual statistics,
      per-profile evidence, and a finite PSF when identifiable.
    Key Steps: Sample local normals with the existing platform detector, retain
      high-contrast near-monotone step fits, and robustly aggregate sigma by
      median while retaining bound-hit telemetry.
    Key Params: Profile geometry is the P10 .5 px/-16..16 px contract; sigma is
      bounded to [.3,4.0] px and fit RMSE is config-limited.
    State/Dependencies: Luma/mask arrays are read-only; no GT or file I/O.
    """
    if y.ndim != 2 or mask.shape != y.shape or not np.isfinite(y).all() or not mask.any():
        return {
            "kernel_status": "kernel_unidentifiable",
            "kernel_reason": "invalid_or_empty_text_mask",
            "kernel_sigma": None,
            "kernel_sigma_mad": None,
            "kernel_segment_count": 0,
            "kernel_profiles_considered": 0,
            "kernel_profiles": [],
            "kernel_direction_stats": {},
            "kernel_fit_residual_stats": {},
            "kernel_bound_hits": {"lower": 0, "upper": 0},
            "kernel_psf": None,
        }
    profile_params = dict(params)
    profile_params["cleanup"] = "none"
    measured = _p10_measure_profiles(y, mask, profile_params)
    candidates = measured.get("profiles", []) if isinstance(measured, dict) else []
    fits = []
    for profile in candidates:
        fit = _p11_fit_gaussian_edge_profile(profile, params)
        if fit is not None:
            fits.append(fit)
    sigma_values = np.asarray([float(item["sigma"]) for item in fits], dtype=np.float64)
    if sigma_values.size == 0:
        return {
            "kernel_status": "kernel_unidentifiable",
            "kernel_reason": "no_profile_passed_gaussian_fit",
            "kernel_sigma": None,
            "kernel_sigma_mad": None,
            "kernel_segment_count": 0,
            "kernel_profiles_considered": len(candidates),
            "kernel_profiles": fits,
            "kernel_direction_stats": {},
            "kernel_fit_residual_stats": {},
            "kernel_bound_hits": {"lower": 0, "upper": 0},
            "kernel_psf": None,
        }
    sigma = float(np.median(sigma_values))
    mad = float(np.median(np.abs(sigma_values - sigma)))
    normal_x = np.asarray([float(item["normal_x"]) for item in fits], dtype=np.float64)
    normal_y = np.asarray([float(item["normal_y"]) for item in fits], dtype=np.float64)
    angles = np.degrees(np.arctan2(normal_y, normal_x))
    horizontal_count = int(np.sum(np.abs(normal_x) >= np.abs(normal_y)))
    vertical_count = len(fits) - horizontal_count
    residuals = np.asarray([float(item["fit_rmse"]) for item in fits], dtype=np.float64)
    psf = _p11_gaussian_psf(sigma)
    return {
        "kernel_status": "computed",
        "kernel_reason": None,
        "kernel_sigma": sigma,
        "kernel_sigma_mad": mad,
        "kernel_segment_count": len(fits),
        "kernel_profiles_considered": len(candidates),
        "kernel_profiles": fits,
        "kernel_direction_stats": {
            "horizontal_normal_count": horizontal_count,
            "vertical_normal_count": vertical_count,
            "angle_mean_deg": float(np.mean(angles)),
            "angle_std_deg": float(np.std(angles)),
            "angle_min_deg": float(np.min(angles)),
            "angle_max_deg": float(np.max(angles)),
        },
        "kernel_fit_residual_stats": {
            "rmse_median": float(np.median(residuals)),
            "rmse_p95": float(np.percentile(residuals, 95.0)),
            "rmse_max": float(np.max(residuals)),
        },
        "kernel_bound_hits": {
            "lower": int(sum(bool(item["sigma_at_lower_bound"]) for item in fits)),
            "upper": int(sum(bool(item["sigma_at_upper_bound"]) for item in fits)),
        },
        "kernel_psf": psf,
    }


def _p11_processing_soft_band(mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """@luna-comment Construct the narrow contour soft band used by P11 stages.

    Input: Boolean HxW text mask and finite equivalent sigma in pixels.
    Output: Float32 [0,1] weight and boolean support map; outside support is
      exactly zero.
    Key Steps: Measure distance to the mask boundary and use full weight through
      2*sigma, then cosine-fade to zero by 3*sigma+1.
    Key Params: The two radii are tied to the measured sigma, preventing global
      or full-ROI processing.
    State/Dependencies: OpenCV distance transforms; no image mutation or I/O.
    """
    sigma = float(sigma)
    if mask.ndim != 2 or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("P11 soft-band inputs are invalid")
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    distance = np.maximum(np.where(mask, inside, outside) - 0.5, 0.0)
    inner = 2.0 * sigma
    outer = 3.0 * sigma + 1.0
    weight = np.where(
        distance <= inner,
        1.0,
        np.where(
            distance < outer,
            0.5 * (1.0 + np.cos(np.pi * (distance - inner) / max(outer - inner, 1.0e-6))),
            0.0,
        ),
    ).astype(np.float32)
    return weight, weight > 0.0


def _p11_wiener_hunt_deblur(
    y: np.ndarray, mask: np.ndarray, sigma: float, regularization: float, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply one reflected-context Wiener-Hunt deblur to local Y.

    Input: Finite HxW Y plane, non-empty text mask, estimated sigma, and positive
      Laplacian regularization.
    Output: Full-size raw deconvolved Y plus crop/context telemetry; no masking
      or RGB reconstruction is performed in this helper.
    Key Steps: Crop text with context, reflect-pad the crop, form FFT responses
      for Gaussian K and a discrete Laplacian L, apply K*/(|K|^2+lambda|L|^2),
      then crop back to the image coordinates.
    Key Params: Context defaults to ceil(3*sigma+1); lambda controls high
      frequency amplification and is never treated as a quality gate.
    State/Dependencies: Numpy FFT/OpenCV padding only; GT is not accessed.
    """
    if y.ndim != 2 or mask.shape != y.shape or not np.isfinite(y).all() or not mask.any():
        raise ValueError("P11 Wiener input must be finite and have a non-empty mask")
    sigma = float(sigma)
    regularization = float(regularization)
    if not (math.isfinite(sigma) and 0.3 <= sigma <= 4.0 and math.isfinite(regularization) and regularization > 0.0):
        raise ValueError("P11 Wiener sigma/lambda is invalid")
    rows, columns = np.nonzero(mask)
    context = max(1, int(math.ceil(3.0 * sigma + 1.0)))
    top = max(0, int(rows.min()) - context)
    bottom = min(y.shape[0], int(rows.max()) + context + 1)
    left = max(0, int(columns.min()) - context)
    right = min(y.shape[1], int(columns.max()) + context + 1)
    crop = np.asarray(y[top:bottom, left:right], dtype=np.float32)
    pad = max(1, int(math.ceil(3.0 * sigma)))
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    kernel = _p11_gaussian_psf(sigma)
    radius = kernel.shape[0] // 2
    psf = np.zeros(padded.shape, dtype=np.float32)
    psf[: kernel.shape[0], : kernel.shape[1]] = kernel
    psf = np.roll(psf, (-radius, -radius), axis=(0, 1))
    laplacian = np.zeros(padded.shape, dtype=np.float32)
    laplacian[:3, :3] = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    laplacian = np.roll(laplacian, (-1, -1), axis=(0, 1))
    response = np.fft.fft2(psf)
    lap_response = np.fft.fft2(laplacian)
    denominator = np.square(np.abs(response)) + regularization * np.square(np.abs(lap_response))
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise ValueError("P11 Wiener denominator is invalid")
    inverse = np.conj(response) / denominator
    restored = np.real(np.fft.ifft2(np.fft.fft2(padded) * inverse)).astype(np.float32)
    restored = restored[pad : pad + crop.shape[0], pad : pad + crop.shape[1]]
    raw = y.copy()
    raw[top:bottom, left:right] = restored
    if not np.isfinite(raw).all():
        raise ValueError("P11 raw Wiener output is non-finite")
    return raw, {
        "wiener_status": "computed",
        "wiener_sigma": sigma,
        "wiener_lambda": regularization,
        "wiener_context": context,
        "wiener_pad": pad,
        "wiener_crop": [top, bottom, left, right],
        "wiener_raw_mean_abs_delta": float(np.mean(np.abs(raw - y))),
        "wiener_raw_p95_abs_delta": float(np.percentile(np.abs(raw - y), 95.0)),
        "_p11_raw_deblur_y": raw,
    }


def _p14_gaussian_field(
    values: np.ndarray, seed: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """@luna-comment Smooth sparse platform samples with normalized Gaussian support.

    Input: Finite HxW luma values, boolean platform seed, and positive smoothing
      scale in pixels.
    Output: A finite normalized value field and a [0,1] support field; pixels
      without enough seed support are represented by zero support.
    Key Steps: Blur the value/support numerators independently and divide only
      where the denominator is positive, preserving spatial luminance trends.
    Key Params: The caller derives sigma from the measured blur kernel; no
      rectangular ROI or global platform constant is introduced here.
    State/Dependencies: OpenCV GaussianBlur and read-only numpy arrays.
    """
    if values.ndim != 2 or seed.shape != values.shape or seed.dtype != np.bool_:
        raise ValueError("P14 Gaussian-field inputs are invalid")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(values).all():
        raise ValueError("P14 Gaussian-field parameters are invalid")
    seed_f = seed.astype(np.float32)
    numerator = cv2.GaussianBlur(values.astype(np.float32) * seed_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    denominator = cv2.GaussianBlur(seed_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    field = np.zeros_like(values, dtype=np.float32)
    valid = denominator > 1.0e-8
    field[valid] = numerator[valid] / denominator[valid]
    support = np.clip(denominator, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(field).all() or not np.isfinite(support).all():
        raise ValueError("P14 Gaussian field is non-finite")
    return field, support


def _p14_estimate_local_platform_fields(
    y: np.ndarray, mask: np.ndarray, sigma: float, params: dict[str, Any]
) -> dict[str, Any]:
    """@luna-comment Estimate spatially varying background/foreground platforms.

    Input: Finite input luma, accepted boolean text mask, equivalent blur sigma,
      and P14 field parameters.
    Output: Per-pixel background/foreground/tolerance fields, normalized seed
      supports, component labels, a contour-derived region mask/feather, and
      JSON-safe status/uncertainty telemetry.
    Key Steps: Seed background away from text, seed each connected text
      component independently, smooth numerator/support pairs, and derive
      local MAD tolerance from both seed populations.
    Key Params: Seed distance is max(2,2*sigma); Gaussian field bandwidth is
      max(6,4*sigma); all remaining thresholds are explicit config values.
    State/Dependencies: CPU OpenCV/numpy only; no GT or file writes.
    """
    if y.ndim != 2 or mask.shape != y.shape or mask.dtype != np.bool_ or not np.isfinite(y).all():
        raise ValueError("P14 platform-field inputs are invalid")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("P14 platform-field sigma is invalid")
    shape = y.shape
    empty = np.zeros(shape, dtype=bool)
    zero = np.zeros(shape, dtype=np.float32)
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return {
            "status": "unidentifiable_empty_text",
            "reason": "empty_text_mask",
            "background_field": zero.copy(),
            "foreground_field": zero.copy(),
            "tau_field": zero.copy(),
            "background_support": zero.copy(),
            "foreground_support": zero.copy(),
            "field_support": zero.copy(),
            "region_mask": empty.copy(),
            "region_feather": zero.copy(),
            "component_labels": np.zeros(shape, dtype=np.int32),
            "background_seed": empty.copy(),
            "foreground_seed": empty.copy(),
            "background_seed_count": 0,
            "foreground_seed_count": 0,
            "component_count": 0,
            "background_y": None,
            "foreground_y": None,
            "background_mad": None,
            "foreground_mad": None,
            "uncertainty": None,
            "region_fraction": 0.0,
        }
    # P15 keeps the background seeds well outside the contour halo.  A wider
    # local window is intentional: it follows slow illumination gradients
    # without turning the whole text crop into one constant platform.
    seed_distance = max(4.0, float(math.ceil(4.0 * sigma)))
    field_sigma = max(8.0, 6.0 * sigma)
    region_radius = max(3, int(math.ceil(3.0 * sigma)) + int(math.ceil(2.0 * sigma)))
    region_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * region_radius + 1, 2 * region_radius + 1)
    )
    region_mask = cv2.dilate(mask.astype(np.uint8), region_kernel, iterations=1) > 0
    # The feather is attached to the actual dilated component set. It reaches
    # one over the interior and fades continuously at the region boundary.
    region_distance = cv2.distanceTransform(region_mask.astype(np.uint8), cv2.DIST_L2, 5)
    feather_width = max(1.0, min(float(region_radius), max(1.0, sigma)))
    region_feather = np.clip(region_distance / feather_width, 0.0, 1.0).astype(np.float32)
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    background_seed = region_mask & ~mask & (outside_distance >= seed_distance)
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    foreground_seed = mask & (inside_distance >= seed_distance)
    background_field, background_support = _p14_gaussian_field(y, background_seed, field_sigma)
    foreground_field = np.zeros_like(y, dtype=np.float32)
    foreground_support = np.zeros_like(y, dtype=np.float32)
    # Each text component gets its own foreground field. At overlaps, the
    # component with the strongest normalized seed support wins.
    for label in range(1, int(component_count)):
        component = component_labels == label
        component_seed = foreground_seed & component
        if not bool(component_seed.any()):
            continue
        local_field, local_support = _p14_gaussian_field(y, component_seed, field_sigma)
        local_region = cv2.dilate(component.astype(np.uint8), region_kernel, iterations=1) > 0
        take = local_region & (local_support > foreground_support)
        foreground_field[take] = local_field[take]
        foreground_support[take] = local_support[take]
    field_support = np.minimum(background_support, foreground_support).astype(np.float32)
    min_support = float(params.get("p14_min_field_support", 1.0e-3))
    if not math.isfinite(min_support) or min_support < 0.0 or min_support > 1.0:
        raise ValueError("P14 field support threshold is invalid")

    def _robust_stats(seed: np.ndarray) -> tuple[float | None, float | None, int]:
        values = y[seed].astype(np.float64)
        if values.size == 0:
            return None, None, 0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, mad, int(values.size)

    background_y, background_mad, background_count = _robust_stats(background_seed)
    foreground_y, foreground_mad, foreground_count = _robust_stats(foreground_seed)
    deviation = np.zeros_like(y, dtype=np.float32)
    if background_y is not None:
        deviation[background_seed] = np.abs(y[background_seed] - background_field[background_seed])
    if foreground_y is not None:
        foreground_deviation = np.zeros_like(y, dtype=np.float32)
        foreground_deviation[foreground_seed] = np.abs(y[foreground_seed] - foreground_field[foreground_seed])
    else:
        foreground_deviation = np.zeros_like(y, dtype=np.float32)
    bg_dev_field, bg_dev_support = _p14_gaussian_field(deviation, background_seed, field_sigma) if bool(background_seed.any()) else (zero.copy(), zero.copy())
    fg_dev_field, fg_dev_support = _p14_gaussian_field(foreground_deviation, foreground_seed, field_sigma) if bool(foreground_seed.any()) else (zero.copy(), zero.copy())
    local_mad = np.maximum(
        np.where(bg_dev_support > 1.0e-8, bg_dev_field, 0.0),
        np.where(fg_dev_support > 1.0e-8, fg_dev_field, 0.0),
    ).astype(np.float32)
    tau_field = np.clip(2.0 * local_mad, 2.0 / 255.0, 6.0 / 255.0).astype(np.float32)
    valid_global = background_count > 0 and foreground_count > 0 and bool((field_support >= min_support).any())
    global_contrast = None if background_y is None or foreground_y is None else abs(float(foreground_y - background_y))
    status = "computed" if valid_global else "uncertain_field_support"
    reason = None if valid_global else "missing_platform_seed_or_support"
    uncertainty = None
    if global_contrast is not None and global_contrast > 1.0e-8:
        uncertainty = float(max(background_mad or 0.0, foreground_mad or 0.0) / global_contrast)
    return {
        "status": status,
        "reason": reason,
        "background_field": background_field,
        "foreground_field": foreground_field,
        "tau_field": tau_field,
        "background_support": background_support,
        "foreground_support": foreground_support,
        "field_support": field_support,
        "region_mask": region_mask,
        "region_feather": region_feather,
        "component_labels": component_labels.astype(np.int32),
        "background_seed": background_seed,
        "foreground_seed": foreground_seed,
        "background_seed_count": background_count,
        "foreground_seed_count": foreground_count,
        "component_count": max(int(component_count) - 1, 0),
        "background_y": background_y,
        "foreground_y": foreground_y,
        "background_mad": background_mad,
        "foreground_mad": foreground_mad,
        "uncertainty": uncertainty,
        "global_contrast": global_contrast,
        "field_sigma": field_sigma,
        "seed_distance": seed_distance,
        "region_radius": region_radius,
        "region_fraction": float(np.mean(region_mask)),
        "field_support_fraction": float(np.mean(field_support >= min_support)),
        "_min_support": min_support,
    }


def _p14_side_interval(
    offsets: np.ndarray,
    mask_values: np.ndarray,
    distance_values: np.ndarray,
    side: str,
) -> tuple[int, int] | None:
    """@luna-comment Find one contour side interval without crossing a rival edge.

    Input: Ordered signed normal offsets, sampled text-class values, distance
      to the nearest contour, and either outside or inside side.
    Output: Inclusive index interval beginning at the contour and ending at the
      first class/medial-axis/image-boundary stop, or None if unavailable.
    Key Steps: Walk outward from the contour, require the expected class, and
      stop at the first competing-contour or distance-turning point.
    Key Params: A single one-sample class mismatch is tolerated at the raster
      contour; no full-width fixed normal interval is forced.
    State/Dependencies: Pure finite arrays; no image mutation.
    """
    if offsets.ndim != 1 or mask_values.shape != offsets.shape or distance_values.shape != offsets.shape:
        raise ValueError("P14 side-interval arrays are invalid")
    zero_candidates = np.flatnonzero(np.abs(offsets) <= 0.5 + 1.0e-6)
    if zero_candidates.size == 0:
        return None
    zero_index = int(zero_candidates[np.argmin(np.abs(offsets[zero_candidates]))])
    expected = side == "inside"
    if side == "outside":
        indices = list(range(zero_index, -1, -1))
    elif side == "inside":
        indices = list(range(zero_index, offsets.size))
    else:
        raise ValueError("P14 side must be outside or inside")
    chosen: list[int] = []
    previous_distance = -1.0
    turning_index: int | None = None
    for position, index in enumerate(indices):
        class_value = bool(mask_values[index])
        if class_value != expected and not chosen:
            # The nearest raster sample can still belong to the other side of
            # a one-pixel contour. Skip that leading sample and start at the
            # first expected-class sample; a later mismatch terminates the
            # single-side interval.
            continue
        if class_value != expected:
            break
        current_distance = float(distance_values[index])
        if previous_distance >= 0.0 and current_distance + 0.35 < previous_distance:
            turning_index = index
            break
        chosen.append(index)
        previous_distance = max(previous_distance, current_distance)
        if position > 1 and current_distance > 0.0 and current_distance <= previous_distance + 0.35:
            # Keep the current sample; the next decreasing sample marks the
            # medial/competing boundary.
            continue
    if len(chosen) < 2:
        return None
    if turning_index is not None and turning_index in chosen:
        chosen = chosen[: chosen.index(turning_index) + 1]
    if not chosen:
        return None
    return min(chosen), max(chosen)


def _p14_sample_field(field: np.ndarray, x: float, y: float) -> float:
    """@luna-comment Bilinearly sample one finite scalar platform field.

    Input: HxW float field and floating-point image coordinate in pixel units.
    Output: Finite interpolated scalar, using reflected border only for the
      one-pixel sampling envelope required by the diagnostic.
    Key Steps: Call OpenCV remap with one coordinate and validate finiteness.
    Key Params: Linear interpolation preserves smooth regional platform drift.
    State/Dependencies: Read-only numpy/OpenCV field.
    """
    if field.ndim != 2 or not np.isfinite(field).all() or not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("P14 scalar-field sample is invalid")
    value = cv2.remap(
        field.astype(np.float32),
        np.asarray([[float(x)]], dtype=np.float32),
        np.asarray([[float(y)]], dtype=np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    result = float(value[0, 0])
    if not math.isfinite(result):
        raise ValueError("P14 scalar-field sample is non-finite")
    return result


def _p14_measure_contour_profiles(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    fields: dict[str, Any],
    sigma: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    """@luna-comment Measure contour-limited normal curves for P14 branch A.

    Input: Finite deblur luma, input text mask, local platform fields, kernel
      sigma, and profile geometry parameters.
    Output: Ordered accepted contour profiles with side-limited offsets and
      bounded monotone target deltas, plus explicit gap/coverage telemetry.
    Key Steps: Enumerate ordered contours, derive signed-distance normals, clip
      outside/inside sides independently at image/rival/medial limits, and
      isotonic-project only the resulting single contour interval.
    Key Params: Search radius is min(32,ceil(4*sigma+4)); contour interpolation
      tolerates only angle <30 degrees and gaps <=4 pixels.
    State/Dependencies: OpenCV remap/contours and numpy; no GT or writes.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or mask.dtype != np.bool_ or not np.isfinite(y_deblur).all():
        raise ValueError("P14 contour-profile inputs are invalid")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("P14 contour-profile sigma is invalid")
    background_field = np.asarray(fields.get("background_field"), dtype=np.float32)
    foreground_field = np.asarray(fields.get("foreground_field"), dtype=np.float32)
    field_support = np.asarray(fields.get("field_support"), dtype=np.float32)
    if background_field.shape != mask.shape or foreground_field.shape != mask.shape or field_support.shape != mask.shape:
        raise ValueError("P14 platform fields have incompatible shapes")
    min_support = float(fields.get("_min_support", params.get("p14_min_field_support", 1.0e-3)))
    default_radius = float(math.ceil(4.0 * sigma + 4.0))
    if bool(params.get("p16_mode", False)) or bool(params.get("p17_mode", False)):
        default_radius = max(default_radius, float(params.get("p16_background_radius", 24.0)))
    max_radius = min(32.0, default_radius)
    configured_max = float(params.get("p14_max_radius", max_radius))
    if not math.isfinite(configured_max) or configured_max <= 0.0:
        raise ValueError("P14 maximum profile radius is invalid")
    max_radius = min(max_radius, configured_max)
    step = float(params.get("p14_sample_step", params.get("sample_step", 0.5)))
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("P14 profile sample step is invalid")
    stride = max(1, int(params.get("p14_boundary_stride", 1)))
    max_profiles = max(1, int(params.get("p14_max_profiles", 4096)))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    signed = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) - cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_boundary = cv2.distanceTransform((cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) == 0).astype(np.uint8), cv2.DIST_L2, 5)
    grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    offsets = np.arange(-max_radius, max_radius + step * 0.5, step, dtype=np.float32)
    profiles: list[dict[str, Any]] = []
    rejected = 0
    gaps = 0
    reverse_before = 0
    reverse_after = 0
    boundary_total = 0
    for contour_id, contour in enumerate(contours):
        points = contour.reshape(-1, 2)
        if points.shape[0] < 2:
            continue
        for contour_order in range(0, points.shape[0], stride):
            column, row = (int(value) for value in points[contour_order])
            boundary_total += 1
            magnitude = float(norm[row, column])
            if magnitude <= 1.0e-5:
                rejected += 1
                gaps += 1
                continue
            normal_x = float(grad_x[row, column] / magnitude)
            normal_y = float(grad_y[row, column] / magnitude)
            map_x = float(column) + offsets * normal_x
            map_y = float(row) + offsets * normal_y
            valid_coord = (map_x >= 0.0) & (map_x <= y_deblur.shape[1] - 1.0) & (map_y >= 0.0) & (map_y <= y_deblur.shape[0] - 1.0)
            if int(np.sum(valid_coord)) < 4:
                rejected += 1
                gaps += 1
                continue
            values = cv2.remap(y_deblur, map_x.reshape(1, -1).astype(np.float32), map_y.reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1).astype(np.float32)
            mask_values = cv2.remap(mask.astype(np.float32), map_x.reshape(1, -1).astype(np.float32), map_y.reshape(1, -1).astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101).reshape(-1) >= 0.5
            distance_values = cv2.remap(distance_to_boundary.astype(np.float32), map_x.reshape(1, -1).astype(np.float32), map_y.reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1)
            outside = _p14_side_interval(offsets, mask_values, distance_values, "outside")
            inside = _p14_side_interval(offsets, mask_values, distance_values, "inside")
            if outside is None or inside is None:
                rejected += 1
                gaps += 1
                continue
            # The active interval must span both measured sides, from the
            # farthest valid outside sample through the contour to the
            # farthest valid inside sample.  The previous max(outside)/
            # min(inside) choice retained only the narrow overlap at the
            # contour and therefore almost never changed a pixel.
            start_index = min(outside[0], outside[1])
            end_index = max(inside[0], inside[1])
            if not start_index < end_index:
                rejected += 1
                gaps += 1
                continue
            b0 = _p14_sample_field(background_field, float(column), float(row))
            f0 = _p14_sample_field(foreground_field, float(column), float(row))
            support0 = _p14_sample_field(field_support, float(column), float(row))
            contrast = f0 - b0
            min_contrast = float(params.get("p14_min_profile_contrast", params.get("min_profile_contrast", 0.12)))
            if not math.isfinite(contrast) or abs(contrast) < min_contrast or support0 < min_support:
                rejected += 1
                gaps += 1
                continue
            active = np.arange(offsets.size) >= start_index
            active &= np.arange(offsets.size) <= end_index
            active &= valid_coord
            active_indices = np.flatnonzero(active)
            if active_indices.size < 3:
                rejected += 1
                gaps += 1
                continue
            oriented = ((values.astype(np.float64) - b0) / contrast).astype(np.float32)
            if not np.isfinite(oriented).all():
                rejected += 1
                gaps += 1
                continue
            raw_active = oriented[active]
            projected_active = np.clip(_p10_isotonic_nondecreasing(np.clip(raw_active, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
            target_values = values.copy()
            target_values[active] = (b0 + contrast * projected_active).astype(np.float32)
            delta = np.where(active, target_values - values, 0.0).astype(np.float32)
            raw_reverse = int(np.sum(np.diff(raw_active) < -1.0 / 255.0)) if raw_active.size > 1 else 0
            projected_reverse = int(np.sum(np.diff(projected_active) < -1.0 / 255.0)) if projected_active.size > 1 else 0
            reverse_before += raw_reverse
            reverse_after += projected_reverse
            # Tangent angle is used by the field stage to prevent interpolation
            # across corners and neighboring contours.
            prior = points[(contour_order - 1) % points.shape[0]].astype(np.float32)
            following = points[(contour_order + 1) % points.shape[0]].astype(np.float32)
            tangent = following - prior
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1.0e-6:
                tangent_x, tangent_y = 1.0, 0.0
            else:
                tangent_x, tangent_y = float(tangent[0] / tangent_norm), float(tangent[1] / tangent_norm)
            profiles.append({
                "contour_id": int(contour_id),
                "contour_order": int(contour_order),
                "row": row,
                "column": column,
                "normal_x": normal_x,
                "normal_y": normal_y,
                "tangent_x": tangent_x,
                "tangent_y": tangent_y,
                "background_y": b0,
                "foreground_y": f0,
                "contrast": contrast,
                "field_support": support0,
                "active_offset": [float(offsets[start_index]), float(offsets[end_index])],
                "active_width_px": float(offsets[end_index] - offsets[start_index]),
                "active_valid_sample_count": int(np.sum(active)) if "active" in locals() else 0,
                "active_raw_min": float(np.min(raw_active)),
                "active_raw_max": float(np.max(raw_active)),
                "active_raw_peak_count": int(np.sum(raw_active > 1.0 + 1.0 / 255.0)),
                "active_raw_valley_count": int(np.sum(raw_active < -1.0 / 255.0)),
                "raw_reverse_steps": raw_reverse,
                "projected_reverse_steps": projected_reverse,
                "offsets": [float(value) for value in offsets],
                "values": [float(value) for value in values],
                "projected_coordinates": [0.0 for _ in range(offsets.size)],
                "target_values": [float(value) for value in target_values],
                "_map_x": map_x.astype(np.float32),
                "_map_y": map_y.astype(np.float32),
                "_delta": delta,
                "_active": active,
            })
            if len(profiles) >= max_profiles:
                break
        if len(profiles) >= max_profiles:
            break
    # Replace the intentionally compact public placeholder with the true
    # projected coordinate array for each accepted profile.
    for profile in profiles:
        values = np.asarray(profile["values"], dtype=np.float32)
        b0 = float(profile["background_y"])
        contrast = float(profile["contrast"])
        active = np.asarray(profile["_active"], dtype=bool)
        projected = np.clip(_p10_isotonic_nondecreasing(np.clip((values[active] - b0) / contrast, 0.0, 1.0)), 0.0, 1.0)
        full = np.zeros_like(values, dtype=np.float32)
        full[active] = projected
        profile["projected_coordinates"] = [float(value) for value in full]
    return {
        "profiles": profiles,
        "profile_count": len(profiles),
        "profile_rejected_count": rejected,
        "profile_gap_count": gaps,
        "profile_geometry_status": "computed" if profiles else "no_valid_profiles",
        "profile_reverse_steps_before": reverse_before,
        "profile_reverse_steps_after": reverse_after,
        "profile_coverage_fraction": float(len(profiles) / max(boundary_total, 1)),
        "profile_gap_fraction": float(gaps / max(boundary_total, 1)),
        "profile_boundary_count": boundary_total,
        "profile_initial_radius": 0.0,
        "profile_max_radius": max_radius,
        "profile_interpolation_gap_limit": 4.0,
        "profile_interpolation_angle_limit": 30.0,
    }


def _p14_apply_curve_field(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    region_mask: np.ndarray,
    profiles: list[dict[str, Any]],
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """@luna-comment Apply the continuous P14 spatial-monotone correction field.

    Input: Finite deblur luma, text mask, contour-derived region mask, accepted
      P14 profiles, and field interpolation parameters.
    Output: Finite target luma, correction/support maps, and exact region/outside
      telemetry. Unsupported long contour gaps remain unchanged.
    Key Steps: Assign each pixel to a nearest contour profile, reject tangent
      mismatches or gaps over four pixels, interpolate its profile delta, and
      apply an independent endpoint fade so profile weights cannot cancel.
    Key Params: Same-contour angle limit is 30 degrees, gap limit is 4 pixels,
      and endpoint feather defaults to one pixel.
    State/Dependencies: OpenCV distance labels and numpy; no RGB state.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or region_mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P14 curve-field inputs are invalid")
    correction = np.zeros_like(y_deblur, dtype=np.float32)
    support = np.zeros_like(y_deblur, dtype=bool)
    if not profiles:
        return y_deblur.copy(), correction, {
            "p14_a_field_support_fraction": 0.0,
            "p14_a_field_region_fraction": float(np.mean(region_mask)),
            "p14_a_field_outside_max_abs_delta": 0.0,
            "p14_a_field_outside_sum_abs_delta": 0.0,
            "p14_a_interpolated_profile_count": 0,
            "p14_a_long_gap_pixels": 0,
            "p14_a_angle_rejected_count": 0,
            "_p14_a_support": support,
            "_p14_a_correction": correction,
        }
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    boundary_coordinates = np.argwhere(boundary)
    if boundary_coordinates.size == 0:
        return y_deblur.copy(), correction, {
            "p14_a_field_support_fraction": 0.0,
            "p14_a_field_region_fraction": float(np.mean(region_mask)),
            "p14_a_field_outside_max_abs_delta": 0.0,
            "p14_a_field_outside_sum_abs_delta": 0.0,
            "p14_a_interpolated_profile_count": 0,
            "p14_a_long_gap_pixels": 0,
            "p14_a_angle_rejected_count": 0,
            "_p14_a_support": support,
            "_p14_a_correction": correction,
        }
    # Labels from DIST_LABEL_PIXEL enumerate zero pixels (the contour) in row
    # major order. Mapping by boundary order keeps this deterministic and does
    # not average sparse profile deltas.
    _, labels = cv2.distanceTransformWithLabels((~boundary).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    signed = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) - cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    coord_to_profile: dict[tuple[int, int], dict[str, Any]] = {
        (int(profile["row"]), int(profile["column"])): profile for profile in profiles
    }
    profile_rows = np.asarray([int(profile["row"]) for profile in profiles], dtype=np.float32)
    profile_cols = np.asarray([int(profile["column"]) for profile in profiles], dtype=np.float32)
    angle_limit_cos = math.cos(math.radians(float(params.get("p14_interpolation_angle_deg", 30.0))))
    gap_limit = float(params.get("p14_interpolation_gap_px", 4.0))
    endpoint_feather = float(params.get("p14_endpoint_feather_px", 1.0))
    if not (math.isfinite(angle_limit_cos) and 0.0 < gap_limit <= 32.0 and math.isfinite(endpoint_feather) and endpoint_feather > 0.0):
        raise ValueError("P14 curve-field interpolation parameters are invalid")
    interpolated_count = 0
    long_gap_pixels = 0
    angle_rejected = 0
    yy, xx = np.nonzero(region_mask)
    for row, column in zip(yy.tolist(), xx.tolist()):
        label = int(labels[row, column])
        candidate: dict[str, Any] | None = None
        if 0 < label <= boundary_coordinates.shape[0]:
            boundary_row, boundary_column = (int(value) for value in boundary_coordinates[label - 1])
            candidate = coord_to_profile.get((boundary_row, boundary_column))
            if candidate is None:
                # P15 deliberately does not use the nearest profile from an
                # arbitrary contour.  A missing exact contour sample is a
                # gap, not permission to cross onto a neighboring stroke.
                long_gap_pixels += 1
        if candidate is None:
            continue
        profile = candidate
        normal_x = float(profile["normal_x"])
        normal_y = float(profile["normal_y"])
        local_nx = float(grad_x[boundary_row, boundary_column])
        local_ny = float(grad_y[boundary_row, boundary_column])
        local_norm = math.hypot(local_nx, local_ny)
        if local_norm > 1.0e-6:
            if abs((local_nx / local_norm) * normal_x + (local_ny / local_norm) * normal_y) < angle_limit_cos:
                angle_rejected += 1
                continue
        profile_row = float(profile["row"])
        profile_column = float(profile["column"])
        offset = (float(column) - profile_column) * normal_x + (float(row) - profile_row) * normal_y
        active_start, active_end = (float(value) for value in profile["active_offset"])
        if offset < active_start or offset > active_end:
            continue
        profile_offsets = np.asarray(profile["offsets"], dtype=np.float32)
        profile_delta = np.asarray(profile["_delta"], dtype=np.float32)
        value = float(np.interp(offset, profile_offsets, profile_delta))
        distance_to_start = max(0.0, offset - active_start)
        distance_to_end = max(0.0, active_end - offset)
        fade = min(1.0, distance_to_start / endpoint_feather, distance_to_end / endpoint_feather)
        fade = 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, fade))))
        correction[row, column] = float(value * fade)
        support[row, column] = fade > 0.0
    output = (y_deblur + correction).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("P14 curve-field output is non-finite")
    outside = ~region_mask
    return output, correction, {
        "p14_a_field_support_fraction": float(np.mean(support)),
        "p14_a_field_region_fraction": float(np.mean(region_mask)),
        "p14_a_field_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p14_a_field_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p14_a_interpolated_profile_count": interpolated_count,
        "p14_a_long_gap_pixels": long_gap_pixels,
        "p14_a_angle_rejected_count": angle_rejected,
        "_p14_a_support": support,
        "_p14_a_correction": correction,
    }


def _p14_apply_local_soft_lut(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    fields: dict[str, Any],
    sigma: float,
    alpha: float,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply P14's locally supported, smoothly feathered shoulders.

    Input: Finite deblur luma, text mask, local LQ platform fields, kernel sigma,
      shoulder strength alpha, and radius/support parameters.
    Output: Finite Y result plus correction, local L/H/tau maps, support mask,
      and compression/edge telemetry.
    Key Steps: Form local L/H from B/F and tau, apply the quadratic shoulder
      residual, multiply contour and region feathers, and leave far background
      exactly unchanged.
    Key Params: Full contour radius is max(3,ceil(3*sigma)); zero radius adds
      max(2,ceil(2*sigma)); s=max(2/255,tau) stabilizes tiny overshoots.
    State/Dependencies: OpenCV distance transforms and numpy only.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P14 soft-LUT inputs are invalid")
    sigma = float(sigma)
    alpha = float(alpha)
    if not math.isfinite(sigma) or sigma <= 0.0 or not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P14 soft-LUT parameters are invalid")
    background_field = np.asarray(fields.get("background_field"), dtype=np.float32)
    foreground_field = np.asarray(fields.get("foreground_field"), dtype=np.float32)
    tau_field = np.asarray(fields.get("tau_field"), dtype=np.float32)
    field_support = np.asarray(fields.get("field_support"), dtype=np.float32)
    region_mask = np.asarray(fields.get("region_mask"), dtype=bool)
    region_feather = np.asarray(fields.get("region_feather"), dtype=np.float32)
    if any(value.shape != y_deblur.shape for value in (background_field, foreground_field, tau_field, field_support, region_mask, region_feather)):
        raise ValueError("P14 soft-LUT fields have incompatible shapes")
    min_support = float(fields.get("_min_support", params.get("p14_min_field_support", 1.0e-3)))
    if not math.isfinite(min_support) or min_support < 0.0 or min_support > 1.0:
        raise ValueError("P14 soft-LUT support threshold is invalid")
    low_platform = np.minimum(background_field, foreground_field)
    high_platform = np.maximum(background_field, foreground_field)
    if bool(params.get("p15_ablation_legacy_tau", False)):
        tau = np.maximum(2.0 / 255.0, tau_field).astype(np.float32)
    else:
        tau = np.clip(tau_field, 2.0 / 255.0, 6.0 / 255.0).astype(np.float32)
    low = low_platform - tau
    high = high_platform + tau
    # The shoulder stabilizer is independent of the local tolerance.  Using
    # tau here made broad-uncertainty regions almost identity transforms.
    s = 2.0 / 255.0
    d_low = np.maximum(low - y_deblur, 0.0)
    d_high = np.maximum(y_deblur - high, 0.0)
    delta = alpha * (np.square(d_low) / (d_low + s) - np.square(d_high) / (d_high + s))
    delta = np.where(field_support >= min_support, delta, 0.0).astype(np.float32)
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    # The two distance transforms are defined on opposite sides of the
    # contour.  Taking their minimum everywhere collapses the distance to
    # zero on both sides and makes the feather behave like a thin halo.  P15
    # selects the distance belonging to the pixel's actual class.
    contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)
    r_full = max(4.0, float(math.ceil(4.0 * sigma)))
    r_zero = r_full + max(2.0, float(math.ceil(2.0 * sigma)))
    contour_weight = np.ones_like(y_deblur, dtype=np.float32)
    mid = (contour_distance > r_full) & (contour_distance < r_zero)
    contour_weight[contour_distance >= r_zero] = 0.0
    if bool(mid.any()):
        ratio = (contour_distance[mid] - r_full) / max(r_zero - r_full, 1.0e-8)
        contour_weight[mid] = (0.5 * (1.0 + np.cos(np.pi * ratio))).astype(np.float32)
    # Support is a validity gate only.  Multiplying the correction by local
    # density was the main reason P14 became visually indistinguishable from
    # deblur in sparse regions.
    support_gate = field_support >= min_support
    weight = contour_weight * region_feather
    weight = np.where(region_mask & support_gate, weight, 0.0).astype(np.float32)
    correction = (weight * delta).astype(np.float32)
    output = (y_deblur + correction).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("P14 soft-LUT output is non-finite")
    outside = ~region_mask
    dark = (d_low > 0.0) & (weight > 0.0)
    bright = (d_high > 0.0) & (weight > 0.0)
    return output, {
        "p14_b_status": "computed",
        "p14_b_alpha": alpha,
        "p14_b_r_full": r_full,
        "p14_b_r_zero": r_zero,
        "p14_b_roi_fraction": float(np.mean(region_mask)),
        "p14_b_weight_coverage_fraction": float(np.mean(weight > 0.0)),
        "p14_b_supported_fraction": float(np.mean(field_support >= min_support)),
        "p14_b_dark_compression_pixels": int(np.sum(dark)),
        "p14_b_bright_compression_pixels": int(np.sum(bright)),
        "p14_b_dark_compression_sum_abs": float(np.sum(np.abs(correction[dark]))),
        "p14_b_bright_compression_sum_abs": float(np.sum(np.abs(correction[bright]))),
        "p14_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p14_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)) if correction.size else 0.0,
        "p14_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p14_b_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p14_b_max_abs_delta": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "p15_b_tau_mode": "legacy_uncapped" if bool(params.get("p15_ablation_legacy_tau", False)) else "clipped_2_6_per_255",
        "p15_b_support_weight_mode": "gate_only",
        "_p14_b_correction": correction,
        "_p14_b_low": low,
        "_p14_b_high": high,
        "_p14_b_tau": tau,
        "_p14_b_weight": weight,
        "_p14_b_delta_raw": delta,
    }


def _p16_apply_patch_shoulder_lut(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply the legacy P13 shoulder LUT using overlapping local patches.

    Input: Deblurred luma, text mask, shoulder strength, and P16 patch limits.
    Output: Luma result and JSON-safe patch/borrow/fusion telemetry.
    Key Steps: Estimate robust local background/foreground medians in 50x50
      patches, borrow missing platform scalars from the nearest valid patch,
      apply the exact P13 LUT per patch, and overlap-add its residual with a
      two-dimensional sin-squared window. Only the 16/24 px contour band writes.
    Key Params: Patch stride defaults to 25 px; background distance is 12 px;
      alpha .5/1.0 is the unchanged P13 LUT strength.
    State/Dependencies: Numpy/OpenCV only; no GT, RGB chroma, or quality gate.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P16 patch LUT inputs are invalid")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P16 patch LUT alpha is invalid")
    height, width = y_deblur.shape
    patch_size = max(3, int(params.get("p16_patch_size", 50)))
    stride = max(1, int(params.get("p16_patch_stride", 25)))
    bg_distance = max(1.0, float(params.get("p16_background_distance", 12.0)))
    fg_distance = max(0.0, float(params.get("p16_foreground_distance", 2.0)))
    full_radius = max(1.0, float(params.get("p16_write_full_radius", 16.0)))
    zero_radius = max(full_radius + 1.0, float(params.get("p16_write_zero_radius", 24.0)))
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)
    starts_y = sorted(set(list(range(0, max(height - patch_size, 0) + 1, stride)) + [max(height - patch_size, 0)]))
    starts_x = sorted(set(list(range(0, max(width - patch_size, 0) + 1, stride)) + [max(width - patch_size, 0)]))
    valid: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for top in starts_y:
        bottom = min(height, top + patch_size)
        for left in starts_x:
            right = min(width, left + patch_size)
            patch_mask = mask[top:bottom, left:right]
            patch_y = y_deblur[top:bottom, left:right]
            patch_inside = inside_distance[top:bottom, left:right]
            patch_outside = outside_distance[top:bottom, left:right]
            bg_values = patch_y[(~patch_mask) & (patch_outside >= bg_distance)]
            fg_values = patch_y[patch_mask & (patch_inside >= fg_distance)]
            item = {
                "top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right),
                "center_y": float((top + bottom - 1) * 0.5), "center_x": float((left + right - 1) * 0.5),
                "background_count": int(bg_values.size), "foreground_count": int(fg_values.size),
                "background_y": float(np.median(bg_values)) if bg_values.size else None,
                "foreground_y": float(np.median(fg_values)) if fg_values.size else None,
                "background_borrowed": False, "foreground_borrowed": False,
                "background_borrow_distance": 0.0, "foreground_borrow_distance": 0.0,
            }
            if bg_values.size and fg_values.size and abs(float(np.median(fg_values)) - float(np.median(bg_values))) > 1.0e-6:
                valid.append(item)
            records.append(item)
    for item in records:
        for field, count_field, borrowed_field, distance_field in (
            ("background_y", "background_count", "background_borrowed", "background_borrow_distance"),
            ("foreground_y", "foreground_count", "foreground_borrowed", "foreground_borrow_distance"),
        ):
            if item[field] is not None:
                continue
            if not valid:
                continue
            nearest = min(valid, key=lambda other: (other["center_y"] - item["center_y"]) ** 2 + (other["center_x"] - item["center_x"]) ** 2)
            item[field] = nearest[field]
            item[borrowed_field] = True
            item[distance_field] = float(math.sqrt((nearest["center_y"] - item["center_y"]) ** 2 + (nearest["center_x"] - item["center_x"]) ** 2))
        if item["background_y"] is None or item["foreground_y"] is None:
            continue
        if abs(float(item["foreground_y"]) - float(item["background_y"])) <= 1.0e-6:
            continue
        item["usable"] = True
    residual_sum = np.zeros_like(y_deblur, dtype=np.float32)
    weight_sum = np.zeros_like(y_deblur, dtype=np.float32)
    bright = np.zeros_like(y_deblur, dtype=bool)
    dark = np.zeros_like(y_deblur, dtype=bool)
    for item in records:
        if not item.get("usable", False):
            continue
        top, left, bottom, right = (int(item[key]) for key in ("top", "left", "bottom", "right"))
        patch = y_deblur[top:bottom, left:right]
        low = min(float(item["background_y"]), float(item["foreground_y"]))
        high = max(float(item["background_y"]), float(item["foreground_y"]))
        target = patch.copy()
        below = patch < low
        above = patch > high
        target[below] = low + (1.0 - alpha) * (patch[below] - low)
        target[above] = high + (1.0 - alpha) * (patch[above] - high)
        ph, pw = patch.shape
        wy = np.sin(np.pi * (np.arange(ph, dtype=np.float32) + 0.5) / max(float(ph), 1.0)) ** 2
        wx = np.sin(np.pi * (np.arange(pw, dtype=np.float32) + 0.5) / max(float(pw), 1.0)) ** 2
        window = wy[:, None] * wx[None, :]
        residual_sum[top:bottom, left:right] += (target - patch) * window
        weight_sum[top:bottom, left:right] += window
        dark[top:bottom, left:right] |= below
        bright[top:bottom, left:right] |= above
    fused = np.divide(residual_sum, np.maximum(weight_sum, 1.0e-8), out=np.zeros_like(y_deblur), where=weight_sum > 0.0)
    write_weight = np.ones_like(y_deblur, dtype=np.float32)
    mid = (contour_distance > full_radius) & (contour_distance < zero_radius)
    write_weight[contour_distance >= zero_radius] = 0.0
    if bool(mid.any()):
        ratio = (contour_distance[mid] - full_radius) / max(zero_radius - full_radius, 1.0e-8)
        write_weight[mid] = (0.5 * (1.0 + np.cos(np.pi * ratio))).astype(np.float32)
    correction = (fused * write_weight).astype(np.float32)
    output = (y_deblur + correction).astype(np.float32)
    outside = write_weight <= 0.0
    if not np.isfinite(output).all():
        raise ValueError("P16 patch LUT output is non-finite")
    return output, {
        "p16_b_status": "computed" if valid else "skipped_no_valid_patch",
        "p16_b_alpha": alpha,
        "p16_b_patch_size": patch_size,
        "p16_b_patch_stride": stride,
        "p16_b_patch_count": len(records),
        "p16_b_valid_patch_count": len(valid),
        "p16_b_background_borrow_count": int(sum(bool(item["background_borrowed"]) for item in records if item.get("usable", False))),
        "p16_b_foreground_borrow_count": int(sum(bool(item["foreground_borrowed"]) for item in records if item.get("usable", False))),
        "p16_b_write_full_radius": full_radius,
        "p16_b_write_zero_radius": zero_radius,
        "p16_b_write_fraction": float(np.mean(write_weight > 0.0)),
        "p16_b_weight_fraction": float(np.mean(weight_sum > 0.0)),
        "p16_b_dark_compression_pixels": int(np.sum(dark & (write_weight > 0.0))),
        "p16_b_bright_compression_pixels": int(np.sum(bright & (write_weight > 0.0))),
        "p16_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p16_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)),
        "p16_b_max_abs_delta": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "p16_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p16_b_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "_p16_b_correction": correction,
        "_p16_b_write_weight": write_weight,
        "_p16_b_weight_sum": weight_sum,
        "_p16_b_patch_records": records,
        "_p16_b_patch_grid": np.asarray([[float(item.get("background_y") or 0.0), float(item.get("foreground_y") or 0.0)] for item in records], dtype=np.float32),
    }


def _p17_robust_otsu_platforms(
    patch_y: np.ndarray, foreground_is_high: bool, params: dict[str, Any]
) -> tuple[float | None, float | None, dict[str, Any]]:
    """@luna-comment Estimate P17 platforms from Otsu classes without seed masks.

    Input: One finite deblurred-Y patch, established global polarity, and P17 bounds.
    Output: Low/high robust class centres or an explicit invalid-distribution reason.
    Key Steps: Blur only for histogram statistics, split with Otsu, robustly centre
      both classes, and reject weak occupancy or class separation.
    Key Params: Both Otsu classes need 12 percent occupancy; centres use p35--p65.
    State/Dependencies: Does not inspect a text mask or distance field.
    """
    sigma = max(0.0, float(params.get("p17_stats_gaussian_sigma", 0.6)))
    stats_y = cv2.GaussianBlur(patch_y.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0.0 else patch_y.astype(np.float32)
    encoded = np.clip(np.rint(stats_y * 255.0), 0, 255).astype(np.uint8)
    threshold, classes = cv2.threshold(encoded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    high_class = classes > 0
    low_values = stats_y[~high_class]
    high_values = stats_y[high_class]
    fraction_low = float(low_values.size / max(stats_y.size, 1))
    fraction_high = float(high_values.size / max(stats_y.size, 1))
    record: dict[str, Any] = {
        "otsu_threshold_y": float(threshold / 255.0),
        "otsu_low_fraction": fraction_low,
        "otsu_high_fraction": fraction_high,
        "global_foreground_is_high": bool(foreground_is_high),
        "validity_reason": None,
    }
    min_fraction = float(params.get("p17_min_class_fraction", 0.12))
    if min(fraction_low, fraction_high) < min_fraction:
        record["validity_reason"] = "class_fraction_below_minimum"
        return None, None, record

    def robust_center(values: np.ndarray) -> float:
        lo, hi = np.percentile(values, [35.0, 65.0])
        return float(np.median(values[(values >= lo) & (values <= hi)]))

    low_center = robust_center(low_values)
    high_center = robust_center(high_values)
    dynamic_range = float(np.max(stats_y) - np.min(stats_y))
    separation = abs(high_center - low_center)
    minimum_separation = max(12.0 / 255.0, 0.08 * dynamic_range)
    record.update({
        "otsu_low_center_y": low_center,
        "otsu_high_center_y": high_center,
        "otsu_dynamic_range": dynamic_range,
        "otsu_separation": separation,
        "otsu_minimum_separation": minimum_separation,
        "foreground_y": high_center if foreground_is_high else low_center,
        "background_y": low_center if foreground_is_high else high_center,
    })
    if separation < minimum_separation:
        record["validity_reason"] = "class_separation_below_minimum"
        return None, None, record
    return min(low_center, high_center), max(low_center, high_center), record


def _p17_apply_regional_otsu_lut(
    y_deblur: np.ndarray, mask: np.ndarray, alpha: float, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply P16's shoulder LUT with P17 regional Otsu platforms.

    Input: Deblurred Y, text mask used only for global polarity/write band, alpha, and P17 configuration.
    Output: Y result plus per-patch Otsu telemetry and correction maps.
    Key Steps: Compute independent regional Otsu platforms, Hann overlap-add residuals,
      and gate the fused correction through P16's 16--24 pixel contour band.
    Key Params: 96/32 and 128/48 statistic domains are fixed candidate variants.
    State/Dependencies: No mask or distance values participate in patch platform statistics.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P17 Otsu LUT inputs are invalid")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P17 Otsu LUT alpha is invalid")
    height, width = y_deblur.shape
    patch_size = max(3, int(params.get("p17_patch_size", 96)))
    stride = max(1, int(params.get("p17_patch_stride", 32)))
    full_radius = max(1.0, float(params.get("p16_write_full_radius", 16.0)))
    zero_radius = max(full_radius + 1.0, float(params.get("p16_write_zero_radius", 24.0)))
    foreground_is_high = bool(np.median(y_deblur[mask]) >= np.median(y_deblur[~mask])) if bool(mask.any()) and bool((~mask).any()) else True
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)
    starts_y = sorted(set(list(range(0, max(height - patch_size, 0) + 1, stride)) + [max(height - patch_size, 0)]))
    starts_x = sorted(set(list(range(0, max(width - patch_size, 0) + 1, stride)) + [max(width - patch_size, 0)]))
    residual_sum = np.zeros_like(y_deblur, dtype=np.float32)
    weight_sum = np.zeros_like(y_deblur, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for top in starts_y:
        bottom = min(height, top + patch_size)
        for left in starts_x:
            right = min(width, left + patch_size)
            patch = y_deblur[top:bottom, left:right]
            low, high, record = _p17_robust_otsu_platforms(patch, foreground_is_high, params)
            record.update({"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right), "usable": low is not None and high is not None})
            records.append(record)
            if low is None or high is None:
                continue
            target = patch.copy()
            below, above = patch < low, patch > high
            target[below] = low + (1.0 - alpha) * (patch[below] - low)
            target[above] = high + (1.0 - alpha) * (patch[above] - high)
            ph, pw = patch.shape
            wy = np.sin(np.pi * (np.arange(ph, dtype=np.float32) + 0.5) / float(ph)) ** 2
            wx = np.sin(np.pi * (np.arange(pw, dtype=np.float32) + 0.5) / float(pw)) ** 2
            window = wy[:, None] * wx[None, :]
            residual_sum[top:bottom, left:right] += (target - patch) * window
            weight_sum[top:bottom, left:right] += window
    fused = np.divide(residual_sum, np.maximum(weight_sum, 1.0e-8), out=np.zeros_like(y_deblur), where=weight_sum > 0.0)
    write_weight = np.ones_like(y_deblur, dtype=np.float32)
    middle = (contour_distance > full_radius) & (contour_distance < zero_radius)
    write_weight[contour_distance >= zero_radius] = 0.0
    if bool(middle.any()):
        ratio = (contour_distance[middle] - full_radius) / max(zero_radius - full_radius, 1.0e-8)
        write_weight[middle] = (0.5 * (1.0 + np.cos(np.pi * ratio))).astype(np.float32)
    correction = (fused * write_weight).astype(np.float32)
    output = (y_deblur + correction).astype(np.float32)
    outside = write_weight <= 0.0
    if not np.isfinite(output).all():
        raise ValueError("P17 Otsu LUT output is non-finite")
    valid = [record for record in records if record["usable"]]
    invalid_reasons: dict[str, int] = {}
    for record in records:
        reason = record.get("validity_reason")
        if reason:
            invalid_reasons[str(reason)] = invalid_reasons.get(str(reason), 0) + 1
    return output, {
        "p17_b_status": "computed" if valid else "skipped_no_valid_patch",
        "p17_b_alpha": alpha, "p17_b_patch_size": patch_size, "p17_b_patch_stride": stride,
        "p17_b_patch_count": len(records), "p17_b_valid_patch_count": len(valid),
        "p17_b_invalid_reason_counts": invalid_reasons,
        "p17_b_global_foreground_is_high": foreground_is_high,
        "p17_b_write_full_radius": full_radius, "p17_b_write_zero_radius": zero_radius,
        "p17_b_write_fraction": float(np.mean(write_weight > 0.0)),
        "p17_b_weight_fraction": float(np.mean(weight_sum > 0.0)),
        "p17_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p17_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)),
        "p17_b_max_abs_delta": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "p17_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p17_b_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "_p17_b_correction": correction, "_p17_b_write_weight": write_weight,
        "_p17_b_weight_sum": weight_sum, "_p17_b_patch_records": records,
    }


def _p18_apply_anchored_excess(
    y_deblur: np.ndarray, mask: np.ndarray, alpha: float, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Correct only side-band excess relative to continuous anchors.

    Input: Finite deblurred Y, a text mask used only for polarity/components and
      contour anchoring, alpha, and P18 parameters.
    Output: Finite corrected Y plus Otsu, anchor, write-mask, and continuity telemetry.
    Key Steps: Reuse regional Otsu ranges, interpolate exterior 24--48 px anchors,
      derive isolated adaptive component foreground anchors, and correct only
      sign-appropriate excess beyond a dead zone with a bounded tanh response.
    Key Params: P18 keeps P17's 96/32 Otsu domain; A uses 16--24 and B 10--28 taper.
    State/Dependencies: No cross-component foreground borrowing and no output fallback.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P18 anchored-excess inputs are invalid")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P18 anchored-excess alpha is invalid")
    height, width = y_deblur.shape
    patch_size = max(3, int(params.get("p17_patch_size", 96)))
    stride = max(1, int(params.get("p17_patch_stride", 32)))
    full_radius = max(1.0, float(params.get("p18_write_full_radius", 16.0)))
    zero_radius = max(full_radius + 1.0, float(params.get("p18_write_zero_radius", 24.0)))
    exterior_near = max(zero_radius, float(params.get("p18_exterior_anchor_near", 24.0)))
    exterior_far = max(exterior_near + 1.0, float(params.get("p18_exterior_anchor_far", 48.0)))
    foreground_is_high = bool(np.median(y_deblur[mask]) >= np.median(y_deblur[~mask])) if bool(mask.any()) and bool((~mask).any()) else True
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)

    # The Otsu field supplies the local two-class separation used by the P18 dead zone.
    starts_y = sorted(set(list(range(0, max(height - patch_size, 0) + 1, stride)) + [max(height - patch_size, 0)]))
    starts_x = sorted(set(list(range(0, max(width - patch_size, 0) + 1, stride)) + [max(width - patch_size, 0)]))
    range_sum = np.zeros_like(y_deblur, dtype=np.float32)
    range_weight = np.zeros_like(y_deblur, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for top in starts_y:
        bottom = min(height, top + patch_size)
        for left in starts_x:
            right = min(width, left + patch_size)
            patch = y_deblur[top:bottom, left:right]
            low, high, record = _p17_robust_otsu_platforms(patch, foreground_is_high, params)
            usable = low is not None and high is not None
            record.update({"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right), "usable": usable})
            records.append(record)
            if not usable:
                continue
            ph, pw = patch.shape
            wy = np.sin(np.pi * (np.arange(ph, dtype=np.float32) + .5) / float(ph)) ** 2
            wx = np.sin(np.pi * (np.arange(pw, dtype=np.float32) + .5) / float(pw)) ** 2
            window = wy[:, None] * wx[None, :]
            range_sum[top:bottom, left:right] += float(high - low) * window
            range_weight[top:bottom, left:right] += window
    range_field = np.divide(range_sum, np.maximum(range_weight, 1.0e-8), out=np.zeros_like(y_deblur), where=range_weight > 0.0)

    # Normalized Gaussian interpolation makes B(x) continuous while being based solely
    # on the same-side exterior annulus.  Empty support is a skip, never a fabricated value.
    exterior_anchor = (~mask) & (outside_distance >= exterior_near) & (outside_distance <= exterior_far)
    bg_seed = np.where(exterior_anchor, y_deblur, 0.0).astype(np.float32)
    bg_support = exterior_anchor.astype(np.float32)
    bg_sigma = max(1.0, float(params.get("p18_background_reference_sigma", 12.0)))
    bg_num = cv2.GaussianBlur(bg_seed, (0, 0), bg_sigma, borderType=cv2.BORDER_REFLECT)
    bg_den = cv2.GaussianBlur(bg_support, (0, 0), bg_sigma, borderType=cv2.BORDER_REFLECT)
    background_field = np.divide(bg_num, np.maximum(bg_den, 1.0e-8), out=np.zeros_like(y_deblur), where=bg_den > 1.0e-5)
    background_valid = bg_den > float(params.get("p18_min_anchor_support", 1.0e-4))

    labels_count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    foreground_field = np.zeros_like(y_deblur, dtype=np.float32)
    foreground_valid = np.zeros_like(mask, dtype=bool)
    component_records: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        component = labels == label
        component_max = float(np.max(inside_distance[component])) if bool(component.any()) else 0.0
        if component_max < 3.0:
            component_records.append({"label": label, "max_inside_distance": component_max, "usable": False, "reason": "thin_component_exterior_only"})
            continue
        core_distance = min(6.0, .5 * component_max)
        core = component & (inside_distance >= core_distance)
        if not bool(core.any()):
            component_records.append({"label": label, "max_inside_distance": component_max, "usable": False, "reason": "no_adaptive_core"})
            continue
        values = y_deblur[core]
        lo, hi = np.percentile(values, [35.0, 65.0])
        value = float(np.median(values[(values >= lo) & (values <= hi)]))
        foreground_field[component] = value
        foreground_valid[component] = True
        component_records.append({"label": label, "max_inside_distance": component_max, "core_distance": core_distance, "foreground_y": value, "usable": True})

    dead_zone = np.maximum(2.0 / 255.0, .10 * range_field).astype(np.float32)
    cap = max(1.0e-6, float(params.get("p18_tanh_cap", .12)))
    exterior = ~mask
    correction_raw = np.zeros_like(y_deblur, dtype=np.float32)
    if foreground_is_high:
        dark_excess = np.maximum(background_field - y_deblur - dead_zone, 0.0)
        bright_excess = np.maximum(y_deblur - foreground_field - dead_zone, 0.0)
        correction_raw[exterior & background_valid] = cap * np.tanh(dark_excess[exterior & background_valid] / cap)
        interior_ok = mask & foreground_valid
        correction_raw[interior_ok] = -cap * np.tanh(bright_excess[interior_ok] / cap)
    else:
        bright_excess = np.maximum(y_deblur - background_field - dead_zone, 0.0)
        dark_excess = np.maximum(foreground_field - y_deblur - dead_zone, 0.0)
        correction_raw[exterior & background_valid] = -cap * np.tanh(bright_excess[exterior & background_valid] / cap)
        interior_ok = mask & foreground_valid
        correction_raw[interior_ok] = cap * np.tanh(dark_excess[interior_ok] / cap)

    write_weight = np.ones_like(y_deblur, dtype=np.float32)
    middle = (contour_distance > full_radius) & (contour_distance < zero_radius)
    write_weight[contour_distance >= zero_radius] = 0.0
    if bool(middle.any()):
        ratio = (contour_distance[middle] - full_radius) / max(zero_radius - full_radius, 1.0e-8)
        write_weight[middle] = (.5 * (1.0 + np.cos(np.pi * ratio))).astype(np.float32)
    correction = (alpha * correction_raw * write_weight).astype(np.float32)
    output = np.clip(y_deblur + correction, 0.0, 1.0).astype(np.float32)
    outside_write = write_weight <= 0.0
    bins: dict[str, float] = {}
    for low, high in ((0., 4.), (4., 10.), (10., 16.), (16., zero_radius)):
        selection = (contour_distance >= low) & (contour_distance < high)
        bins[f"{int(low)}_{int(high)}"] = float(np.mean(np.abs(correction[selection]))) if bool(selection.any()) else 0.0
    status = "computed" if bool(exterior_anchor.any()) and bool(range_weight.any()) else "skipped_missing_exterior_anchor"
    if status != "computed":
        correction.fill(0.0)
        output = y_deblur.copy()
    if not np.isfinite(output).all():
        raise ValueError("P18 anchored-excess output is non-finite")
    return output, {
        "p18_b_status": status, "p18_b_alpha": alpha,
        "p18_b_patch_size": patch_size, "p18_b_patch_stride": stride,
        "p18_b_patch_count": len(records), "p18_b_valid_patch_count": int(sum(item["usable"] for item in records)),
        "p18_b_global_foreground_is_high": foreground_is_high,
        "p18_b_write_full_radius": full_radius, "p18_b_write_zero_radius": zero_radius,
        "p18_b_exterior_anchor_near": exterior_near, "p18_b_exterior_anchor_far": exterior_far,
        "p18_b_exterior_anchor_fraction": float(np.mean(exterior_anchor)),
        "p18_b_background_reference_coverage": float(np.mean(background_valid)),
        "p18_b_foreground_reference_coverage": float(np.mean(foreground_valid)),
        "p18_b_distance_bin_mean_abs_correction": bins,
        "p18_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p18_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)),
        "p18_b_max_abs_delta": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "p18_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside_write]))) if bool(outside_write.any()) else 0.0,
        "_p18_b_correction": correction, "_p18_b_write_weight": write_weight,
        "_p18_b_background_field": background_field, "_p18_b_foreground_field": foreground_field,
        "_p18_b_background_anchor": exterior_anchor, "_p18_b_foreground_anchor": foreground_valid,
        "_p18_b_anchor_coverage": (background_valid | foreground_valid),
        "_p18_b_range_field": range_field, "_p18_b_patch_records": records,
        "_p18_b_component_records": component_records,
        "_p18_b_continuity_profile": {"background_field_mean": float(np.mean(background_field[background_valid])) if bool(background_valid.any()) else None, "write_band_mean_abs": bins},
    }


def _p14_kernel_ab(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Run P14 local-kernel deblur and continuous A/B branches.

    Input: Authoritative finite HxWx3 RGB tensor and one frozen P14 candidate.
    Output: First finite A output plus both lambda deblur controls, continuous
      spatial-monotone A outputs, local soft-shoulder B outputs, and telemetry.
    Key Steps: Reuse P11 segmentation/kernel/Wiener, estimate LQ platform fields,
      measure side-clipped contours, and export both post-deblur branches.
    Key Params: Lambda defaults [.01,.03], B alpha defaults [.5,1.0], and all
      contour/field limits are candidate-configured; no quality fallback exists.
    State/Dependencies: CPU numpy/OpenCV only; GT and chroma planes are unused.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("invalid P14 RGB input")
    y_input = _rgb_to_luma(rgb)
    mask, segmentation = _p10_segment_text(y_input, params)
    empty_mask = np.zeros_like(y_input, dtype=bool)
    empty_weight = np.zeros_like(y_input, dtype=np.float32)
    base_meta: dict[str, Any] = {
        "fallback": False,
        "fallback_reason": None,
        "p14_status": "kernel_unidentifiable",
        "p14_kernel_status": "kernel_unidentifiable",
        "p14_kernel_reason": None,
        "p14_kernel_sigma": None,
        "p14_kernel_sigma_mad": None,
        "p14_kernel_segment_count": 0,
        "p14_kernel_profiles_considered": 0,
        "p14_field_status": "skipped_no_mask",
        "p14_field_reason": None,
        "p14_field_sigma": None,
        "p14_field_seed_distance": None,
        "p14_field_background_y": None,
        "p14_field_foreground_y": None,
        "p14_field_uncertainty": None,
        "p14_field_support_fraction": 0.0,
        "p14_field_region_fraction": 0.0,
        "p14_a_status": "skipped_no_mask",
        "p14_a_profile_count": 0,
        "p14_a_rejected_count": 0,
        "p14_a_gap_count": 0,
        "p14_a_coverage_fraction": 0.0,
        "p14_a_gap_fraction": 1.0,
        "p14_a_support_fraction": 0.0,
        "p14_a_reverse_steps_before": 0,
        "p14_a_reverse_steps_after": 0,
        "p14_b_status": "skipped_no_field",
        "p14_b_roi_fraction": 0.0,
        "p14_stage_count": 0,
        "p14_stage_telemetry": [],
        "_p14_text_mask": empty_mask,
        "_p14_region_mask": empty_mask,
        "_p14_fields": {},
        "_p14_stage_outputs": [],
    }
    if mask is None:
        base_meta["p14_kernel_reason"] = segmentation.get("segmentation_reason", "no_text_mask")
        base_meta.update(segmentation)
        return rgb.copy(), base_meta
    base_meta["_p14_text_mask"] = mask
    kernel = _p11_estimate_equivalent_kernel(y_input, mask, params)
    sigma = kernel.get("kernel_sigma")
    base_meta.update({
        "p14_kernel_status": str(kernel.get("kernel_status", "kernel_unidentifiable")),
        "p14_kernel_reason": kernel.get("kernel_reason"),
        "p14_kernel_sigma": sigma,
        "p14_kernel_sigma_mad": kernel.get("kernel_sigma_mad"),
        "p14_kernel_segment_count": int(kernel.get("kernel_segment_count", 0)),
        "p14_kernel_profiles_considered": int(kernel.get("kernel_profiles_considered", 0)),
        "p14_kernel_direction_stats": kernel.get("kernel_direction_stats", {}),
        "p14_kernel_fit_residual_stats": kernel.get("kernel_fit_residual_stats", {}),
        "p14_kernel_bound_hits": kernel.get("kernel_bound_hits", {}),
        "_p14_kernel_psf": kernel.get("kernel_psf"),
    })
    if sigma is None:
        base_meta.update(segmentation)
        return rgb.copy(), base_meta
    sigma = float(sigma)
    fields = _p14_estimate_local_platform_fields(y_input, mask, sigma, params)
    if bool(params.get("p16_mode", False)):
        inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)
        p16_region = contour_distance <= float(params.get("p16_a_write_radius", 24.0))
        fields["region_mask"] = p16_region
        fields["region_feather"] = np.ones_like(y_input, dtype=np.float32)
    base_meta.update({
        "p14_field_status": fields.get("status"),
        "p14_field_reason": fields.get("reason"),
        "p14_field_sigma": fields.get("field_sigma"),
        "p14_field_seed_distance": fields.get("seed_distance"),
        "p14_field_background_y": fields.get("background_y"),
        "p14_field_foreground_y": fields.get("foreground_y"),
        "p14_field_uncertainty": fields.get("uncertainty"),
        "p14_field_support_fraction": fields.get("field_support_fraction", 0.0),
        "p14_field_region_fraction": fields.get("region_fraction", 0.0),
        "_p14_fields": fields,
        "_p14_region_mask": fields.get("region_mask", empty_mask),
    })
    deblur_weight, _ = _p11_processing_soft_band(mask, sigma)
    base_meta["_p14_processing_weight"] = deblur_weight
    lambda_candidates = params.get("lambda_candidates", [params.get("lambda", 0.01), 0.03])
    alpha_candidates = params.get("alpha_candidates", [0.5, 1.0])
    if not isinstance(lambda_candidates, (list, tuple)) or not isinstance(alpha_candidates, (list, tuple)):
        raise ValueError("P14 lambda_candidates and alpha_candidates must be lists")
    lambdas = [float(value) for value in lambda_candidates]
    alphas = [float(value) for value in alpha_candidates]
    if not lambdas or not alphas or any(not math.isfinite(value) or value <= 0.0 for value in lambdas) or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("P14 lambda/alpha candidate values are invalid")
    stage_outputs: list[dict[str, Any]] = []
    for regularization in lambdas:
        raw_deblur_y, wiener_meta = _p11_wiener_hunt_deblur(y_input, mask, sigma, regularization, params)
        deblur_y = (y_input + deblur_weight * (raw_deblur_y - y_input)).astype(np.float32)
        if not np.isfinite(deblur_y).all():
            raise ValueError("P14 deblur output is non-finite")
        deblur_rgb, deblur_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, deblur_y)
        profile_meta = _p14_measure_contour_profiles(deblur_y, mask, fields, sigma, params)
        profiles = profile_meta.get("profiles", [])
        a_y, a_correction, field_meta = _p14_apply_curve_field(
            deblur_y, mask, fields.get("region_mask", empty_mask), profiles, params
        )
        profile_meta.update(field_meta)
        profile_meta["profile_support_fraction"] = float(field_meta.get("p14_a_field_support_fraction", 0.0))
        a_y = np.clip(a_y, 0.0, 1.0).astype(np.float32)
        a_rgb, a_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, a_y)
        b_stages: list[dict[str, Any]] = []
        b_ablation_stages: list[dict[str, Any]] = []
        for alpha in alphas:
            if bool(params.get("p18_mode", False)):
                b_y, b_meta = _p18_apply_anchored_excess(deblur_y, mask, alpha, params)
            elif bool(params.get("p17_mode", False)):
                b_y, b_meta = _p17_apply_regional_otsu_lut(deblur_y, mask, alpha, params)
            elif bool(params.get("p16_mode", False)):
                b_y, b_meta = _p16_apply_patch_shoulder_lut(deblur_y, mask, alpha, params)
            else:
                b_y, b_meta = _p14_apply_local_soft_lut(deblur_y, mask, fields, sigma, alpha, params)
            b_rgb, b_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, b_y)
            b_stages.append({
                "alpha": alpha,
                "status": b_meta.get("p18_b_status", b_meta.get("p17_b_status", b_meta.get("p16_b_status", b_meta.get("p14_b_status", "computed")))),
                "output": b_rgb,
                "y": b_y,
                "rgb_meta": b_rgb_meta,
                "meta": b_meta,
            })
            # Fixed 112_v2 ablation: keep the pre-P15 uncapped tau but retain
            # the P15 support-gate-only behavior.  It is diagnostic only and
            # never participates in the nine-column main matrix.
            ablation_params = dict(params)
            ablation_params["p15_ablation_legacy_tau"] = True
            ab_y, ab_meta = _p14_apply_local_soft_lut(deblur_y, mask, fields, sigma, alpha, ablation_params)
            ab_rgb, ab_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, ab_y)
            b_ablation_stages.append({
                "alpha": alpha,
                "status": ab_meta.get("p14_b_status", "computed"),
                "output": ab_rgb,
                "y": ab_y,
                "rgb_meta": ab_rgb_meta,
                "meta": ab_meta,
            })
        stage_outputs.append({
            "lambda": regularization,
            "deblur_status": "computed",
            "deblur_output": deblur_rgb,
            "deblur_y": deblur_y,
            "raw_deblur_y": raw_deblur_y,
            "deblur_rgb_meta": deblur_rgb_meta,
            "wiener_meta": wiener_meta,
            "a_status": "computed" if profiles else "computed_no_profiles",
            "a_output": a_rgb,
            "a_y": a_y,
            "a_correction": a_correction,
            "a_rgb_meta": a_rgb_meta,
            "a_meta": profile_meta,
            "b_stages": b_stages,
            "b_ablation_stages": b_ablation_stages,
        })
    if not stage_outputs:
        raise ValueError("P14 produced no finite deblur stage")
    stage_telemetry: list[dict[str, Any]] = []
    for stage in stage_outputs:
        stage_telemetry.append({
            "lambda": float(stage["lambda"]),
            "deblur_status": stage["deblur_status"],
            "p14_a_status": stage["a_status"],
            "p14_a_profile_count": int(stage["a_meta"].get("profile_count", 0)),
            "p14_a_rejected_count": int(stage["a_meta"].get("profile_rejected_count", 0)),
            "p14_a_gap_count": int(stage["a_meta"].get("profile_gap_count", 0)),
            "p14_a_coverage_fraction": float(stage["a_meta"].get("profile_coverage_fraction", 0.0)),
            "p14_a_support_fraction": float(stage["a_meta"].get("profile_support_fraction", 0.0)),
            "p14_a_reverse_steps_before": int(stage["a_meta"].get("profile_reverse_steps_before", 0)),
            "p14_a_reverse_steps_after": int(stage["a_meta"].get("profile_reverse_steps_after", 0)),
            "p14_deblur_mean_abs_delta": float(stage["wiener_meta"].get("wiener_raw_mean_abs_delta", 0.0)),
            "p14_deblur_p95_abs_delta": float(stage["wiener_meta"].get("wiener_raw_p95_abs_delta", 0.0)),
            "b_stages": [
                {
                    "alpha": float(item["alpha"]),
                    "status": item["status"],
                    **{key: value for key, value in item["meta"].items() if not str(key).startswith("_")},
                }
                for item in stage["b_stages"]
            ],
        })
    selected = stage_outputs[0]
    selected_a_meta = selected["a_meta"]
    selected_b = selected["b_stages"][0] if selected["b_stages"] else None
    meta = dict(base_meta)
    meta.update({
        "p14_status": "computed",
        "p14_kernel_status": "computed",
        "p14_deblur_status": "computed",
        "p14_a_status": selected["a_status"],
        "p14_a_profile_count": int(selected_a_meta.get("profile_count", 0)),
        "p14_a_rejected_count": int(selected_a_meta.get("profile_rejected_count", 0)),
        "p14_a_gap_count": int(selected_a_meta.get("profile_gap_count", 0)),
        "p14_a_coverage_fraction": float(selected_a_meta.get("profile_coverage_fraction", 0.0)),
        "p14_a_gap_fraction": float(selected_a_meta.get("profile_gap_fraction", 0.0)),
        "p14_a_support_fraction": float(selected_a_meta.get("profile_support_fraction", 0.0)),
        "p14_a_reverse_steps_before": int(selected_a_meta.get("profile_reverse_steps_before", 0)),
        "p14_a_reverse_steps_after": int(selected_a_meta.get("profile_reverse_steps_after", 0)),
        "p14_a_field_region_fraction": float(selected_a_meta.get("p14_a_field_region_fraction", 0.0)),
        "p14_a_field_outside_max_abs_delta": float(selected_a_meta.get("p14_a_field_outside_max_abs_delta", 0.0)),
        "p14_a_field_outside_sum_abs_delta": float(selected_a_meta.get("p14_a_field_outside_sum_abs_delta", 0.0)),
        "p14_a_interpolated_profile_count": int(selected_a_meta.get("p14_a_interpolated_profile_count", 0)),
        "p14_a_long_gap_pixels": int(selected_a_meta.get("p14_a_long_gap_pixels", 0)),
        "p14_a_angle_rejected_count": int(selected_a_meta.get("p14_a_angle_rejected_count", 0)),
        "p14_b_status": selected_b["status"] if selected_b is not None else "skipped_no_stage",
        "p14_b_roi_fraction": float(selected_b["meta"].get("p14_b_roi_fraction", 0.0)) if selected_b is not None else 0.0,
        "p14_stage_count": len(stage_outputs),
        "p14_stage_telemetry": stage_telemetry,
        "_p14_stage_outputs": stage_outputs,
        "_p14_profiles": selected_a_meta.get("profiles", []),
        "_p14_a_correction": selected["a_correction"],
        "_p14_a_support": selected_a_meta.get("_p14_a_support", np.zeros_like(y_input, dtype=bool)),
        "_p14_output_minus_deblur": selected["a_output"] - selected["deblur_output"],
    })
    if selected_b is not None:
        meta.update({key: value for key, value in selected_b["meta"].items() if not str(key).startswith("_")})
    meta.update(segmentation)
    return selected["a_output"], meta


def _robust_stats(seed: np.ndarray) -> tuple[float | None, float | None, int]:
        values = y[seed].astype(np.float64)
        if values.size == 0:
            return None, None, 0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, mad, int(values.size)


def robust_center(values: np.ndarray) -> float:
        lo, hi = np.percentile(values, [35.0, 65.0])
        return float(np.median(values[(values >= lo) & (values <= hi)]))



