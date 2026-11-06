"""CUDA-only numerical-compatibility candidate for the frozen Lenovo P17-B path."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def _gaussian_blur(value: Tensor, sigma: float) -> Tensor:
    """@luna-comment Blur BCHW luma with reflect-padded separable Gaussian kernels on its CUDA device."""
    if sigma <= 0.0:
        return value
    radius = max(1, int(math.ceil(3.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-axis.square() / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    horizontal = kernel.reshape(1, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1)
    value = F.pad(value, (radius, radius, 0, 0), mode="reflect")
    value = F.conv2d(value, horizontal)
    value = F.pad(value, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(value, vertical)


def _otsu_threshold(value: Tensor) -> tuple[Tensor, Tensor]:
    """@luna-comment Compute hard 8-bit Otsu thresholds and per-item histograms entirely on CUDA."""
    if value.ndim != 3:
        raise ValueError("Otsu input must be BHW")
    batches, height, width = value.shape
    encoded = value.mul(255.0).round().clamp(0, 255).long().reshape(batches, height * width)
    histogram = torch.zeros((batches, 256), device=value.device, dtype=value.dtype)
    histogram.scatter_add_(1, encoded, torch.ones_like(encoded, dtype=value.dtype))
    probability = histogram / float(height * width)
    bins = torch.arange(256, device=value.device, dtype=value.dtype).reshape(1, -1)
    omega = probability.cumsum(dim=1)
    mean = (probability * bins).cumsum(dim=1)
    total_mean = mean[:, -1:].expand_as(mean)
    denominator = (omega * (1.0 - omega)).clamp_min(torch.finfo(value.dtype).eps)
    between = (total_mean * omega - mean).square() / denominator
    threshold = between.argmax(dim=1).to(value.dtype) / 255.0
    return threshold, histogram


def _segment_text(y: Tensor, params: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """@luna-comment
    Input: CUDA B1HW luma in [0, 1] and normalized mask parameters.
    Output: A conservative hard B1HW foreground mask and one polarity flag per batch.
    Key Steps: Smooths luma, Otsu-quantizes it, infers foreground polarity from
    border coverage, removes singleton foreground pixels, and rejects invalid coverage.
    Key Params: mask_sigma controls smoothing and mask_border_share resolves polarity.
    State/Dependencies: CUDA Torch Otsu and morphology only; no global state.
    """
    smooth = _gaussian_blur(y, float(params.get("mask_sigma", 0.6)))
    threshold, _ = _otsu_threshold(smooth[:, 0])
    encoded = smooth[:, 0].mul(255.0).round().clamp(0, 255)
    high = encoded > threshold[:, None, None].mul(255.0).round()
    batches, height, width = high.shape
    border_width = max(1, round(min(height, width) * 0.03))
    border = torch.zeros_like(high)
    border[:, :border_width] = True
    border[:, -border_width:] = True
    border[:, :, :border_width] = True
    border[:, :, -border_width:] = True
    high_border = high[border].reshape(batches, -1).float().mean(dim=1)
    high_roi = high.float().mean(dim=(1, 2))
    border_share = float(params.get("mask_border_share", 0.60))
    foreground_is_high = torch.where(
        high_border >= border_share,
        torch.zeros_like(high_border, dtype=torch.bool),
        torch.where(1.0 - high_border >= border_share, torch.ones_like(high_border, dtype=torch.bool), high_roi < 0.5),
    )
    foreground = torch.where(foreground_is_high[:, None, None], high, ~high)
    min_area = max(2, min(8, round(height * width * 0.00002)))
    if min_area == 2:
        neighbourhood = F.conv2d(
            foreground[:, None].float(),
            torch.ones((1, 1, 3, 3), device=y.device, dtype=y.dtype),
            padding=1,
        )[:, 0]
        foreground = foreground & (neighbourhood >= 2.0)
    coverage = foreground.float().mean(dim=(1, 2))
    accepted = (coverage >= 0.01) & (coverage <= 0.65)
    foreground = foreground & accepted[:, None, None]
    return foreground[:, None], foreground_is_high


def _torch_l2_chamfer_distance(nonzero: Tensor, radius: int) -> Tensor:
    """@luna-comment Compute a bounded CUDA 5x5 L2-chamfer distance to zero pixels.

    Input: Boolean ``B1HW`` map whose false pixels are distance-zero sources.
    Output: A finite CUDA distance field capped at ``radius`` pixels.
    Key Steps: Run fixed-count min-plus relaxations with the OpenCV 5x5 L2
      chamfer axial, diagonal, and knight-step costs; constant padding keeps
      the image exterior from becoming a false source.
    Key Params: ``radius`` bounds work to the maximum profile/feather support.
    State/Dependencies: Torch CUDA tensors only; no CPU, OpenCV, or NumPy path.
    """
    cap = float(max(1, int(radius)))
    distance = torch.where(nonzero, torch.full_like(nonzero, cap, dtype=torch.float32), torch.zeros_like(nonzero, dtype=torch.float32))
    distance = distance.to(dtype=torch.float32)
    costs = torch.tensor(
        ((float("inf"), 2.1969, 1.4, 2.1969, float("inf")),
         (2.1969, 1.4, 1.0, 1.4, 2.1969),
         (1.4, 1.0, 0.0, 1.0, 1.4),
         (2.1969, 1.4, 1.0, 1.4, 2.1969),
         (float("inf"), 2.1969, 1.4, 2.1969, float("inf"))),
        device=nonzero.device,
        dtype=distance.dtype,
    ).reshape(1, 25, 1, 1)
    height, width = nonzero.shape[-2:]
    for _ in range(max(1, int(radius))):
        neighbourhood = F.unfold(F.pad(distance, (2, 2, 2, 2), mode="constant", value=cap), 5).reshape(distance.shape[0], 25, height, width)
        distance = (neighbourhood + costs).amin(dim=1, keepdim=True).clamp_max(cap)
    return distance


def _torch_signed_distance(mask: Tensor, radius: int) -> tuple[Tensor, Tensor]:
    """@luna-comment Build bounded CUDA signed L2-chamfer distances and hard contour.

    Input: Boolean ``B1HW`` text mask on the active CUDA device.
    Output: Signed ``B1HW`` distance proxy (positive inside) and the one-pixel
      hard morphological contour used as profile seeds.
    Key Steps: Compute opposite-class 5x5 L2-chamfer fields and use their
      signed difference; max-pool dilation/erosion supplies hard contour seeds.
    Key Params: ``radius`` bounds the distance work to the profile support.
    State/Dependencies: Torch only; no CPU transfer, OpenCV, NumPy, or state.
    """
    foreground = mask.float()
    inside = _torch_l2_chamfer_distance(mask, radius).to(dtype=foreground.dtype)
    outside = _torch_l2_chamfer_distance(~mask, radius).to(dtype=foreground.dtype)
    signed = torch.where(mask, inside, -outside)
    dilated = F.max_pool2d(foreground, 3, 1, 1)
    eroded = 1.0 - F.max_pool2d(1.0 - foreground, 3, 1, 1)
    boundary = (dilated - eroded) > 0.0
    return signed, boundary


def _torch_profile_platform(values: Tensor, offsets: Tensor, side: str, params: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """@luna-comment Find first stable foreground/background platform per CUDA profile.

    Input: ``BMS`` bilinear Y profiles, ordered offsets, and P11 platform
      parameters; ``M`` is the bounded seed count and ``S`` the 65 samples.
    Output: Platform level and an accepted boolean for every ``B,M`` profile.
    Key Steps: Scan contiguous 2-pixel windows from the contour outward,
      compare P90-P10 spread to absolute/relative tolerance, then choose the
      first stable window median.
    Key Params: Default run/scan are 2/10 px; tolerance is max(2/255, .03
      times the profile P90-P10 robust contrast).
    State/Dependencies: Torch quantiles and reductions execute on CUDA only.
    """
    min_run = float(params.get("min_platform_run", 2.0))
    scan_limit = float(params.get("platform_scan_limit", 10.0))
    abs_tolerance = float(params.get("platform_tolerance_abs", 2.0 / 255.0))
    rel_tolerance = float(params.get("platform_tolerance_rel", 0.03))
    step = float(params.get("sample_step", 0.5))
    min_samples = max(2, int(math.ceil(min_run / max(step, 1.0e-6))))
    if side == "outside":
        eligible = torch.nonzero((offsets <= -min_run) & (offsets >= -scan_limit), as_tuple=False).flatten().flip(0)
    elif side == "inside":
        eligible = torch.nonzero((offsets >= min_run) & (offsets <= scan_limit), as_tuple=False).flatten()
    else:
        raise ValueError("P22 platform side must be outside or inside")
    if eligible.numel() < min_samples:
        shape = values.shape[:2]
        return torch.zeros(shape, device=values.device, dtype=values.dtype), torch.zeros(shape, device=values.device, dtype=torch.bool)
    windows = eligible.unfold(0, min_samples, 1)
    window_values = values[..., windows]
    spread = torch.quantile(window_values, 0.90, dim=-1) - torch.quantile(window_values, 0.10, dim=-1)
    robust_contrast = torch.quantile(values, 0.90, dim=-1) - torch.quantile(values, 0.10, dim=-1)
    tolerance = torch.maximum(torch.full_like(robust_contrast, abs_tolerance), rel_tolerance * robust_contrast)
    stable = spread <= tolerance.unsqueeze(-1)
    accepted = stable.any(dim=-1)
    first = stable.float().argmax(dim=-1)
    medians = torch.quantile(window_values, 0.5, dim=-1)
    level = medians.gather(-1, first.unsqueeze(-1)).squeeze(-1)
    return level, accepted


def _estimate_sigma(y: Tensor, mask: Tensor, params: Mapping[str, Any]) -> Tensor:
    """@luna-comment Fit bounded Gaussian edge profiles entirely on CUDA.

    Input: Float ``B1HW`` luma, hard ``B1HW`` text mask, and frozen P11 fit
      parameters. The mask is treated as authoritative; no CPU/OpenCV path is
      permitted.
    Output: One finite equivalent Gaussian sigma per batch item, equal to the
      median of accepted profile fits or an explicit bounded default when no
      profile is identifiable.
    Key Steps: Build signed-distance normals; choose row-major first
      ``max_profiles`` contour seeds; bilinearly sample -16..16 px at .5 px;
      reject unstable sides/reversals; grid-search sigma .3:.05:4 and centre
      -4:.25:4 with Huber loss; aggregate accepted fits by median.
    Key Params: P11 bounds/thresholds are read from ``params``; defaults match
      the frozen CPU implementation (run 2 px, scan 10 px, contrast .12,
      max reverse steps 8, fit RMSE .08).
    State/Dependencies: All tensors, reductions, ``grid_sample`` and fitting
      loops remain on the input CUDA device under the caller's no-grad scope.
    """
    if y.ndim != 4 or y.shape[1] != 1 or mask.shape != y.shape:
        raise ValueError("P22 profile fit requires matching B1HW luma and mask")
    batches, _, height, width = y.shape
    sigma_min = float(params.get("kernel_sigma_min", 0.3))
    sigma_max = float(params.get("kernel_sigma_max", 4.0))
    sigma_step = float(params.get("kernel_sigma_step", 0.05))
    centre_limit = float(params.get("kernel_center_limit", 4.0))
    centre_step = float(params.get("kernel_center_step", 0.25))
    huber_delta = float(params.get("kernel_huber_delta", 0.05))
    max_reverse = int(params.get("kernel_max_reverse_steps", 8))
    max_fit_rmse = float(params.get("kernel_max_fit_rmse", 0.08))
    min_contrast = float(params.get("kernel_min_contrast", params.get("min_profile_contrast", 0.12)))
    half_width = float(params.get("profile_half_width", 16.0))
    sample_step = float(params.get("sample_step", 0.5))
    max_profiles = max(1, int(params.get("max_profiles", 128)))
    if not (0.0 < sigma_min <= sigma_max and sigma_step > 0.0 and centre_limit > 0.0 and centre_step > 0.0 and huber_delta > 0.0):
        raise ValueError("invalid P22 Gaussian profile-fit parameters")
    signed, boundary = _torch_signed_distance(mask, int(math.ceil(half_width + 2.0)))
    sobel_x = torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)), device=y.device, dtype=y.dtype).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    signed_for_sobel = F.pad(signed, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(signed_for_sobel, sobel_x)
    grad_y = F.conv2d(signed_for_sobel, sobel_y)
    norm = torch.sqrt(grad_x.square() + grad_y.square())
    flat_boundary = boundary[:, 0].reshape(batches, -1)
    positions = torch.arange(height * width, device=y.device, dtype=torch.long).reshape(1, -1).expand(batches, -1)
    sentinel = torch.full_like(positions, height * width)
    ordered = torch.where(flat_boundary, positions, sentinel)
    stride = max(1, int(params.get("boundary_stride", 1)))
    count = min(max_profiles, height * width)
    seed_positions = ordered.topk(height * width, dim=1, largest=False, sorted=True).values[:, ::stride][:, :count]
    valid_seed = seed_positions < height * width
    seed_rows = torch.div(seed_positions.clamp_max(height * width - 1), width, rounding_mode="floor")
    seed_columns = seed_positions.clamp_max(height * width - 1).remainder(width)
    batch_indices = torch.arange(batches, device=y.device).reshape(-1, 1).expand(-1, count)
    seed_grad_x = grad_x[:, 0][batch_indices, seed_rows, seed_columns]
    seed_grad_y = grad_y[:, 0][batch_indices, seed_rows, seed_columns]
    seed_norm = norm[:, 0][batch_indices, seed_rows, seed_columns]
    valid_seed = valid_seed & (seed_norm > 1.0e-5)
    normal_x = seed_grad_x / seed_norm.clamp_min(1.0e-5)
    normal_y = seed_grad_y / seed_norm.clamp_min(1.0e-5)
    offsets = torch.arange(-half_width, half_width + sample_step * 0.5, sample_step, device=y.device, dtype=y.dtype)
    sample_x = seed_columns.to(y.dtype).unsqueeze(-1) + offsets.reshape(1, 1, -1) * normal_x.unsqueeze(-1)
    sample_y = seed_rows.to(y.dtype).unsqueeze(-1) + offsets.reshape(1, 1, -1) * normal_y.unsqueeze(-1)
    valid_seed = valid_seed & (sample_x.amin(-1) >= 1.0) & (sample_x.amax(-1) <= width - 2.0) & (sample_y.amin(-1) >= 1.0) & (sample_y.amax(-1) <= height - 2.0)
    sample_x_q = (sample_x * 32.0).round() / 32.0
    sample_y_q = (sample_y * 32.0).round() / 32.0
    grid = torch.stack((sample_x_q.clamp(0.0, width - 1.0) / max(width - 1, 1) * 2.0 - 1.0, sample_y_q.clamp(0.0, height - 1.0) / max(height - 1, 1) * 2.0 - 1.0), dim=-1)
    mask_grid = torch.stack((sample_x.clamp(0.0, width - 1.0) / max(width - 1, 1) * 2.0 - 1.0, sample_y.clamp(0.0, height - 1.0) / max(height - 1, 1) * 2.0 - 1.0), dim=-1)
    grid_for_sampling = grid.reshape(batches, count * offsets.numel(), 1, 2)
    mask_grid_for_sampling = mask_grid.reshape(batches, count * offsets.numel(), 1, 2)
    values = F.grid_sample(y, grid_for_sampling, mode="bilinear", padding_mode="reflection", align_corners=True)[:, 0, :, 0].reshape(batches, count, offsets.numel())
    mask_values = F.grid_sample(mask.float(), mask_grid_for_sampling, mode="nearest", padding_mode="reflection", align_corners=True)[:, 0, :, 0].reshape(batches, count, offsets.numel()) >= 0.5
    outside_idx = offsets <= -2.0
    inside_idx = offsets >= 2.0
    side_outside = (~mask_values[..., outside_idx]).float().mean(-1) >= 0.75
    side_inside = mask_values[..., inside_idx].float().mean(-1) >= 0.75
    background, background_ok = _torch_profile_platform(values, offsets, "outside", params)
    foreground, foreground_ok = _torch_profile_platform(values, offsets, "inside", params)
    contrast_signed = foreground - background
    contrast = contrast_signed.abs()
    denominator = torch.where(contrast >= 1.0e-6, contrast_signed, torch.ones_like(contrast_signed))
    oriented = ((values - background.unsqueeze(-1)) / denominator.unsqueeze(-1)).clamp(0.0, 1.0)
    reverse_steps = (oriented[..., 1:] - oriented[..., :-1] < -1.0 / 255.0).sum(-1)
    fit_valid = valid_seed & side_outside & side_inside & background_ok & foreground_ok & (contrast >= min_contrast) & (reverse_steps <= max_reverse)
    sigma_grid = torch.arange(sigma_min, sigma_max + sigma_step * 0.5, sigma_step, device=y.device, dtype=y.dtype)
    centre_grid = torch.arange(-centre_limit, centre_limit + centre_step * 0.5, centre_step, device=y.device, dtype=y.dtype)
    best_loss = torch.full((batches, count), float("inf"), device=y.device, dtype=y.dtype)
    best_sigma = torch.full((batches, count), sigma_min, device=y.device, dtype=y.dtype)
    best_centre = torch.zeros_like(best_sigma)
    for current_sigma in sigma_grid:
        cdf = 0.5 * (1.0 + torch.erf((offsets.unsqueeze(0) - centre_grid.unsqueeze(1)) / (math.sqrt(2.0) * current_sigma)))
        prediction = background.unsqueeze(-1).unsqueeze(-1) + contrast_signed.unsqueeze(-1).unsqueeze(-1) * cdf.unsqueeze(0).unsqueeze(0)
        residual = prediction - values.unsqueeze(-2)
        absolute = residual.abs()
        loss = torch.where(absolute <= huber_delta, 0.5 * absolute.square(), huber_delta * (absolute - 0.5 * huber_delta)).mean(-1)
        current_loss, current_index = loss.min(-1)
        improved = current_loss < best_loss
        best_loss = torch.where(improved, current_loss, best_loss)
        best_sigma = torch.where(improved, current_sigma, best_sigma)
        best_centre = torch.where(improved, centre_grid[current_index], best_centre)
    best_cdf = 0.5 * (1.0 + torch.erf((offsets.unsqueeze(0).unsqueeze(0) - best_centre.unsqueeze(-1)) / (math.sqrt(2.0) * best_sigma.unsqueeze(-1))))
    residual = background.unsqueeze(-1) + contrast_signed.unsqueeze(-1) * best_cdf - values
    fit_rmse = torch.sqrt(residual.square().mean(-1))
    accepted = fit_valid & torch.isfinite(fit_rmse) & (fit_rmse <= max_fit_rmse)
    accepted_count = accepted.sum(-1)
    sorted_sigma = best_sigma.masked_fill(~accepted, float("inf")).sort(dim=-1).values
    lower_index = ((accepted_count - 1).clamp_min(0) // 2).unsqueeze(-1)
    upper_index = (accepted_count.clamp_min(0) // 2).unsqueeze(-1)
    median_sigma = 0.5 * (sorted_sigma.gather(-1, lower_index).squeeze(-1) + sorted_sigma.gather(-1, upper_index).squeeze(-1))
    default_sigma = float(params.get("kernel_sigma_default", 1.0))
    return torch.where(accepted_count > 0, median_sigma, torch.full_like(median_sigma, min(max(default_sigma, sigma_min), sigma_max)))


def _local_source_envelope(source: Tensor, radius: int, epsilon: float) -> tuple[Tensor, Tensor]:
    """@luna-comment
    Input: CUDA B1HW source luma, positive integer radius, and nonnegative luma epsilon.
    Output: Per-pixel lower and upper source-local bounds expanded by epsilon.
    Key Steps: Reflect-pads source data, then computes square-window min and max via
    max pooling so crop boundaries do not introduce artificial zero-valued bounds.
    Key Params: radius is capped by the caller; epsilon permits normal sub-quantum variation.
    State/Dependencies: CUDA Torch pooling only; does not access preprocessing state.
    """
    if radius < 1 or epsilon < 0.0:
        raise ValueError("anti-ringing envelope requires radius >= 1 and epsilon >= 0")
    padded = F.pad(source, (radius, radius, radius, radius), mode="reflect")
    kernel = 2 * radius + 1
    minimum = -F.max_pool2d(-padded, kernel_size=kernel, stride=1)
    maximum = F.max_pool2d(padded, kernel_size=kernel, stride=1)
    return minimum - epsilon, maximum + epsilon


def _wiener_deblur(
    y: Tensor,
    mask: Tensor,
    sigma: Tensor,
    regularization: float,
    params: Mapping[str, Any] | None = None,
    foreground_is_high: Tensor | None = None,
    *,
    return_debug: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """@luna-comment
    Input: CUDA B1HW luma, hard foreground mask, fitted sigma per batch, positive
    Wiener regularization, normalized P17 parameters, and foreground polarity flags.
    Output: Contour-weighted deblurred luma; debug mode also returns raw and guarded
    Wiener maps before Otsu statistics.
    Key Steps: Performs reflect-padded crop-local FFT Wiener inversion, optionally
    rejects opposite-polarity deltas and source-envelope overshoot, then applies the
    existing contour soft band unchanged.
    Key Params: p17_wiener_bypass_enabled is an explicit diagnostic control;
    p17_wiener_antiring_* are disabled by default; delta floor ignores weak noise,
    radius is sigma-scaled/capped, and epsilon widens source bounds.
    State/Dependencies: CUDA Torch FFT/pooling only; disabled flags retain the frozen
    arithmetic path and do not alter Otsu or RGB reconstruction.
    """
    if regularization <= 0.0:
        raise ValueError("P17 Torch regularization must be positive")
    params = {} if params is None else params
    if bool(params.get("p17_wiener_bypass_enabled", False)):
        if not return_debug:
            return y
        zero = torch.zeros_like(y)
        return y, {
            "raw_wiener_local_y": y,
            "direction_rejected_delta": zero,
            "envelope_rejected_delta": zero,
            "global_clip_delta": zero,
        }
    batches, _, height, width = y.shape
    restored: list[Tensor] = []
    raw_wiener: list[Tensor] = []
    direction_rejected: list[Tensor] = []
    envelope_rejected: list[Tensor] = []
    global_clip_rejected: list[Tensor] = []
    direction_enabled = bool(params.get("p17_wiener_antiring_direction_enabled", False))
    envelope_enabled = bool(params.get("p17_wiener_antiring_envelope_enabled", False))
    global_clip_enabled = bool(params.get("p17_wiener_global_clip_enabled", False))
    delta_floor = float(params.get("p17_wiener_antiring_delta_floor", 2.0 / 255.0))
    envelope_epsilon = float(params.get("p17_wiener_antiring_envelope_epsilon", 2.0 / 255.0))
    radius_sigma = float(params.get("p17_wiener_antiring_radius_sigma", 3.0))
    radius_max = int(params.get("p17_wiener_antiring_radius_max", 12))
    if delta_floor < 0.0 or envelope_epsilon < 0.0 or radius_sigma <= 0.0 or radius_max < 1:
        raise ValueError("invalid P17 Wiener anti-ringing parameters")
    for index in range(batches):
        current_sigma = sigma[index]
        coordinates = torch.nonzero(mask[index, 0], as_tuple=False)
        if coordinates.numel() == 0:
            restored.append(y[index, 0])
            if return_debug:
                raw_wiener.append(y[index, 0])
                direction_rejected.append(torch.zeros_like(y[index, 0]))
                envelope_rejected.append(torch.zeros_like(y[index, 0]))
                global_clip_rejected.append(torch.zeros_like(y[index, 0]))
            continue
        context = max(1, int(math.ceil(3.0 * float(current_sigma.item()) + 1.0)))
        top = max(0, int(coordinates[:, 0].amin().item()) - context)
        bottom = min(height, int(coordinates[:, 0].amax().item()) + context + 1)
        left = max(0, int(coordinates[:, 1].amin().item()) - context)
        right = min(width, int(coordinates[:, 1].amax().item()) + context + 1)
        radius = max(1, int(math.ceil(3.0 * float(current_sigma.item()))))
        crop = y[index : index + 1, :, top:bottom, left:right]
        padded = F.pad(crop, (radius, radius, radius, radius), mode="reflect")[0, 0]
        padded_height, padded_width = padded.shape
        axis = torch.arange(-radius, radius + 1, device=y.device, dtype=y.dtype)
        gaussian = torch.exp(-(axis[:, None].square() + axis[None, :].square()) / (2.0 * current_sigma.square()))
        gaussian = gaussian / gaussian.sum().clamp_min(torch.finfo(y.dtype).eps)
        psf = torch.zeros((padded_height, padded_width), device=y.device, dtype=y.dtype)
        psf[: 2 * radius + 1, : 2 * radius + 1] = gaussian
        psf = torch.roll(psf, shifts=(-radius, -radius), dims=(0, 1))
        laplacian = torch.zeros_like(psf)
        laplacian[:3, :3] = torch.tensor(((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)), device=y.device, dtype=y.dtype)
        laplacian = torch.roll(laplacian, shifts=(-1, -1), dims=(0, 1))
        response = torch.fft.rfft2(psf)
        lap_fft = torch.fft.rfft2(laplacian)
        inverse = response.conj() / (response.abs().square() + regularization * lap_fft.abs().square()).clamp_min(torch.finfo(y.dtype).eps)
        local = torch.fft.irfft2(torch.fft.rfft2(padded) * inverse, s=(padded_height, padded_width))[radius : radius + bottom - top, radius : radius + right - left]
        crop_mask = mask[index : index + 1, :, top:bottom, left:right]
        crop_source = crop
        direction_delta = torch.zeros_like(local)
        envelope_delta = torch.zeros_like(local)
        clip_delta = torch.zeros_like(local)
        safe_local = local
        if global_clip_enabled:
            clipped = safe_local.clamp(0.0, 1.0)
            clip_delta = clipped - safe_local
            safe_local = clipped
        if direction_enabled:
            if foreground_is_high is None:
                raise ValueError("P17 Wiener direction guard requires foreground polarity")
            foreground_sign = 1.0 if bool(foreground_is_high[index].item()) else -1.0
            desired = torch.where(crop_mask[0, 0], torch.full_like(safe_local, foreground_sign), torch.full_like(safe_local, -foreground_sign))
            delta = safe_local - crop_source[0, 0]
            rejected = (delta.abs() >= delta_floor) & (delta * desired < 0.0)
            guarded = torch.where(rejected, crop_source[0, 0], safe_local)
            direction_delta = guarded - safe_local
            safe_local = guarded
        if envelope_enabled:
            envelope_radius = min(max(1, int(math.ceil(radius_sigma * float(current_sigma.item())))), radius_max)
            lower, upper = _local_source_envelope(crop_source, envelope_radius, envelope_epsilon)
            bounded = torch.minimum(torch.maximum(safe_local, lower[0, 0]), upper[0, 0])
            envelope_delta = bounded - safe_local
            safe_local = bounded
        raw = y[index, 0].clone()
        raw[top:bottom, left:right] = safe_local
        full_radius = max(1, int(math.floor(2.0 * float(current_sigma.item()))))
        zero_radius = max(full_radius + 1, int(math.ceil(3.0 * float(current_sigma.item()) + 1.0)))
        weight = _contour_weight(mask[index : index + 1], full_radius, zero_radius)[0, 0]
        restored.append(y[index, 0] + weight * (raw - y[index, 0]))
        if return_debug:
            raw_map = y[index, 0].clone()
            direction_map = torch.zeros_like(raw_map)
            envelope_map = torch.zeros_like(raw_map)
            clip_map = torch.zeros_like(raw_map)
            raw_map[top:bottom, left:right] = local
            direction_map[top:bottom, left:right] = direction_delta
            envelope_map[top:bottom, left:right] = envelope_delta
            clip_map[top:bottom, left:right] = clip_delta
            raw_wiener.append(raw_map)
            direction_rejected.append(direction_map)
            envelope_rejected.append(envelope_map)
            global_clip_rejected.append(clip_map)
    output = torch.stack(restored, dim=0).unsqueeze(1)
    if not return_debug:
        return output
    return output, {
        "raw_wiener_local_y": torch.stack(raw_wiener, dim=0).unsqueeze(1),
        "direction_rejected_delta": torch.stack(direction_rejected, dim=0).unsqueeze(1),
        "envelope_rejected_delta": torch.stack(envelope_rejected, dim=0).unsqueeze(1),
        "global_clip_delta": torch.stack(global_clip_rejected, dim=0).unsqueeze(1),
    }


def _contour_weight(mask: Tensor, full_radius: int, zero_radius: int) -> Tensor:
    """@luna-comment Approximate P17's contour-distance feather through CUDA morphological expansion rings."""
    foreground = mask.float()
    eroded = 1.0 - F.max_pool2d(1.0 - foreground, 3, 1, 1)
    dilated = F.max_pool2d(foreground, 3, 1, 1)
    frontier = (dilated - eroded).clamp(0.0, 1.0)
    reached = frontier > 0.0
    result = frontier.clone()
    for radius in range(1, zero_radius + 1):
        frontier = F.max_pool2d(frontier, 3, 1, 1)
        ring = (frontier > 0.0) & ~reached
        reached |= ring
        if radius <= full_radius:
            weight = 1.0
        else:
            ratio = (radius - full_radius) / max(zero_radius - full_radius, 1)
            weight = 0.5 * (1.0 + math.cos(math.pi * ratio))
        result = torch.where(ring, torch.full_like(result, weight), result)
    return result.clamp(0.0, 1.0)


def _regional_otsu_lut(
    y: Tensor, mask: Tensor, alpha: float, params: Mapping[str, Any], *, return_debug: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """@luna-comment
    Input: CUDA BCHW deblurred luma in [0, 1], a hard BCHW text mask, the P17 alpha,
    and normalized P17 patch/write parameters.
    Output: Corrected luma in [0, 1]; when return_debug is true, also returns the
    exact overlap-add maps and per-patch Otsu platform statistics used for that result.
    Key Steps: Computes per-patch blurred statistics and quantised Otsu classes,
    forms valid patch corrections, Hann-fuses them, then applies contour write weight.
    Key Params: p17_patch_size/stride control overlap; alpha controls correction;
    p17_min_class_fraction and P16 write radii retain frozen validity/write behavior.
    State/Dependencies: CUDA Torch only; no global state or CPU fallback.
    """
    batches, _, height, width = y.shape
    patch_size = max(3, int(params.get("p17_patch_size", 96)))
    stride = max(1, int(params.get("p17_patch_stride", 32)))
    starts_y = sorted(set(list(range(0, max(height - patch_size, 0) + 1, stride)) + [max(height - patch_size, 0)]))
    starts_x = sorted(set(list(range(0, max(width - patch_size, 0) + 1, stride)) + [max(width - patch_size, 0)]))
    residual_sum = torch.zeros_like(y)
    weight_sum = torch.zeros_like(y)
    min_fraction = float(params.get("p17_min_class_fraction", 0.12))
    patch_xyxy: list[tuple[int, int, int, int]] = []
    patch_low: list[Tensor] = []
    patch_high: list[Tensor] = []
    patch_valid: list[Tensor] = []
    for top in starts_y:
        bottom = min(height, top + patch_size)
        for left in starts_x:
            right = min(width, left + patch_size)
            patch = y[:, :, top:bottom, left:right]
            stats = _gaussian_blur(patch, float(params.get("p17_stats_gaussian_sigma", 0.6)))[:, 0]
            threshold, _ = _otsu_threshold(stats)
            encoded = stats.mul(255.0).round().clamp(0, 255).long()
            high = encoded > threshold.mul(255.0).round().long()[:, None, None]
            low_weight = (~high).float()
            high_weight = high.float()
            low_fraction = low_weight.mean((1, 2))
            high_fraction = high_weight.mean((1, 2))
            low_centers: list[Tensor] = []
            high_centers: list[Tensor] = []
            for index in range(batches):
                low_values = stats[index][~high[index]]
                high_values = stats[index][high[index]]
                if low_values.numel() == 0 or high_values.numel() == 0:
                    low_centers.append(torch.zeros((), device=y.device, dtype=y.dtype))
                    high_centers.append(torch.zeros((), device=y.device, dtype=y.dtype))
                    continue
                low_lo, low_hi = torch.quantile(low_values, torch.tensor((0.35, 0.65), device=y.device, dtype=y.dtype))
                high_lo, high_hi = torch.quantile(high_values, torch.tensor((0.35, 0.65), device=y.device, dtype=y.dtype))
                # On quantized or nearly constant short-line patches, the
                # float bounds can exclude every sample despite both classes
                # being non-empty.  Keep the trimmed median when available,
                # otherwise use the class median rather than failing P17.
                low_middle = low_values[(low_values >= low_lo) & (low_values <= low_hi)]
                high_middle = high_values[(high_values >= high_lo) & (high_values <= high_hi)]
                low_centers.append(torch.quantile(low_middle if low_middle.numel() else low_values, 0.5))
                high_centers.append(torch.quantile(high_middle if high_middle.numel() else high_values, 0.5))
            low_center = torch.stack(low_centers)
            high_center = torch.stack(high_centers)
            dynamic_range = stats.amax((1, 2)) - stats.amin((1, 2))
            valid = (low_fraction >= min_fraction) & (high_fraction >= min_fraction) & ((high_center - low_center).abs() >= torch.maximum(torch.full_like(dynamic_range, 12.0 / 255.0), 0.08 * dynamic_range))
            low = torch.minimum(low_center, high_center)[:, None, None, None]
            high_value = torch.maximum(low_center, high_center)[:, None, None, None]
            correction = alpha * (low - patch).clamp_min(0.0) + alpha * (high_value - patch).clamp_max(0.0)
            correction = correction * valid[:, None, None, None]
            patch_height, patch_width = patch.shape[-2:]
            axis_y = torch.arange(patch_height, device=y.device, dtype=y.dtype)
            axis_x = torch.arange(patch_width, device=y.device, dtype=y.dtype)
            window = torch.sin(math.pi * (axis_y + 0.5) / patch_height).square()[:, None] * torch.sin(math.pi * (axis_x + 0.5) / patch_width).square()[None, :]
            residual_sum[:, :, top:bottom, left:right] += correction * window
            weight_sum[:, :, top:bottom, left:right] += window * valid[:, None, None, None]
            if return_debug:
                patch_xyxy.append((left, top, right, bottom))
                patch_low.append(low_center.detach())
                patch_high.append(high_center.detach())
                patch_valid.append(valid.detach())
    fused = residual_sum / weight_sum.clamp_min(torch.finfo(y.dtype).eps)
    write = _contour_weight(mask, int(params.get("p16_write_full_radius", 16.0)), int(params.get("p16_write_zero_radius", 24.0)))
    corrected = y + fused * write
    if not return_debug:
        return corrected
    patch_count = len(patch_xyxy)
    patch_table = torch.tensor(patch_xyxy, device=y.device, dtype=torch.int32) if patch_count else torch.empty((0, 4), device=y.device, dtype=torch.int32)
    return corrected, {
        "regional_residual_sum": residual_sum,
        "regional_weight_sum": weight_sum,
        "fused_correction": fused,
        "write_weight": write,
        "weighted_correction": fused * write,
        "patch_xyxy": patch_table,
        "patch_low": torch.stack(patch_low, dim=1) if patch_count else torch.empty((batches, 0), device=y.device, dtype=y.dtype),
        "patch_high": torch.stack(patch_high, dim=1) if patch_count else torch.empty((batches, 0), device=y.device, dtype=y.dtype),
        "patch_valid": torch.stack(patch_valid, dim=1) if patch_count else torch.empty((batches, 0), device=y.device, dtype=torch.bool),
    }


def preprocess_p17_torch_gpu(lq_rgb: Tensor, params: Mapping[str, Any]) -> Tensor:
    """@luna-comment
    Input: Finite CUDA RGB BCHW tensor in [0, 1] and normalized enabled P17 parameters.
    Output: Same-shape, same-dtype CUDA RGB tensor after frozen P17-B preprocessing.
    Key Steps: Derives luma, segments text, estimates blur, performs Wiener deblur,
    applies regional Otsu correction, then shares the gamut-limited luma delta in RGB.
    Key Params: lambda regularizes Wiener deblur; alpha and P17 patch/write fields
    control the frozen regional correction.
    State/Dependencies: CUDA Torch kernels only; no CPU round trip or global state.
    """
    if not lq_rgb.is_cuda:
        raise ValueError("lq_preprocess implementation torch_gpu requires a CUDA tensor")
    if lq_rgb.ndim != 4 or lq_rgb.shape[1] != 3:
        raise ValueError("Torch P17 requires RGB BCHW")
    original_dtype = lq_rgb.dtype
    with torch.no_grad():
        rgb = lq_rgb.float()
        y = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]).clamp(0.0, 1.0)
        mask, polarity = _segment_text(y, params)
        sigma = _estimate_sigma(y, mask, params)
        deblurred = _wiener_deblur(y, mask, sigma, float(params["lambda"]), params, polarity)
        corrected = _regional_otsu_lut(deblurred, mask, float(params["alpha"]), params)
        requested = corrected - y
        lower = -rgb.amin(dim=1, keepdim=True)
        upper = 1.0 - rgb.amax(dim=1, keepdim=True)
        output = rgb + torch.minimum(torch.maximum(requested, lower), upper)
    if not torch.isfinite(output).all():
        raise ValueError("Torch P17 produced non-finite RGB")
    return output.to(dtype=original_dtype)


def preprocess_p17_torch_gpu_with_debug(lq_rgb: Tensor, params: Mapping[str, Any]) -> tuple[Tensor, dict[str, Any]]:
    """@luna-comment
    Input: Finite CUDA RGB BCHW tensor in [0, 1] and normalized enabled P17 parameters.
    Output: The exact P17 output plus stage tensors for diagnostic export; tensors remain
    on CUDA and no production preprocessing behavior is altered.
    Key Steps: Runs the same luma, segmentation, Wiener, Otsu overlap-add, and RGB
    gamut-limit sequence as the production path while retaining each stage result.
    Key Params: Uses the supplied frozen P17 parameters unchanged.
    State/Dependencies: Calls the same CUDA helpers as preprocess_p17_torch_gpu;
    diagnostic tables are allocated only for this explicit debug entry point.
    """
    if not lq_rgb.is_cuda:
        raise ValueError("Torch P17 debug requires a CUDA tensor")
    if lq_rgb.ndim != 4 or lq_rgb.shape[1] != 3:
        raise ValueError("Torch P17 debug requires RGB BCHW")
    original_dtype = lq_rgb.dtype
    with torch.no_grad():
        rgb = lq_rgb.float()
        y = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]).clamp(0.0, 1.0)
        mask, polarity = _segment_text(y, params)
        sigma = _estimate_sigma(y, mask, params)
        deblurred, wiener_debug = _wiener_deblur(y, mask, sigma, float(params["lambda"]), params, polarity, return_debug=True)
        corrected, regional = _regional_otsu_lut(deblurred, mask, float(params["alpha"]), params, return_debug=True)
        requested = corrected - y
        lower = -rgb.amin(dim=1, keepdim=True)
        upper = 1.0 - rgb.amax(dim=1, keepdim=True)
        applied = torch.minimum(torch.maximum(requested, lower), upper)
        output = rgb + applied
        output_y = (0.299 * output[:, 0:1] + 0.587 * output[:, 1:2] + 0.114 * output[:, 2:3]).clamp(0.0, 1.0)
    if not torch.isfinite(output).all():
        raise ValueError("Torch P17 debug produced non-finite RGB")
    debug = {
        "input_y": y,
        "text_mask": mask,
        "polarity": polarity,
        "sigma": sigma,
        "deblurred_y": deblurred,
        "safe_deblurred_y": deblurred,
        "deblur_delta": deblurred - y,
        **wiener_debug,
        "corrected_y": corrected,
        "requested_y_delta": requested,
        "gamut_lower_delta": lower,
        "gamut_upper_delta": upper,
        "applied_y_delta": applied,
        "output_y": output_y,
        **regional,
    }
    return output.to(dtype=original_dtype), debug


__all__ = ["preprocess_p17_torch_gpu", "preprocess_p17_torch_gpu_with_debug"]
