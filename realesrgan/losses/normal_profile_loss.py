"""GT-mask guided boundary-normal luminance profile loss.

The loss is intentionally training-only.  It consumes the ground-truth text
and validity masks, but it never changes a model input or inference path.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def _as_bchw(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a mask/image tensor to BCHW without changing its values.

    @shanlan-comment
    Input: a tensor in BCHW or BHW form and a diagnostic name.
    Output: the same tensor with an explicit singleton channel, or the original BCHW tensor.
    Key Steps: validate rank and add the channel dimension for BHW inputs.
    Key Params: ``name`` is used only to make validation errors actionable.
    State/Dependencies: no mutable state; dtype and device are preserved.
    """
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,H,W] or [B,H,W], got {tuple(value.shape)}")
    return value


def _unit_mask(
    value: torch.Tensor | None,
    *,
    reference: torch.Tensor,
    name: str,
    size: tuple[int, int],
    channel_mode: str,
) -> torch.Tensor:
    """Convert an RGB/gray mask to a detached BCHW mask in the unit range.

    @shanlan-comment
    Input: optional mask in BCHW/BHW, a prediction reference, target size, and
        ``channel_mode`` (``mean`` for text or ``min`` for validity).
    Output: detached ``[B,1,H,W]`` float mask in ``[0,1]``; missing text is zero
        and missing validity is one.
    Key Steps: validate batch shape, resize with nearest-neighbor semantics,
        collapse RGB channels, normalize byte masks, and clamp.
    Key Params: ``channel_mode`` keeps a partially invalid RGB validity mask
        conservative while averaging replicated RGB text masks.
    State/Dependencies: the result is GT metadata and is detached from autograd.
    """
    batch_size = int(reference.shape[0])
    if value is None:
        fill = 0.0 if name.startswith("text") else 1.0
        return reference.new_full((batch_size, 1, *size), fill)
    mask = _as_bchw(value, name).to(device=reference.device, dtype=reference.dtype)
    if mask.shape[0] != batch_size:
        raise ValueError(f"{name} batch size {mask.shape[0]} does not match prediction {batch_size}")
    mask = torch.nan_to_num(mask.detach(), nan=0.0, posinf=1.0, neginf=0.0)
    if tuple(mask.shape[-2:]) != tuple(size):
        mask = F.interpolate(mask, size=size, mode="nearest")
    if mask.shape[1] > 1:
        if channel_mode == "min":
            mask = mask.amin(dim=1, keepdim=True)
        elif channel_mode == "mean":
            mask = mask.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"unsupported mask channel mode: {channel_mode}")
    else:
        mask = mask[:, :1]
    byte_scale = torch.where(
        mask.detach().amax() > 1.5,
        mask.new_tensor(255.0),
        mask.new_tensor(1.0),
    )
    return (mask / byte_scale).clamp(0.0, 1.0)


def _central_mask_gradient(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute x/y finite differences of the detached binary GT mask.

    @shanlan-comment
    Input: detached binary ``[B,1,H,W]`` text mask.
    Output: centered x and y mask gradients with the same spatial shape.
    Key Steps: apply fixed three-tap central-difference kernels with zero image
        boundary padding; these gradients are never computed from prediction.
    Key Params: the fixed ``[-0.5,0,0.5]`` stencil defines pixel-coordinate normals.
    State/Dependencies: uses only the supplied GT mask and torch convolution.
    """
    kernel_x = mask.new_tensor([[-0.5, 0.0, 0.5]]).view(1, 1, 1, 3)
    kernel_y = mask.new_tensor([[-0.5], [0.0], [0.5]]).view(1, 1, 3, 1)
    return (
        F.conv2d(mask, kernel_x, padding=(0, 1)),
        F.conv2d(mask, kernel_y, padding=(1, 0)),
    )


def _orientation_coherence(
    normal_x: torch.Tensor,
    normal_y: torch.Tensor,
    *,
    kernel_size: int,
    eps: float,
) -> torch.Tensor:
    """Estimate sign-invariant local normal coherence from direction moments.

    @shanlan-comment
    Input: normalized mask-normal components and an odd neighborhood size.
    Output: ``[B,1,H,W]`` coherence in ``[0,1]``.
    Key Steps: average the second direction moments and evaluate their anisotropy;
        opposite normals on the two sides of a stroke remain coherent.
    Key Params: ``kernel_size`` controls the curvature/junction neighborhood and
        ``eps`` stabilizes low-gradient regions.
    State/Dependencies: only GT-derived normal directions are read.
    """
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    pad = size // 2
    xx = F.avg_pool2d(normal_x.square(), size, stride=1, padding=pad)
    yy = F.avg_pool2d(normal_y.square(), size, stride=1, padding=pad)
    xy = F.avg_pool2d(normal_x * normal_y, size, stride=1, padding=pad)
    anisotropy = torch.sqrt((xx - yy).square() + 4.0 * xy.square() + eps * eps)
    energy = (xx + yy).clamp_min(eps)
    return (anisotropy / energy).clamp(0.0, 1.0)


def _normalize_profile(profile: torch.Tensor, eps: float) -> torch.Tensor:
    """Remove local profile mean and normalize by local RMS contrast."""
    centered = profile - profile.mean(dim=-1, keepdim=True)
    contrast = torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + eps * eps)
    return centered / contrast


def _luma(value: torch.Tensor) -> torch.Tensor:
    """Return Rec.601 luminance while preserving prediction gradients."""
    if value.shape[1] == 1:
        return value
    if value.shape[1] != 3:
        raise ValueError(f"profile inputs must have one or three channels, got {value.shape[1]}")
    return 0.299 * value[:, :1] + 0.587 * value[:, 1:2] + 0.114 * value[:, 2:3]


def _charbonnier(value: torch.Tensor, enabled: bool, eps: float) -> torch.Tensor:
    """Apply either Charbonnier or absolute residual per profile sample."""
    if enabled:
        return torch.sqrt(value.square() + eps * eps)
    return value.abs()


@LOSS_REGISTRY.register()
class BoundaryNormalProfileLoss(nn.Module):
    """Compare normalized Y profiles sampled along reliable GT-mask normals.

    The support is derived from ``text_mask_rgb`` and ``valid_mask_rgb`` only.
    A boundary point is retained when a morphology band and mask gradient agree,
    while a sign-invariant direction-coherence score softly suppresses corners,
    junctions, and strongly curved regions.  Prediction gradients can therefore
    only originate from valid profile samples around accepted GT boundaries.

    Args:
        radius: profile half-width in pixels.
        samples: number of profile taps, default ``2 * radius + 1``.
        tolerance_pixels: half-width of the registration tolerance in pixels.
        tolerance_samples: number of softmin candidate shifts.
        profile_weight: weight of normalized profile residuals.
        difference_weight: weight of first-order profile-difference residuals.
        charb: use Charbonnier residuals instead of L1.
        loss_weight: module-local multiplier.
    """

    def __init__(
        self,
        radius: int = 3,
        samples: int | None = None,
        tolerance_pixels: float = 0.75,
        tolerance_samples: int = 3,
        profile_weight: float = 1.0,
        difference_weight: float = 1.0,
        charb: bool = True,
        charbonnier: bool | None = None,
        charbonnier_eps: float = 1e-3,
        contrast_eps: float = 1e-3,
        softmin_temperature: float = 0.05,
        threshold: float = 0.5,
        valid_threshold: float | None = None,
        edge_width: int = 1,
        normal_gradient_threshold: float = 1e-3,
        coherence_kernel: int = 3,
        min_coherence: float = 0.35,
        y_channel: int = 0,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        """Initialize profile geometry, robust residual, and reduction settings.

        @shanlan-comment
        Input: numeric profile/support options; no image or mask tensors are
            retained in module state.
        Output: a registry-compatible loss module.
        Key Steps: validate radius/tap/tolerance/reduction settings and store
            immutable scalar controls.
        Key Params: tolerance enables soft registration forgiveness; coherence
            and edge thresholds gate unreliable GT boundary directions.
        State/Dependencies: all tensors are created per forward on input device.
        """
        super().__init__()
        self.radius = int(radius)
        self.samples = int(2 * self.radius + 1 if samples is None else samples)
        self.tolerance_pixels = float(tolerance_pixels)
        self.tolerance_samples = int(tolerance_samples)
        self.profile_weight = float(profile_weight)
        self.difference_weight = float(difference_weight)
        self.charb = bool(charb if charbonnier is None else charbonnier)
        self.charbonnier_eps = float(charbonnier_eps)
        self.contrast_eps = float(contrast_eps)
        self.softmin_temperature = float(softmin_temperature)
        self.threshold = float(threshold)
        self.valid_threshold = float(threshold if valid_threshold is None else valid_threshold)
        self.edge_width = int(edge_width)
        self.normal_gradient_threshold = float(normal_gradient_threshold)
        self.coherence_kernel = int(coherence_kernel)
        self.min_coherence = float(min_coherence)
        self.y_channel = int(y_channel)
        self.loss_weight = float(loss_weight)
        self.reduction = str(reduction)
        self.eps = float(eps)
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.samples < 3:
            raise ValueError("samples must be at least 3")
        if self.tolerance_pixels < 0.0 or self.tolerance_samples < 1:
            raise ValueError("tolerance_pixels must be non-negative and tolerance_samples positive")
        if self.profile_weight < 0.0 or self.difference_weight < 0.0:
            raise ValueError("profile_weight and difference_weight must be non-negative")
        if self.profile_weight + self.difference_weight <= 0.0:
            raise ValueError("at least one profile term must be enabled")
        if self.charbonnier_eps <= 0.0 or self.contrast_eps <= 0.0 or self.softmin_temperature <= 0.0:
            raise ValueError("eps/temperature values must be positive")
        if not 0.0 <= self.threshold <= 1.0 or not 0.0 <= self.valid_threshold <= 1.0:
            raise ValueError("mask thresholds must be in [0,1]")
        if self.edge_width < 0 or self.coherence_kernel < 1:
            raise ValueError("edge_width must be non-negative and coherence_kernel positive")
        if not 0.0 <= self.min_coherence < 1.0:
            raise ValueError("min_coherence must be in [0,1)")
        if self.loss_weight < 0.0 or self.eps <= 0.0:
            raise ValueError("loss_weight must be non-negative and eps positive")
        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    def _support(
        self,
        text_mask_rgb: torch.Tensor | None,
        valid_mask_rgb: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build detached boundary weights, normals, and binary validity support.

        @shanlan-comment
        Input: GT text/valid masks and a BCHW Y reference defining batch/spatial
            geometry and dtype/device.
        Output: flattened boundary weight, x/y normals, valid mask, and boundary
            band, all detached from prediction autograd.
        Key Steps: threshold masks, form a morphology band, differentiate the
            mask, compute sign-invariant coherence, and suppress invalid points.
        Key Params: edge width, gradient threshold, and coherence threshold
            implement the robust junction/high-curvature simplification.
        State/Dependencies: no model state or inference tensor is touched.
        """
        size = (int(reference.shape[-2]), int(reference.shape[-1]))
        text = _unit_mask(
            text_mask_rgb,
            reference=reference,
            name="text_mask_rgb",
            size=size,
            channel_mode="mean",
        )
        valid = _unit_mask(
            valid_mask_rgb,
            reference=reference,
            name="valid_mask_rgb",
            size=size,
            channel_mode="min",
        )
        text_binary = (text >= self.threshold).to(reference.dtype)
        valid_binary = (valid >= self.valid_threshold).to(reference.dtype)
        morphology_size = 2 * self.edge_width + 1
        if morphology_size > 1:
            dilated = F.max_pool2d(text_binary, morphology_size, stride=1, padding=self.edge_width)
            eroded = -F.max_pool2d(-text_binary, morphology_size, stride=1, padding=self.edge_width)
            boundary_band = (dilated - eroded).clamp(0.0, 1.0)
        else:
            boundary_band = text_binary

        grad_x, grad_y = _central_mask_gradient(text_binary)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + self.eps * self.eps)
        normal_x = grad_x / magnitude
        normal_y = grad_y / magnitude
        coherence = _orientation_coherence(
            normal_x,
            normal_y,
            kernel_size=self.coherence_kernel,
            eps=self.eps,
        )
        coherence_weight = ((coherence - self.min_coherence) / (1.0 - self.min_coherence)).clamp(0.0, 1.0)
        gradient_weight = (magnitude > self.normal_gradient_threshold).to(reference.dtype)
        point_weight = boundary_band * valid_binary * coherence_weight * gradient_weight
        point_weight = point_weight.detach().reshape(reference.shape[0], -1)
        normal_x = normal_x.detach().reshape(reference.shape[0], -1)
        normal_y = normal_y.detach().reshape(reference.shape[0], -1)
        valid_binary = valid_binary.detach()
        return point_weight, normal_x, normal_y, valid_binary, boundary_band.detach()

    def _sample(
        self,
        value: torch.Tensor,
        base_x: torch.Tensor,
        base_y: torch.Tensor,
        normal_x: torch.Tensor,
        normal_y: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly sample one BCHW channel at boundary points along normals."""
        height, width = int(value.shape[-2]), int(value.shape[-1])
        grid_x = base_x.unsqueeze(-1) + normal_x.unsqueeze(-1) * offsets.view(1, 1, -1)
        grid_y = base_y.unsqueeze(-1) + normal_y.unsqueeze(-1) * offsets.view(1, 1, -1)
        grid = torch.stack(
            (
                2.0 * grid_x / max(width - 1, 1) - 1.0,
                2.0 * grid_y / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        )
        return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]

    def _softmin(self, candidate: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Take a non-negative, validity-aware soft minimum over shifts."""
        huge = torch.finfo(candidate.dtype).max / 32.0
        masked = torch.where(valid, candidate, candidate.new_full((), huge))
        count = valid.sum(dim=-1)
        temperature = self.softmin_temperature
        log_mean_exp = torch.logsumexp(-masked / temperature, dim=-1) - torch.log(count.clamp_min(1).to(candidate.dtype))
        result = -temperature * log_mean_exp
        result = torch.where(count > 0, result, torch.zeros_like(result))
        return result, count

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        text_mask_rgb: torch.Tensor | None = None,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        return_details: bool = False,
        return_stats: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compute the robust GT-boundary normal profile loss on Y only.

        @shanlan-comment
        Input: matching BCHW prediction/target tensors, optional RGB/gray GT
            masks, and an optional per-pixel external weight.
        Output: scalar ``loss_weight * mean`` (or configured reduction), with
            optional detached diagnostics; no masks are returned to a network.
        Key Steps: select Y, build GT support/normals, sample ±radius profiles,
            normalize local contrast, compare profiles and first differences,
            soft-min registration candidates, and reduce over valid support.
        Key Params: tolerance/temperature control registration softness; profile
            and difference weights balance shape and edge-transition agreement.
        State/Dependencies: autograd flows only to ``pred`` through grid_sample.
        """
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError("BoundaryNormalProfileLoss expects BCHW prediction and target tensors")
        if pred.shape[0] != target.shape[0] or pred.shape[-2:] != target.shape[-2:]:
            raise ValueError("prediction and target batch/spatial shapes must match")
        if pred.shape[1] not in (1, 3) or target.shape[1] not in (1, 3):
            raise ValueError("BoundaryNormalProfileLoss expects one-channel Y or three-channel RGB tensors")
        pred_y = _luma(pred)
        target_y = _luma(target).detach()
        if mask is not None and text_mask_rgb is None:
            text_mask_rgb = mask
        point_weight, normal_x, normal_y, valid_binary, boundary_band = self._support(
            text_mask_rgb,
            valid_mask_rgb,
            pred_y,
        )

        batch_size, _, height, width = pred_y.shape
        pixel_y, pixel_x = torch.meshgrid(
            torch.arange(height, device=pred_y.device, dtype=pred_y.dtype),
            torch.arange(width, device=pred_y.device, dtype=pred_y.dtype),
            indexing="ij",
        )
        base_x = pixel_x.reshape(1, -1).expand(batch_size, -1)
        base_y = pixel_y.reshape(1, -1).expand(batch_size, -1)
        offsets = torch.linspace(
            -float(self.radius),
            float(self.radius),
            self.samples,
            device=pred_y.device,
            dtype=pred_y.dtype,
        )
        pred_profile = self._sample(pred_y, base_x, base_y, normal_x, normal_y, offsets)
        pred_profile = _normalize_profile(pred_profile, self.contrast_eps)
        target_profiles = []
        valid_profiles = []
        if self.tolerance_pixels == 0.0 or self.tolerance_samples == 1:
            shifts = offsets.new_zeros(1)
        else:
            shifts = torch.linspace(
                -self.tolerance_pixels,
                self.tolerance_pixels,
                self.tolerance_samples,
                device=pred_y.device,
                dtype=pred_y.dtype,
            )
        for shift in shifts:
            candidate_offsets = offsets + shift
            target_profiles.append(
                _normalize_profile(
                    self._sample(target_y, base_x, base_y, normal_x, normal_y, candidate_offsets),
                    self.contrast_eps,
                )
            )
            valid_profiles.append(
                self._sample(
                    valid_binary,
                    base_x,
                    base_y,
                    normal_x,
                    normal_y,
                    candidate_offsets,
                ).amin(dim=-1) >= (1.0 - 1e-6)
            )
        target_profile = torch.stack(target_profiles, dim=-1)
        candidate_valid = torch.stack(valid_profiles, dim=-1)
        profile_error = _charbonnier(
            pred_profile.unsqueeze(-1) - target_profile,
            self.charb,
            self.charbonnier_eps,
        ).mean(dim=2)
        pred_difference = pred_profile[..., 1:] - pred_profile[..., :-1]
        target_difference = target_profile[..., 1:, :] - target_profile[..., :-1, :]
        difference_error = _charbonnier(
            pred_difference.unsqueeze(-1) - target_difference,
            self.charb,
            self.charbonnier_eps,
        ).mean(dim=2)
        term_total = self.profile_weight + self.difference_weight
        candidate_loss = (
            self.profile_weight * profile_error + self.difference_weight * difference_error
        ) / term_total
        point_loss, valid_count = self._softmin(candidate_loss, candidate_valid)

        if weight is not None:
            external = _unit_mask(
                weight,
                reference=pred_y,
                name="weight",
                size=(height, width),
                channel_mode="mean",
            ).reshape(batch_size, -1)
            point_weight = point_weight * external
        point_weight = point_weight * (valid_count > 0).to(point_weight.dtype)
        numerator = (point_loss * point_weight).sum()
        denominator = point_weight.sum()
        if self.reduction == "none":
            result = (point_loss * point_weight).reshape(batch_size, 1, height, width) * self.loss_weight
        elif self.reduction == "sum":
            result = numerator * self.loss_weight
        else:
            result = numerator / denominator.clamp_min(self.eps) * self.loss_weight
        if not (return_details or return_stats):
            return result
        details = {
            "point_weight": point_weight.detach().reshape(batch_size, 1, height, width),
            "boundary_band": boundary_band.detach(),
            "valid_mask": valid_binary.detach(),
            "normal_x": normal_x.detach().reshape(batch_size, 1, height, width),
            "normal_y": normal_y.detach().reshape(batch_size, 1, height, width),
            "valid_profile_fraction": (valid_count > 0).to(pred_y.dtype).mean().detach(),
            "boundary_weight_sum": denominator.detach(),
            "accepted_shift_count": valid_count.detach().to(pred_y.dtype).mean().detach(),
        }
        return result, details


# Short aliases keep the loss discoverable for configs written as either name.
NormalProfileLoss = BoundaryNormalProfileLoss
GTBoundaryNormalProfileLoss = BoundaryNormalProfileLoss


__all__ = [
    "BoundaryNormalProfileLoss",
    "GTBoundaryNormalProfileLoss",
    "NormalProfileLoss",
]
