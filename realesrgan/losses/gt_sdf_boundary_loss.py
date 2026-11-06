"""Ground-truth signed-distance narrow-band loss for text boundaries.

The module is intentionally training-only and has no dependency on a text
mask or an SDF at inference time.  The target text mask and RGB target provide
detached morphology and local dark/light anchors; only the predicted Y plane
is part of the autograd path.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def _as_bchw_mask(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    name: str,
    *,
    default: float | None = None,
) -> torch.Tensor:
    """@shanlan-comment Normalize a GT mask to detached BCHW float values.

    Input: optional mask in ``[B,1,H,W]`` or ``[B,H,W]`` and a floating
        prediction reference whose device and spatial dimensions are used.
    Output: detached ``[B,1,H,W]`` mask on the reference device, or a full
        default mask when ``value`` is absent; malformed inputs raise.
    Key Steps: move/cast the mask, add a channel when required, and validate
        batch/channel/spatial dimensions before detaching it.
    Key Params: ``default`` is used only for the optional valid-mask fallback.
    State/Dependencies: torch tensor metadata only; no mutable shared state.
    """
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return reference.new_full((reference.shape[0], 1, *reference.shape[-2:]), float(default))
    result = value.to(device=reference.device, dtype=reference.dtype)
    if result.ndim == 3:
        result = result.unsqueeze(1)
    if result.ndim != 4 or result.shape[0] != reference.shape[0] or result.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W], got {tuple(result.shape)}")
    if tuple(result.shape[-2:]) != tuple(reference.shape[-2:]):
        raise ValueError(
            f"{name} spatial shape {tuple(result.shape[-2:])} does not match "
            f"reference {tuple(reference.shape[-2:])}"
        )
    return result.detach()


def _luma(value: torch.Tensor) -> torch.Tensor:
    """@shanlan-comment Convert one- or three-channel values to Rec.601 Y.

    Input: floating BCHW tensor with one luminance channel or three RGB
        channels in the same numeric range as the prediction.
    Output: BCHW one-channel luminance tensor; gradients are preserved.
    Key Steps: pass one-channel values through or apply fixed RGB weights.
    Key Params: Rec.601 coefficients keep GT polarity consistent with the Y
        output used by the loss.
    State/Dependencies: torch arithmetic only; input is not mutated.
    """
    if value.ndim != 4 or value.shape[1] not in (1, 3):
        raise ValueError(f"value must be BCHW with one or three channels, got {tuple(value.shape)}")
    if value.shape[1] == 1:
        return value
    return 0.299 * value[:, :1] + 0.587 * value[:, 1:2] + 0.114 * value[:, 2:3]


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """@shanlan-comment Dilate a BCHW mask with zero-valued image boundaries.

    Input: detached binary/soft BCHW mask and a non-negative pixel radius.
    Output: same-shaped max-pooled dilation in the input dtype.
    Key Steps: use an odd square max-pool kernel with stride one.
    Key Params: ``radius`` is measured in image pixels and controls local GT
        context or the outside SDF shells.
    State/Dependencies: ``torch.nn.functional.max_pool2d`` only.
    """
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)


def _erode_zero_boundary(mask: torch.Tensor) -> torch.Tensor:
    """@shanlan-comment Erode a BCHW mask while treating outside as background.

    Input: binary/soft BCHW mask in the loss working dtype.
    Output: one-pixel erosion with the original shape; image-edge pixels are
        removed when their full neighborhood would cross the canvas.
    Key Steps: explicitly zero-pad, negate, max-pool, and negate back.
    Key Params: fixed 3x3 neighborhood defines one pixel of SDF distance.
    State/Dependencies: torch pooling only; no prediction tensor is involved.
    """
    padded = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0.0)
    return -F.max_pool2d(-padded, kernel_size=3, stride=1)


@torch.no_grad()
def build_gt_sdf_band(
    text_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    radius: int = 3,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """@shanlan-comment Build a detached signed SDF and valid narrow-band gate.

    Input: full-resolution GT text and optional valid/padding masks as BCHW or
        BHW tensors; binary thresholding is applied to both masks.
    Output: ``(sdf, band, text, valid)`` with detached BCHW float tensors.
        SDF values are signed pixel distances in a square (L-infinity)
        neighborhood: text is negative, outside is positive, and distances
        beyond ``radius`` are clamped to ``+/- (radius + 1)``.
    Key Steps: intersect text with validity, peel inside shells by erosion,
        grow outside shells by dilation, and keep only ``abs(sdf) <= radius``.
    Key Params: ``radius`` is the inclusive boundary-band width in pixels;
        ``threshold`` defines binary GT/valid support.
    State/Dependencies: detached GT morphology only; no prediction or model
        state is read or changed.
    """
    radius = int(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not isinstance(text_mask, torch.Tensor):
        text_mask = torch.as_tensor(text_mask)
    if text_mask.ndim == 3:
        text_mask = text_mask.unsqueeze(1)
    if text_mask.ndim != 4 or text_mask.shape[1] != 1:
        raise ValueError("text_mask must have shape [B,1,H,W] or [B,H,W]")
    text = (text_mask.detach().float() > float(threshold)).float()
    if valid_mask is None:
        valid = torch.ones_like(text)
    else:
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = torch.as_tensor(valid_mask)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.shape != text.shape:
            raise ValueError("valid_mask must match text_mask shape")
        valid = (valid_mask.detach().to(device=text.device, dtype=text.dtype) > float(threshold)).float()
    text = text * valid

    far = text.new_tensor(float(radius + 1))
    sdf = torch.where(text > 0.0, -far, far)

    current = text
    for distance in range(1, radius + 1):
        eroded = _erode_zero_boundary(current)
        shell = (current - eroded).clamp(0.0, 1.0)
        sdf = torch.where(shell > 0.0, sdf.new_tensor(float(-distance)), sdf)
        current = eroded

    current = text
    for distance in range(1, radius + 1):
        dilated = _dilate(current, 1)
        shell = (dilated - current).clamp(0.0, 1.0)
        sdf = torch.where(shell > 0.0, sdf.new_tensor(float(distance)), sdf)
        current = dilated

    band = ((sdf.abs() <= float(radius)) & (valid > 0.0)).to(sdf.dtype)
    return sdf.detach(), band.detach(), text.detach(), valid.detach()


@torch.no_grad()
def _gt_local_anchors(
    target_luma: torch.Tensor,
    text: torch.Tensor,
    valid: torch.Tensor,
    *,
    radius: int,
    min_contrast: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """@shanlan-comment Estimate detached dark/light anchors from GT only.

    Input: detached target luminance and binary text/valid masks in BCHW.
    Output: per-sample polarity, midpoint, signed contrast, active structure
        gate, and the near-text background ring.  Polarity ``+1`` means dark
        ink (background brighter); ``-1`` means bright ink.
    Key Steps: form a valid non-text ring, compute foreground/background means,
        orient the midpoint, and disable empty/low-contrast samples.
    Key Params: ``radius`` controls ring context and ``min_contrast`` rejects
        unreliable polarity estimates.
    State/Dependencies: GT-only reductions; all outputs are detached.
    """
    text = text * valid
    ring = (_dilate(text, max(1, int(radius))) > 0.0).to(text.dtype)
    ring = ring * (1.0 - text) * valid
    text_mass = text.sum(dim=(2, 3), keepdim=True)
    ring_mass = ring.sum(dim=(2, 3), keepdim=True)
    outside = (1.0 - text) * valid
    outside_mass = outside.sum(dim=(2, 3), keepdim=True)
    # Prefer a local ring.  A farther valid background remains valid when the
    # ring is empty, but a fully-text/fully-padding tile stays inactive.
    background = torch.where(ring_mass > 0.0, ring, outside)
    background_mass = background.sum(dim=(2, 3), keepdim=True)
    fg_mean = (target_luma * text).sum(dim=(2, 3), keepdim=True) / text_mass.clamp_min(1.0)
    bg_mean = (target_luma * background).sum(dim=(2, 3), keepdim=True) / background_mass.clamp_min(1.0)
    contrast = bg_mean - fg_mean
    polarity = torch.where(contrast >= 0.0, torch.ones_like(contrast), -torch.ones_like(contrast))
    active = (
        (text_mass > 0.0)
        & (background_mass > 0.0)
        & (outside_mass > 0.0)
        & (contrast.abs() >= float(min_contrast))
    ).to(target_luma.dtype)
    midpoint = 0.5 * (fg_mean + bg_mean)
    return polarity.detach(), midpoint.detach(), contrast.detach(), active.detach(), ring.detach()


def soft_ink_probability(
    value_y: torch.Tensor,
    midpoint: torch.Tensor,
    polarity: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """@shanlan-comment Convert predicted Y into differentiable GT-oriented ink.

    Input: BCHW predicted Y and broadcastable GT-derived midpoint/polarity;
        ``temperature`` is expressed in the same Y units as the tensors.
    Output: BCHW probability in ``[0,1]`` with gradients to ``value_y``.
    Key Steps: orient the midpoint residual according to dark/bright polarity
        and apply a sigmoid; no mask or SDF enters this inference-like map.
    Key Params: smaller positive temperature makes the transition sharper.
    State/Dependencies: torch sigmoid only; anchor tensors are not mutated.
    """
    temperature = float(temperature)
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return torch.sigmoid(polarity * (midpoint - value_y) / temperature)


@LOSS_REGISTRY.register()
class GTSDFBoundaryLoss(nn.Module):
    """Differentiable GT-SDF profile loss restricted to a valid narrow band.

    The signed SDF is negative inside text and positive outside.  A detached
    GT local foreground/background anchor maps predicted Y to a stable ink
    probability for both dark and light text.  Only the selected band pixels
    contribute, so invalid right-padding and the rest of the tile receive
    exactly zero prediction gradient.
    """

    def __init__(
        self,
        radius: int = 3,
        sdf_temperature: float = 1.0,
        ink_temperature: float = 0.05,
        anchor_radius: int | None = None,
        min_contrast: float = 0.03,
        mask_threshold: float = 0.5,
        profile_loss: str = "bce",
        linear_boundary_weight: float = 0.0,
        boundary_width: int = 1,
        charbonnier_eps: float = 1.0e-3,
        loss_weight: float = 1.0,
        eps: float = 1.0e-6,
    ) -> None:
        """@shanlan-comment Configure the standalone GT-SDF boundary loss.

        Input: scalar band, temperature, GT-anchor, profile, and weighting
            options; values are validated before the module is used.
        Output: initialized loss module with no dataset/model state.
        Key Steps: normalize options, choose the anchor ring radius, and
            validate positive temperatures and non-negative weights.
        Key Params: ``radius=3`` is the default pixel band; ``profile_loss``
            selects BCE or probability-space Charbonnier; the optional linear
            term compares a piecewise-linear boundary target on inner shells.
        State/Dependencies: torch.nn.Module state only.
        """
        super().__init__()
        self.radius = int(radius)
        self.sdf_temperature = float(sdf_temperature)
        self.ink_temperature = float(ink_temperature)
        self.anchor_radius = self.radius if anchor_radius is None else int(anchor_radius)
        self.min_contrast = float(min_contrast)
        self.mask_threshold = float(mask_threshold)
        self.profile_loss = str(profile_loss).strip().lower()
        self.linear_boundary_weight = float(linear_boundary_weight)
        self.boundary_width = int(boundary_width)
        self.charbonnier_eps = float(charbonnier_eps)
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        if self.radius <= 0 or self.anchor_radius < 0:
            raise ValueError("radius must be positive and anchor_radius must be non-negative")
        if self.boundary_width < 1 or self.boundary_width > self.radius:
            raise ValueError("boundary_width must be in [1, radius]")
        if not torch.isfinite(torch.tensor(self.sdf_temperature)) or self.sdf_temperature <= 0.0:
            raise ValueError("sdf_temperature must be finite and positive")
        if not torch.isfinite(torch.tensor(self.ink_temperature)) or self.ink_temperature <= 0.0:
            raise ValueError("ink_temperature must be finite and positive")
        if self.min_contrast < 0.0 or not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("min_contrast must be non-negative and mask_threshold must be in [0,1]")
        if self.profile_loss not in {"bce", "charbonnier"}:
            raise ValueError("profile_loss must be 'bce' or 'charbonnier'")
        if self.linear_boundary_weight < 0.0 or self.loss_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.charbonnier_eps <= 0.0 or self.eps <= 0.0:
            raise ValueError("charbonnier_eps and eps must be positive")

    def forward(
        self,
        pred_y: torch.Tensor,
        gt_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        return_details: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """@shanlan-comment Compute the GT-SDF profile loss and diagnostics.

        Input: predicted full-resolution Y (or RGB, reduced to Y), full-
            resolution GT RGB, GT text mask, and optional valid mask.  Masks,
            SDF, polarity, and anchor statistics are training-only inputs.
        Output: scalar differentiable loss, optionally paired with detached
            support/profile/polarity maps and scalar diagnostics.
        Key Steps: validate shapes, build detached SDF/support, estimate GT
            local anchors, map prediction Y to ink probability, and average
            profile plus optional linear-boundary errors over active pixels.
        Key Params: constructor temperatures and weights control profile
            softness, narrow-band width, and optional inner-boundary emphasis.
        State/Dependencies: autograd flows only through ``pred_y``; no model
            input mask/SDF is consumed and no shared state is mutated.
        """
        if not isinstance(pred_y, torch.Tensor) or not isinstance(gt_rgb, torch.Tensor):
            raise ValueError("pred_y and gt_rgb must be tensors")
        if pred_y.ndim != 4 or gt_rgb.ndim != 4:
            raise ValueError("pred_y and gt_rgb must be BCHW tensors")
        if pred_y.shape[0] != gt_rgb.shape[0] or tuple(pred_y.shape[-2:]) != tuple(gt_rgb.shape[-2:]):
            raise ValueError("pred_y and gt_rgb must have matching batch and spatial shapes")
        if pred_y.shape[1] not in (1, 3) or gt_rgb.shape[1] != 3:
            raise ValueError("pred_y must have one/three channels and gt_rgb must have three channels")
        if not pred_y.is_floating_point():
            pred_y = pred_y.float()
        if not gt_rgb.is_floating_point():
            gt_rgb = gt_rgb.float()

        prediction_y = _luma(pred_y)
        target_luma = _luma(gt_rgb.detach())
        text_reference = prediction_y.detach()
        text = _as_bchw_mask(text_mask_rgb, text_reference, "text_mask_rgb")
        valid = _as_bchw_mask(valid_mask_rgb, text_reference, "valid_mask_rgb", default=1.0)
        sdf, band, text, valid = build_gt_sdf_band(
            text,
            valid,
            radius=self.radius,
            threshold=self.mask_threshold,
        )
        polarity, midpoint, contrast, active, ring = _gt_local_anchors(
            target_luma,
            text,
            valid,
            radius=self.anchor_radius,
            min_contrast=self.min_contrast,
        )
        active_band = band * active
        active_pixels = active_band.sum()
        target_profile = torch.sigmoid(-sdf / float(self.sdf_temperature))
        ink_logit = polarity.to(dtype=prediction_y.dtype) * (
            midpoint.to(dtype=prediction_y.dtype) - prediction_y
        ) / float(self.ink_temperature)
        ink_probability = torch.sigmoid(ink_logit)

        if self.profile_loss == "bce":
            profile_map = F.binary_cross_entropy_with_logits(
                ink_logit,
                target_profile.to(dtype=ink_logit.dtype),
                reduction="none",
            )
        else:
            profile_map = torch.sqrt(
                (ink_probability - target_profile.to(dtype=ink_probability.dtype)).square()
                + self.charbonnier_eps * self.charbonnier_eps
            )

        denominator = active_pixels.detach().clamp_min(self.eps)
        profile_value = (profile_map * active_band).sum() / denominator

        linear_target = (
            0.5 - sdf / float(2 * max(1, self.radius))
        ).clamp(0.0, 1.0)
        boundary_band = active_band * (sdf.abs() <= float(self.boundary_width)).to(active_band.dtype)
        linear_error = torch.sqrt(
            (ink_probability - linear_target.to(dtype=ink_probability.dtype)).square()
            + self.charbonnier_eps * self.charbonnier_eps
        )
        boundary_value = (linear_error * boundary_band).sum() / boundary_band.sum().detach().clamp_min(self.eps)

        if active_pixels.detach().item() <= 0.0:
            loss = prediction_y.sum() * 0.0
            profile_value = loss
            boundary_value = loss
        else:
            loss = self.loss_weight * profile_value + self.linear_boundary_weight * boundary_value

        if not return_details:
            return loss
        details: dict[str, torch.Tensor] = {
            "sdf": sdf.detach(),
            "band": band.detach(),
            "active_band": active_band.detach(),
            "text": text.detach(),
            "valid": valid.detach(),
            "ring": ring.detach(),
            "profile_target": target_profile.detach(),
            "linear_target": linear_target.detach(),
            "ink_probability": ink_probability.detach(),
            "polarity": polarity.reshape(polarity.shape[0]).detach(),
            "midpoint": midpoint.reshape(midpoint.shape[0]).detach(),
            "contrast": contrast.reshape(contrast.shape[0]).detach(),
            "structure_active": active.reshape(active.shape[0]).detach(),
            "active_pixels": active_band.sum(dim=(1, 2, 3)).detach(),
            "profile_loss": profile_value.detach(),
            "linear_boundary_loss": boundary_value.detach(),
        }
        return loss, details


# Descriptive aliases make config/search terminology flexible without
# registering multiple classes under different names.
GTSDFNarrowBandLoss = GTSDFBoundaryLoss
SignedDistanceBoundaryLoss = GTSDFBoundaryLoss


__all__ = [
    "GTSDFBoundaryLoss",
    "GTSDFNarrowBandLoss",
    "SignedDistanceBoundaryLoss",
    "build_gt_sdf_band",
    "soft_ink_probability",
]
