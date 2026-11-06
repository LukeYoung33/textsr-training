"""Training-only thin-stroke skeleton recall/precision loss.

The loss uses a detached GT text mask to build a one-pixel morphological
skeleton and a small tolerance tube.  The prediction is never skeletonized:
its luminance is converted to a soft ink probability using the GT tile's
foreground/background polarity and midpoint.  A weighted centerline recall is
combined with an occupancy precision term so that external spurs and moderate
overgrowth remain costly.

This module is deliberately independent of the V4 trainer.  The trainer may
instantiate :class:`ThinSkeletonFbetaLoss` after the BG-LF integration is
stable, while the loss itself remains usable by a direct smoke or unit test.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def _as_bchw(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    name: str,
    *,
    default: float | None = None,
) -> torch.Tensor:
    """@luna-comment Normalize a value to the detached BCHW mask contract.

    Input: optional scalar mask/value in ``[B,1,H,W]`` or ``[B,H,W]`` and a
    floating BCHW reference whose device and spatial shape are authoritative.
    Output: detached float ``[B,1,H,W]`` on the reference device, or a full
    valid-mask default when ``value`` is absent; invalid shapes raise.
    Key Steps: convert dtype/device, add a channel when needed, and validate
    batch/channel/spatial dimensions before detaching the result.
    Key Params: ``default`` enables the all-valid fallback for valid masks.
    State/Dependencies: torch tensor/device state only; no shared state.
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


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """@luna-comment Convert one- or three-channel RGB values to luminance.

    Input: floating BCHW prediction/target with one or three channels in the
    training range; one-channel input is already treated as luminance.
    Output: BCHW one-channel Rec.601 luminance, preserving autograd links.
    Key Steps: pass through one channel or apply fixed RGB coefficients.
    Key Params: coefficients are fixed to match the local-polarity contract.
    State/Dependencies: torch only; no mutation or shared state.
    """
    if rgb.shape[1] == 1:
        return rgb
    if rgb.shape[1] != 3:
        raise ValueError(f"RGB input must have one or three channels, got {rgb.shape[1]}")
    return 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """@luna-comment Apply zero-boundary max-pool dilation to a BCHW mask.

    Input: binary or soft BCHW mask and a non-negative integer pixel radius.
    Output: same-shaped dilated mask; zero-valued outside-image context is
    implicit in max-pooling and gradients follow the selected maxima.
    Key Steps: use an odd square kernel with stride one and symmetric padding.
    Key Params: ``radius`` controls tolerated neighborhood size.
    State/Dependencies: torch.nn.functional.max_pool2d; no shared state.
    """
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)


def _erode(mask: torch.Tensor) -> torch.Tensor:
    """@luna-comment Apply one 3x3 erosion with background image boundaries.

    Input: binary/soft BCHW mask in the loss working dtype.
    Output: same-shaped eroded mask, with boundary pixels requiring a full
    neighborhood and therefore treating outside-image values as background.
    Key Steps: complement, zero-boundary max-pool, then complement back.
    Key Params: fixed three-by-three neighborhood gives one-pixel skeleton
    layers.
    State/Dependencies: torch.nn.functional.max_pool2d; no shared state.
    """
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)


@torch.no_grad()
def build_gt_skeleton(
    text_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    threshold: float = 0.5,
    max_iterations: int = 96,
    tube_radius: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """@luna-comment Build detached GT skeleton, tube, weights, and text support.

    Args:
        text_mask: full-resolution GT text mask, ``[B,1,H,W]`` or ``[B,H,W]``.
        valid_mask: full-resolution valid/padding mask.  ``None`` means all
            pixels are valid.
        threshold: threshold applied to the supplied masks.
        max_iterations: upper bound on erosion iterations.  The V4 canvas is
            at most 96 pixels high, so the default is bounded and finite.
        tube_radius: radius in pixels for the tolerated centerline tube.

    Returns:
        ``(skeleton, tube, skeleton_weight, text)``.  All outputs are detached
        float masks with shape ``[B,1,H,W]``.  ``skeleton_weight`` is one on
        average over each non-empty skeleton and gives thinner components a
        larger recall contribution.
    Key Steps: threshold and intersect GT text with validity, peel bounded
    morphological shells, fall back for degenerate masks, then dilate the
    skeleton into its tolerance tube and normalize inverse-radius weights.
    Key Params: ``threshold`` binarizes masks, ``max_iterations`` bounds work,
    and ``tube_radius`` sets tolerated centerline distance.
    State/Dependencies: torch morphology only; outputs are detached and no
    dataset, model, or global state is changed.
    """
    if not isinstance(text_mask, torch.Tensor):
        text_mask = torch.as_tensor(text_mask)
    if text_mask.ndim == 3:
        text_mask = text_mask.unsqueeze(1)
    if text_mask.ndim != 4 or text_mask.shape[1] != 1:
        raise ValueError("text_mask must be BCHW or BHW with one channel")
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

    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    max_iterations = min(max_iterations, max(1, int(max(text.shape[-2:]))))
    skeleton = torch.zeros_like(text)
    radius_map = torch.zeros_like(text)
    current = text
    for step in range(1, max_iterations + 1):
        eroded = _erode(current)
        # Morphological skeleton layer: (current - opening(current)).
        opened = _dilate(eroded, 1)
        skeleton = torch.maximum(skeleton, (current - opened).clamp(0.0, 1.0))
        shell = (current - eroded).clamp(0.0, 1.0)
        radius_map = torch.where(shell > 0.0, radius_map.new_tensor(float(step)), radius_map)
        current = eroded * valid
        if not bool(torch.any(current > 0.0)):
            break

    # Degenerate one-pixel or very small masks can have no opening layer on a
    # backend with a different border convention.  Falling back to the text
    # mask retains a finite, useful recall target.
    empty = skeleton.sum(dim=(1, 2, 3), keepdim=True) <= 0.0
    skeleton = torch.where(empty, text, skeleton)
    radius_map = torch.where((skeleton > 0.0) & (radius_map <= 0.0), radius_map.new_ones(()), radius_map)
    tube_radius = int(tube_radius)
    if tube_radius < 0:
        raise ValueError("tube_radius must be non-negative")
    tube = _dilate(skeleton, tube_radius) * valid

    # Inverse radius emphasizes the centerlines of thin strokes.  The
    # per-sample normalization keeps this a weighting change, not a hidden
    # sample-size change.
    inverse_radius = (1.0 / radius_map.clamp_min(1.0)) * skeleton
    mean_weight = inverse_radius.sum(dim=(1, 2, 3), keepdim=True) / skeleton.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    skeleton_weight = torch.where(skeleton > 0.0, inverse_radius / mean_weight.clamp_min(1e-6), torch.zeros_like(skeleton))
    return skeleton.detach(), tube.detach(), skeleton_weight.detach(), text.detach()


@torch.no_grad()
def _local_polarity_midpoint(
    target_luma: torch.Tensor,
    text: torch.Tensor,
    valid: torch.Tensor,
    *,
    radius: int,
    min_contrast: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """@luna-comment Estimate GT-local foreground polarity and midpoint.

    Input: detached target luminance plus binary text/valid BCHW masks.
    Output: per-sample polarity, midpoint, signed contrast, active gate, and
    nearby non-text ring masks; statistics are detached from prediction.
    Key Steps: prefer a dilated near-text ring, fall back to valid background,
    compute foreground/background means, and gate low-contrast samples.
    Key Params: ``radius`` selects local context; ``min_contrast`` suppresses
    polarity guesses on flat or unusable crops.
    State/Dependencies: torch pooling/reduction only; no shared state.
    """
    text = text * valid
    ring = (_dilate(text, int(radius)) > 0.0).to(text.dtype) * (1.0 - text) * valid
    outside = (1.0 - text) * valid
    ring_mass = ring.sum(dim=(2, 3), keepdim=True)
    outside_mass = outside.sum(dim=(2, 3), keepdim=True)
    # Prefer the nearby ring, then all valid non-text pixels.  If a crop is
    # fully text, the final fallback keeps the tile inactive rather than
    # inventing polarity from padding or from a fabricated background.
    background_support = torch.where(ring_mass > 0.0, ring, outside)
    background_mass = background_support.sum(dim=(2, 3), keepdim=True)
    valid_mass = valid.sum(dim=(2, 3), keepdim=True)
    background_support = torch.where(
        (background_mass <= 0.0) & (valid_mass > 0.0), valid, background_support
    )
    background_mass = background_support.sum(dim=(2, 3), keepdim=True)
    text_mass = text.sum(dim=(2, 3), keepdim=True)
    fg_mean = (target_luma * text).sum(dim=(2, 3), keepdim=True) / text_mass.clamp_min(1e-6)
    bg_mean = (target_luma * background_support).sum(dim=(2, 3), keepdim=True) / background_mass.clamp_min(1e-6)
    contrast = bg_mean - fg_mean
    polarity = torch.where(contrast >= 0.0, torch.ones_like(contrast), -torch.ones_like(contrast))
    active = (text_mass > 0.0) & (valid_mass > 0.0) & (contrast.abs() >= float(min_contrast))
    midpoint = 0.5 * (fg_mean + bg_mean)
    return polarity, midpoint, contrast, active.to(target_luma.dtype), ring


def soft_ink_probability(
    value_luma: torch.Tensor,
    midpoint: torch.Tensor,
    polarity: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """@luna-comment Convert predicted luminance into differentiable ink probability.

    Input: BCHW predicted luminance and broadcastable GT-derived midpoint and
    polarity tensors; all values are floating point.
    Output: BCHW probabilities in ``[0,1]`` with gradients to value_luma.
    Key Steps: orient midpoint distance by local polarity and apply sigmoid.
    Key Params: positive ``temperature`` controls softness around the midpoint.
    State/Dependencies: torch sigmoid only; midpoint/polarity are not mutated.
    """
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid(polarity * (midpoint - value_luma) / temperature)


@LOSS_REGISTRY.register()
class ThinSkeletonFbetaLoss(nn.Module):
    """Differentiable thin-stroke centerline recall/precision objective.

    ``beta > 1`` gives recall more influence than precision.  The default
    ``beta=1.5`` is intentionally conservative: a missing centerline is costly
    while a prediction that grows outside the GT text occupancy still lowers
    precision.  All masks and GT-derived polarity statistics are detached.
    """

    def __init__(
        self,
        beta: float = 1.5,
        tube_radius: int = 1,
        skeleton_threshold: float = 0.5,
        max_skeleton_iterations: int = 96,
        polarity_radius: int = 3,
        min_polarity_contrast: float = 0.03,
        temperature: float = 0.05,
        overgrowth_weight: float = 0.10,
        loss_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """@luna-comment Configure the standalone SkelFbeta objective.

        Input: scalar hyperparameters controlling F-beta emphasis, morphology,
        polarity estimation, overgrowth cost, and module-local scaling.
        Output: initialized loss module; invalid or non-positive settings raise
        before any training state is created.
        Key Steps: coerce options to stable Python scalars, save them for the
        forward contract, then validate bounds and numerical stabilizers.
        Key Params: beta defaults to 1.5, tube radius to one pixel, and the
        overgrowth weight keeps occupancy precision active.
        State/Dependencies: torch.nn.Module state only; no optimizer/global
        or dataset state is touched.
        """
        super().__init__()
        self.beta = float(beta)
        self.tube_radius = int(tube_radius)
        self.skeleton_threshold = float(skeleton_threshold)
        self.max_skeleton_iterations = int(max_skeleton_iterations)
        self.polarity_radius = int(polarity_radius)
        self.min_polarity_contrast = float(min_polarity_contrast)
        self.temperature = float(temperature)
        self.overgrowth_weight = float(overgrowth_weight)
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        if not torch.isfinite(torch.tensor(self.beta)) or self.beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        if self.tube_radius < 0 or self.polarity_radius < 0:
            raise ValueError("tube_radius and polarity_radius must be non-negative")
        if self.max_skeleton_iterations <= 0:
            raise ValueError("max_skeleton_iterations must be positive")
        if self.min_polarity_contrast < 0.0 or self.temperature <= 0.0:
            raise ValueError("min_polarity_contrast must be non-negative and temperature positive")
        if self.overgrowth_weight < 0.0 or self.loss_weight < 0.0 or self.eps <= 0.0:
            raise ValueError("overgrowth_weight/loss_weight must be non-negative and eps positive")

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        return_details: bool = False,
        return_maps: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """@luna-comment Compute centerline-recall/occupancy-precision F-beta.

        Input: predicted and GT RGB/luma BCHW tensors plus full-resolution GT
        text and optional valid masks; optional flags request detached details.
        Output: scalar differentiable loss, or that loss with per-sample maps
        and diagnostics; prediction receives gradients while GT/masks do not.
        Key Steps: build GT skeleton/tube, infer GT-local polarity, convert
        prediction luminance to soft ink, combine weighted recall and precision,
        and gate inactive low-contrast samples.
        Key Params: constructor morphology/polarity/temperature and beta tune
        tolerance, recall emphasis, and false/overgrowth penalties.
        State/Dependencies: torch autograd and morphology helpers only; no
        trainer, generator-input mask, optimizer, or shared mutable state.
        """
        if not isinstance(pred_rgb, torch.Tensor) or not isinstance(target_rgb, torch.Tensor):
            raise ValueError("pred_rgb and target_rgb must be tensors")
        if pred_rgb.ndim != 4 or target_rgb.ndim != 4 or pred_rgb.shape != target_rgb.shape:
            raise ValueError("pred_rgb and target_rgb must have matching BCHW shapes")
        if pred_rgb.shape[1] not in (1, 3):
            raise ValueError("pred_rgb and target_rgb must have one or three channels")
        if not pred_rgb.is_floating_point():
            pred_rgb = pred_rgb.float()
        if not target_rgb.is_floating_point():
            target_rgb = target_rgb.float()
        valid = _as_bchw(valid_mask_rgb, pred_rgb, "valid_mask_rgb", default=1.0)
        text = _as_bchw(text_mask_rgb, pred_rgb, "text_mask_rgb")
        valid = (valid > float(self.skeleton_threshold)).to(pred_rgb.dtype)
        text = (text > float(self.skeleton_threshold)).to(pred_rgb.dtype) * valid

        skeleton, tube, skeleton_weight, text = build_gt_skeleton(
            text,
            valid,
            threshold=0.5,
            max_iterations=self.max_skeleton_iterations,
            tube_radius=self.tube_radius,
        )
        target_luma = _luma(target_rgb.detach())
        polarity, midpoint, contrast, active, ring = _local_polarity_midpoint(
            target_luma,
            text,
            valid,
            radius=self.polarity_radius,
            min_contrast=self.min_polarity_contrast,
        )
        prediction_luma = _luma(pred_rgb)
        ink_probability = soft_ink_probability(
            prediction_luma,
            midpoint.to(dtype=prediction_luma.dtype),
            polarity.to(dtype=prediction_luma.dtype),
            temperature=self.temperature,
        )

        # Centerline recall: inverse-radius weighting makes a 1--2 px GT
        # component contribute more than a broad, already-easy component.
        skeleton_mass = (skeleton_weight * skeleton).sum(dim=(1, 2, 3)).clamp_min(self.eps)
        recall = (ink_probability * skeleton_weight * skeleton).sum(dim=(1, 2, 3)) / skeleton_mass

        # Occupancy precision counts external ink as false positive.  A small
        # explicit overgrowth term also discourages filling the whole GT glyph
        # when the tolerated tube is much thinner than its occupancy envelope.
        true_positive = (ink_probability * text).sum(dim=(1, 2, 3))
        false_external = (ink_probability * valid * (1.0 - text)).sum(dim=(1, 2, 3))
        overgrowth = (ink_probability * text * (1.0 - tube)).sum(dim=(1, 2, 3))
        precision_denominator = (
            true_positive + false_external + self.overgrowth_weight * overgrowth
        ).clamp_min(self.eps)
        precision = true_positive / precision_denominator
        beta_squared = self.beta * self.beta
        fbeta = (1.0 + beta_squared) * precision * recall / (
            beta_squared * precision + recall
        ).clamp_min(self.eps)
        per_sample_loss = (1.0 - fbeta) * float(self.loss_weight)
        active_vector = active.reshape(active.shape[0], -1).amax(dim=1)
        loss = (per_sample_loss * active_vector).sum() / active_vector.sum().clamp_min(1.0)
        # Keep zero-text/low-contrast batches connected to the graph without
        # allowing inactive samples to contribute a hidden constant.
        loss = loss + pred_rgb.sum() * 0.0

        if not return_details:
            return loss
        details: dict[str, torch.Tensor] = {
            "skeleton_recall": recall.detach(),
            "occupancy_precision": precision.detach(),
            "fbeta": fbeta.detach(),
            "active": active_vector.detach(),
            "polarity": polarity.reshape(polarity.shape[0]).detach(),
            "midpoint": midpoint.reshape(midpoint.shape[0]).detach(),
            "contrast": contrast.reshape(contrast.shape[0]).detach(),
            "ring": ring.detach(),
            "skeleton_pixels": skeleton.sum(dim=(1, 2, 3)).detach(),
            "tube_pixels": tube.sum(dim=(1, 2, 3)).detach(),
            "false_external_mass": false_external.detach(),
            "overgrowth_mass": overgrowth.detach(),
            "valid_pixels": valid.sum(dim=(1, 2, 3)).detach(),
            "text_pixels": text.sum(dim=(1, 2, 3)).detach(),
        }
        if return_maps:
            details.update({
                "skeleton": skeleton.detach(),
                "tube": tube.detach(),
                "text": text.detach(),
                "valid": valid.detach(),
                "ink_probability": ink_probability.detach(),
            })
        return loss, details


# Short alias for config/document terminology.  The long class name remains
# the registry key so the purpose is explicit in BasicSR logs.
SkelFbetaLoss = ThinSkeletonFbetaLoss


__all__ = [
    "ThinSkeletonFbetaLoss",
    "SkelFbetaLoss",
    "build_gt_skeleton",
    "soft_ink_probability",
]
