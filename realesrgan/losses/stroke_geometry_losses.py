"""Training-only differentiable stroke topology and width losses.

The losses in this module consume the already available RGB restoration output,
the paired GT RGB target, and detached text/valid masks.  They do not add a
foreground head and they never feed masks into the generator.  GT morphology
is detached; autograd is retained only through the predicted luminance.

``SoftCLDiceLoss`` follows the toolkit definition: GT skeleton coverage is the
recall direction and prediction skeleton precision is the anti-spur direction.
``StrokeWidthLoss`` compares soft erosion distances at detached GT skeleton
pixels, which directly penalizes a locally too-thin or too-wide stroke.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

from .thin_skeleton_fbeta_loss import (
    _luma,
    _local_polarity_midpoint,
    build_gt_skeleton,
)


def _as_bchw_mask(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    name: str,
    *,
    default: float | None = None,
) -> torch.Tensor:
    """@luna-comment Normalize a detached structure mask to BCHW.

    Input: an optional BHW/B1HW mask and a floating prediction reference.
    Output: detached float B1HW mask on the reference device.
    Key Steps: cast, add a singleton channel, validate dimensions, detach.
    Key Params: ``default`` is used only for an omitted valid mask.
    State/Dependencies: tensor metadata only; no model or dataset state.
    """
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return reference.new_full((reference.shape[0], 1, *reference.shape[-2:]), float(default))
    result = value.to(device=reference.device, dtype=reference.dtype)
    if result.ndim == 3:
        result = result.unsqueeze(1)
    expected = (reference.shape[0], 1, *reference.shape[-2:])
    if tuple(result.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(result.shape)}")
    return result.detach()


@LOSS_REGISTRY.register()
class StrokeProbabilityBCELoss(nn.Module):
    """Training-only BCE supervision for the V4 1x1 stroke-logit branch."""

    def __init__(self, mask_threshold: float = 0.5, loss_weight: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.mask_threshold, self.loss_weight, self.eps = float(mask_threshold), float(loss_weight), float(eps)

    def forward(self, logits, text_mask_rgb, valid_mask_rgb=None, *, return_details=False, **_):
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("stroke logits must be [B,1,H,W]")
        text = _as_bchw_mask(text_mask_rgb, logits, "text_mask_rgb")
        valid = _as_bchw_mask(valid_mask_rgb, logits, "valid_mask_rgb", default=1.0)
        gate = (valid > self.mask_threshold).to(logits.dtype)
        target = (text > self.mask_threshold).to(logits.dtype) * gate
        denominator = gate.sum().detach().clamp_min(self.eps)
        loss = self.loss_weight * (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * gate).sum() / denominator
        if not return_details:
            return loss
        pred = (logits.sigmoid() >= 0.5).to(logits.dtype) * gate
        inter = (pred * target).sum(dim=(1, 2, 3)); pred_mass = pred.sum(dim=(1, 2, 3)); target_mass = target.sum(dim=(1, 2, 3))
        return loss, {
            "active_pixels": gate.sum(dim=(1, 2, 3)).detach(),
            "precision": (inter / pred_mass.clamp_min(self.eps)).detach(),
            "recall": (inter / target_mass.clamp_min(self.eps)).detach(),
            "iou": (inter / (pred_mass + target_mass - inter).clamp_min(self.eps)).detach(),
        }


def _erode_zero_boundary(mask: torch.Tensor) -> torch.Tensor:
    """@luna-comment Apply one differentiable 3x3 erosion with zero outside.

    Input: BCHW binary or soft occupancy map.
    Output: same-shaped erosion with image-boundary pixels requiring full support.
    Key Steps: zero-pad the map, max-pool its negative, then negate back.
    Key Params: fixed 3x3 neighborhood defines one pixel of soft distance.
    State/Dependencies: ``torch.nn.functional`` pooling only.
    """
    padded = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0.0)
    return -F.max_pool2d(-padded, kernel_size=3, stride=1)


def _dilate_zero_boundary(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """@luna-comment Apply square max-pool dilation without image wraparound.

    Input: BCHW binary or soft occupancy map and non-negative radius.
    Output: same-shaped dilated map; values outside the canvas are zero.
    Key Steps: zero-pad explicitly before max pooling and crop to the canvas.
    Key Params: radius controls the morphological neighborhood.
    State/Dependencies: pooling only; no shared mutable state.
    """
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask
    padded = F.pad(mask, (radius, radius, radius, radius), mode="constant", value=0.0)
    return F.max_pool2d(padded, kernel_size=2 * radius + 1, stride=1)


def soft_distance(occupancy: torch.Tensor, max_radius: int = 8) -> torch.Tensor:
    """@luna-comment Estimate a differentiable Chebyshev soft half-width.

    Input: BCHW soft occupancy in [0,1] and positive iteration cap.
    Output: sum of repeatedly eroded occupancies, with gradients to occupancy.
    Key Steps: repeatedly erode and accumulate; unlike a hard distance
        transform, max-pool argmax routing remains differentiable.
    Key Params: ``max_radius`` caps work and the largest supervised width.
    State/Dependencies: only the prediction occupancy participates in autograd.
    """
    max_radius = int(max_radius)
    if max_radius <= 0:
        raise ValueError("max_radius must be positive")
    result = torch.zeros_like(occupancy)
    current = occupancy
    for _ in range(max_radius):
        current = _erode_zero_boundary(current)
        result = result + current
    return result


def soft_skeleton(occupancy: torch.Tensor, iterations: int = 6) -> torch.Tensor:
    """@luna-comment Extract a differentiable morphological soft skeleton.

    Input: BCHW soft occupancy and a positive number of peeling iterations.
    Output: BCHW soft centerline map with gradients to occupancy.
    Key Steps: retain each erosion/opening shell and suppress duplicate shells.
    Key Params: iterations should cover the largest expected half-stroke width.
    State/Dependencies: pooling/autograd only; no thresholded prediction mask.
    """
    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    current = occupancy
    eroded = _erode_zero_boundary(current)
    skeleton = F.relu(current - _dilate_zero_boundary(eroded, 1))
    for _ in range(iterations):
        current = _erode_zero_boundary(current)
        eroded = _erode_zero_boundary(current)
        shell = F.relu(current - _dilate_zero_boundary(eroded, 1))
        skeleton = skeleton + F.relu(shell - skeleton * shell)
    return skeleton


def _prepare_inputs(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    text_mask_rgb: torch.Tensor,
    valid_mask_rgb: torch.Tensor | None,
    *,
    mask_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """@luna-comment Validate RGB/mask shapes and create GT-derived maps.

    Input: predicted/GT RGB tensors and paired text/valid masks.
    Output: predicted luminance, detached GT luminance, binary text, valid,
        and GT-local activity gate.
    Key Steps: validate full-resolution alignment, threshold masks, estimate
        detached polarity/midpoint/contrast, and return the activity gate.
    Key Params: mask_threshold binarizes the supplied dataset masks.
    State/Dependencies: target/masks are detached; prediction luminance stays
        connected to autograd.
    """
    if not isinstance(pred_rgb, torch.Tensor) or not isinstance(gt_rgb, torch.Tensor):
        raise ValueError("pred_rgb and gt_rgb must be tensors")
    if pred_rgb.ndim != 4 or gt_rgb.ndim != 4 or pred_rgb.shape != gt_rgb.shape:
        raise ValueError("pred_rgb and gt_rgb must have matching BCHW shapes")
    if pred_rgb.shape[1] not in (1, 3):
        raise ValueError("pred_rgb and gt_rgb must have one or three channels")
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("mask_threshold must be in [0,1]")
    prediction_luma = _luma(pred_rgb)
    target_luma = _luma(gt_rgb.detach())
    text = (_as_bchw_mask(text_mask_rgb, prediction_luma, "text_mask_rgb") > float(mask_threshold)).to(prediction_luma.dtype)
    valid = (_as_bchw_mask(valid_mask_rgb, prediction_luma, "valid_mask_rgb", default=1.0) > float(mask_threshold)).to(prediction_luma.dtype)
    text = text * valid
    _, _, contrast, active, _ = _local_polarity_midpoint(
        target_luma,
        text,
        valid,
        radius=3,
        min_contrast=0.0,
    )
    # Re-evaluate the hard activity conditions here so low-contrast/empty
    # tiles contribute an exact graph-connected zero and no fabricated stroke.
    text_mass = text.sum(dim=(2, 3), keepdim=True)
    valid_mass = valid.sum(dim=(2, 3), keepdim=True)
    active = ((text_mass > 0.0) & (valid_mass > 0.0) & (contrast.abs() >= 0.01)).to(prediction_luma.dtype)
    return prediction_luma, target_luma, text, valid, active


def _local_ink_probability(
    prediction_luma: torch.Tensor,
    target_luma: torch.Tensor,
    text: torch.Tensor,
    valid: torch.Tensor,
    *,
    anchor_radius: int,
    min_contrast: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """@luna-comment Convert prediction luminance to a GT-oriented ink map.

    Input: prediction/GT luminance and detached binary text/valid masks.
    Output: differentiable ink probability plus detached polarity, midpoint,
        contrast, and active sample gate.
    Key Steps: estimate GT-local anchors and apply the existing sigmoid map.
    Key Params: anchor radius, minimum contrast, and temperature control the
        polarity support and softness.
    State/Dependencies: only prediction luminance carries gradients.
    """
    polarity, midpoint, contrast, active, _ = _local_polarity_midpoint(
        target_luma,
        text,
        valid,
        radius=max(0, int(anchor_radius)),
        min_contrast=float(min_contrast),
    )
    # Normalize temperature by the detached local contrast.  Without this
    # normalization a perfectly reproduced low-contrast GT stroke maps to an
    # intermediate probability (for example 0.88), causing the auxiliary loss
    # to punish the GT itself.  At the GT foreground/background anchors the
    # normalized logit is approximately +/-1/(2*temperature), so identity is
    # close to zero while the map remains differentiable for predictions.
    scale = float(temperature) * contrast.abs().clamp_min(float(min_contrast)).to(
        dtype=prediction_luma.dtype
    )
    ink_probability = torch.sigmoid(
        polarity.to(dtype=prediction_luma.dtype)
        * (midpoint.to(dtype=prediction_luma.dtype) - prediction_luma)
        / scale.clamp_min(torch.finfo(prediction_luma.dtype).eps)
    ) * valid
    return ink_probability, polarity, midpoint, contrast, active


@LOSS_REGISTRY.register()
class SoftCLDiceLoss(nn.Module):
    """Bidirectional differentiable topology loss for text strokes.

    The GT skeleton coverage term penalizes broken/missing strokes.  The
    prediction skeleton precision term penalizes spurs and false branches.
    An optional soft occupancy Dice blend keeps the objective stable on tiny
    strokes, matching the TextHD toolkit implementation.
    """

    def __init__(
        self,
        skeleton_iterations: int = 6,
        alpha: float = 0.4,
        anchor_radius: int = 3,
        # Keep the gate below the regime of visually weak but annotated text.
        # Contrast normalization preserves the GT fixed point and scales the
        # residual gradient instead of dropping a 1--3% stroke altogether.
        min_contrast: float = 0.01,
        ink_temperature: float = 0.05,
        mask_threshold: float = 0.5,
        max_skeleton_iterations: int = 96,
        tube_radius: int = 0,
        loss_weight: float = 1.0,
        eps: float = 1.0e-6,
    ) -> None:
        """@luna-comment Configure the clDice morphology and GT anchor gate."""
        super().__init__()
        self.skeleton_iterations = int(skeleton_iterations)
        self.alpha = float(alpha)
        self.anchor_radius = int(anchor_radius)
        self.min_contrast = float(min_contrast)
        self.ink_temperature = float(ink_temperature)
        self.mask_threshold = float(mask_threshold)
        self.max_skeleton_iterations = int(max_skeleton_iterations)
        self.tube_radius = int(tube_radius)
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        if self.skeleton_iterations <= 0 or self.max_skeleton_iterations <= 0:
            raise ValueError("skeleton iteration limits must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        if self.anchor_radius < 0 or self.tube_radius < 0:
            raise ValueError("anchor_radius/tube_radius must be non-negative")
        if self.min_contrast < 0.0 or self.ink_temperature <= 0.0:
            raise ValueError("min_contrast must be non-negative and ink_temperature positive")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0,1]")
        if self.loss_weight < 0.0 or self.eps <= 0.0:
            raise ValueError("loss_weight must be non-negative and eps positive")

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        pred_probability: torch.Tensor | None = None,
        return_details: bool = False,
        return_maps: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """@luna-comment Compute GT-skeleton recall and prediction precision.

        Input: paired RGB restoration tensors and detached full-resolution
            structure masks.
        Output: scalar differentiable loss, optionally with detached metrics.
        Key Steps: derive GT anchors, build detached GT skeleton, extract a
            soft prediction skeleton, then average gated bidirectional terms.
        Key Params: constructor morphology and anchor settings.
        State/Dependencies: no model/mask input path is created; only pred RGB
            participates in autograd.
        """
        prediction_luma, target_luma, text, valid, _ = _prepare_inputs(
            pred_rgb,
            gt_rgb,
            text_mask_rgb,
            valid_mask_rgb,
            mask_threshold=self.mask_threshold,
        )
        ink, polarity, midpoint, contrast, active = _local_ink_probability(
            prediction_luma,
            target_luma,
            text,
            valid,
            anchor_radius=self.anchor_radius,
            min_contrast=self.min_contrast,
            temperature=self.ink_temperature,
        )
        if pred_probability is not None:
            if tuple(pred_probability.shape) != tuple(ink.shape):
                raise ValueError("pred_probability must match full-resolution luma shape")
            ink = pred_probability.clamp(0.0, 1.0) * valid
        gt_skeleton, gt_tube, _, _ = build_gt_skeleton(
            text,
            valid,
            threshold=0.5,
            max_iterations=self.max_skeleton_iterations,
            tube_radius=self.tube_radius,
        )
        prediction_skeleton = soft_skeleton(ink, self.skeleton_iterations) * valid
        gt_mass = gt_skeleton.sum(dim=(1, 2, 3))
        pred_mass = prediction_skeleton.sum(dim=(1, 2, 3))
        gt_coverage = (gt_skeleton * ink).sum(dim=(1, 2, 3)) / gt_mass.clamp_min(self.eps)
        pred_precision = (prediction_skeleton * text).sum(dim=(1, 2, 3)) / pred_mass.clamp_min(self.eps)
        cldice_score = 2.0 * gt_coverage * pred_precision / (
            gt_coverage + pred_precision
        ).clamp_min(self.eps)
        dice_inter = (ink * text).sum(dim=(1, 2, 3))
        dice_score = 2.0 * dice_inter / (
            ink.sum(dim=(1, 2, 3)) + text.sum(dim=(1, 2, 3))
        ).clamp_min(self.eps)
        per_sample = (1.0 - self.alpha) * (1.0 - dice_score) + self.alpha * (1.0 - cldice_score)
        active_vector = active.reshape(active.shape[0], -1).amax(dim=1)
        loss = self.loss_weight * (per_sample * active_vector).sum() / active_vector.sum().clamp_min(1.0)
        loss = loss + prediction_luma.sum() * 0.0
        if not return_details:
            return loss
        details: dict[str, torch.Tensor] = {
            "gt_coverage": gt_coverage.detach(),
            "pred_precision": pred_precision.detach(),
            "cldice_score": cldice_score.detach(),
            "dice_score": dice_score.detach(),
            "active": active_vector.detach(),
            "polarity": polarity.reshape(polarity.shape[0]).detach(),
            "midpoint": midpoint.reshape(midpoint.shape[0]).detach(),
            "contrast": contrast.reshape(contrast.shape[0]).detach(),
            "gt_skeleton_pixels": gt_mass.detach(),
            "pred_skeleton_mass": pred_mass.detach(),
            "valid_pixels": valid.sum(dim=(1, 2, 3)).detach(),
            "text_pixels": text.sum(dim=(1, 2, 3)).detach(),
        }
        if return_maps:
            details.update({
                "gt_skeleton": gt_skeleton.detach(),
                "gt_tube": gt_tube.detach(),
                "prediction_skeleton": prediction_skeleton.detach(),
                "ink_probability": ink.detach(),
                "text": text.detach(),
                "valid": valid.detach(),
            })
        return loss, details


@LOSS_REGISTRY.register()
class StrokeWidthLoss(nn.Module):
    """GT-skeleton soft distance width consistency loss.

    The detached GT skeleton selects where width is measured.  Soft erosion
    distances remain connected to predicted luminance, so the loss does not
    turn into a brittle thresholded width metric.
    """

    def __init__(
        self,
        max_radius: int = 8,
        anchor_radius: int = 3,
        # Match SoftCLDiceLoss: retain supervision for weak, but still
        # measurable, annotated strokes rather than treating them as absent.
        min_contrast: float = 0.01,
        ink_temperature: float = 0.05,
        mask_threshold: float = 0.5,
        max_skeleton_iterations: int = 96,
        tube_radius: int = 0,
        loss_weight: float = 1.0,
        eps: float = 1.0e-6,
    ) -> None:
        """@luna-comment Configure soft-distance width supervision."""
        super().__init__()
        self.max_radius = int(max_radius)
        self.anchor_radius = int(anchor_radius)
        self.min_contrast = float(min_contrast)
        self.ink_temperature = float(ink_temperature)
        self.mask_threshold = float(mask_threshold)
        self.max_skeleton_iterations = int(max_skeleton_iterations)
        self.tube_radius = int(tube_radius)
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        if self.max_radius <= 0 or self.max_skeleton_iterations <= 0:
            raise ValueError("max_radius and max_skeleton_iterations must be positive")
        if self.anchor_radius < 0 or self.tube_radius < 0:
            raise ValueError("anchor_radius/tube_radius must be non-negative")
        if self.min_contrast < 0.0 or self.ink_temperature <= 0.0:
            raise ValueError("min_contrast must be non-negative and ink_temperature positive")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0,1]")
        if self.loss_weight < 0.0 or self.eps <= 0.0:
            raise ValueError("loss_weight must be non-negative and eps positive")

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        pred_probability: torch.Tensor | None = None,
        return_details: bool = False,
        return_maps: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """@luna-comment Compare predicted and GT soft widths on GT skeleton.

        Input: paired RGB output/target and full-resolution structure masks.
        Output: scalar differentiable width loss with optional detached maps.
        Key Steps: derive GT-oriented ink, build GT skeleton, calculate soft
            erosion distances, and average absolute width error on valid pixels.
        Key Params: max_radius controls the supervised half-width range.
        State/Dependencies: GT morphology is detached; prediction gradients
            flow through luminance, erosion, and the skeleton-weighted error.
        """
        prediction_luma, target_luma, text, valid, _ = _prepare_inputs(
            pred_rgb,
            gt_rgb,
            text_mask_rgb,
            valid_mask_rgb,
            mask_threshold=self.mask_threshold,
        )
        ink, polarity, midpoint, contrast, active = _local_ink_probability(
            prediction_luma,
            target_luma,
            text,
            valid,
            anchor_radius=self.anchor_radius,
            min_contrast=self.min_contrast,
            temperature=self.ink_temperature,
        )
        if pred_probability is not None:
            if tuple(pred_probability.shape) != tuple(ink.shape):
                raise ValueError("pred_probability must match full-resolution luma shape")
            ink = pred_probability.clamp(0.0, 1.0) * valid
        gt_skeleton, gt_tube, _, gt_text = build_gt_skeleton(
            text,
            valid,
            threshold=0.5,
            max_iterations=self.max_skeleton_iterations,
            tube_radius=self.tube_radius,
        )
        predicted_distance = soft_distance(ink, self.max_radius)
        gt_distance = soft_distance(gt_text, self.max_radius).detach()
        width_error = (predicted_distance - gt_distance).abs()
        skeleton_mass = gt_skeleton.sum(dim=(1, 2, 3))
        per_sample = (width_error * gt_skeleton).sum(dim=(1, 2, 3)) / skeleton_mass.clamp_min(self.eps)
        active_vector = active.reshape(active.shape[0], -1).amax(dim=1)
        loss = self.loss_weight * (per_sample * active_vector).sum() / active_vector.sum().clamp_min(1.0)
        loss = loss + prediction_luma.sum() * 0.0
        if not return_details:
            return loss
        details: dict[str, torch.Tensor] = {
            "width_abs_error": per_sample.detach(),
            "active": active_vector.detach(),
            "polarity": polarity.reshape(polarity.shape[0]).detach(),
            "midpoint": midpoint.reshape(midpoint.shape[0]).detach(),
            "contrast": contrast.reshape(contrast.shape[0]).detach(),
            "gt_skeleton_pixels": skeleton_mass.detach(),
            "predicted_width_at_gt_skeleton": (predicted_distance * gt_skeleton).sum(dim=(1, 2, 3)).div(skeleton_mass.clamp_min(self.eps)).detach(),
            "gt_width_at_gt_skeleton": (gt_distance * gt_skeleton).sum(dim=(1, 2, 3)).div(skeleton_mass.clamp_min(self.eps)).detach(),
            "valid_pixels": valid.sum(dim=(1, 2, 3)).detach(),
            "text_pixels": text.sum(dim=(1, 2, 3)).detach(),
        }
        if return_maps:
            details.update({
                "gt_skeleton": gt_skeleton.detach(),
                "gt_tube": gt_tube.detach(),
                "predicted_distance": predicted_distance.detach(),
                "gt_distance": gt_distance.detach(),
                "ink_probability": ink.detach(),
                "text": text.detach(),
                "valid": valid.detach(),
            })
        return loss, details


__all__ = ["StrokeProbabilityBCELoss", "SoftCLDiceLoss", "StrokeWidthLoss", "soft_distance", "soft_skeleton"]
