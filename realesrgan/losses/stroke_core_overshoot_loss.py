"""Training-only centerline and local-envelope losses for text restoration.

The two losses in this module deliberately consume the already available
full-resolution RGB target and structure masks.  They do not add a model
head and are not used by inference/export code:

* :class:`StrokeCoreLoss` anchors the predicted luminance on a detached GT
  skeleton, which directly supervises fading/uneven ink on the stroke core.
* :class:`OvershootLoss` penalizes only values outside a GT local min/max
  envelope.  The envelope is built with a validity-aware pool so right
  padding cannot become a false reference or receive prediction gradients.

Both implementations keep all morphology/reference tensors detached.  The
only autograd path is from the prediction tensor to the scalar loss.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

from .thin_skeleton_fbeta_loss import build_gt_skeleton


def _as_bchw_mask(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    name: str,
    *,
    default: float | None = None,
) -> torch.Tensor:
    """Normalize a one-channel mask and detach it from autograd."""

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
    """Convert a one- or three-channel tensor to Rec.601 luminance."""

    if value.ndim != 4 or value.shape[1] not in (1, 3):
        raise ValueError(f"value must be BCHW with one or three channels, got {tuple(value.shape)}")
    if value.shape[1] == 1:
        return value
    return 0.299 * value[:, :1] + 0.587 * value[:, 1:2] + 0.114 * value[:, 2:3]


def _validate_pair(pred: torch.Tensor, target: torch.Tensor, name: str) -> None:
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError(f"{name} inputs must be tensors")
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(f"{name} inputs must be BCHW tensors")
    if pred.shape[0] != target.shape[0] or tuple(pred.shape[-2:]) != tuple(target.shape[-2:]):
        raise ValueError(f"{name} prediction and target must have matching batch/spatial shapes")
    if pred.shape[1] not in (1, 3) or target.shape[1] not in (1, 3):
        raise ValueError(f"{name} prediction and target must have one or three channels")


def _zero_connected(value: torch.Tensor) -> torch.Tensor:
    """Return a differentiable scalar zero tied to ``value``."""

    return value.sum() * 0.0


@LOSS_REGISTRY.register()
class StrokeCoreLoss(nn.Module):
    """GT-skeleton luminance L1 for preventing weak or uneven stroke cores.

    ``text_mask_rgb`` and ``valid_mask_rgb`` are thresholded and detached,
    then a bounded morphological GT skeleton is built.  The loss is exactly
    the mean luminance absolute error over skeleton pixels; padding and
    non-skeleton pixels contribute neither value nor prediction gradient.
    """

    def __init__(
        self,
        max_skeleton_iterations: int = 96,
        skeleton_threshold: float = 0.5,
        loss_weight: float = 1.0,
        weight: float | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.max_skeleton_iterations = int(max_skeleton_iterations)
        self.skeleton_threshold = float(skeleton_threshold)
        self.loss_weight = float(loss_weight if weight is None else weight)
        self.eps = float(eps)
        if self.max_skeleton_iterations <= 0:
            raise ValueError("max_skeleton_iterations must be positive")
        if not 0.0 <= self.skeleton_threshold <= 1.0:
            raise ValueError("skeleton_threshold must be in [0, 1]")
        if self.loss_weight < 0.0 or self.eps <= 0.0:
            raise ValueError("loss_weight must be non-negative and eps must be positive")

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        return_details: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        _validate_pair(pred_rgb, gt_rgb, "StrokeCoreLoss")
        prediction_y = _luma(pred_rgb)
        target_y = _luma(gt_rgb.detach())
        text = _as_bchw_mask(text_mask_rgb, prediction_y, "text_mask_rgb")
        valid = _as_bchw_mask(valid_mask_rgb, prediction_y, "valid_mask_rgb", default=1.0)
        valid_binary = (valid > self.skeleton_threshold).to(prediction_y.dtype)
        text_binary = (text > self.skeleton_threshold).to(prediction_y.dtype) * valid_binary

        skeleton, _, _, _ = build_gt_skeleton(
            text_binary,
            valid_binary,
            threshold=0.5,
            max_iterations=self.max_skeleton_iterations,
            tube_radius=0,
        )
        skeleton = skeleton.to(dtype=prediction_y.dtype).detach()
        skeleton_pixels = skeleton.sum().detach()
        error = (prediction_y - target_y).abs()
        denominator = skeleton_pixels.clamp_min(self.eps)
        loss = self.loss_weight * (error * skeleton).sum() / denominator
        if float(skeleton_pixels.item()) <= 0.0:
            loss = _zero_connected(prediction_y)

        if not return_details:
            return loss
        per_sample_pixels = skeleton.sum(dim=(1, 2, 3)).detach()
        per_sample_error = (error * skeleton).sum(dim=(1, 2, 3)).detach() / per_sample_pixels.clamp_min(self.eps)
        details: dict[str, torch.Tensor] = {
            "skeleton": skeleton,
            "valid": valid_binary.detach(),
            "text": text_binary.detach(),
            "skeleton_pixels": per_sample_pixels,
            "core_abs_error": per_sample_error,
            "active_samples": (per_sample_pixels > 0.0).to(prediction_y.dtype),
        }
        return loss, details


def _valid_local_max_min(
    reference: torch.Tensor,
    valid: torch.Tensor,
    kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a validity-aware local max/min envelope from a detached target."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    valid = (valid > 0.5).to(dtype=reference.dtype)
    neg_inf = torch.finfo(reference.dtype).min
    pos_inf = torch.finfo(reference.dtype).max
    max_source = torch.where(valid > 0.0, reference, reference.new_full((), neg_inf))
    min_source = torch.where(valid > 0.0, reference, reference.new_full((), pos_inf))
    radius = kernel_size // 2
    local_max = F.max_pool2d(max_source, kernel_size, stride=1, padding=radius)
    local_min = -F.max_pool2d(-min_source, kernel_size, stride=1, padding=radius)
    # A fully invalid window has no meaningful envelope.  Keep a finite
    # placeholder there; the caller gates these pixels out, but avoiding
    # +/-finfo.max also prevents ``0 * inf`` from becoming NaN before the
    # gate is reduced.
    support = F.max_pool2d(valid, kernel_size, stride=1, padding=radius)
    local_max = torch.where(support > 0.0, local_max, reference)
    local_min = torch.where(support > 0.0, local_min, reference)
    return local_min.detach(), local_max.detach()


@LOSS_REGISTRY.register()
class OvershootLoss(nn.Module):
    """One-sided penalty for prediction values outside a GT local envelope.

    The default gate is the valid text neighborhood: GT text is dilated by
    ``mask_radius`` so both the stroke and its immediate halo are checked,
    while ``valid_mask_rgb`` removes right padding.  Set ``mask_mode='valid'``
    to reproduce a full-valid-area envelope loss, or pass ``mask=...`` for an
    additional custom gate.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        tol: float = 2.0 / 255.0,
        mask_radius: int = 2,
        mask_mode: str = "text_dilate",
        mask_threshold: float = 0.5,
        channel: str = "luma",
        loss_weight: float = 1.0,
        weight: float | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.tol = float(tol)
        self.mask_radius = int(mask_radius)
        self.mask_mode = str(mask_mode).strip().lower()
        self.mask_threshold = float(mask_threshold)
        self.channel = str(channel).strip().lower()
        self.loss_weight = float(loss_weight if weight is None else weight)
        self.eps = float(eps)
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.tol < 0.0 or self.mask_radius < 0:
            raise ValueError("tol and mask_radius must be non-negative")
        if self.mask_mode not in {"text", "text_dilate", "valid"}:
            raise ValueError("mask_mode must be 'text', 'text_dilate', or 'valid'")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        if self.channel not in {"luma", "rgb"}:
            raise ValueError("channel must be 'luma' or 'rgb'")
        if self.loss_weight < 0.0 or self.eps <= 0.0:
            raise ValueError("loss_weight must be non-negative and eps must be positive")

    def _active_gate(
        self,
        text: torch.Tensor,
        valid: torch.Tensor,
        custom_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid_binary = (valid > self.mask_threshold).to(dtype=valid.dtype)
        if self.mask_mode == "valid":
            gate = valid_binary
        else:
            text_binary = (text > self.mask_threshold).to(dtype=valid.dtype) * valid_binary
            if self.mask_mode == "text":
                gate = text_binary
            elif self.mask_radius == 0:
                gate = text_binary
            else:
                gate = F.max_pool2d(
                    text_binary,
                    kernel_size=2 * self.mask_radius + 1,
                    stride=1,
                    padding=self.mask_radius,
                )
                gate = (gate > 0.0).to(dtype=valid.dtype) * valid_binary
        if custom_mask is not None:
            custom = _as_bchw_mask(custom_mask, valid, "mask")
            gate = gate * (custom > self.mask_threshold).to(dtype=gate.dtype)
        return gate.detach()

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
        return_details: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        _validate_pair(pred_rgb, gt_rgb, "OvershootLoss")
        prediction = _luma(pred_rgb) if self.channel == "luma" else pred_rgb
        reference = _luma(gt_rgb.detach()) if self.channel == "luma" else gt_rgb.detach()
        text = _as_bchw_mask(text_mask_rgb, prediction[:, :1], "text_mask_rgb")
        valid = _as_bchw_mask(valid_mask_rgb, prediction[:, :1], "valid_mask_rgb", default=1.0)
        gate = self._active_gate(text, valid, mask)
        local_min, local_max = _valid_local_max_min(reference, valid, self.kernel_size)

        upper = F.relu(prediction - (local_max + self.tol))
        lower = F.relu((local_min - self.tol) - prediction)
        over = upper + lower
        gate_expanded = gate.expand(-1, over.shape[1], -1, -1)
        active_pixels = gate_expanded.sum().detach()
        loss = self.loss_weight * (over * gate_expanded).sum() / active_pixels.clamp_min(self.eps)
        if float(active_pixels.item()) <= 0.0:
            loss = _zero_connected(prediction)

        if not return_details:
            return loss
        details: dict[str, torch.Tensor] = {
            "gate": gate.detach(),
            "valid": (valid > self.mask_threshold).to(dtype=gate.dtype).detach(),
            "text": (text > self.mask_threshold).to(dtype=gate.dtype).detach(),
            "local_min": local_min.detach(),
            "local_max": local_max.detach(),
            "upper_overshoot": (upper * gate_expanded).detach(),
            "lower_overshoot": (lower * gate_expanded).detach(),
            "active_pixels": gate.sum(dim=(1, 2, 3)).detach(),
            "upper_active_fraction": ((upper > 0.0).to(gate_expanded.dtype) * gate_expanded).sum(
                dim=(1, 2, 3)
            ).detach()
            / gate_expanded.sum(dim=(1, 2, 3)).clamp_min(self.eps).detach(),
            "lower_active_fraction": ((lower > 0.0).to(gate_expanded.dtype) * gate_expanded).sum(
                dim=(1, 2, 3)
            ).detach()
            / gate_expanded.sum(dim=(1, 2, 3)).clamp_min(self.eps).detach(),
        }
        return loss, details


# Descriptive alias for configs/documentation that prefer a stroke-prefixed
# name.  The registry key remains the reference-compatible ``OvershootLoss``.
StrokeOvershootLoss = OvershootLoss


__all__ = ["StrokeCoreLoss", "OvershootLoss", "StrokeOvershootLoss"]
