"""P0 loss contracts for BASIC 2.0 line-segment training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY


def _spatial_mask(mask, reference, name):
    """@shanlan-comment Normalize one spatial mask to float BCHW with one channel."""
    if mask is None:
        return torch.ones_like(reference[:, :1])
    value = mask.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[0] != reference.shape[0] or value.shape[2:] != reference.shape[2:]:
        raise ValueError(f"{name} must be [B,1,H,W] or [B,H,W], got {tuple(value.shape)}")
    if value.shape[1] != 1:
        value = value[:, :1]
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value.clamp(0.0, 1.0)


def _channel_weights(weights, reference):
    """@shanlan-comment Convert fixed YUV channel weights to broadcastable BCHW."""
    if len(weights) != reference.shape[1]:
        raise ValueError(f"channel_weights length {len(weights)} != channels {reference.shape[1]}")
    value = reference.new_tensor([float(item) for item in weights]).view(1, -1, 1, 1)
    if (value < 0).any():
        raise ValueError("channel_weights must be non-negative")
    return value


@LOSS_REGISTRY.register()
class BoxAwareMaskedL1LossV2(nn.Module):
    """P0 baseline with valid ROI, text-box weighting, and YUV channel weighting."""

    def __init__(self, loss_weight=1.0, reduction="mean", fg_weight=1.0, bg_weight=0.5, channel_weight=(1.0, 0.25, 0.25), eps=1e-6):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unsupported reduction: {reduction}")
        self.loss_weight = float(loss_weight); self.reduction = reduction
        self.fg_weight = float(fg_weight); self.bg_weight = float(bg_weight)
        self.channel_weight = tuple(float(item) for item in channel_weight); self.eps = float(eps)

    def forward(self, pred, target, valid_mask=None, text_mask=None, mask=None, weight=None, **_):
        """@shanlan-comment Compute effective-weight-normalized box-aware YUV L1."""
        if pred.shape != target.shape or pred.ndim != 4:
            raise ValueError("pred and target must match BCHW")
        valid = _spatial_mask(valid_mask if valid_mask is not None else mask, pred, "valid_mask")
        text = _spatial_mask(text_mask, pred, "text_mask") * valid
        region = valid * (text * self.fg_weight + (1.0 - text) * self.bg_weight)
        effective = region * _channel_weights(self.channel_weight, pred)
        error = (pred - target).abs() * effective
        if weight is not None:
            extra = weight.to(device=pred.device, dtype=pred.dtype)
            while extra.ndim < 4: extra = extra.unsqueeze(1)
            error = error * extra; effective = effective * extra
        if self.reduction == "none": return error * self.loss_weight
        if self.reduction == "sum": return error.sum() * self.loss_weight
        return error.sum() / (effective.sum() + self.eps) * self.loss_weight


@LOSS_REGISTRY.register()
class YOnlySobelMaskedLossV2(nn.Module):
    """P0 Y-only Sobel magnitude loss normalized over valid text pixels."""

    def __init__(self, loss_weight=1.0, reduction="mean", eps=1e-6):
        super().__init__()
        if reduction not in ("mean", "sum", "none"): raise ValueError(f"unsupported reduction: {reduction}")
        self.loss_weight = float(loss_weight); self.reduction = reduction; self.eps = float(eps)
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sx.t().view(1, 1, 3, 3), persistent=False)

    def _magnitude(self, value):
        """@shanlan-comment Compute replicated-boundary Sobel magnitude for Y."""
        value = value[:, :1]; padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(padded, self.sobel_x.to(dtype=value.dtype)); gy = F.conv2d(padded, self.sobel_y.to(dtype=value.dtype))
        return torch.sqrt(gx.square() + gy.square() + 1e-12)

    def forward(self, pred, target, valid_mask=None, text_mask=None, mask=None, **_):
        """@shanlan-comment Compare Y Sobel magnitudes only inside valid text pixels."""
        if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] < 1: raise ValueError("invalid YUV tensors")
        valid = _spatial_mask(valid_mask if valid_mask is not None else mask, pred, "valid_mask")
        effective = valid * _spatial_mask(text_mask, pred, "text_mask")
        error = (self._magnitude(pred) - self._magnitude(target)).abs() * effective
        if self.reduction == "none": return error * self.loss_weight
        if self.reduction == "sum": return error.sum() * self.loss_weight
        return error.sum() / (effective.sum() + self.eps) * self.loss_weight


@dataclass(frozen=True)
class CPLineEligibilityV2:
    """Auditable CP eligibility state for one segment record."""
    eligible: bool
    reason: str = ""
    orientation: str = ""
    tile_count: int = 1
    is_crop_truncated: bool = False
    polygon_valid: bool = True
    crop_width: int = 0


def cp_eligibility_v2(metadata: Mapping[str, Any]) -> CPLineEligibilityV2:
    """@shanlan-comment Allow CP only for complete horizontal/oblique single tiles."""
    orientation = str(metadata.get("orientation", "")); tile_count = int(metadata.get("tile_count", 1)); truncated = bool(metadata.get("is_crop_truncated", False)); polygon_valid = bool(metadata.get("polygon_valid", True)); valid_hw = metadata.get("valid_hw", (0, 0)); width = int(metadata.get("crop_width", valid_hw[1] if valid_hw else 0))
    if not polygon_valid: return CPLineEligibilityV2(False, "INVALID_POLYGON", orientation, tile_count, truncated, False, width)
    if tile_count != 1: return CPLineEligibilityV2(False, "SEGMENTED_LINE", orientation, tile_count, truncated, True, width)
    if truncated: return CPLineEligibilityV2(False, "CROP_TRUNCATED", orientation, tile_count, truncated, True, width)
    if orientation not in ("HORIZONTAL", "OBLIQUE"): return CPLineEligibilityV2(False, "VERTICAL_OR_UNSUPPORTED_ORIENTATION", orientation, tile_count, truncated, True, width)
    if width < 16 or width > 1024: return CPLineEligibilityV2(False, "CRNN_WIDTH_OUT_OF_RANGE", orientation, tile_count, truncated, True, width)
    return CPLineEligibilityV2(True, "", orientation, tile_count, truncated, True, width)


class EligibleCPAdapterV2(nn.Module):
    """Apply an existing CP loss to eligible lines while recording skipped reasons."""
    def __init__(self, cp_loss: nn.Module, loss_weight=1.0):
        super().__init__(); self.cp_loss = cp_loss; self.loss_weight = float(loss_weight); self.last_stats = {}

    def forward(self, output_rgb, target_rgb, lines):
        """@shanlan-comment Filter metadata, invoke frozen CP, and keep zero connected otherwise."""
        states = [cp_eligibility_v2(item) for item in lines]; eligible = [item for item, state in zip(lines, states) if state.eligible]; skipped = [state for state in states if not state.eligible]
        self.last_stats = {"eligible_lines": len(eligible), "skipped_lines": len(skipped), "skip_reasons": {reason: sum(state.reason == reason for state in skipped) for reason in sorted({state.reason for state in skipped})}}
        if not eligible: return output_rgb.sum() * 0.0
        return self.cp_loss(output_rgb, target_rgb, [eligible]) * self.loss_weight


__all__ = ["BoxAwareMaskedL1LossV2", "YOnlySobelMaskedLossV2", "CPLineEligibilityV2", "cp_eligibility_v2", "EligibleCPAdapterV2"]
