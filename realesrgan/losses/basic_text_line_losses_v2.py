"""Padding-aware BASIC 2.0 losses."""

from __future__ import annotations

import torch
from torch import nn
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MaskedL1LossV2Basic(nn.Module):
    """Normalize reconstruction error only over valid, non-padding pixels."""

    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean", require_valid_mask: bool = True, eps: float = 1e-6):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.require_valid_mask = bool(require_valid_mask)
        self.eps = float(eps)

    def forward(self, pred, target, mask=None, valid_mask=None, text_mask=None, weight=None, **kwargs):
        """@shanlan-comment Compute channel-aware L1 with valid_mask-only normalization."""
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
        if valid_mask is None:
            valid_mask = mask
        if valid_mask is None and self.require_valid_mask:
            raise ValueError("valid_mask is required")
        region = torch.ones_like(pred[:, :1]) if valid_mask is None else valid_mask.to(pred)
        while region.ndim < pred.ndim:
            region = region.unsqueeze(1)
        if region.shape[1] != 1:
            region = region[:, :1]
        if text_mask is not None:
            text = text_mask.to(pred)
            while text.ndim < pred.ndim:
                text = text.unsqueeze(1)
            region_weight = torch.where(text > 0, torch.ones_like(text), torch.ones_like(text))
            region = region * region_weight
        error = (pred - target).abs() * region
        if weight is not None:
            error = error * weight.to(pred)
        if self.reduction == "none":
            return error * self.loss_weight
        if self.reduction == "sum":
            return error.sum() * self.loss_weight
        denom = region.sum() * pred.shape[1] + self.eps
        return error.sum() / denom * self.loss_weight


@LOSS_REGISTRY.register()
class YStructureCharbonnierV2(nn.Module):
    """Apply Charbonnier to Y/structure regions while preserving valid-mask semantics."""

    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-3, text_weight: float = 1.0, background_weight: float = 1.0):
        super().__init__()
        self.loss_weight, self.eps = float(loss_weight), float(eps)
        self.text_weight, self.background_weight = float(text_weight), float(background_weight)

    def forward(self, pred, target, valid_mask=None, text_mask=None, **kwargs):
        """@shanlan-comment Compute Y-channel Charbonnier over valid pixels."""
        if pred.shape != target.shape or pred.shape[1] < 1:
            raise ValueError("pred and target must match and contain Y")
        region = torch.ones_like(pred[:, :1]) if valid_mask is None else valid_mask.to(pred)
        text = torch.ones_like(region) if text_mask is None else text_mask.to(pred)
        weight = torch.where(text > 0, self.text_weight, self.background_weight) * region
        value = torch.sqrt((pred[:, :1] - target[:, :1]).square() + self.eps ** 2)
        return (value * weight).sum() / (weight.sum() + 1e-6) * self.loss_weight
