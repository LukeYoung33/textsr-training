"""Recall-aware structure terms for paired text restoration with stroke masks."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """@luna-comment Input: BCHW RGB. Output: Rec601 Y. Key Steps: fixed weighted sum.
    Key Params: .299/.587/.114. State/Dependencies: differentiable tensor arithmetic only.
    """
    return 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _soft_hinge(value: torch.Tensor, margin: float, temperature: float) -> torch.Tensor:
    """@luna-comment Input: signed residual, margin, temperature. Output: smooth one-sided penalty.
    Key Steps: scaled softplus. Key Params: margin tolerates small mismatch; temperature controls smoothness.
    State/Dependencies: differentiable torch arithmetic only.
    """
    return float(temperature) * F.softplus((value - float(margin)) / float(temperature))


def recall_aware_structure_losses(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    text_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    ring_radius: int = 3,
    margin: float = 0.03,
    temperature: float = 0.01,
    min_polarity_contrast: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """@luna-comment Return missing-stroke and false-stroke losses without thresholding prediction RGB.

    Input: paired BCHW RGB and text/valid masks. Output: text-only missing and outside-ring false losses.
    Key Steps: derive GT polarity/contrast, apply signed smooth hinge and separate region normalization.
    Key Params: ring radius, residual margin, temperature and minimum GT contrast.
    State/Dependencies: torch and local luma/hinge helpers; prediction remains differentiable.

    The GT foreground/ring luminance determines whether the tile contains dark
    or light text.  In the true stroke mask, movement toward the local
    background is penalized.  In the immediately outside ring, movement toward
    the text polarity is penalized.  Thus a weak true stroke and a hallucinated
    neighboring stroke receive different one-sided gradients.
    """
    if text_mask.ndim != 4 or valid_mask.ndim != 4:
        raise ValueError("text_mask and valid_mask must be BCHW")
    text = (text_mask[:, :1] > 0.5).float()
    valid = (valid_mask[:, :1] > 0.5).float()
    ring = (F.max_pool2d(text, 2 * ring_radius + 1, 1, ring_radius) > 0.5).float() * (1.0 - text) * valid
    text = text * valid
    y_pred, y_gt = _luma(pred_rgb), _luma(gt_rgb)
    eps = y_gt.new_tensor(1e-6)
    text_mass = text.sum((2, 3), keepdim=True).clamp_min(eps)
    ring_mass = ring.sum((2, 3), keepdim=True).clamp_min(eps)
    fg_mean = (y_gt * text).sum((2, 3), keepdim=True) / text_mass
    ring_mean = (y_gt * ring).sum((2, 3), keepdim=True) / ring_mass
    contrast = ring_mean - fg_mean
    polarity = torch.where(contrast >= 0, torch.ones_like(contrast), -torch.ones_like(contrast))
    active = (contrast.abs() >= float(min_polarity_contrast)).float()
    # e>0 means that the prediction has moved a true stroke toward its local background.
    e = polarity * (y_pred - y_gt)
    missing_map = _soft_hinge(e, margin, temperature) * text * active
    false_map = _soft_hinge(-e, margin, temperature) * ring * active
    missing = missing_map.sum() / (text * active).sum().clamp_min(eps)
    false = false_map.sum() / (ring * active).sum().clamp_min(eps)
    return missing, false, {"ring": ring, "polarity": polarity, "active": active}


__all__ = ["recall_aware_structure_losses"]
