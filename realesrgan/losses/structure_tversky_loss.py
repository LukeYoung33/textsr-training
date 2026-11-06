"""Explicit text-stroke occupancy supervision for paired restoration."""

from __future__ import annotations

import torch


def tversky_structure_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one occupancy objective with independently priced FP and FN.

    ``beta > alpha`` deliberately makes removal of a true stroke more costly
    than a nearby false-positive stroke.  Reduction is per sample so a long
    line cannot dominate a short line merely through pixel count.
    """
    if logits.shape != target.shape or target.shape != valid_mask.shape:
        raise ValueError("logits, target and valid_mask must have identical BCHW shape")
    prediction = logits.sigmoid()
    target = target.clamp(0.0, 1.0) * valid_mask
    prediction = prediction * valid_mask
    dims = (1, 2, 3)
    true_positive = (prediction * target).sum(dims)
    false_positive = (prediction * (1.0 - target) * valid_mask).sum(dims)
    false_negative = ((1.0 - prediction) * target).sum(dims)
    score = (true_positive + eps) / (true_positive + float(alpha) * false_positive + float(beta) * false_negative + eps)
    return (1.0 - score).mean()


__all__ = ["tversky_structure_loss"]
