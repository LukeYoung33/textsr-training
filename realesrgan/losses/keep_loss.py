"""Evidence-gated thin-stroke preservation loss for BASIC V2.

The Keep objective is deliberately a loss-side operation.  It does not add a
generator input or branch, and it never infers a mask from a prediction.  A
pixel is eligible only when it is on a GT text skeleton/tolerance tube, the GT
and LQ local contrasts agree with the per-sample polarity, and the LQ has a
measurable signed stroke contrast.  The one-sided target is the measured LQ
contrast, so a missing LQ stroke is not turned into a hallucination target.
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
    """Normalize an optional single-channel mask to detached BCHW values.

    Input: mask in [B,1,H,W] or [B,H,W] and a BCHW image reference.
    Output: detached float mask on the reference device and shape. Missing
    values use default when supplied; malformed masks fail explicitly.
    Key Params: default=1 is used only for the valid-area mask.
    State/Dependencies: torch tensor conversion; no global/model state.
    """

    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return reference.new_full((reference.shape[0], 1, *reference.shape[-2:]), float(default))
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 3:
        result = result.unsqueeze(1)
    expected = (reference.shape[0], 1, *reference.shape[-2:])
    if tuple(result.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(result.shape)}")
    return result.detach()


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """Convert one- or three-channel RGB tensors to differentiable luma.

    Input: floating BCHW image. Output: BCHW one-channel Rec.601 luma.
    Three-channel inputs are required to be RGB; one-channel inputs are
    treated as already-luminance. The operation preserves prediction grads.
    """

    if rgb.ndim != 4 or rgb.shape[1] not in (1, 3):
        raise ValueError("RGB input must be BCHW with one or three channels")
    if rgb.shape[1] == 1:
        return rgb
    return 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _masked_ring_contrast(
    image: torch.Tensor,
    valid: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Measure center-minus-ring contrast without reading invalid padding.

    The square neighborhood excludes its center. avg_pool2d is converted back
    to a true masked sum/count before the center is removed, which avoids both
    image-boundary and right-padding bias. The result is zero outside valid.
    """

    radius = int(radius)
    if radius < 1:
        raise ValueError("ring_radius must be >= 1")
    kernel = 2 * radius + 1
    value = image * valid.to(dtype=image.dtype)
    local_sum_mean = F.avg_pool2d(value, kernel, 1, radius, count_include_pad=False)
    local_count_mean = F.avg_pool2d(
        valid.to(dtype=image.dtype), kernel, 1, radius, count_include_pad=False
    )
    window_area = F.avg_pool2d(
        torch.ones_like(valid, dtype=image.dtype), kernel, 1, radius, count_include_pad=False
    ) * float(kernel * kernel)
    local_sum = local_sum_mean * window_area
    local_count = local_count_mean * window_area
    center = valid.to(dtype=image.dtype)
    ring_sum = local_sum - value
    ring_count = local_count - center
    ring_mean = torch.where(ring_count > 1e-6, ring_sum / ring_count.clamp_min(1e-6), image)
    return (image - ring_mean) * center


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply a zero-boundary binary square dilation to a BCHW mask."""

    radius = int(radius)
    if radius <= 0:
        return mask.bool()
    return F.max_pool2d(mask.float(), 2 * radius + 1, 1, radius).ge(0.5)


def _skeleton(mask: torch.Tensor, max_iterations: int | None = None) -> torch.Tensor:
    """Build a detached morphological skeleton from a binary GT text mask.

    Each iteration peels one erosion layer and records current-opening. Thin
    one-pixel components are retained. The implementation is torch-only and
    intended for the no-grad support-selection path.
    """

    work = mask.bool()
    result = torch.zeros_like(work)
    height, width = work.shape[-2:]
    iterations = max(1, min(int(max_iterations or max(height, width)), max(height, width)))
    for _ in range(iterations):
        work_float = work.float()
        eroded = (-F.max_pool2d(-work_float, 3, 1, 1)).ge(0.5)
        opened = F.max_pool2d(eroded.float(), 3, 1, 1).ge(0.5)
        result |= work & ~opened
        work = eroded
        if not bool(work.any()):
            break
    # Very small components can be lost under backend boundary conventions.
    # Retaining the original component is safer than dropping a credible thin
    # target entirely.
    empty = result.flatten(1).sum(1) <= 0
    result = torch.where(empty[:, None, None, None], mask.bool(), result)
    return result


def _infer_polarity(
    contrast: torch.Tensor,
    skeleton: torch.Tensor,
    valid: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Infer one bright/dark polarity per sample from strong GT skeleton contrast."""

    quantile = float(quantile)
    if not 0.0 <= quantile < 1.0:
        raise ValueError("polarity_quantile must be in [0, 1)")
    result = contrast.new_ones((contrast.shape[0],))
    for index in range(contrast.shape[0]):
        values = contrast[index][skeleton[index] & valid[index]]
        if values.numel() == 0:
            continue
        cutoff = torch.quantile(values.float().abs(), quantile)
        strong = values.abs() >= cutoff
        positive = values[strong & (values > 0)]
        negative = -values[strong & (values < 0)]
        positive_score = positive.mean() if positive.numel() else values.new_zeros(())
        negative_score = negative.mean() if negative.numel() else values.new_zeros(())
        if bool(negative_score > positive_score):
            result[index] = -1.0
    return result


@LOSS_REGISTRY.register()
class KeepLoss(nn.Module):
    """Evidence-gated local-contrast preservation loss.

    The caller supplies verified GT text and valid masks. structure_available
    optionally gates rows whose structure asset is missing; no inferred mask is
    used. forward returns a scalar by default, or (loss, details) when
    return_details=True. Details are detached diagnostics and do not
    participate in backpropagation.
    """

    def __init__(
        self,
        ring_radius: int = 3,
        gt_min_contrast: float = 0.04,
        lq_min_contrast: float = 0.02,
        support_radius: int = 1,
        temperature: float = 0.02,
        loss_weight: float = 1.0,
        polarity_quantile: float = 0.90,
        max_skeleton_iterations: int | None = None,
        lost_contrast: float | None = None,
        thin_recall_ratio: float = 0.50,
        output_min_contrast: float | None = None,
        mask_threshold: float = 0.5,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.ring_radius = int(ring_radius)
        self.gt_min_contrast = float(gt_min_contrast)
        self.lq_min_contrast = float(lq_min_contrast)
        self.support_radius = int(support_radius)
        self.temperature = float(temperature)
        self.loss_weight = float(loss_weight)
        self.polarity_quantile = float(polarity_quantile)
        self.max_skeleton_iterations = max_skeleton_iterations
        self.lost_contrast = lost_contrast
        self.thin_recall_ratio = float(thin_recall_ratio)
        self.output_min_contrast = output_min_contrast
        self.mask_threshold = float(mask_threshold)
        self.enabled = bool(enabled)
        if self.ring_radius < 1 or self.support_radius < 0:
            raise ValueError("ring_radius must be >=1 and support_radius must be non-negative")
        if self.gt_min_contrast <= 0.0 or self.lq_min_contrast <= 0.0:
            raise ValueError("GT/LQ contrast thresholds must be positive")
        if self.temperature <= 0.0 or self.loss_weight < 0.0:
            raise ValueError("temperature must be positive and loss_weight non-negative")
        if not 0.0 <= self.polarity_quantile < 1.0:
            raise ValueError("polarity_quantile must be in [0,1)")
        if not 0.0 <= self.thin_recall_ratio <= 1.0:
            raise ValueError("thin_recall_ratio must be in [0,1]")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0,1]")

    @staticmethod
    def _gate_structure_rows(text: torch.Tensor, structure_available: Any) -> torch.Tensor:
        """Zero GT text support for rows lacking a verified structure asset."""

        if structure_available is None:
            return text
        values = list(structure_available) if isinstance(structure_available, (list, tuple)) else [structure_available]
        if len(values) != int(text.shape[0]):
            raise ValueError("structure_available must have one entry per batch sample")
        result = text.clone()
        for index, available in enumerate(values):
            if not bool(available):
                result[index] = 0.0
        return result

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        lq_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        structure_available: Any = None,
        return_details: bool = False,
        return_maps: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compute Keep on GT/LQ evidence and return optional diagnostics.

        Input: differentiable RGB prediction plus GT/LQ RGB pairs and verified
        full-resolution text/valid masks. Output: scalar differentiable loss;
        prediction is the sole gradient source. GT/LQ evidence, skeleton,
        polarity, and eligibility are selected under no_grad.
        """

        if not all(isinstance(value, torch.Tensor) for value in (pred_rgb, target_rgb, lq_rgb)):
            raise ValueError("pred_rgb, target_rgb, and lq_rgb must be tensors")
        if pred_rgb.ndim != 4 or target_rgb.shape != pred_rgb.shape or lq_rgb.shape != pred_rgb.shape:
            raise ValueError("pred_rgb, target_rgb, and lq_rgb must have matching BCHW shapes")
        if pred_rgb.shape[1] not in (1, 3):
            raise ValueError("KeepLoss expects one or three RGB channels")
        if not pred_rgb.is_floating_point():
            pred_rgb = pred_rgb.float()
        target_rgb = target_rgb.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
        lq_rgb = lq_rgb.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
        valid = _as_bchw_mask(valid_mask_rgb, pred_rgb, "valid_mask_rgb", default=1.0)
        text = _as_bchw_mask(text_mask_rgb, pred_rgb, "text_mask_rgb")
        valid = (valid > self.mask_threshold)
        text = (text > self.mask_threshold) & valid
        text = self._gate_structure_rows(text, structure_available)

        with torch.no_grad():
            gt_luma = _luma(target_rgb.detach())
            lq_luma = _luma(lq_rgb.detach())
            gt_contrast = _masked_ring_contrast(gt_luma, valid, self.ring_radius)
            lq_contrast = _masked_ring_contrast(lq_luma, valid, self.ring_radius)
            skeleton = _skeleton(text, self.max_skeleton_iterations)
            polarity = _infer_polarity(gt_contrast, skeleton, valid, self.polarity_quantile)
            polarity_map = polarity.view(-1, 1, 1, 1)
            gt_signed = polarity_map * gt_contrast
            lq_signed = polarity_map * lq_contrast
            gt_credible = skeleton & (gt_signed >= self.gt_min_contrast) & valid
            support = _dilate(gt_credible, self.support_radius) & valid
            gt_consistent = gt_signed > 0.0
            lq_consistent = lq_signed > 0.0
            selected = support & gt_consistent & lq_consistent & (lq_signed >= self.lq_min_contrast)
            visible_skeleton = skeleton & gt_credible & gt_consistent & lq_consistent & (lq_signed >= self.lq_min_contrast)
            target_lq = lq_signed[selected].detach()
            selected_count = selected.sum()
            support_count = support.sum()
            skeleton_count = skeleton.sum()
            visible_count = visible_skeleton.sum()
            valid_count = valid.sum()

        if int(selected_count.item()) > 0 and self.enabled and self.loss_weight > 0.0:
            pred_luma = _luma(pred_rgb)
            pred_contrast = _masked_ring_contrast(pred_luma, valid, self.ring_radius)
            pred_signed = polarity.view(-1, 1, 1, 1).to(pred_contrast.dtype) * pred_contrast
            gap = target_lq - pred_signed[selected]
            loss = F.softplus(gap / self.temperature).mul(self.temperature).mean().mul(self.loss_weight)
        else:
            loss = pred_rgb.sum() * 0.0
            pred_signed = None

        with torch.no_grad():
            if pred_signed is None:
                pred_luma_diag = _luma(pred_rgb.detach())
                pred_contrast_diag = _masked_ring_contrast(pred_luma_diag, valid, self.ring_radius)
                pred_signed_diag = polarity.view(-1, 1, 1, 1) * pred_contrast_diag
            else:
                pred_signed_diag = pred_signed.detach()
            if int(visible_count.item()) > 0:
                visible_pred = pred_signed_diag[visible_skeleton]
                visible_lq = lq_signed[visible_skeleton]
                lost_threshold = self.lq_min_contrast if self.lost_contrast is None else float(self.lost_contrast)
                lost_count = (visible_pred < lost_threshold).sum()
                thin_kept = (visible_pred >= visible_lq * self.thin_recall_ratio).sum()
                mean_lq = visible_lq.mean()
                mean_pred = visible_pred.mean()
            else:
                lost_count = torch.zeros((), dtype=torch.long, device=pred_rgb.device)
                thin_kept = torch.zeros((), dtype=torch.long, device=pred_rgb.device)
                mean_lq = pred_rgb.new_zeros(())
                mean_pred = pred_rgb.new_zeros(())
            output_threshold = self.lq_min_contrast if self.output_min_contrast is None else float(self.output_min_contrast)
            pred_stroke = (pred_signed_diag >= output_threshold) & valid
            gt_tube = _dilate(gt_credible, self.support_radius) & valid
            unsupported = pred_stroke & ~support & valid
            background = pred_stroke & ~gt_tube & valid
            gt_support_count = (support & gt_consistent).sum()
            polarity_count = (support & gt_consistent & lq_consistent).sum()
            positive_count = pred_stroke.sum()
            details: dict[str, torch.Tensor] = {
                "keep_support_count": selected_count.float(),
                "keep_support_coverage": selected_count.float() / valid_count.float().clamp_min(1.0),
                "keep_gt_skeleton_count": skeleton_count.float(),
                "keep_visible_skeleton_count": visible_count.float(),
                "keep_gt_support_count": support_count.float(),
                "keep_visible_support_rate": selected_count.float() / support_count.float().clamp_min(1.0),
                "keep_polarity_consistency_rate": polarity_count.float() / gt_support_count.float().clamp_min(1.0),
                "keep_visible_to_lost_rate": lost_count.float() / visible_count.float().clamp_min(1.0),
                "keep_thin_recall": thin_kept.float() / visible_count.float().clamp_min(1.0),
                "keep_mean_lq_contrast": mean_lq,
                "keep_mean_pred_contrast": mean_pred,
                "keep_false_positive_rate": unsupported.float().sum() / valid_count.float().clamp_min(1.0),
                "keep_spur_rate": unsupported.float().sum() / positive_count.float().clamp_min(1.0),
                "keep_background_activation_rate": background.float().sum() / valid_count.float().clamp_min(1.0),
                "keep_valid_pixel_count": valid_count.float(),
                "keep_polarity_bright_fraction": (polarity > 0).float().mean(),
            }
            if return_maps:
                details.update({
                    "keep_skeleton": skeleton.float(),
                    "keep_support": support.float(),
                    "keep_selected": selected.float(),
                    "keep_visible_skeleton": visible_skeleton.float(),
                    "keep_valid": valid.float(),
                    "keep_gt_signed_contrast": gt_signed,
                    "keep_lq_signed_contrast": lq_signed,
                    "keep_pred_signed_contrast": pred_signed_diag,
                    "keep_polarity": polarity.view(-1, 1, 1, 1).float(),
                })
            details["keep_loss"] = loss.detach()

        if return_details:
            return loss, details
        return loss


__all__ = ["KeepLoss"]
