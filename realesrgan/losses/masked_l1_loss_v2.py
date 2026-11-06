"""
masked_l1_loss_v2.py

Sprint-1 improvements over masked_l1_loss.py:

  TextRegionPerceptualLossV2
    - Per-layer feature normalisation: each CRNN feature layer is divided by
      its own mean absolute value before computing L1.  This collapses the
      ~1800x magnitude gap between l_rec and l_pix that exists in the original
      TextRegionPerceptualLoss (empirically measured: l_rec≈440, l_pix≈0.24
      with loss_weight=1.0).
    - Everything else is identical to TextRegionPerceptualLoss so the two can
      be swapped in the yml with a single line change.

Imports all helpers from masked_l1_loss to avoid duplication.
"""

import torch
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY

from .masked_l1_loss import (
    _crnn_init,
    _parse_crnn_data,
    _ensure_mask,
    _match_channels,
    _reduce_loss,
    _check_reduction,
)


@LOSS_REGISTRY.register()
class TextRegionPerceptualLossV2(torch.nn.Module):
    """CRNN-feature perceptual loss with per-layer magnitude normalisation.

    Normalisation prevents any single CRNN layer from dominating the total
    loss purely because its activations happen to have a larger scale.

    Args:
        loss_weight (float): Overall scalar applied to the total loss.
            With normalisation in place, a value of 0.1–1.0 is reasonable
            (compared to ~0.001 needed for the unnormalised version).
        reduction (str): 'mean' | 'sum' | 'none'.
        recognizer_path (str | None): Path to CRNN weights.  None uses the
            default path configured in _crnn_init.
        ratio_keep (bool): Whether to preserve aspect ratio when resizing the
            patch for CRNN input.
        detach_target (bool): Detach GT features from the computation graph
            (standard practice; avoids gradient flow through the frozen CRNN).
        background_fill (float): Value used to fill non-text regions before
            sending to CRNN.
        norm_eps (float): Epsilon added to the per-layer scale denominator to
            prevent division by zero.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = 'mean',
        recognizer_path=None,
        ratio_keep: bool = True,
        detach_target: bool = True,
        background_fill: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        _check_reduction(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.ratio_keep = ratio_keep
        self.detach_target = detach_target
        self.background_fill = background_fill
        self.norm_eps = norm_eps
        self.recognizer = _crnn_init(recognizer_path=recognizer_path)

    def forward(self, pred, target, mask=None, weight=None, **kwargs):
        channels = pred.shape[1]
        target = target[:, :channels]

        if mask is None:
            mask = torch.ones_like(pred[:, :1])
        else:
            mask = _ensure_mask(mask, pred)
        mask = _match_channels(mask, channels)

        pred_region   = pred   * mask + self.background_fill * (1.0 - mask)
        target_region = target * mask + self.background_fill * (1.0 - mask)

        pred_region   = _parse_crnn_data(pred_region[:, :3],   ratio_keep=self.ratio_keep)
        target_region = _parse_crnn_data(target_region[:, :3], ratio_keep=self.ratio_keep)

        recognizer = self.recognizer.to(pred.device)
        pred_feats   = recognizer.get_feature(pred_region)
        target_feats = recognizer.get_feature(target_region)

        total_loss = 0.0
        for pred_feat, target_feat in zip(pred_feats, target_feats):
            tgt = target_feat.detach() if self.detach_target else target_feat

            # Per-layer normalisation: scale ≈ mean |activation| of this layer.
            # Dividing by this makes each layer contribute on a ~[0,1] scale
            # regardless of the raw activation magnitude.
            scale = tgt.abs().mean().clamp(min=self.norm_eps)

            feat_loss = torch.abs(pred_feat - tgt) / scale
            total_loss = total_loss + _reduce_loss(feat_loss, self.reduction)

        return total_loss * self.loss_weight
