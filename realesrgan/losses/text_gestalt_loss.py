"""Masked shallow-CRNN feature loss for the V4 Text-Gestalt-style arm."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY

from .crnn_content_perceptual_loss import FrozenCRNNFeatureExtractor


@LOSS_REGISTRY.register()
class MaskedTextGestaltLoss(nn.Module):
    """Match frozen shallow CRNN features with soft text-focused attention."""

    def __init__(self, recognizer_path, alphabet_path, feature_taps=(0, 1, 2),
                 feature_weights=(1.4, 1.4, 1.4), attention_gain=3.0,
                 attention_dilate_radius=1, attention_blur_sigma=0.8, loss_weight=1.0):
        super().__init__()
        names = ("pooling0", "pooling1", "pooling2", "pooling3", "relu6")
        indexes = tuple(int(index) for index in feature_taps)
        if not indexes or any(index < 0 or index >= len(names) for index in indexes):
            raise ValueError("feature_taps must select shallow CRNN taps")
        if len(indexes) != len(feature_weights) or any(float(weight) <= 0 for weight in feature_weights):
            raise ValueError("feature_weights must be positive and align with feature_taps")
        self.layer_names = tuple(names[index] for index in indexes)
        self.layer_weights = tuple(float(weight) for weight in feature_weights)
        self.attention_gain = float(attention_gain)
        self.radius = int(attention_dilate_radius)
        self.loss_weight = float(loss_weight)
        self.extractor = FrozenCRNNFeatureExtractor(recognizer_path, alphabet_path, self.layer_names)
        self.extractor.eval()
        self.last_stats = {}

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()
        return self

    def forward(self, pred_rgb, gt_rgb, text_mask_rgb, valid_mask_rgb):
        if pred_rgb.shape != gt_rgb.shape or pred_rgb.ndim != 4 or pred_rgb.shape[1] != 3:
            raise ValueError("Text-Gestalt expects matching RGB BCHW tensors")
        if text_mask_rgb.shape[:2] != (pred_rgb.shape[0], 1) or valid_mask_rgb.shape != text_mask_rgb.shape:
            raise ValueError("Text-Gestalt expects full-resolution B1HW masks")
        # Reuse the CRNN H=32 input contract; 96x480 becomes 32x160.
        size = (32, max(16, int(round(pred_rgb.shape[-1] * 32 / pred_rgb.shape[-2]))))
        pred = F.interpolate(pred_rgb, size=size, mode="bilinear", align_corners=False)
        target = F.interpolate(gt_rgb, size=size, mode="bilinear", align_corners=False)
        text = text_mask_rgb.float()
        if self.radius:
            text = F.max_pool2d(text, 2 * self.radius + 1, stride=1, padding=self.radius)
        text = F.avg_pool2d(text, 3, stride=1, padding=1)
        text = F.interpolate(text, size=size, mode="bilinear", align_corners=False).clamp(0, 1)
        valid = F.interpolate(valid_mask_rgb.float(), size=size, mode="nearest")
        pred_features = self.extractor(pred)
        with torch.no_grad():
            target_features = self.extractor(target)
        total = pred.new_zeros(())
        stats = {}
        for name, weight, out, ref in zip(self.layer_names, self.layer_weights, pred_features, target_features):
            attention = F.interpolate(valid * (1.0 + self.attention_gain * text), size=out.shape[-2:], mode="bilinear", align_corners=False)
            raw = ((out - ref.detach()).square() * attention).sum() / (attention.sum() * out.shape[1]).clamp_min(1e-6)
            total = total + weight * raw
            stats[name] = float(raw.detach())
        self.last_stats = {"coverage": float((text * valid).mean().detach()), "layers": stats}
        return total * self.loss_weight
