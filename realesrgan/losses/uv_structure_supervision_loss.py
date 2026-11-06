"""Training-only UV supervision for chroma blotches and colour spread."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY


def _uv_mask(mask, ref):
    """@luna-comment
    Input: optional one/six-channel valid/text mask and UV reference. Output: clamped UV-grid mask.
    Key Steps: align dtype/device, select chroma membership and pool full-resolution masks.
    Key Params:2x spatial reduction only. State/Dependencies: torch; inputs are not mutated.
    """
    if mask is None:
        return torch.ones_like(ref[:, :1])
    mask = mask.to(ref.device, ref.dtype)
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[1] == 6:
        # Native masks follow Y4UV channel layout.  UV supervision consumes
        # the shared chroma-plane validity/membership mask only.
        mask = mask[:, 4:5]
    elif mask.shape[1] != 1:
        raise ValueError("UV supervision mask must have one or six channels")
    if mask.shape[-2:] != ref.shape[-2:]:
        if tuple(v // 2 for v in mask.shape[-2:]) != ref.shape[-2:]:
            raise ValueError("UV supervision mask must match UV or full-resolution grid")
        mask = F.avg_pool2d(mask, 2, 2)
    return mask.clamp(0, 1)


@LOSS_REGISTRY.register()
class UVStrongSupervisionLoss(nn.Module):
    """GT-targeted regional UV loss; filters only prediction residuals."""
    def __init__(self, uv_weight=.25, multiscale_weight=.10, ring_radius=6,
                 sigmas=(1., 3., 6.), region_weights=(2., 3., .5), eps=1e-6):
        """@luna-comment
        Input: UV/multiscale coefficients, ring radius, blur scales and region weights.
        Output: configured training-only module. Key Steps: store scalar/tuple options.
        Key Params: region order text/ring/far. State/Dependencies: nn.Module, no model parameters.
        """
        super().__init__()
        self.uv_weight, self.multiscale_weight = float(uv_weight), float(multiscale_weight)
        self.ring_radius, self.sigmas, self.region_weights, self.eps = int(ring_radius), tuple(sigmas), tuple(region_weights), float(eps)

    def _mean(self, value, mask):
        """@luna-comment
        Input: BCHW residual values and membership mask. Output: batch-mean masked channel average.
        Key Steps: normalize each sample by mask mass times channels. Key Params: eps guards empty regions.
        State/Dependencies: tensor reductions only; gradients flow through values.
        """
        denom = mask.sum((1, 2, 3)).clamp_min(self.eps) * value.shape[1]
        return ((value * mask).sum((1, 2, 3)) / denom).mean()

    def _blur(self, value, mask, sigma):
        """@luna-comment
        Input: BCHW residual, region mask and Gaussian sigma. Output: masked normalized blur.
        Key Steps: construct separable Gaussian outer product; grouped convolution; divide by blurred mask.
        Key Params: radius ceil(3sigma), eps floor. State/Dependencies: torch functional convolution.
        """
        radius = max(1, int(math.ceil(3 * sigma)))
        axis = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
        kernel = torch.exp(-axis.square() / (2 * sigma * sigma)); kernel /= kernel.sum()
        k2 = (kernel[:, None] * kernel[None, :])[None, None]
        numerator = F.conv2d(value * mask, k2.expand(value.shape[1], 1, -1, -1), padding=radius, groups=value.shape[1])
        denominator = F.conv2d(mask, k2, padding=radius).clamp_min(self.eps)
        return numerator / denominator

    def forward(self, pred, target, valid_mask=None, text_mask=None, return_details=False, **_):
        """@luna-comment
        Input: matching six-channel packed prediction/target and valid/text masks.
        Output: regional UV plus multiscale residual penalty, optionally detached diagnostics.
        Key Steps: partition text/ring/far; weighted Charbonnier mean; masked regional frequency bands.
        Key Params: normalized region weights and fixed multiscale shares .4/.5/.1.
        State/Dependencies: UV mask/blur helpers; no image or network state is mutated.
        """
        if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 6:
            raise ValueError("UVStrongSupervisionLoss requires matching six-channel Y4UV tensors")
        p, t = pred[:, 4:6], target[:, 4:6]
        valid, text = _uv_mask(valid_mask, p), _uv_mask(text_mask, p)
        dilated = F.max_pool2d(text, 2 * self.ring_radius + 1, 1, self.ring_radius)
        ring, far = (dilated - text).clamp_min(0), (1 - dilated).clamp_min(0)
        regions = (text * valid, ring * valid, far * valid)
        weights = valid * sum(w * m for w, m in zip(self.region_weights, regions))
        residual = p - t
        charbonnier = torch.sqrt(residual.square() + 1e-6) - 1e-3
        l_uv = self._mean(charbonnier, weights)
        l_ms = p.new_zeros(())
        for region, coefficient in zip(regions, (.4, .5, .1)):
            blurs = [self._blur(residual, region, sigma) for sigma in self.sigmas]
            bands = (blurs[0] - blurs[1], blurs[1] - blurs[2], blurs[2])
            l_ms = l_ms + coefficient * sum(w * self._mean(torch.sqrt(x.square() + 1e-6) - 1e-3, region) for w, x in zip((.35, .35, .30), bands))
        total = self.uv_weight * l_uv + self.multiscale_weight * l_ms
        if return_details:
            return total, {"uv_raw": l_uv.detach(), "ms_raw": l_ms.detach(), "text_coverage": text.mean().detach(), "ring_coverage": ring.mean().detach(), "far_coverage": far.mean().detach()}
        return total
