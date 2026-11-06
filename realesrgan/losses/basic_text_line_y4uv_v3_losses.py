"""Native and decoded-reference losses for the isolated Y4UV420 path."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY

from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_mask_to_yuv444, y4uv_to_yuv444, yuv420_mask_to_y4uv


def _mask_bchw(mask, reference: torch.Tensor, name: str) -> torch.Tensor:
    """@luna-comment
    Input: optional BHW/BCHW mask and reference. Output: device/dtype-aligned compatible mask.
    Key Steps: validate batch/channels/spatial size; pack full-resolution masks when needed.
    Key Params: six-channel half-resolution Y4UV. State/Dependencies: torch/packer; no input mutation.
    """
    if mask is None:
        return torch.ones_like(reference[:, :1])
    value = mask.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} must be BCHW or BHW, got {tuple(value.shape)}")
    if value.shape[1] == 1:
        if value.shape[-2:] == reference.shape[-2:]:
            return value
        if reference.shape[1] == 6 and value.shape[-2] == reference.shape[-2] * 2 and value.shape[-1] == reference.shape[-1] * 2:
            return yuv420_mask_to_y4uv(value)
        raise ValueError(f"{name} spatial shape {tuple(value.shape[-2:])} does not match {tuple(reference.shape[-2:])}")
    if value.shape[1] != reference.shape[1] or value.shape[-2:] != reference.shape[-2:]:
        raise ValueError(f"{name} must match reference shape, got {tuple(value.shape)} vs {tuple(reference.shape)}")
    return value


def _weights(weights: Iterable[float], reference: torch.Tensor) -> torch.Tensor:
    """@luna-comment
    Input: nonnegative channel weights and BCHW reference. Output: 1xCx1x1 tensor.
    Key Steps: validate length/sign, allocate on reference device/dtype. Key Params: matching channel count.
    State/Dependencies: torch; no persistent state.
    """
    values = tuple(float(item) for item in weights)
    if len(values) != reference.shape[1]:
        raise ValueError(f"channel_weight length {len(values)} != channels {reference.shape[1]}")
    result = reference.new_tensor(values).view(1, -1, 1, 1)
    if (result < 0).any():
        raise ValueError("channel_weight values must be non-negative")
    return result


@LOSS_REGISTRY.register()
class Y4UVNativeMaskedL1LossV3(nn.Module):
    """Six-channel masked L1 loss with channel-aligned Y4 and UV weighting."""

    def __init__(
        self,
        loss_weight=1.0,
        reduction="mean",
        fg_weight=1.0,
        bg_weight=0.5,
        channel_weight=(1.0, 1.0, 1.0, 1.0, 0.25, 0.25),
        eps=1e-6,
    ):
        """@luna-comment
        Input: scalar loss/region weights, channel weights, reduction and denominator epsilon.
        Output: configured native L1 module. Key Steps: validate reduction and store parameters.
        Key Params: mean/sum/none; channel count checked in forward. State/Dependencies: nn.Module configuration.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unsupported reduction: {reduction}")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.fg_weight = float(fg_weight)
        self.bg_weight = float(bg_weight)
        self.channel_weight = tuple(float(item) for item in channel_weight)
        self.eps = float(eps)

    def forward(self, pred, target, valid_mask=None, text_mask=None, mask=None, weight=None, **_):
        """@luna-comment
        Input: matching six-channel tensors, optional valid/text masks and sample weights. Output: weighted L1.
        Key Steps: combine valid/region/channel weights and reduce absolute error.
        Key Params: loss_weight/reduction/fg/bg/channel weights/eps. State/Dependencies: torch/helpers; no state mutation.
        """
        if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 6:
            raise ValueError("pred and target must match BCHW with six Y4UV channels")
        valid = _mask_bchw(valid_mask if valid_mask is not None else mask, pred, "valid_mask")
        text = _mask_bchw(text_mask, pred, "text_mask") * valid
        region = valid * (text * self.fg_weight + (1.0 - text) * self.bg_weight)
        effective = region * _weights(self.channel_weight, pred)
        if weight is not None:
            extra = weight.to(device=pred.device, dtype=pred.dtype)
            while extra.ndim < 4:
                extra = extra.unsqueeze(1)
            effective = effective * extra
        error = (pred - target).abs() * effective
        if self.reduction == "none":
            return error * self.loss_weight
        if self.reduction == "sum":
            return error.sum() * self.loss_weight
        return error.sum() / (effective.sum() + self.eps) * self.loss_weight


@LOSS_REGISTRY.register()
class YUV444ReferenceMaskedL1LossV3(nn.Module):
    """Decode the same Y4UV tensors to temporary YUV444 and compare there."""

    def __init__(
        self,
        loss_weight=1.0,
        reduction="mean",
        fg_weight=1.0,
        bg_weight=0.5,
        channel_weight=(1.0, 0.25, 0.25),
        eps=1e-6,
    ):
        """@luna-comment
        Input: scalar loss/region weights, channel weights, reduction and denominator epsilon.
        Output: configured decoded-reference L1 module. Key Steps: validate reduction and store parameters.
        Key Params: mean/sum/none; channel count checked in forward. State/Dependencies: nn.Module configuration.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unsupported reduction: {reduction}")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.fg_weight = float(fg_weight)
        self.bg_weight = float(bg_weight)
        self.channel_weight = tuple(float(item) for item in channel_weight)
        self.eps = float(eps)

    def _full_mask(self, mask, reference: torch.Tensor, name: str) -> torch.Tensor:
        """@luna-comment
        Input: optional BHW/BCHW mask and full-resolution reference. Output: compatible full-resolution mask.
        Key Steps: validate geometry; nearest-expand half-size one-channel masks or unpack six channels.
        Key Params: exact native/full spatial ratios. State/Dependencies: torch/decoder; no persistent mutation.
        """
        if mask is None:
            return torch.ones_like(reference[:, :1])
        value = mask.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 3:
            value = value.unsqueeze(1)
        if value.ndim != 4 or value.shape[0] != reference.shape[0]:
            raise ValueError(f"{name} must be BCHW or BHW")
        if value.shape[1] == 1:
            if value.shape[-2:] == reference.shape[-2:]:
                return value
            if value.shape[-2] * 2 == reference.shape[-2] and value.shape[-1] * 2 == reference.shape[-1]:
                return F.interpolate(value, size=reference.shape[-2:], mode="nearest")
            raise ValueError(f"{name} does not match full reference shape")
        if value.shape[1] != 6:
            raise ValueError(f"{name} must have one or six channels")
        return y4uv_mask_to_yuv444(value, full_hw=tuple(reference.shape[-2:]))

    def forward(self, pred, target, valid_mask=None, text_mask=None, mask=None, weight=None, **_):
        """@luna-comment
        Input: matching native tensors and optional masks/weights. Output: reduced decoded-YUV444 absolute error.
        Key Steps: decode tensors/masks, combine valid/region/channel weights and reduce.
        Key Params: weights/reduction/eps. State/Dependencies: torch/native decoders; no model-state mutation.
        """
        if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 6:
            raise ValueError("pred and target must match BCHW with six Y4UV channels")
        pred444, target444 = y4uv_to_yuv444(pred), y4uv_to_yuv444(target)
        valid = self._full_mask(valid_mask if valid_mask is not None else mask, pred444, "valid_mask")
        text = self._full_mask(text_mask, pred444, "text_mask") * valid
        region = valid * (text * self.fg_weight + (1.0 - text) * self.bg_weight)
        effective = region * _weights(self.channel_weight, pred444)
        if weight is not None:
            extra = weight.to(device=pred444.device, dtype=pred444.dtype)
            while extra.ndim < 4:
                extra = extra.unsqueeze(1)
            effective = effective * extra
        error = (pred444 - target444).abs() * effective
        if self.reduction == "none":
            return error * self.loss_weight
        if self.reduction == "sum":
            return error.sum() * self.loss_weight
        return error.sum() / (effective.sum() + self.eps) * self.loss_weight


def compare_y4uv_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
    native_loss: nn.Module | None = None,
    reference_loss: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """@luna-comment
    Input: matching native tensors, masks and optional loss modules. Output: native/reference/difference scalars.
    Key Steps: instantiate missing modules and evaluate both on identical tensors.
    Key Params: modules own weighting/reduction. State/Dependencies: torch losses; no optimizer/state mutation.
    """
    native = native_loss or Y4UVNativeMaskedL1LossV3()
    reference = reference_loss or YUV444ReferenceMaskedL1LossV3()
    l_native = native(pred, target, valid_mask=valid_mask, text_mask=text_mask)
    l_reference = reference(pred, target, valid_mask=valid_mask, text_mask=text_mask)
    return {
        "l_native": l_native,
        "l_reference": l_reference,
        "l_compare_abs": (l_native - l_reference).abs(),
    }


__all__ = [
    "Y4UVNativeMaskedL1LossV3",
    "YUV444ReferenceMaskedL1LossV3",
    "compare_y4uv_losses",
]
