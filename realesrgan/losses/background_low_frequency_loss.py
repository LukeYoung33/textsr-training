"""Training-only low-frequency reconstruction losses for far background pixels.

``BackgroundLowFrequencyLoss`` is the historical masked Gaussian loss.  The
strict ``BackgroundLowFrequencyGuardLoss`` variant keeps the same GT-derived
support but masks the prediction error *before* the Gaussian operation.  That
ordering is important: it prevents the blur kernel from creating a prediction
gradient in text, near-text, or invalid padding pixels.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def _as_mask(mask: torch.Tensor | None, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a mask to detached ``[B,1,H,W]`` values on ``reference``."""
    if mask is None:
        return torch.ones_like(reference[:, :1])
    value = mask.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[0] != reference.shape[0] or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W] or [B,H,W], got {tuple(value.shape)}")
    if value.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"{name} spatial shape {tuple(value.shape[-2:])} does not match "
            f"prediction shape {tuple(reference.shape[-2:])}"
        )
    return value.detach().clamp(0.0, 1.0)


def far_background_mask(
    text_mask_rgb: torch.Tensor | None,
    valid_mask_rgb: torch.Tensor | None,
    *,
    radius: int = 7,
    threshold: float = 0.5,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``V * (1 - dilate(M, radius))`` at full RGB resolution.

    ``reference`` is only needed when one of the masks is ``None``.  Keeping
    the helper public gives smoke/evaluation tools the exact same support
    construction as training.
    """
    if text_mask_rgb is None and valid_mask_rgb is None and reference is None:
        raise ValueError("reference is required when both masks are None")
    if reference is None:
        reference = text_mask_rgb if text_mask_rgb is not None else valid_mask_rgb
        if reference is None:  # guarded above; helps static type checkers
            raise ValueError("reference is required")
        if reference.ndim == 3:
            reference = reference.unsqueeze(1)
        reference = reference.new_zeros((reference.shape[0], 3, reference.shape[-2], reference.shape[-1]))
    valid = (_as_mask(valid_mask_rgb, reference, "valid_mask_rgb") > float(threshold)).to(reference.dtype)
    text = (_as_mask(text_mask_rgb, reference, "text_mask_rgb") > float(threshold)).to(reference.dtype)
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius:
        text = F.max_pool2d(text, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (valid * (1.0 - text).clamp_min(0.0)).detach()


@LOSS_REGISTRY.register()
class BackgroundLowFrequencyLoss(nn.Module):
    """Masked Gaussian low-frequency L1 loss on decoded RGB/luminance.

    Args:
        radius: HR pixels used to dilate the GT text mask before subtraction.
        sigma: Gaussian blur sigma in HR pixels.
        channel: ``"luma"`` (default) or ``"rgb"``.  Luma is the calibrated
            default and avoids giving chroma three independent opportunities
            to dominate the native Y4UV objective.
        threshold: binary threshold for the supplied GT masks.
        loss_weight: optional module-local multiplier; experiment-level
            weighting is applied by the trainer via ``bg_lf_weight``.
        eps: denominator stabilizer.
    """

    def __init__(
        self,
        radius: int = 7,
        sigma: float = 4.0,
        channel: str = "luma",
        threshold: float = 0.5,
        loss_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.sigma = float(sigma)
        self.channel = str(channel).strip().lower()
        self.threshold = float(threshold)
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        if not torch.isfinite(torch.tensor(self.sigma)) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if self.channel not in {"luma", "rgb"}:
            raise ValueError("channel must be 'luma' or 'rgb'")
        if self.loss_weight < 0.0 or self.eps <= 0.0:
            raise ValueError("loss_weight must be non-negative and eps must be positive")

    def _blur(self, value: torch.Tensor) -> torch.Tensor:
        """Blur each channel independently without introducing zero-padding edges."""
        blur_radius = max(1, int(torch.ceil(torch.tensor(3.0 * self.sigma)).item()))
        coords = torch.arange(-blur_radius, blur_radius + 1, device=value.device, dtype=value.dtype)
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * self.sigma * self.sigma))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(torch.finfo(value.dtype).eps)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        channels = int(value.shape[1])
        kernel = kernel_2d.expand(channels, 1, -1, -1)
        pad = (blur_radius,) * 4
        # Replication matches the datasets' right-padding convention and keeps
        # a valid edge from acquiring an artificial zero-valued blur halo.
        padded = F.pad(value, pad, mode="replicate")
        return F.conv2d(padded, kernel, groups=channels)

    @staticmethod
    def _luma(value: torch.Tensor) -> torch.Tensor:
        return 0.299 * value[:, :1] + 0.587 * value[:, 1:2] + 0.114 * value[:, 2:3]

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor | None = None,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        return_details: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compute raw bgLF and, optionally, support diagnostics."""
        if pred_rgb.shape != target_rgb.shape or pred_rgb.ndim != 4 or pred_rgb.shape[1] != 3:
            raise ValueError("pred_rgb and target_rgb must match BCHW with three RGB channels")
        far = far_background_mask(
            text_mask_rgb,
            valid_mask_rgb,
            radius=self.radius,
            threshold=self.threshold,
            reference=pred_rgb,
        )
        if self.channel == "luma":
            pred_value, target_value = self._luma(pred_rgb), self._luma(target_rgb)
        else:
            pred_value, target_value = pred_rgb, target_rgb
        error = (self._blur(pred_value) - self._blur(target_value)).abs()
        if self.channel == "rgb":
            error = error.mean(dim=1, keepdim=True)
        denominator = far.sum()
        if denominator.detach().item() <= 0.0:
            raw = pred_rgb.sum() * 0.0
        else:
            raw = (error * far).sum() / denominator.clamp_min(self.eps)
            raw = raw * self.loss_weight
        if not return_details:
            return raw
        valid = _as_mask(valid_mask_rgb, pred_rgb, "valid_mask_rgb") > self.threshold
        details = {
            "far_mask": far,
            "far_coverage": far.mean(),
            "far_over_valid": denominator / valid.float().sum().clamp_min(self.eps),
            "valid_coverage": valid.float().mean(),
            "text_coverage": (_as_mask(text_mask_rgb, pred_rgb, "text_mask_rgb") > self.threshold).float().mean(),
        }
        return raw, details


@LOSS_REGISTRY.register()
class BackgroundLowFrequencyGuardLoss(BackgroundLowFrequencyLoss):
    """Strict masked Gaussian L1 loss with a prediction-gradient firewall.

    The exact guarded formulation is::

        B = V * (1 - dilate(M, radius))
        residual = Gaussian(B * (Ypred - Ygt), sigma) / (Gaussian(B) + eps)
        L = mean_B(abs(residual))

    ``B`` is detached because it is derived only from GT annotations.  The
    numerator and denominator use the same normalized, replicate-padded
    Gaussian kernel.  Multiplying the error by ``B`` before convolution, and
    multiplying the final residual by ``B`` for the masked mean, makes the
    derivative with respect to the prediction exactly zero outside ``B``.
    In particular, the Gaussian kernel cannot leak a gradient into text,
    ``radius``-pixel near-text support, or invalid right padding.

    Args are identical to :class:`BackgroundLowFrequencyLoss`.  The class is
    separate so historical BG-LF experiments retain their original semantics.
    """

    variant = "guard"

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        text_mask_rgb: torch.Tensor | None = None,
        valid_mask_rgb: torch.Tensor | None = None,
        *,
        return_details: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compute guarded bgLF and optional support/normalizer diagnostics."""
        if pred_rgb.shape != target_rgb.shape or pred_rgb.ndim != 4 or pred_rgb.shape[1] != 3:
            raise ValueError("pred_rgb and target_rgb must match BCHW with three RGB channels")
        far = far_background_mask(
            text_mask_rgb,
            valid_mask_rgb,
            radius=self.radius,
            threshold=self.threshold,
            reference=pred_rgb,
        )
        if self.channel == "luma":
            pred_value, target_value = self._luma(pred_rgb), self._luma(target_rgb)
        else:
            pred_value, target_value = pred_rgb, target_rgb

        # The support is detached and broadcast over RGB/luma channels.  This
        # is the gradient firewall: no prediction value outside B enters the
        # convolution, so no smooth-kernel halo can return there.
        masked_error = (pred_value - target_value) * far
        numerator = self._blur(masked_error)
        normalizer = self._blur(far)
        residual = numerator / (normalizer + self.eps)
        if self.channel == "rgb":
            residual = residual.mean(dim=1, keepdim=True)

        denominator = far.sum()
        if denominator.detach().item() <= 0.0:
            raw = pred_rgb.sum() * 0.0
        else:
            # ``far`` is one-channel, so this is mean_B(abs(residual)) for
            # both luma and RGB (RGB is averaged over channels above).
            raw = (residual.abs() * far).sum() / denominator.clamp_min(self.eps)
            raw = raw * self.loss_weight
        if not return_details:
            return raw

        valid = _as_mask(valid_mask_rgb, pred_rgb, "valid_mask_rgb") > self.threshold
        details = {
            "far_mask": far,
            "far_coverage": far.mean(),
            "far_over_valid": denominator / valid.float().sum().clamp_min(self.eps),
            "valid_coverage": valid.float().mean(),
            "text_coverage": (_as_mask(text_mask_rgb, pred_rgb, "text_mask_rgb") > self.threshold).float().mean(),
            "guard_normalizer_min": normalizer.detach().amin(),
            "guard_normalizer_max": normalizer.detach().amax(),
            "guard_normalizer_mean": normalizer.detach().mean(),
        }
        return raw, details


__all__ = ["BackgroundLowFrequencyLoss", "BackgroundLowFrequencyGuardLoss", "far_background_mask"]
