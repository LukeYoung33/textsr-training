"""Deterministic photometric augmentation for cached native Y4UV crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


AUGMENTATION_VERSION = "y4uv_v4_cached_luma_aug_v1"


@dataclass(frozen=True)
class CachedY4UVAugmentConfig:
    """@shanlan-comment Configuration for the two isolated cached-crop luminance branches."""

    inversion_enabled: bool = True
    inversion_probability: float = 0.20
    contrast_enabled: bool = True
    contrast_probability: float = 0.35
    contrast_factor_range: tuple[float, float] = (1.05, 1.20)

    def __post_init__(self) -> None:
        """@shanlan-comment Validate probability and contrast bounds before a worker samples them."""
        for name, value in (
            ("inversion_probability", self.inversion_probability),
            ("contrast_probability", self.contrast_probability),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        low, high = (float(value) for value in self.contrast_factor_range)
        if low < 1.0 or high < low:
            raise ValueError(
                "contrast_factor_range must satisfy 1 <= low <= high, "
                f"got {self.contrast_factor_range}"
            )


def _as_y4uv(value: np.ndarray, name: str) -> np.ndarray:
    """@shanlan-comment Validate and copy one native six-channel Y4UV crop."""
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[0] != 6:
        raise ValueError(f"{name} must be a NumPy [6,H,W] array, got {type(value).__name__}")
    if value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError(f"{name} has invalid spatial shape {value.shape[1:]}")
    result = np.ascontiguousarray(value, dtype=np.float32).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _as_mask(value: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Validate one full-resolution mask used for luma-region statistics."""
    if not isinstance(value, np.ndarray) or value.ndim != 2 or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must be a full-resolution {shape} mask")
    result = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.clip(result, 0.0, 1.0)


def _mask_to_y4(mask: np.ndarray) -> np.ndarray:
    """@shanlan-comment Map a full-resolution mask to TL/TR/BL/BR native luma channels."""
    height, width = mask.shape
    if height % 2 or width % 2:
        raise ValueError(f"full-resolution mask must have even shape, got {mask.shape}")
    # The first axis order is exactly torch.pixel_unshuffle's TL, TR, BL, BR.
    return mask.reshape(height // 2, 2, width // 2, 2).transpose(1, 3, 0, 2).reshape(4, height // 2, width // 2)


def _rng_random(rng: Any) -> float:
    """@shanlan-comment Read one deterministic unit interval draw from a worker RNG."""
    if rng is None:
        rng = np.random
    return float(rng.random())


def _rng_uniform(rng: Any, low: float, high: float) -> float:
    """@shanlan-comment Read one deterministic uniform draw from a worker RNG."""
    if rng is None:
        rng = np.random
    return float(rng.uniform(float(low), float(high)))


def _contrast_mean(
    gt_luma: np.ndarray,
    valid_mask: np.ndarray,
    text_mask: np.ndarray,
) -> tuple[float, str]:
    """@shanlan-comment Compute the valid text-region mean with deterministic fallbacks."""
    valid4 = _mask_to_y4(valid_mask)
    text4 = _mask_to_y4(text_mask)
    weights = np.clip(valid4 * text4, 0.0, 1.0)
    region = "valid_text"
    if float(weights.sum()) <= 1e-8:
        weights = np.clip(valid4, 0.0, 1.0)
        region = "valid"
    if float(weights.sum()) <= 1e-8:
        weights = np.ones_like(gt_luma, dtype=np.float32)
        region = "all"
    denominator = float(weights.sum())
    mean = float((gt_luma * weights).sum() / max(denominator, 1e-8))
    return float(np.clip(mean, 0.0, 1.0)), region


def apply_cached_y4uv_augmentation(
    lq: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    text_mask: np.ndarray,
    rng: Any = None,
    config: CachedY4UVAugmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply ordered paired inversion and target-only contrast to native Y4UV.

    Input: native ``[6,H,W]`` LQ/GT tensors and full-resolution ``[2H,2W]``
    valid/text masks, all normalized to float32.  Output: copied native tensors
    with only Y4 luma channels changed plus auditable branch metadata.  The
    inversion decision is joint; contrast is sampled afterward and only alters
    GT luma around the valid text-region mean.  U/V and both masks are untouched.
    The caller supplies a DataLoader-worker-seeded NumPy RNG.
    """
    config = config or CachedY4UVAugmentConfig()
    lq_out = _as_y4uv(lq, "lq")
    gt_out = _as_y4uv(gt, "gt")
    if lq_out.shape != gt_out.shape:
        raise ValueError(f"lq and gt shapes must match, got {lq_out.shape} and {gt_out.shape}")
    mask_shape = (lq_out.shape[1] * 2, lq_out.shape[2] * 2)
    valid = _as_mask(valid_mask, "valid_mask", mask_shape)
    text = _as_mask(text_mask, "text_mask", mask_shape)

    inversion_applied = bool(
        config.inversion_enabled and _rng_random(rng) < float(config.inversion_probability)
    )
    if inversion_applied:
        # Native Y4 channels are the four luma sub-pixels.  Chroma is copied.
        lq_out[:4] = 1.0 - lq_out[:4]
        gt_out[:4] = 1.0 - gt_out[:4]

    contrast_applied = bool(
        config.contrast_enabled and _rng_random(rng) < float(config.contrast_probability)
    )
    contrast_factor = 1.0
    contrast_mean = 0.0
    contrast_region = "disabled"
    if contrast_applied:
        contrast_mean, contrast_region = _contrast_mean(gt_out[:4], valid, text)
        contrast_factor = _rng_uniform(
            rng,
            config.contrast_factor_range[0],
            config.contrast_factor_range[1],
        )
        gt_out[:4] = np.clip(
            contrast_mean + contrast_factor * (gt_out[:4] - contrast_mean),
            0.0,
            1.0,
        )

    changed = bool(inversion_applied or contrast_applied)
    metadata = {
        "augmentation_version": AUGMENTATION_VERSION,
        "luminance_inversion_enabled": bool(config.inversion_enabled),
        "luminance_inversion_applied": inversion_applied,
        "luma_inversion_applied": inversion_applied,
        "augmentation_inversion_applied": inversion_applied,
        "luminance_polarity_inverted": inversion_applied,
        "luminance_inversion_probability": float(config.inversion_probability),
        "contrast_enhancement_enabled": bool(config.contrast_enabled),
        "contrast_enhancement_applied": contrast_applied,
        "contrast_applied": contrast_applied,
        "augmentation_contrast_applied": contrast_applied,
        "contrast_enhancement_probability": float(config.contrast_probability),
        "contrast_factor": float(contrast_factor),
        "contrast_factor_range": [
            float(config.contrast_factor_range[0]),
            float(config.contrast_factor_range[1]),
        ],
        "contrast_reference_mean": float(contrast_mean),
        "contrast_reference_region": contrast_region,
        "augmentation_applied": changed,
        "augmentation_identity": not changed,
        "gt_rgb_regenerated": changed,
        "lq_rgb_regenerated": inversion_applied,
    }
    return lq_out, gt_out, metadata


def cached_y4uv_augment_config(options: dict[str, Any] | None) -> CachedY4UVAugmentConfig:
    """@shanlan-comment Parse nested/flat options without changing legacy dataset defaults."""
    options = dict(options or {})
    nested = options.get("augmentation", {})
    if nested is True:
        nested = {}
    if not isinstance(nested, dict):
        raise ValueError("dataset augmentation must be a mapping")
    global_enabled = bool(nested.get("enabled", options.get("augmentation_enabled", True)))

    def _value(keys: tuple[str, ...], default: Any) -> Any:
        """@shanlan-comment Resolve one augmentation option from nested then flat keys."""
        for key in keys:
            if key in nested:
                return nested[key]
            if key in options:
                return options[key]
        return default

    inversion_enabled = bool(
        _value(
            (
                "luminance_inversion_enabled",
                "enable_luminance_inversion",
                "inversion_enabled",
            ),
            global_enabled,
        )
    )
    inversion_probability = float(
        _value(
            (
                "luminance_inversion_probability",
                "luminance_inversion_prob",
                "inversion_probability",
                "inversion_prob",
            ),
            0.20,
        )
    )
    contrast_enabled = bool(
        _value(
            (
                "contrast_enhancement_enabled",
                "enable_contrast_enhancement",
                "contrast_enabled",
            ),
            global_enabled,
        )
    )
    contrast_probability = float(
        _value(
            (
                "contrast_enhancement_probability",
                "contrast_enhancement_prob",
                "contrast_probability",
                "contrast_prob",
            ),
            0.35,
        )
    )
    factor_range = _value(
        ("contrast_factor_range", "contrast_enhancement_factor_range"),
        (1.05, 1.20),
    )
    if len(factor_range) != 2:
        raise ValueError("contrast_factor_range must contain [low, high]")
    return CachedY4UVAugmentConfig(
        inversion_enabled=inversion_enabled,
        inversion_probability=inversion_probability,
        contrast_enabled=contrast_enabled,
        contrast_probability=contrast_probability,
        contrast_factor_range=(float(factor_range[0]), float(factor_range[1])),
    )


__all__ = [
    "AUGMENTATION_VERSION",
    "CachedY4UVAugmentConfig",
    "apply_cached_y4uv_augmentation",
    "cached_y4uv_augment_config",
]
