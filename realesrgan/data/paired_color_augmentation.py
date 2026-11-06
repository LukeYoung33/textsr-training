"""Paired luminance-preserving color augmentation for RealCE training."""

from dataclasses import asdict, dataclass

import numpy as np


LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class ColorAugmentParams:
    temperature: float = 0.0
    tint: float = 0.0
    saturation: float = 1.0

    def manifest(self):
        return asdict(self)


def srgb_to_linear(encoded):
    encoded = np.asarray(encoded, dtype=np.float32)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(linear):
    linear = np.maximum(np.asarray(linear, dtype=np.float32), 0.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def _white_balance_gains(params):
    gains = np.exp(np.asarray([
        params.temperature + 0.5 * params.tint,
        -params.tint,
        -params.temperature + 0.5 * params.tint,
    ], dtype=np.float32))
    gains /= float(np.dot(LUMA_WEIGHTS, gains))
    return gains.astype(np.float32)


def _project_gamut_preserve_luma(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    y = np.sum(rgb * LUMA_WEIGHTS, axis=-1, keepdims=True)
    y = np.clip(y, 0.0, 1.0)
    delta = rgb - y
    limit = np.ones_like(rgb, dtype=np.float32)
    np.divide(1.0 - y, delta, out=limit, where=delta > 1e-8)
    np.divide(-y, delta, out=limit, where=delta < -1e-8)
    scale = np.minimum(1.0, np.min(np.maximum(limit, 0.0), axis=-1, keepdims=True))
    projected = y + delta * scale
    return np.clip(projected, 0.0, 1.0), float(np.mean(scale[..., 0] < 0.999))


def apply_color_augmentation_bgr(image_bgr, params):
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('expected uint8 HWC BGR image')
    rgb = srgb_to_linear(image[..., ::-1].astype(np.float32) / 255.0)
    gains = _white_balance_gains(params)
    rgb *= gains.reshape(1, 1, 3)
    y = np.sum(rgb * LUMA_WEIGHTS, axis=-1, keepdims=True)
    rgb = y + float(params.saturation) * (rgb - y)
    rgb, projection_ratio = _project_gamut_preserve_luma(rgb)
    encoded = linear_to_srgb(rgb)
    output = np.rint(np.clip(encoded, 0.0, 1.0)[..., ::-1] * 255.0).astype(np.uint8)
    return output, {
        'params': params.manifest(),
        'linear_rgb_gains': gains.tolist(),
        'gamut_projection_ratio': projection_ratio,
    }


def apply_paired_color_augmentation(lq_bgr, gt_bgr, params):
    lq_out, lq_info = apply_color_augmentation_bgr(lq_bgr, params)
    gt_out, gt_info = apply_color_augmentation_bgr(gt_bgr, params)
    return lq_out, gt_out, {'lq': lq_info, 'gt': gt_info}


def sample_safe_paired_color_augmentation(
        lq_bgr,
        gt_bgr,
        rng,
        probability=0.85,
        max_projection_ratio=0.05,
        max_attempts=8,
        temperature_range=(-0.15, 0.03),
        tint_range=(-0.12, 0.02),
        saturation_range=(0.85, 1.05)):
    if float(rng.random()) >= float(probability):
        params = ColorAugmentParams()
        lq_out, gt_out, diagnostics = apply_paired_color_augmentation(lq_bgr, gt_bgr, params)
        diagnostics.update(applied=False, attempts=1)
        return lq_out, gt_out, params, diagnostics

    for attempt in range(1, int(max_attempts) + 1):
        params = ColorAugmentParams(
            temperature=float(rng.uniform(*temperature_range)),
            tint=float(rng.uniform(*tint_range)),
            saturation=float(rng.uniform(*saturation_range)),
        )
        lq_out, gt_out, diagnostics = apply_paired_color_augmentation(lq_bgr, gt_bgr, params)
        projection = max(
            diagnostics['lq']['gamut_projection_ratio'],
            diagnostics['gt']['gamut_projection_ratio'],
        )
        if projection <= float(max_projection_ratio):
            diagnostics.update(applied=True, attempts=attempt)
            return lq_out, gt_out, params, diagnostics

    params = ColorAugmentParams()
    lq_out, gt_out, diagnostics = apply_paired_color_augmentation(lq_bgr, gt_bgr, params)
    diagnostics.update(
        applied=False,
        attempts=int(max_attempts),
        fallback_reason='projection_ratio_limit',
    )
    return lq_out, gt_out, params, diagnostics

