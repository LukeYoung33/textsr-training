"""Shared frozen-r7 OpenCV atoms. BGR uint8; no framework approximation."""
from __future__ import annotations
from typing import Callable
import cv2
import numpy as np
ATOMIC_VERSION = "preview_r7_shared_v1"
PARAMETER_SCHEMA = "fixed_r7_v1"

def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Convert uint8 sRGB BGR to linear-light float BGR."""
    value = image.astype(np.float32) / 255.0
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert linear-light float BGR to uint8 sRGB BGR."""
    value = np.clip(image, 0.0, 1.0)
    srgb = np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1.0 / 2.4) - 0.055)
    return np.round(np.clip(srgb * 255.0, 0, 255)).astype(np.uint8)


def _midfreq_chroma(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Add zero-mean DoG chroma variation while keeping the Y channel unchanged."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    field = rng.normal(0.0, 1.0, image.shape[:2]).astype(np.float32)
    low = cv2.GaussianBlur(field, (0, 0), 1.2)
    very_low = cv2.GaussianBlur(field, (0, 0), 5.0)
    band = low - very_low
    band /= max(float(np.std(band)), 1e-6)
    chroma_direction = rng.normal(0.0, 1.0, 2).astype(np.float32)
    chroma_direction /= max(float(np.linalg.norm(chroma_direction)), 1e-6)
    ycrcb[:, :, 1] += band * strength * chroma_direction[0]
    ycrcb[:, :, 2] += band * strength * chroma_direction[1]
    return cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def gaussian_blur(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C1a: visible but realistic Gaussian lens/processing blur."""
    sigma = (params or {}).get("sigma", 1.55)
    return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101), {"gaussian_sigma": sigma}


def motion_blur(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C1b: camera-motion blur with an explicit direction and visible length."""
    length, angle = int((params or {}).get("length", 11)), float(rng.uniform(20, 160))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1.0), (length, length))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101), {"motion_length_px": length, "angle_degrees": round(angle, 2)}


def zoom_resample(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C1c: loss from a low-resolution digital-zoom source resampled to output size."""
    h, w = image.shape[:2]
    scale = (params or {}).get("source_scale", 0.52)
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), {"source_scale": scale, "upscale_interpolation": "bicubic"}


def defocus_blur(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C1d: disk-shaped defocus blur, distinct from Gaussian and motion kernels."""
    radius = int((params or {}).get("radius", 3))
    size = radius * 2 + 1
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101), {"disk_radius_px": radius}


def fine_structure_loss(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C2: linear-light PSF, pixel integration and phase-dependent sampling."""
    h, w = image.shape[:2]
    # The same optical PSF attenuates narrow strokes more than wide strokes:
    # wide centres still receive mainly foreground light, while a narrow stroke
    # or endpoint is mixed with background on several sides.  Work in linear
    # light so that this mixing retains the intended physical contrast.
    linear = _srgb_to_linear(image)
    sigma = (params or {}).get("optical_psf_sigma", 0.82)
    blurred = cv2.GaussianBlur(linear, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)
    # A sub-pixel phase then changes how the already softened coverage falls
    # into the sampling cells.  No artificial endpoint erasure, noise, denoise
    # or codec is used in this minimal validation chain.
    phase_xy = rng.uniform(-(params or {}).get("phase_px", .42), (params or {}).get("phase_px", .42), size=2).astype(np.float32)
    phase = np.float32([[1.0, 0.0, phase_xy[0]], [0.0, 1.0, phase_xy[1]]])
    shifted = cv2.warpAffine(blurred, phase, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    scale = (params or {}).get("sample_scale", .56)
    sampled = cv2.resize(shifted, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    reconstructed = cv2.resize(sampled, (w, h), interpolation=cv2.INTER_CUBIC)
    return _linear_to_srgb(reconstructed), {
        "linear_light": True,
        "optical_psf_sigma": sigma,
        "sample_scale": scale,
        "subpixel_phase_xy": [round(float(phase_xy[0]), 3), round(float(phase_xy[1]), 3)],
        "reconstruction": "bicubic",
    }


def flat_luma_noise(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C3: low-amplitude mid-frequency chroma residue concentrated on dark strokes."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycrcb[:, :, 0]
    # Flat bright areas are aggressively smoothed by a camera ISP.  Chroma
    # residual instead survives where dark ink and its luma edge reduce the
    # denoiser's confidence: use darkness plus a narrow gradient support.
    local_y = cv2.GaussianBlur(y, (0, 0), 3.2)
    # A relative-darkness mask identifies black ink on a pale surface.  It does
    # not mistake a uniformly dark board for ink, which was why the previous
    # absolute-luma mask visibly contaminated large flat dark backgrounds.
    dark_stroke = np.clip((local_y - y - 7.0) / 42.0, 0.0, 1.0)
    dark_stroke *= np.clip((195.0 - y) / 75.0, 0.0, 1.0)
    mask = cv2.GaussianBlur(dark_stroke, (0, 0), 0.42)
    field_cr = rng.normal(0.0, 1.0, image.shape[:2]).astype(np.float32)
    field_cb = rng.normal(0.0, 1.0, image.shape[:2]).astype(np.float32)
    band_cr = cv2.GaussianBlur(field_cr, (0, 0), 0.7) - cv2.GaussianBlur(field_cr, (0, 0), 3.0)
    band_cb = cv2.GaussianBlur(field_cb, (0, 0), 0.7) - cv2.GaussianBlur(field_cb, (0, 0), 3.0)
    band_cr /= max(float(np.std(band_cr)), 1e-6)
    band_cb /= max(float(np.std(band_cb)), 1e-6)
    strength = (params or {}).get("strength", 12.0)
    ycrcb[:, :, 1] += band_cr * mask * strength
    ycrcb[:, :, 2] += band_cb * mask * strength
    return cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR), {
        "dark_stroke_chroma_sigma": strength,
        "bandpass_sigmas": [0.7, 3.0],
        "bright_flat_area_mask": "near_zero",
    }


def color_anomaly(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C4: UV-only over-smoothing makes coloured writing bleed into its background."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # The luma channel remains sharp while UV is smoothed independently.  This
    # intentionally reproduces a chroma-denoise failure mode: blue ink can
    # leave a pale coloured cloud outside a comparatively sharp luma edge.
    sigma = (params or {}).get("uv_blur_sigma", 4.0)
    strength = (params or {}).get("midfreq_strength", 3.0)
    ycrcb[:, :, 1:] = cv2.GaussianBlur(ycrcb[:, :, 1:], (0, 0), sigma)
    diffused = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return _midfreq_chroma(diffused, rng, strength), {"uv_only_blur_sigma": sigma, "midfreq_chroma_strength": strength}


def halo_ringing(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C5: wider black/white bands from a large-radius USM overshoot."""
    sigma = (params or {}).get("usm_sigma", 2.3)
    amount = (params or {}).get("usm_amount", 1.55)
    quality = int((params or {}).get("jpeg_quality", 82))
    base = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, base, -amount, 0)
    return _jpeg(sharpened, quality), {"usm_sigma": sigma, "usm_amount": amount, "jpeg_quality": quality}


def edge_irregularity(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """C6: phase-dependent aliasing and reconstruction around near-Nyquist edges."""
    h, w = image.shape[:2]
    linear = _srgb_to_linear(image)
    phase_xy = rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
    phase = np.float32([[1.0, 0.0, phase_xy[0]], [0.0, 1.0, phase_xy[1]]])
    shifted = cv2.warpAffine(linear, phase, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    prefiltered = cv2.GaussianBlur(shifted, (0, 0), 0.40, borderType=cv2.BORDER_REFLECT101)
    scale = 0.42
    sampled = cv2.resize(prefiltered, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    # An edge-directed reconstruction need not perfectly restore the continuous
    # contour.  Blending its cubic result with grid-preserving reconstruction
    # gives staircase/contour deformation only where the scene has an edge;
    # flat fields remain constant without synthetic edge-band grain.
    cubic = cv2.resize(sampled, (w, h), interpolation=cv2.INTER_CUBIC)
    grid = cv2.resize(sampled, (w, h), interpolation=cv2.INTER_NEAREST)
    reconstructed = 0.42 * cubic + 0.58 * grid
    return _linear_to_srgb(reconstructed), {"linear_light": True, "sample_scale": scale, "subpixel_phase_xy": [round(float(phase_xy[0]), 3), round(float(phase_xy[1]), 3)], "reconstruction": "0.42_cubic_plus_0.58_grid"}


def directional_ghosting(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C7: explicit directional double-image, distinct from ordinary blur."""
    shift_x = (params or {}).get("shift_x", 2.6)
    shift_y = (params or {}).get("shift_y", 1.2)
    weight = (params or {}).get("weight", .35)
    matrix = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
    shifted = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT101)
    result = cv2.addWeighted(image, 1 - weight, shifted, weight, 0)
    return result, {"ghost_shift_xy": [shift_x, shift_y], "ghost_weight": weight}


def secondary_capture(image: np.ndarray, rng: np.random.Generator, params=None) -> tuple[np.ndarray, dict]:
    """C8: display/source downsampling then a restrained second camera capture."""
    h, w = image.shape[:2]
    scale = (params or {}).get("display_scale", .62)
    q1 = int((params or {}).get("display_jpeg_quality", 90))
    q2 = int((params or {}).get("final_jpeg_quality", 86))
    sigma = (params or {}).get("camera_blur_sigma", .7)
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    displayed = cv2.resize(_jpeg(small, q1), (w, h), interpolation=cv2.INTER_CUBIC)
    margin = (params or {}).get("perspective_fraction", .008)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + rng.uniform(-margin, margin, (4, 2)).astype(np.float32) * np.float32([w, h])
    warped = cv2.warpPerspective(displayed, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT101)
    result = _jpeg(cv2.GaussianBlur(warped, (0, 0), sigma), q2)
    return result, {"display_scale": scale, "display_jpeg_quality": q1, "perspective_fraction": margin, "camera_blur_sigma": sigma, "final_jpeg_quality": q2}


CATEGORIES: dict[str, tuple[str, Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, dict]]]] = {
    "c1a_gaussian_blur": ("C1a 高斯模糊", gaussian_blur),
    "c1b_motion_blur": ("C1b 运动模糊", motion_blur),
    "c1c_zoom_resample": ("C1c Zoom 重采样", zoom_resample),
    "c1d_defocus_blur": ("C1d 失焦模糊", defocus_blur),
    "c2_fine_structure_loss": ("C2 细结构对比度与形状损失", fine_structure_loss),
    "c3_dark_stroke_chroma_noise": ("C3 深色笔画中频彩噪", flat_luma_noise),
    "c4_color_anomaly": ("C4 色度异常与中频彩噪", color_anomaly),
    "c5_halo_ringing": ("C5 亮暗边、halo 与振铃", halo_ringing),
    "c6_edge_irregularity": ("C6 沿边不规则伪影", edge_irregularity),
    "c7_directional_ghosting": ("C7 方向性重影", directional_ghosting),
    "c8_secondary_capture": ("C8 二次成像", secondary_capture),
}


