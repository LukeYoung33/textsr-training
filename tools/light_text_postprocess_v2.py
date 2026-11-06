"""Conservative, switchable classical text postprocess for rectified camera frames."""
from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class V2Config:
    """@luna-comment Input: scalar gains and module switches. Output: immutable operator settings. Key Steps: hold strength and safety limits. Key Params: gains are dimensionless; max_delta is BGR code values. State/Dependencies: no global mutable state."""
    contrast: float = .60
    halo: float = .60
    aa: float = .20
    sharp: float = .30
    max_delta: float = 20.
    noise: bool = True
    polarity: bool = True
    adaptive_feather: bool = True
    dehalo: bool = True
    directional_aa: bool = True
    sharpening: bool = True


PRESETS = {
    'low': V2Config(contrast=.35, halo=.35, aa=.12, sharp=.20, max_delta=12.),
    'standard': V2Config(),
    'off': V2Config(contrast=0, halo=0, aa=0, sharp=0, max_delta=0),
}


def robust_threshold(y):
    """@luna-comment Input: float encoded luma 0..255. Output: detail threshold and sampling telemetry. Key Steps: highpass, select flat 32px cells, sample four phases and median their MAD. Key Params: 0.8px blur, 2..12 threshold; sparse fallback. State/Dependencies: OpenCV/NumPy; no sensor-noise claim."""
    fine = y-cv2.GaussianBlur(y, (0, 0), .8)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    flat = cv2.boxFilter(np.abs(gx)+np.abs(gy), -1, (32, 32)) < 12
    phases = []
    counts = []
    for oy, ox in ((0, 0), (0, 4), (4, 0), (4, 4)):
        sample = fine[oy::8, ox::8]
        select = flat[oy::8, ox::8]
        values = sample[select]
        if values.size < 32:
            values = sample.ravel()
        median = float(np.median(values)) if values.size else 0.
        phases.append(float(1.4826*np.median(np.abs(values-median))) if values.size else 0.)
        counts.append(int(select.sum()))
    sigma = float(np.median(phases))
    return float(np.clip(1.5*sigma, 2., 12.)), {'noise_sigma': sigma, 'noise_phase_mad': phases, 'flat_samples': counts}


def adaptive_support(shape, boxes):
    """@luna-comment Input: image HW and optional half-open OCR boxes. Output: float alpha inside boxes only. Key Steps: per-box inward distance smoothstep and max union. Key Params: feather min(16,max(2,.25 short side)). State/Dependencies: None means full image; empty list identity support."""
    if boxes is None:
        return np.ones(shape, np.float32)
    alpha = np.zeros(shape, np.float32)
    for x0, y0, x1, y1 in boxes:
        if not 0 <= x0 < x1 <= shape[1] or not 0 <= y0 < y1 <= shape[0]:
            raise ValueError('Invalid support box')
        h, w = y1-y0, x1-x0
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        distance = np.minimum(np.minimum(xx+1, w-xx), np.minimum(yy+1, h-yy))
        feather = min(16., max(2., .25*min(w, h)))
        t = np.clip((distance-1)/feather, 0, 1)
        np.maximum(alpha[y0:y1, x0:x1], t*t*(3-2*t), out=alpha[y0:y1, x0:x1])
    return alpha


def edge_field(y):
    """@luna-comment Input: float luma. Output: unit normal and edge trust mask. Key Steps: Sobel gradients, locally averaged structure tensor coherence and straightness. Key Params: 3px tensor window, gradient>14. State/Dependencies: OpenCV; tangent is perpendicular to returned normal."""
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)/8
    mag = cv2.magnitude(gx, gy)
    nx, ny = gx/(mag+1e-6), gy/(mag+1e-6)
    a = cv2.boxFilter(gx*gx, -1, (5, 5))
    b = cv2.boxFilter(gy*gy, -1, (5, 5))
    c = cv2.boxFilter(gx*gy, -1, (5, 5))
    coherence = np.sqrt((a-b)**2+4*c*c)/(a+b+1e-6)
    trust = (mag > 14)&(coherence > .72)
    return nx, ny, trust, mag


def normal_sample(y, nx, ny, distance):
    """@luna-comment Input: luma, unit normals, signed pixel offset. Output: bilinear normal sample. Key Steps: remap reflected border. Key Params: offsets are in pixels. State/Dependencies: OpenCV float maps."""
    h, w = y.shape
    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    return cv2.remap(y, xx+distance*nx, yy+distance*ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def trusted_plateau_envelope(y, threshold):
    """@luna-comment Input: float luma and noise threshold. Output: side-platform lower/upper bounds and trusted edge mask. Key Steps: sample normals at ±3/±5, demand stable platforms and single-edge monotonic trend. Key Params: separation>24, side drift<max(5,2T). State/Dependencies: heuristic; ambiguous thin/complex structures deliberately excluded."""
    nx, ny, trust, mag = edge_field(y)
    minus3 = normal_sample(y, nx, ny, -3)
    minus5 = normal_sample(y, nx, ny, -5)
    plus3 = normal_sample(y, nx, ny, 3)
    plus5 = normal_sample(y, nx, ny, 5)
    stable = (np.abs(minus3-minus5)<max(5, 2*threshold))&(np.abs(plus3-plus5)<max(5, 2*threshold))
    low = np.minimum(minus5, plus5)-max(2., threshold)
    high = np.maximum(minus5, plus5)+max(2., threshold)
    near_minus = normal_sample(y, nx, ny, -1)
    near_plus = normal_sample(y, nx, ny, 1)
    monotonic = (plus5-minus5)*(near_plus-near_minus)>0
    # Opposing nearby gradients break monotonicity; this excludes many thin strokes.
    trusted = trust&stable&monotonic&(np.abs(plus5-minus5)>24)&(mag>14)
    return low, high, trusted


def dehalo_luma(y, strength, threshold, envelope=None):
    """@luna-comment Input: float luma, correction gain/noise gate, optional trusted envelope. Output: corrected luma and hit count. Key Steps: clip only overshoot beyond stable opposite-side platforms at trustworthy single edges. Key Params: correction gain and tolerance from envelope. State/Dependencies: wide/complex halos can be missed."""
    if strength <= 0:
        return y, 0
    low, high, trusted = envelope if envelope is not None else trusted_plateau_envelope(y, threshold)
    hit = trusted&((y<low)|(y>high))
    corrected = np.where(hit, y+strength*(np.clip(y, low, high)-y), y)
    return corrected.astype(np.float32), int(hit.sum())


def polarity_contrast(y, background, threshold, gain, max_delta):
    """@luna-comment Input: luma, coarse background, noise threshold and gain/cap. Output: confidence-gated luma and hit count. Key Steps: compare local signed residual energy, suppress ambiguous/flat windows, softly cap increment. Key Params: 31px support, dominance margin .25. State/Dependencies: contrast is heuristic, not OCR."""
    if gain <= 0 or max_delta <= 0:
        return y, 0
    residual = y-background
    dark = cv2.boxFilter(np.maximum(-residual-threshold, 0), -1, (31, 31))
    bright = cv2.boxFilter(np.maximum(residual-threshold, 0), -1, (31, 31))
    dominance = (dark-bright)/(dark+bright+threshold)
    confidence = np.clip((np.abs(dominance)-.25)/.55, 0, 1)
    confidence *= np.clip((dark+bright-threshold)/(3*threshold+1e-6), 0, 1)
    signed = np.where(dominance>0, -np.maximum(-residual-threshold, 0), np.maximum(residual-threshold, 0))
    q = gain*confidence*signed
    delta = q/(1+np.abs(q)/max_delta)
    return (y+delta).astype(np.float32), int(np.count_nonzero(np.abs(delta)>.5))


def directional_antialias(y, gain):
    """@luna-comment Input: adjusted luma and weak gain. Output: tangent-smoothed luma and hit count. Key Steps: gate diagonal coherent edge, bilinear tangent samples, suppress opposing-edge thin strokes, bound change. Key Params: gain<=.2, <=2-code change. State/Dependencies: heuristic AA; cannot reconstruct missing subpixels."""
    if gain <= 0:
        return y, 0
    nx, ny, trust, _ = edge_field(y)
    a = normal_sample(y, -ny, nx, 1)
    b = normal_sample(y, -ny, nx, -1)
    normal_a = normal_sample(y, nx, ny, 2)
    normal_b = normal_sample(y, nx, ny, -2)
    diagonal = (np.abs(nx)>.25)&(np.abs(ny)>.25)
    isolated = np.abs(normal_a-normal_b)>20
    staircase = np.abs((a+b)*.5-y)>1.5
    hit = trust&diagonal&isolated&staircase
    change = np.clip(gain*((a+b)*.5-y), -2., 2.)
    return np.where(hit, y+change, y).astype(np.float32), int(hit.sum())


def weak_sharpen(y, threshold, gain, envelope=None):
    """@luna-comment Input: float luma, noise threshold and trusted side-platform envelope. Output: weak-sharpened luma/hit count. Key Steps: soft threshold detail only on trusted edges and clamp to platform bounds. Key Params: gain<=.3. State/Dependencies: skips ambiguous pixels; cannot repair unavailable platform evidence."""
    if gain <= 0:
        return y, 0
    detail = y-cv2.GaussianBlur(y, (0, 0), .8)
    detail = np.sign(detail)*np.maximum(np.abs(detail)-threshold*.5, 0)
    if envelope is None:
        envelope = trusted_plateau_envelope(y, threshold)
    low, high, trusted = envelope
    result = np.where(trusted, np.clip(y+gain*detail, low, high), y)
    return result.astype(np.float32), int(np.count_nonzero(np.abs(result-y)>.5))


def enhance_bgr_v2(image, preset='standard', boxes=None, config=None):
    """@luna-comment Input: BGR8, named preset, optional OCR boxes/config. Output: BGR8 and path-hit telemetry. Key Steps: robust noise, trusted halo correction, confident contrast, optional directional AA, weak sharp, adaptive feather, shared integer delta/gamut. Key Params: V2Config switches; off is exact identity. State/Dependencies: OpenCV/NumPy CPU; no model inference or source mutation."""
    started = time.perf_counter()
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected BGR8')
    if config is None:
        config = PRESETS[preset]
    if config.max_delta <= 0:
        return image.copy(), {'preset': preset, 'seconds': time.perf_counter()-started, 'changed_pixels': 0, 'halo_hits': 0, 'aa_hits': 0, 'contrast_hits': 0, 'sharp_hits': 0}
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    if config.noise:
        threshold, stats = robust_threshold(y)
    else:
        threshold, stats = 2., {'noise_sigma': 0., 'noise_phase_mad': [], 'flat_samples': []}
    if config.adaptive_feather:
        alpha = adaptive_support(y.shape, boxes)
    else:
        from tools.light_text_postprocess import support_alpha
        alpha = support_alpha(y.shape, boxes)
    original_envelope = trusted_plateau_envelope(y, threshold)
    target, halo_hits = dehalo_luma(y, config.halo if config.dehalo else 0, threshold, original_envelope)
    small = cv2.resize(target, (max(1, (y.shape[1]+7)//8), max(1, (y.shape[0]+7)//8)), interpolation=cv2.INTER_AREA)
    background = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5), (y.shape[1], y.shape[0]))
    target, contrast_hits = polarity_contrast(target, background, threshold, config.contrast if config.polarity else 0, config.max_delta)
    target, aa_hits = directional_antialias(target, config.aa if config.directional_aa else 0)
    target, sharp_hits = weak_sharpen(target, threshold, config.sharp if config.sharpening else 0, original_envelope)
    delta = np.clip((target-y)*alpha, -config.max_delta, config.max_delta)
    lower = -image.min(2).astype(np.float32)
    upper = 255-image.max(2).astype(np.float32)
    capped = np.minimum(np.maximum(delta, lower), upper)
    integer = np.rint(capped).astype(np.int16)
    output = (image.astype(np.int16)+integer[:, :, None]).astype(np.uint8)
    telemetry = {'preset': preset, 'seconds': time.perf_counter()-started, **stats,
                 'halo_hits': halo_hits, 'contrast_hits': contrast_hits, 'aa_hits': aa_hits, 'sharp_hits': sharp_hits,
                 'trusted_edge_pixels': int(np.count_nonzero(original_envelope[2]& (alpha>0))),
                 'support_pixels': int(np.count_nonzero(alpha)), 'changed_pixels': int(np.count_nonzero(integer)),
                 'gamut_cap_hits': int(np.count_nonzero(np.abs(capped-delta)>.01)),
                 'max_abs_delta': int(np.abs(integer).max()), 'outside_support_unchanged': bool(np.array_equal(output[alpha==0], image[alpha==0])),
                 'common_channel_delta': True}
    return output, telemetry
