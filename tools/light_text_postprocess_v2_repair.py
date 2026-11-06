"""Isolated pilot: frozen V1 contrast-stage delta inside otherwise frozen V2 stages."""
from dataclasses import dataclass
import time

import cv2
import numpy as np

from tools.light_text_postprocess_v2 import (adaptive_support, dehalo_luma,
    directional_antialias, robust_threshold, trusted_plateau_envelope, weak_sharpen)


@dataclass(frozen=True)
class RepairConfig:
    """@luna-comment Input: pilot lambda 0..1. Output: immutable pilot contrast scale. Key Steps: retain V2 standard surrounding stages. Key Params: .5/1 tested, zero for stage equivalence. State/Dependencies: no mutable state."""
    contrast_scale: float


def v1_contrast_delta(y):
    """@luna-comment Input: original rectified float32 encoded Y. Output: frozen V1 standard float contrast delta and stage telemetry. Key Steps: 1/8 coarse background, single-phase MAD threshold, paper-polarity residual and softcap. Key Params: V1 gain .75, softcap24, T2..6. State/Dependencies: exact formulas of frozen tools/light_text_postprocess.py::enhance_bgr; no image mutation."""
    h, w = y.shape
    small = cv2.resize(y, (max(1, (w+7)//8), max(1, (h+7)//8)), interpolation=cv2.INTER_AREA)
    background = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT_101),
                            (w, h), interpolation=cv2.INTER_LINEAR)
    fine = y-cv2.GaussianBlur(y, (0, 0), .8, borderType=cv2.BORDER_REFLECT_101)
    samples = fine[::8, ::8]
    sigma = float(np.median(np.abs(samples-np.median(samples)))*1.4826)
    threshold = float(np.clip(1.5*sigma, 2., 6.))
    residual = y-background
    paper = np.clip((background-80.)/96., 0, 1)
    paper = paper*paper*(3-2*paper)
    dark = -np.maximum(-residual-threshold, 0)*paper
    bright = np.maximum(residual-threshold, 0)*(1-paper)
    contrast = .75*(dark+bright)
    delta = contrast/(1+np.abs(contrast)/24.)
    return delta.astype(np.float32), {'v1_noise_sigma': sigma, 'v1_noise_threshold': threshold,
                                      'v1_contrast_float_hits': int(np.count_nonzero(np.abs(delta)>.5))}


def enhance_bgr_repair(image, boxes, config):
    """@luna-comment Input: BGR8 OCR warp, boxes and contrast lambda. Output: BGR8 candidate and stage telemetry. Key Steps: frozen V2 standard robust threshold/support/envelope/dehalo, insert lambda*frozen-V1 delta from original Y, then frozen V2 AA/sharp/common integer BGR cap. Key Params: V2 halo .60, AA .20, sharp .30, cap20. State/Dependencies: imports frozen V2 helpers; never edits source/core."""
    started = time.perf_counter()
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected BGR8')
    if not 0 <= config.contrast_scale <= 1:
        raise ValueError('Contrast scale must be in [0,1]')
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    threshold, stats = robust_threshold(y)
    alpha = adaptive_support(y.shape, boxes)
    original_envelope = trusted_plateau_envelope(y, threshold)
    target, halo_hits = dehalo_luma(y, .60, threshold, original_envelope)
    contrast_delta, v1_stats = v1_contrast_delta(y)
    target = (target+config.contrast_scale*contrast_delta).astype(np.float32)
    target, aa_hits = directional_antialias(target, .20)
    target, sharp_hits = weak_sharpen(target, threshold, .30, original_envelope)
    delta = np.clip((target-y)*alpha, -20., 20.)
    lower = -image.min(2).astype(np.float32)
    upper = 255-image.max(2).astype(np.float32)
    capped = np.minimum(np.maximum(delta, lower), upper)
    integer = np.rint(capped).astype(np.int16)
    output = (image.astype(np.int16)+integer[:, :, None]).astype(np.uint8)
    report = {'contrast_scale': config.contrast_scale, 'seconds': time.perf_counter()-started,
              **stats, **v1_stats, 'contrast_path': 'exact_V1_standard_stage_delta_from_original_y',
              'v1_contrast_scaled_float_hits': int(np.count_nonzero(np.abs(config.contrast_scale*contrast_delta)>.5)),
              'halo_hits': halo_hits, 'aa_hits': aa_hits, 'sharp_hits': sharp_hits,
              'support_pixels': int(np.count_nonzero(alpha)),
              'trusted_edge_pixels': int(np.count_nonzero(original_envelope[2]&(alpha>0))),
              'changed_pixels': int(np.count_nonzero(integer)),
              'gamut_cap_hits': int(np.count_nonzero(np.abs(capped-delta)>.01)),
              'max_abs_delta': int(np.abs(integer).max()),
              'outside_support_unchanged': bool(np.array_equal(output[alpha==0], image[alpha==0])),
              'common_channel_delta': bool(np.array_equal(output[:, :, 0].astype(np.int16)-image[:, :, 0].astype(np.int16),
                                                            output[:, :, 1].astype(np.int16)-image[:, :, 1].astype(np.int16))
                                           and np.array_equal(output[:, :, 1].astype(np.int16)-image[:, :, 1].astype(np.int16),
                                                              output[:, :, 2].astype(np.int16)-image[:, :, 2].astype(np.int16)))}
    return output, report
