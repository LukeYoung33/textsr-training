"""PNG-derived Y reference for the deployed local-background contrast curve.

This is a host numerical approximation of a Y-only native420 stage. PNG input
does not contain the fused native Y/UV planes, so its RGB preview cannot prove
NEON byte identity or native UV preservation.
"""
from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CurveTrial:
    """@luna-comment Input: residual gate and gain/cap changes. Output: immutable trial settings. Key Steps: keep deployed polarity/deadband and background fixed. Key Params: R0=16, R1=48, eta=.10, DeltaD=2. State/Dependencies: trial only, not production tuning."""
    r0: float = 16.0
    r1: float = 48.0
    eta: float = .10
    delta_d: float = 2.0


TRIAL = CurveTrial()


def smoothstep01(value):
    """@luna-comment Input: scalar or float32 array. Output: clamped smoothstep in [0,1]. Key Steps: t*t*(3-2*t). Key Params: cubic end slopes zero. State/Dependencies: no writes."""
    t = np.clip(value, 0.0, 1.0)
    return t*t*(3.0-2.0*t)


def cell_background(y):
    """@luna-comment Input: uint8 full-range Y plane. Output: twice-smoothed 16-cell mean grid. Key Steps: exact integer partial-cell sums, actual-count mean, replicated 121 horizontal/vertical passes. Key Params: two rounds. State/Dependencies: C++ deployed scalar reference is read-only."""
    if y.ndim != 2 or y.dtype != np.uint8:
        raise ValueError('Expected uint8 Y plane')
    h, w = y.shape
    if min(h, w) < 1:
        raise ValueError('Empty Y plane')
    yy = np.arange(0, h, 16)
    xx = np.arange(0, w, 16)
    sums = np.add.reduceat(np.add.reduceat(y.astype(np.uint32), yy, axis=0), xx, axis=1)
    heights = np.minimum(16, h-yy)
    widths = np.minimum(16, w-xx)
    grid = (sums / (heights[:, None]*widths[None, :])).astype(np.float32)
    for _ in range(2):
        horizontal = np.pad(grid, ((0, 0), (1, 1)), mode='edge')
        scratch = (horizontal[:, :-2]+2.0*horizontal[:, 1:-1]+horizontal[:, 2:])*.25
        vertical = np.pad(scratch, ((1, 1), (0, 0)), mode='edge')
        grid = (vertical[:-2]+2.0*vertical[1:-1]+vertical[2:])*.25
    return grid


def background_band(grid, y0, y1, width):
    """@luna-comment Input: smoothed grid and native row interval. Output: float32 half-pixel bilinear background for that band. Key Steps: clamp grid indices, interpolate y then x. Key Params: (coordinate+.5)/16-.5. State/Dependencies: bounded band avoids full-frame float intermediates."""
    gh, gw = grid.shape
    ys = np.arange(y0, y1, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    gy = (ys+.5)/16.0-.5
    gx = (xs+.5)/16.0-.5
    by = np.floor(gy).astype(np.int32)
    bx = np.floor(gx).astype(np.int32)
    fy = (gy-by).astype(np.float32)
    fx = (gx-bx).astype(np.float32)
    rows = grid[np.clip(by, 0, gh-1)]*(1.0-fy[:, None]) + grid[np.clip(by+1, 0, gh-1)]*fy[:, None]
    return rows[:, np.clip(bx, 0, gw-1)]*(1.0-fx[None, :]) + rows[:, np.clip(bx+1, 0, gw-1)]*fx[None, :]


def correction_pair(y, background, trial=TRIAL):
    """@luna-comment Input: same-shape float32 Y/background and fixed trial. Output: current/proposed signed rounded Y deltas plus branch counts. Key Steps: common polarity/deadband, residual smoothstep gain/cap gate, Y-only clamp, ties-even rounding. Key Params: deployed gain .75/cap24; trial R0/R1/eta/DeltaD. State/Dependencies: no RGB gamut limit here."""
    if trial.r1 <= trial.r0 or trial.eta < 0 or trial.delta_d < 0:
        raise ValueError('Invalid curve trial')
    y = np.asarray(y, np.float32)
    background = np.asarray(background, np.float32)
    residual = y-background
    paper = smoothstep01((background-80.0)/96.0)
    signed = -np.maximum(-residual-2.0, 0.0)*paper + np.maximum(residual-2.0, 0.0)*(1.0-paper)
    h = smoothstep01((np.abs(residual)-trial.r0)/(trial.r1-trial.r0))
    q0 = .75*signed
    q1 = (.75*(1.0+trial.eta*h))*signed
    d0 = q0/(1.0+np.abs(q0)/24.0)
    cap1 = 24.0+trial.delta_d*h
    d1 = (cap1*q1)/(cap1+np.abs(q1))
    if trial.eta == 0.0 and trial.delta_d == 0.0:
        d1 = d0
    else:
        d1 = np.where(h==0.0, d0, d1)
    low, high = -y, 255.0-y
    current = np.rint(np.clip(d0, low, high)).astype(np.int16)
    proposed = np.rint(np.clip(d1, low, high)).astype(np.int16)
    return current, proposed, {
        'residual_gate_float_hits': int(np.count_nonzero(h>0.0)),
        'residual_gate_full_hits': int(np.count_nonzero(h>=1.0)),
        'signed_formula_float_hits': int(np.count_nonzero(signed)),
        'current_y_cap_float_hits': int(np.count_nonzero((d0<low)|(d0>high))),
        'proposed_y_cap_float_hits': int(np.count_nonzero((d1<low)|(d1>high))),
        'proposed_minus_current_integer_hits': int(np.count_nonzero(proposed-current)),
        'proposed_minus_current_max_abs': int(np.max(np.abs(proposed-current))),
    }


def preview_bgr(source, signed_y_delta):
    """@luna-comment Input: original BGR8 PNG and signed Y change. Output: display BGR8 and RGB clipping count. Key Steps: common integer delta to each BGR channel, clip only for PNG gamut. Key Params: no artificial common-delta gamut restriction. State/Dependencies: this preview cannot establish native420 UV byte identity."""
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('Expected BGR8 source')
    if signed_y_delta.shape != source.shape[:2]:
        raise ValueError('Mismatched delta shape')
    total = source.astype(np.int16)+signed_y_delta[:, :, None]
    clipped_channels = int(np.count_nonzero((total<0)|(total>255)))
    return np.clip(total, 0, 255).astype(np.uint8), clipped_channels


def process_pair(source, trial=TRIAL, band_rows=128):
    """@luna-comment Input: one immutable native BGR8 PNG. Output: current/proposed display PNG arrays and telemetry. Key Steps: derive BT.601 full-range Y once, shared cell grid, bandwise paired corrections and RGB preview. Key Params: 128-row bands by default. State/Dependencies: no OCR warp, inference, extra sharpening or source mutation."""
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3 or band_rows < 1:
        raise ValueError('Expected BGR8 source and positive band size')
    started = time.perf_counter()
    y = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    grid = cell_background(y)
    background_seconds = time.perf_counter()-started
    current = np.empty_like(source)
    proposed = np.empty_like(source)
    counts = {}
    rgb_clip_current = rgb_clip_proposed = 0
    h, w = y.shape
    kernel_started = time.perf_counter()
    for top in range(0, h, band_rows):
        end = min(top+band_rows, h)
        bg = background_band(grid, top, end, w)
        d0, d1, band_counts = correction_pair(y[top:end], bg, trial)
        current[top:end], c0 = preview_bgr(source[top:end], d0)
        proposed[top:end], c1 = preview_bgr(source[top:end], d1)
        rgb_clip_current += c0
        rgb_clip_proposed += c1
        for key, value in band_counts.items():
            counts[key] = max(counts.get(key, 0), value) if key.endswith('_max_abs') else counts.get(key, 0)+value
    counts.update({'rgb_clipped_channels_current': rgb_clip_current,
                   'rgb_clipped_channels_proposed': rgb_clip_proposed,
                   'changed_pixels_current_from_source': int(np.count_nonzero(np.any(current!=source, axis=2))),
                   'changed_pixels_proposed_from_source': int(np.count_nonzero(np.any(proposed!=source, axis=2))),
                   'changed_pixels_proposed_from_current': int(np.count_nonzero(np.any(proposed!=current, axis=2))),
                   'background_seconds_host': background_seconds,
                   'paired_curve_and_preview_seconds_host': time.perf_counter()-kernel_started})
    return current, proposed, counts
