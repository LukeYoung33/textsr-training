"""Experimental dark-side anchored Y curve with frozen deployed bright branch."""
from dataclasses import dataclass
import time

import cv2
import numpy as np

from tools.native_y_curve_four_images import (CurveTrial, background_band,
    cell_background, correction_pair, preview_bgr, smoothstep01)


ZERO_TRIAL = CurveTrial(16.0, 48.0, 0.0, 0.0)


@dataclass(frozen=True)
class Arm:
    """@luna-comment Input: gamma, dark blend strength, anchor choice and offset. Output: immutable research arm. Key Steps: isolate curve strength from upper-anchor choice. Key Params: current zero branch; candidate offset12 after synthetic gates. State/Dependencies: no production parameter claim."""
    gamma: int
    alpha: float
    upper: bool
    anchor_offset: float = 4.0


ARMS = {
    'current': Arm(2, 0.0, False),
    'weak': Arm(2, .35, False, 12.0),
    'mean_g3': Arm(3, 1.0, False, 12.0),
    'upper_g3': Arm(3, 1.0, True, 12.0),
    'upper_g4': Arm(4, 1.0, True, 12.0),
}


def raw_cell_grid(y):
    """@luna-comment Input: uint8 full-range Y. Output: actual-count unsmoothed16x16 mean grid. Key Steps: exact integer sums including partial edge cells. Key Params: cell16. State/Dependencies: read-only source; separate from frozen core so its behavior is unchanged."""
    if y.dtype != np.uint8 or y.ndim != 2 or min(y.shape) < 1:
        raise ValueError('Expected nonempty uint8 Y')
    h, w = y.shape
    ys, xs = np.arange(0, h, 16), np.arange(0, w, 16)
    sums = np.add.reduceat(np.add.reduceat(y.astype(np.uint32), ys, axis=0), xs, axis=1)
    return (sums/(np.minimum(16, h-ys)[:, None]*np.minimum(16, w-xs)[None, :])).astype(np.float32)


def smooth_grid(raw):
    """@luna-comment Input: float32 cell grid. Output: two replicated separable121 rounds. Key Steps: horizontal then vertical each round. Key Params: positive normalized kernel. State/Dependencies: identical geometry to frozen mean-background function."""
    grid = np.asarray(raw, np.float32).copy()
    for _ in range(2):
        horizontal = np.pad(grid, ((0, 0), (1, 1)), mode='edge')
        scratch = (horizontal[:, :-2]+2.0*horizontal[:, 1:-1]+horizontal[:, 2:])*.25
        vertical = np.pad(scratch, ((1, 1), (0, 0)), mode='edge')
        grid = (vertical[:-2]+2.0*vertical[1:-1]+vertical[2:])*.25
    return grid


def make_grids(y, upper):
    """@luna-comment Input: Y8 and upper-anchor switch. Output: deployed mean M grid and optional closing7 upper U grid. Key Steps: call frozen mean estimator; close raw grid before matching two-round smoothing. Key Params: rectangular7x7, replicated boundary. State/Dependencies: upper is only a research anchor, not semantic paper detection."""
    mean = cell_background(y)
    if not upper:
        return mean, mean
    raw = raw_cell_grid(y)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_REPLICATE)
    upper_grid = smooth_grid(closed)
    if float(np.min(upper_grid-mean)) < -1e-3:
        raise AssertionError('Upper grid fell below mean grid')
    return mean, upper_grid


def anchored_dark(y, upper, gamma, anchor_offset=4.0):
    """@luna-comment Input: float32 Y/upper background, gamma2..4 and nonnegative anchor offset. Output: F-Y dark correction before polarity weight. Key Steps: A=max(U-offset,0), masked safe division, analytic integer power with C1 join at A. Key Params: exploratory offset4/8/12/16/20. State/Dependencies: no divide by zero or pixelwise pow."""
    if gamma not in (2, 3, 4):
        raise ValueError('Supported exploratory gamma is 2, 3 or 4')
    if not 0.0 <= anchor_offset <= 255.0:
        raise ValueError('Anchor offset must be in [0,255]')
    y = np.asarray(y, np.float32)
    upper = np.asarray(upper, np.float32)
    anchor = np.maximum(upper-anchor_offset, 0.0)
    active = (anchor>0.0)&(y<anchor)
    t = np.zeros_like(y, np.float32)
    np.divide(y, anchor, out=t, where=active)
    power = t*t
    if gamma >= 3:
        power = power*t
    if gamma == 4:
        power = power*t
    mapped = anchor*power*(gamma-(gamma-1)*t)
    return np.where(active, mapped-y, 0.0).astype(np.float32)


def curve_band(y, mean, upper, arm):
    """@luna-comment Input: Y/M/U bands and frozen arm. Output: current/candidate integer Y deltas and overlap telemetry. Key Steps: deployed current from frozen core, separate softcapped old dark/bright, alpha blend only dark, Y-headroom clamp/ties-even. Key Params: p(M), deadband2, old gain.75/cap24. State/Dependencies: alpha0 explicitly returns frozen current byte-for-byte."""
    current, _, current_stats = correction_pair(y, mean, ZERO_TRIAL)
    if arm.alpha == 0.0:
        return current, current, {**current_stats, 'old_bright_new_dark_overlap': 0,
                                  'overlap_cancellation_sum': 0.0,
                                  'new_dark_float_hits': 0}
    y = np.asarray(y, np.float32)
    mean = np.asarray(mean, np.float32)
    paper = smoothstep01((mean-80.0)/96.0)
    residual = y-mean
    qdark = -.75*np.maximum(-residual-2.0, 0.0)*paper
    qbright = .75*np.maximum(residual-2.0, 0.0)*(1.0-paper)
    old_dark = qdark/(1.0+np.abs(qdark)/24.0)
    old_bright = qbright/(1.0+np.abs(qbright)/24.0)
    new_dark = paper*anchored_dark(y, upper, arm.gamma, arm.anchor_offset)
    combined = old_bright+(1.0-arm.alpha)*old_dark+arm.alpha*new_dark
    candidate = np.rint(np.clip(combined, -y, 255.0-y)).astype(np.int16)
    both = (old_bright>1e-6)&(new_dark<-1e-6)
    cancellation = np.minimum(old_bright, -arm.alpha*new_dark)
    stats = {**current_stats,
             'old_bright_new_dark_overlap': int(np.count_nonzero(both)),
             'overlap_cancellation_sum': float(np.sum(cancellation[both], dtype=np.float64)),
             'new_dark_float_hits': int(np.count_nonzero(new_dark < -.1)),
             'candidate_minus_current_integer_hits': int(np.count_nonzero(candidate-current)),
             'candidate_minus_current_max_abs': int(np.max(np.abs(candidate-current)))}
    return current, candidate, stats


def process_arm(source, arm, band_rows=128):
    """@luna-comment Input: immutable BGR8 source and one research arm. Output: current/candidate RGB-preview arrays plus branch/host telemetry. Key Steps: derive Y once, shared M/U grids, bandwise curve and common BGR delta. Key Params: serial128-row bands. State/Dependencies: PNG-derived Y preview only; native UV/NEON/time not proved."""
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3 or band_rows < 1:
        raise ValueError('Expected BGR8 source')
    began = time.perf_counter()
    y = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mean_grid, upper_grid = make_grids(y, arm.upper)
    grid_seconds = time.perf_counter()-began
    current = np.empty_like(source)
    candidate = np.empty_like(source)
    stats = {}
    rgb_clip_current = rgb_clip_candidate = 0
    h, w = y.shape
    curve_began = time.perf_counter()
    for top in range(0, h, band_rows):
        end = min(h, top+band_rows)
        mean = background_band(mean_grid, top, end, w)
        upper = background_band(upper_grid, top, end, w) if arm.upper else mean
        if float(np.min(upper-mean)) < -1e-3:
            raise AssertionError('Interpolated upper fell below mean')
        d0, d1, detail = curve_band(y[top:end], mean, upper, arm)
        current[top:end], c0 = preview_bgr(source[top:end], d0)
        candidate[top:end], c1 = preview_bgr(source[top:end], d1)
        rgb_clip_current += c0
        rgb_clip_candidate += c1
        for key, value in detail.items():
            stats[key] = max(stats.get(key, 0), value) if key.endswith('_max_abs') else stats.get(key, 0)+value
    stats.update({'rgb_clipped_channels_current': rgb_clip_current,
                  'rgb_clipped_channels_candidate': rgb_clip_candidate,
                  'changed_pixels_candidate_vs_current': int(np.count_nonzero(np.any(candidate!=current, axis=2))),
                  'changed_pixels_candidate_vs_source': int(np.count_nonzero(np.any(candidate!=source, axis=2))),
                  'grid_seconds_host': grid_seconds,
                  'curve_preview_seconds_host': time.perf_counter()-curve_began})
    return current, candidate, stats
