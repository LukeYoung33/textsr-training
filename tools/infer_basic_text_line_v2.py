"""Pure NumPy helpers for BASIC V2 segment and full-frame fusion."""

from __future__ import annotations

import numpy as np


def merge_line_segments(segments, ranges, frame_hw):
    """@shanlan-comment Merge same-line outputs with center-weighted overlap windows."""
    height, width = map(int, frame_hw)
    if not segments or len(segments) != len(ranges):
        raise ValueError("segments and ranges must have equal nonzero length")
    channels = segments[0].shape[0]
    output = np.zeros((channels, height, width), np.float32)
    weights = np.zeros((1, height, width), np.float32)
    for segment, (x0, x1) in zip(segments, ranges):
        if segment.ndim != 3 or segment.shape[1] != height:
            raise ValueError("segment shape mismatch")
        local_width = min(segment.shape[2], int(x1 - x0))
        if local_width <= 0 or x0 < 0 or x1 > width:
            raise ValueError("invalid segment range")
        window = np.ones(local_width, np.float32)
        if local_width > 1:
            window *= np.sin(np.linspace(0, np.pi, local_width, dtype=np.float32))
            window = np.maximum(window, 1e-3)
        output[:, :, x0:x0+local_width] += segment[:, :, :local_width] * window[None, None]
        weights[:, :, x0:x0+local_width] += window[None, None]
    return output / np.maximum(weights, 1e-6)


def alpha_blend_yuv444(base, restored, alpha):
    """@shanlan-comment Blend restored YUV444 only where alpha is nonzero."""
    base = np.asarray(base, np.float32); restored = np.asarray(restored, np.float32)
    if base.shape != restored.shape or base.ndim != 3 or base.shape[0] != 3:
        raise ValueError("base/restored must both be [3,H,W]")
    alpha = np.asarray(alpha, np.float32)
    if alpha.ndim == 2: alpha = alpha[None]
    if alpha.shape != (1, base.shape[1], base.shape[2]):
        raise ValueError("alpha must be [H,W] or [1,H,W]")
    return alpha * restored + (1.0 - alpha) * base
