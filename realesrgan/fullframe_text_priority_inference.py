"""Native-size, phase-aligned full-frame inference without text detection."""

from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.utils import YUV2RGB


def _pair(value, name):
    """@luna-comment
    Input: integer or pair of RGB pixel counts.
    Output: validated integer pair, or ValueError.
    Key Steps: normalize scalar and reject noninteger values.
    Key Params: name identifies the invalid setting.
    State/Dependencies: no shared state.
    """
    try:
        values = (value, value) if isinstance(value, int) else tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer or two integers") from error
    if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError(f"{name} must be an integer or two integers")
    return values


def tile_geometry(canvas_hw, halo, stride=None):
    """@luna-comment
    Input: RGB canvas, context halo, optional center step in pixels.
    Output: canvas, halo, center, stride pairs; invalid geometry raises.
    Key Steps: require phase alignment and overlapping or adjoining centers.
    Key Params: every origin/extent uses phase 8; halo may be zero.
    State/Dependencies: V5 spatial-phase contract; no shared state.
    """
    canvas = _pair(canvas_hw, "canvas_hw")
    context = _pair(halo, "halo")
    center = tuple(c - 2 * h for c, h in zip(canvas, context))
    step = center if stride is None else _pair(stride, "stride")
    if any(c <= 0 or c % 8 for c in canvas):
        raise ValueError("canvas_hw must contain positive multiples of 8")
    if any(h < 0 or h % 8 for h in context) or any(c <= 0 for c in center):
        raise ValueError("halo must contain nonnegative multiples of 8 and leave a positive center")
    if any(s <= 0 or s % 8 or s > c for s, c in zip(step, center)):
        raise ValueError("stride must contain positive multiples of 8 no greater than the center")
    return canvas, context, center, step


@torch.inference_mode()
def restore_fullframe_rgb(network, rgb, *, canvas_hw=(256, 256), halo=32,
                          stride=None, batch_size=1, device=None, clamp=False,
                          return_report=False):
    """@luna-comment
    Input: network accepting native Y4UV420; floating CHW/BCHW RGB in [0,1].
    Output: same-rank native-size RGB on input device; optional coverage report.
    Key Steps: replicate-pad global grid, infer context tiles, average positive centers.
    Key Params: canvas/halo/stride are phase-8 pixels; batch_size controls tile batches;
        clamp clips final RGB only, disabled by default.
    State/Dependencies: temporarily sets network eval and restores its training flag;
        network is expected on device; uses existing Y4UV color helpers.
    """
    canvas, context, center, step = tile_geometry(canvas_hw, halo, stride)
    if not isinstance(rgb, torch.Tensor) or rgb.ndim not in (3, 4):
        raise ValueError("rgb must be a CHW or BCHW tensor")
    single = rgb.ndim == 3
    value = rgb.unsqueeze(0) if single else rgb
    if value.shape[1] != 3 or min(value.shape[0], *value.shape[-2:]) <= 0:
        raise ValueError("rgb requires nonempty three-channel images")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError("rgb must be finite floating point")
    if value.min() < 0 or value.max() > 1:
        raise ValueError("input RGB must lie in [0,1]")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    parameter = next(network.parameters(), None)
    compute_device = torch.device(device) if device is not None else (
        parameter.device if parameter is not None else value.device)
    compute_dtype = parameter.dtype if parameter is not None else value.dtype
    if hasattr(network, "canvas_hw") and tuple(network.canvas_hw) != canvas:
        raise ValueError("inference canvas_hw differs from network canvas_hw")
    value = value.to(device=compute_device, dtype=compute_dtype)
    height, width = value.shape[-2:]
    # Include the first grid center that covers the final pixel; never shift origins.
    ys = list(range(0, math.ceil(max(height - center[0], 0) / step[0]) * step[0] + 1, step[0]))
    xs = list(range(0, math.ceil(max(width - center[1], 0) / step[1]) * step[1] + 1, step[1]))
    padded = F.pad(value, (context[1], xs[-1] + canvas[1] - context[1] - width,
                           context[0], ys[-1] + canvas[0] - context[0] - height), mode="replicate")
    jobs = [(b, y, x) for b in range(value.shape[0]) for y in ys for x in xs]
    sums = torch.zeros_like(value)
    weights = value.new_zeros((value.shape[0], 1, height, width))
    training = network.training
    network.eval()
    try:
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            tiles = torch.stack([padded[b, :, y:y + canvas[0], x:x + canvas[1]] for b, y, x in batch])
            native = rgb_to_y4uv(tiles)
            prediction = network(native)
            if not isinstance(prediction, torch.Tensor) or prediction.shape != native.shape:
                raise ValueError("network must return the same native Y4UV tensor shape")
            if not torch.isfinite(prediction).all():
                raise ValueError("network returned nonfinite values")
            decoded = YUV2RGB(y4uv_to_yuv444(prediction), format="RGB", is_train=True, uv_normalized=True)
            for i, (b, y, x) in enumerate(batch):
                h, w = min(center[0], height - y), min(center[1], width - x)
                crop = decoded[i, :, context[0]:context[0] + h, context[1]:context[1] + w]
                sums[b, :, y:y + h, x:x + w] += crop
                weights[b, :, y:y + h, x:x + w] += 1
    finally:
        network.train(training)
    if not torch.all(weights > 0):
        raise RuntimeError("inference grid left uncovered real pixels")
    result = sums / weights
    if clamp:
        result = result.clamp(0, 1)
    result = result.to(device=rgb.device, dtype=rgb.dtype)
    result = result[0] if single else result
    report = {"input_hw": [height, width], "canvas_hw": list(canvas), "halo": list(context),
              "center_hw": list(center), "stride": list(step), "tile_count": len(jobs),
              "batch_size": batch_size, "coverage_min": weights.min().item(),
              "coverage_max": weights.max().item(), "clamp_output": bool(clamp),
              "padding": "replicate", "phase": 8}
    return (result, report) if return_report else result
