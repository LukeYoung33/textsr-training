"""Approved inference-only crop8/smooth16 fusion for native256/224 patches."""
import numpy as np


def axis_weights(origin, extent):
    """@luna-comment Input:tile axis origin/native extent. Output:256 float weights.
    Key Steps:discard outer8 predictions, smoothstep remaining16 overlap, retain
    physical-frame boundary coverage. Key Params:256 canvas224 stride.
    State/Dependencies:NumPy; matches user-approved030 crop8 candidate exactly.
    """
    i = np.arange(256)
    t = np.clip((np.minimum(i, 255 - i) - 8) / 15, 0, 1)
    t = t * t * (3 - 2 * t)
    if origin == 0:
        t[:128] = 1
    if origin + 256 >= extent:
        t[128:] = 1
    return t.astype(np.float32)


def restore_block(rgb, network, device):
    """@luna-comment Input:RGB8/native network/device. Output:RGB8 and stitch telemetry.
    Key Steps:unchanged even256/224 tiles/edge padding/model transforms; discard
    outer8 output pixels and smooth16/product weights; normalize all contributions.
    Key Params:batch8; no OCR/resize/tone/source blend. State/Dependencies:CUDA and
    existing model transforms; does not change training or historical stitcher.
    """
    import torch
    from tools.auto_hx_v3_native import positions
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    h, w = rgb.shape[:2]
    grid = [(x, y) for y in positions(h, 256, 224) for x in positions(w, 256, 224)]
    total = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w), np.float32)
    for start in range(0, len(grid), 8):
        selected = grid[start:start + 8]
        patches = []
        for x, y in selected:
            patch = rgb[y:y + 256, x:x + 256]
            patches.append(np.pad(patch, ((0, 256 - patch.shape[0]),
                                         (0, 256 - patch.shape[1]), (0, 0)), mode='edge'))
        batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2).copy()).to(device).float() / 255
        with torch.inference_mode():
            decoded = YUV2RGB(y4uv_to_yuv444(network(rgb_to_y4uv(batch))), is_train=True).clamp(0, 1)
        if decoded.shape != batch.shape or not torch.isfinite(decoded).all():
            raise ValueError('Invalid block prediction')
        for (x, y), value in zip(selected, decoded.permute(0, 2, 3, 1).cpu().numpy()):
            hh, ww = min(256, h - y), min(256, w - x)
            window = axis_weights(y, h)[:, None] * axis_weights(x, w)[None, :]
            total[y:y + hh, x:x + ww] += value[:hh, :ww] * window[:hh, :ww, None]
            weights[y:y + hh, x:x + ww] += window[:hh, :ww]
    if not np.isfinite(weights).all() or not np.all(weights > 0):
        raise ValueError('Uncovered full-frame pixels')
    final = np.clip(np.rint(total / weights[..., None] * 255), 0, 255).astype(np.uint8)
    return final, dict(canvas_hw=[256, 256], stride_hw=[224, 224], patches=len(grid),
                       ocr_cropping=False, stitch_policy='crop8_smooth16', crop_pixels=8,
                       blend_pixels=16, axis_combination='product', weight_min=float(weights.min()))
