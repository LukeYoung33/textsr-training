"""Tensor helpers for the isolated BASIC V2 Y4UV420 training path.

The legacy BASIC path exposes packed NV21 input and a YUV444 output.  This
module deliberately uses a different contract: four luma sub-pixels followed
by the native half-resolution U and V planes, all at half-resolution spatial
coordinates.  No helper in this file mutates the legacy conversion code.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F


Y4UV_CHANNELS = 6
Y4_CHANNELS = 4
UV_CHANNELS = 2


def _plane_to_bchw(value: torch.Tensor, name: str) -> torch.Tensor:
    """@luna-comment
    Normalize a single YUV plane to ``[B,1,H,W]``.

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 2:
        return value.unsqueeze(0).unsqueeze(0)
    if value.ndim == 3:
        if value.shape[0] == 1:
            return value.unsqueeze(0)
        return value.unsqueeze(1)
    if value.ndim == 4 and value.shape[1] == 1:
        return value
    raise ValueError(f"{name} must be HW, BHW, CHW, or B1HW; got {tuple(value.shape)}")


def _restore_planes(value: torch.Tensor, original_ndim: int, original_shape: tuple[int, ...]):
    """@luna-comment
    Restore planar output rank while keeping the BCHW path canonical.

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    if original_ndim == 2:
        return value[0, 0]
    if original_ndim == 3 and original_shape[0] == 1:
        return value[0]
    if original_ndim == 3:
        return value[:, 0]
    return value


def yuv420_to_y4uv(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """@luna-comment
    Pack YUV420 planes as ``[Y_TL,Y_TR,Y_BL,Y_BR,U,V]``.
    
        The Y plane must be exactly twice the height and width of U/V.  A 2-D
        input returns ``[6,h,w]`` for convenient dataset samples; batched input
        returns ``[B,6,h,w]``.
        

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    y_original_ndim, y_original_shape = y.ndim, tuple(y.shape)
    y_bchw = _plane_to_bchw(y, "y")
    u_bchw = _plane_to_bchw(u, "u").to(device=y_bchw.device, dtype=y_bchw.dtype)
    v_bchw = _plane_to_bchw(v, "v").to(device=y_bchw.device, dtype=y_bchw.dtype)
    if y_bchw.shape[0] != u_bchw.shape[0] or u_bchw.shape != v_bchw.shape:
        raise ValueError(
            "Y/U/V batch and chroma shapes must match, got "
            f"{tuple(y_bchw.shape)}, {tuple(u_bchw.shape)}, {tuple(v_bchw.shape)}"
        )
    if y_bchw.shape[-2] != 2 * u_bchw.shape[-2] or y_bchw.shape[-1] != 2 * u_bchw.shape[-1]:
        raise ValueError("YUV420 requires Y height/width to be twice U/V height/width")
    if y_bchw.shape[-2] % 2 or y_bchw.shape[-1] % 2:
        raise ValueError("Y4UV packing requires even Y height and width")
    y4 = F.pixel_unshuffle(y_bchw, downscale_factor=2)
    packed = torch.cat((y4, u_bchw, v_bchw), dim=1)
    if y_original_ndim == 2 or (y_original_ndim == 3 and y_original_shape[0] == 1):
        return packed[0]
    return packed


def y4uv_to_yuv420(y4uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """@luna-comment
    Unpack a 6-channel tensor into Y, U, and V planes without upsampling.

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    if not isinstance(y4uv, torch.Tensor) or y4uv.ndim not in (3, 4):
        raise ValueError("y4uv must be CHW or BCHW")
    original_ndim = y4uv.ndim
    if y4uv.shape[-3] != Y4UV_CHANNELS:
        raise ValueError(f"y4uv must have six channels, got {y4uv.shape[-3]}")
    value = y4uv.unsqueeze(0) if original_ndim == 3 else y4uv
    y = F.pixel_shuffle(value[:, :Y4_CHANNELS], upscale_factor=2)
    u, v = value[:, 4:5], value[:, 5:6]
    if original_ndim == 3:
        return y[0], u[0], v[0]
    return y, u, v


def y4uv_to_yuv444(y4uv: torch.Tensor) -> torch.Tensor:
    """@luna-comment
    Decode Y4UV420 to a temporary YUV444 tensor for comparison losses only.

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    y, u, v = y4uv_to_yuv420(y4uv)
    batched = y.ndim == 4
    if not batched:
        y, u, v = y.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0)
    uv_size = y.shape[-2:]
    uv = F.interpolate(torch.cat((u, v), dim=1), size=uv_size, mode="bilinear", align_corners=False)
    output = torch.cat((y, uv), dim=1)
    return output if batched else output[0]


def yuv420_mask_to_y4uv(mask: torch.Tensor) -> torch.Tensor:
    """@luna-comment
    Convert a full-resolution single-channel mask to a channel-aligned mask.
    
        The first four mask channels follow Y pixel-unshuffle exactly.  The two
        chroma masks use area pooling, matching the native half-resolution U/V
        planes.  A six-channel half-resolution mask is validated and returned.
        

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim not in (2, 3, 4):
        raise ValueError("mask must be HW, BHW/CHW, or BCHW")
    if mask.ndim == 4 and mask.shape[1] == Y4UV_CHANNELS:
        if mask.shape[-2] <= 0 or mask.shape[-1] <= 0:
            raise ValueError("six-channel mask has invalid spatial shape")
        return mask
    if mask.ndim == 3 and mask.shape[0] == Y4UV_CHANNELS:
        return mask
    value = _plane_to_bchw(mask, "mask")
    if value.shape[-2] % 2 or value.shape[-1] % 2:
        raise ValueError("full-resolution mask dimensions must be even")
    y4 = F.pixel_unshuffle(value, downscale_factor=2)
    uv = F.interpolate(value, size=(value.shape[-2] // 2, value.shape[-1] // 2), mode="area")
    output = torch.cat((y4, uv, uv), dim=1).clamp(0.0, 1.0)
    if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[0] == 1):
        return output[0]
    return output


def y4uv_mask_to_yuv444(mask: torch.Tensor, full_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """@luna-comment
    Decode a Y4UV mask to three full-resolution YUV masks.

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    original_ndim = mask.ndim
    if mask.ndim == 3:
        value = mask.unsqueeze(0)
    elif mask.ndim == 4:
        value = mask
    else:
        raise ValueError("mask must be CHW or BCHW")
    if value.shape[1] == 1:
        if full_hw is not None and tuple(value.shape[-2:]) != tuple(full_hw):
            raise ValueError("one-channel mask does not match requested full resolution")
        output = value.expand(-1, 3, -1, -1)
    elif value.shape[1] == Y4UV_CHANNELS:
        y = F.pixel_shuffle(value[:, :4], upscale_factor=2)
        if full_hw is not None and tuple(y.shape[-2:]) != tuple(full_hw):
            raise ValueError("Y4UV mask does not decode to requested full resolution")
        uv = F.interpolate(value[:, 4:6], size=y.shape[-2:], mode="bilinear", align_corners=False)
        output = torch.cat((y, uv), dim=1)
    else:
        raise ValueError(f"mask must have one or six channels, got {value.shape[1]}")
    return output if original_ndim == 4 else output[0]


def rgb_to_y4uv(rgb: torch.Tensor) -> torch.Tensor:
    """@luna-comment
    Convert RGB ``[0,1]`` CHW/BCHW data to normalized Y4UV420.

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    if not isinstance(rgb, torch.Tensor) or rgb.ndim not in (3, 4):
        raise ValueError("rgb must be CHW or BCHW")
    original_ndim = rgb.ndim
    value = rgb.unsqueeze(0) if original_ndim == 3 else rgb
    if value.shape[1] != 3:
        raise ValueError("rgb must have three channels")
    if value.shape[-2] % 2 or value.shape[-1] % 2:
        raise ValueError("RGB dimensions must be even for YUV420 conversion")
    matrix = value.new_tensor(
        [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
    )
    yuv = torch.einsum("ij,bjhw->bihw", matrix, value)
    yuv[:, 1:] = yuv[:, 1:] + 0.5
    u = F.avg_pool2d(yuv[:, 1:2], kernel_size=2, stride=2)
    v = F.avg_pool2d(yuv[:, 2:3], kernel_size=2, stride=2)
    packed = yuv420_to_y4uv(yuv[:, :1], u, v)
    return packed if original_ndim == 4 else packed[0]


def numpy_rgb_to_y4uv(rgb: np.ndarray) -> torch.Tensor:
    """@luna-comment
    Convert an RGB HWC NumPy image in ``[0,1]`` to a CHW Y4UV tensor.

    Input: Tensor planes or image values with ranks validated below.
    Output: Converted tensor with the documented spatial/channel layout.
    Key Steps: Validate rank and shape, convert layout or color as implemented.
    Key Params: Y4UV uses four luma phases and native half-resolution U/V.
    State/Dependencies: PyTorch functional operations; no persistent state.
    """
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be an HWC NumPy array with three channels")
    value = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32)).permute(2, 0, 1)
    return rgb_to_y4uv(value)


__all__ = [
    "Y4UV_CHANNELS",
    "Y4_CHANNELS",
    "UV_CHANNELS",
    "yuv420_to_y4uv",
    "y4uv_to_yuv420",
    "y4uv_to_yuv444",
    "yuv420_mask_to_y4uv",
    "y4uv_mask_to_yuv444",
    "rgb_to_y4uv",
    "numpy_rgb_to_y4uv",
]
