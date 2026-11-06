"""Fixed-shape BASIC 2.0 model with a small residual YUV path."""

from __future__ import annotations

import torch
from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BasicTextLineRestoreNetV2(nn.Module):
    """Static-shape 1x YUV420 strip to planar YUV444 residual network."""

    def __init__(self, canvas_hw=(96, 320), in_channel=1, out_channel=3, features=32, shape_frozen=False, **kwargs):
        super().__init__()
        self.canvas_hw = tuple(int(v) for v in canvas_hw)
        if len(self.canvas_hw) != 2 or any(v <= 0 or v % 32 for v in self.canvas_hw):
            raise ValueError("canvas_hw must be positive multiples of 32")
        self.shape_frozen = bool(shape_frozen)
        self.body = nn.Sequential(
            nn.Conv2d(3, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, 3, 3, padding=1),
        )

    def forward(self, input_yuv420: torch.Tensor) -> torch.Tensor:
        """@shanlan-comment Validate fixed NV21 shape, upsample chroma, and emit YUV444 residual output."""
        if input_yuv420.ndim != 4 or input_yuv420.shape[1] != 1:
            raise ValueError("input must be packed NV21 [B,1,H+H/2,W]")
        h, w = self.canvas_hw
        if input_yuv420.shape[-2:] != (h + h // 2, w):
            raise ValueError(f"expected packed shape {(h + h // 2, w)}, got {tuple(input_yuv420.shape[-2:])}")
        y = input_yuv420[:, :, :h]
        # NV21 stores interleaved VU samples across each chroma row.
        uv = input_yuv420[:, :, h:].reshape(input_yuv420.shape[0], 1, h // 2, w // 2, 2)
        uv = uv.permute(0, 1, 4, 2, 3).reshape(input_yuv420.shape[0], 2, h // 2, w // 2)
        v, u = uv[:, :1], uv[:, 1:]
        u = torch.nn.functional.interpolate(u, size=(h, w), mode="bilinear", align_corners=False)
        v = torch.nn.functional.interpolate(v, size=(h, w), mode="bilinear", align_corners=False)
        base = torch.cat([y, u, v], dim=1)
        return base + self.body(base)
