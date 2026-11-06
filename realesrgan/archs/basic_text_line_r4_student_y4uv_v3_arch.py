"""Parallel six-channel BASIC V2 R4 Student architecture.

This module intentionally does not replace the legacy R4 adapter.  Its public
input, skip, and output are all ``[B,6,H/2,W/2]`` Y4UV420 tensors.  The
``haar_switch`` selects the frequency transform at the graph boundary:
enabled mode reproduces the existing four-band luma Haar equations, while
disabled mode sends the complete six-channel tensor through one ordinary
learnable convolution without Y/UV channel packing operations.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from .ecbsr_Unet_arch_server import HaarLikeToY4, Y4ToHaarLikeWavelet, _ECBSRUnetBranch


Y4UV_CHANNELS = 6
R4_Y4UV_EXPAND_CHANNEL = (0.5, 1.0, 2.0, 4.0)
R4_Y4UV_LAYER_NUM = 3
R4_Y4UV_DEPTH_FACTOR = 1
R4_Y4UV_SCALE_FACTOR = 2


def _coerce_switch(value: Any) -> bool:
    """@luna-comment Input: YAML boolean/string. Output: validated boolean.
    Key Steps: parse explicit true/false spellings; reject unknown strings.
    Key Params: case-insensitive recognized literals. State/Dependencies: no shared state.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"invalid haar_switch value: {value!r}")
    return bool(value)


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV3(nn.Module):
    """Fixed-canvas R4 Student with native Y4UV420 residual skip/output."""

    def __init__(
        self,
        canvas_hw=(96, 320),
        in_channel=Y4UV_CHANNELS,
        out_channel=Y4UV_CHANNELS,
        layer_num=R4_Y4UV_LAYER_NUM,
        expand_channel=R4_Y4UV_EXPAND_CHANNEL,
        depth_factor=R4_Y4UV_DEPTH_FACTOR,
        scale_factor=R4_Y4UV_SCALE_FACTOR,
        with_idt=False,
        act_type="prelu",
        scale=2,
        with_proj=True,
        haar_switch=True,
        learnable_haar=False,
        **kwargs,
    ):
        """@luna-comment Input: fixed canvas, width/depth and Haar/activation options.
        Output: validated native six-channel residual model with zero-initialized projection.
        Key Steps: validate fixed V3 geometry, build branch and selected boundary transforms.
        Key Params: three levels, widths3/6/12/24, no spatial upsampler.
        State/Dependencies: existing ECB branch, torch modules and random initialization.
        """
        super().__init__()
        if len(canvas_hw) != 2:
            raise ValueError("canvas_hw must contain height and width")
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        if any(value <= 0 or value % 32 for value in self.canvas_hw):
            raise ValueError("canvas_hw values must be positive multiples of 32")
        if int(in_channel) != Y4UV_CHANNELS or int(out_channel) != Y4UV_CHANNELS:
            raise ValueError("Y4UV R4 Student requires in_channel=out_channel=6")
        if int(layer_num) != R4_Y4UV_LAYER_NUM:
            raise ValueError("Y4UV R4 Student requires layer_num=3")
        expand = tuple(float(value) for value in expand_channel)
        if expand != R4_Y4UV_EXPAND_CHANNEL:
            raise ValueError(f"expand_channel must be {R4_Y4UV_EXPAND_CHANNEL}")
        if int(depth_factor) != R4_Y4UV_DEPTH_FACTOR or int(scale_factor) != R4_Y4UV_SCALE_FACTOR:
            raise ValueError("Y4UV R4 Student requires depth_factor=1 and scale_factor=2")
        self.haar_switch = _coerce_switch(haar_switch)
        self.learnable_haar = bool(learnable_haar)

        # The branch is reused as a public implementation component.  It
        # remains at half resolution because use_upsampler=False; unlike the
        # legacy V6 graph, no output PixelShuffle or chroma interpolation is
        # present here.
        self.backbone = _ECBSRUnetBranch(
            in_channel=Y4UV_CHANNELS,
            out_channel=Y4UV_CHANNELS,
            layer_num=R4_Y4UV_LAYER_NUM,
            expand_channel=list(expand),
            depth_factor=R4_Y4UV_DEPTH_FACTOR,
            scale_factor=R4_Y4UV_SCALE_FACTOR,
            with_idt=with_idt,
            act_type=act_type,
            scale=scale,
            with_proj=with_proj,
            use_upsampler=False,
            input_conv=False,
        )

        if self.haar_switch:
            self.y4_to_haar: nn.Module | None = Y4ToHaarLikeWavelet(learnable_scale=self.learnable_haar)
            self.haar_to_y4: nn.Module | None = HaarLikeToY4(learnable_scale=self.learnable_haar)
            self.input_transform: nn.Module | None = None
        else:
            self.y4_to_haar = None
            self.haar_to_y4 = None
            # Disabled mode deliberately receives all six channels in one
            # tensor.  There is no Y4/UV split, repeat, or concatenation.
            self.input_transform = nn.Conv2d(Y4UV_CHANNELS, Y4UV_CHANNELS, kernel_size=3, padding=1)

        self.residual_projection = nn.Conv2d(Y4UV_CHANNELS, Y4UV_CHANNELS, kernel_size=3, padding=1)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    @property
    def expected_input_hw(self) -> tuple[int, int]:
        """@luna-comment Input: stored canvas. Output: expected packed spatial dimensions.
        Key Steps: divide canvas by2. Key Params: Y4UV packing factor2.
        State/Dependencies: immutable constructor geometry.
        """
        return self.canvas_hw[0] // 2, self.canvas_hw[1] // 2

    def _prepare_features(self, input_y4uv: torch.Tensor) -> torch.Tensor:
        """@luna-comment Input: native six-channel packed tensor. Output: transformed branch input.
        Key Steps: transform only Y4 and preserve UV, or use direct learned convolution.
        Key Params: haar_switch. State/Dependencies: constructed boundary modules.
        """
        if not self.haar_switch:
            assert self.input_transform is not None
            return self.input_transform(input_y4uv)
        assert self.y4_to_haar is not None
        y4 = self.y4_to_haar(input_y4uv[:, :4])
        return torch.cat((y4, input_y4uv[:, 4:]), dim=1)

    def _decode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """@luna-comment Input: branch six-channel residual. Output: native Y4UV residual.
        Key Steps: inverse-transform Y4 while preserving UV or pass direct mode through.
        Key Params: haar_switch. State/Dependencies: constructed inverse boundary module.
        """
        if not self.haar_switch:
            return residual
        assert self.haar_to_y4 is not None
        return torch.cat((self.haar_to_y4(residual[:, :4]), residual[:, 4:]), dim=1)

    def forward(self, input_y4uv: torch.Tensor) -> torch.Tensor:
        """@luna-comment Input: BCHW six-channel tensor at fixed packed canvas dimensions.
        Output: same-shaped residual-plus-input Y4UV prediction.
        Key Steps: validate, prepare features, run branch/projection, decode and add identity skip.
        Key Params: fixed canvas/Haar choice. State/Dependencies: learned modules and input tensor.
        """
        if not isinstance(input_y4uv, torch.Tensor) or input_y4uv.ndim != 4:
            raise ValueError("input must be [B,6,H/2,W/2] Y4UV420")
        if input_y4uv.shape[1] != Y4UV_CHANNELS:
            raise ValueError(f"input must have six channels, got {input_y4uv.shape[1]}")
        if tuple(input_y4uv.shape[-2:]) != self.expected_input_hw:
            raise ValueError(
                f"expected half-resolution shape {self.expected_input_hw}, "
                f"got {tuple(input_y4uv.shape[-2:])}"
            )
        features = self._prepare_features(input_y4uv)
        residual = self.residual_projection(self.backbone(features))
        residual = self._decode_residual(residual)
        output = input_y4uv + residual
        if output.shape != input_y4uv.shape:
            raise RuntimeError(f"Y4UV output contract failed: {tuple(output.shape)}")
        return output


__all__ = [
    "BasicTextLineR4StudentY4UVV3",
    "R4_Y4UV_EXPAND_CHANNEL",
    "R4_Y4UV_LAYER_NUM",
    "R4_Y4UV_DEPTH_FACTOR",
    "R4_Y4UV_SCALE_FACTOR",
]
