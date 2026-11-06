"""R4 ECBSR Student adapter for BASIC 2.0 fixed text-line strips.

This module adds a new registry name around the existing ECBSR Student graph.
It does not alter the legacy architecture or its registrations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY
from .ecbsr_Unet_arch_server import ECBSRUnet_StdV6 as _R4StudentGraph


# The server-side V6 graph is the validated R4 Student implementation. It is
# imported under a new adapter name without changing its legacy registration.


R4_STUDENT_EXPAND_CHANNEL = (0.5, 1.0, 2.0, 4.0)
R4_STUDENT_LAYER_NUM = 3
R4_STUDENT_DEPTH_FACTOR = 1
R4_STUDENT_SCALE_FACTOR = 2


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentV2(_R4StudentGraph):
    """Fixed-shape R4 ECBSR Student using canonical packed NV21 input."""

    def __init__(
        self,
        canvas_hw=(96, 320),
        in_channel=12,
        out_channel=12,
        layer_num=R4_STUDENT_LAYER_NUM,
        expand_channel=R4_STUDENT_EXPAND_CHANNEL,
        depth_factor=R4_STUDENT_DEPTH_FACTOR,
        scale_factor=R4_STUDENT_SCALE_FACTOR,
        with_idt=False,
        act_type="prelu",
        scale=2,
        with_proj=True,
        **kwargs,
    ):
        """@shanlan-comment
        Input: fixed canvas Hc/Wc and packed NV21 tensor contract.
        Output: same-size planar YUV444 tensor with three Y/U/V channels.
        Key Steps: validate the R4 Student hyperparameters, construct the existing
        ECBSR U-Net graph, and enforce static packed input dimensions in forward.
        Key Params: layer_num=3, expand=[0.5,1,2,4], depth=1, scale=2.
        State/Dependencies: delegates graph operations to the existing ECBSR Student.
        """
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        if len(self.canvas_hw) != 2 or any(value <= 0 or value % 32 for value in self.canvas_hw):
            raise ValueError("canvas_hw values must be positive multiples of 32")
        if int(in_channel) != 12 or int(out_channel) != 12:
            raise ValueError("R4 Student adapter requires in_channel=12 and out_channel=12")
        if int(layer_num) != R4_STUDENT_LAYER_NUM:
            raise ValueError("BASIC V2 R4 Student requires layer_num=3")
        expand = tuple(float(value) for value in expand_channel)
        if expand != R4_STUDENT_EXPAND_CHANNEL:
            raise ValueError(f"R4 Student expand_channel must be {R4_STUDENT_EXPAND_CHANNEL}")
        super().__init__(
            in_channel=12,
            out_channel=12,
            layer_num=layer_num,
            expand_channel=list(expand),
            depth_factor=depth_factor,
            scale_factor=scale_factor,
            with_idt=with_idt,
            act_type=act_type,
            scale=scale,
            with_proj=with_proj,
        )

    def forward(self, input_yuv420: torch.Tensor) -> torch.Tensor:
        """@shanlan-comment
        Input: packed NV21 tensor [B,1,H+H/2,W] at the configured fixed canvas.
        Output: YUV444 tensor [B,3,H,W] with finite values, or a contract error.
        Key Steps: validate the packed geometry, run the ECBSR graph, collapse
        Y4UV output when present, then resize chroma to the luma canvas.
        Key Params: canvas_hw fixes H/W and prevents accidental dynamic-canvas use.
        State/Dependencies: uses the graph built by _R4StudentGraph.__init__.
        """
        height, width = self.canvas_hw
        expected = (height + height // 2, width)
        if input_yuv420.ndim != 4 or input_yuv420.shape[1] != 1:
            raise ValueError("input must be [B,1,H+H/2,W] packed NV21")
        if tuple(input_yuv420.shape[-2:]) != expected:
            raise ValueError(f"expected packed NV21 spatial shape {expected}, got {tuple(input_yuv420.shape[-2:])}")
        output = super().forward(input_yuv420)
        if output.shape[1] == 12:
            # The validated Student emits Y4UV at scale=2; collapse its four
            # luma subpixels to same-size Y and upsample U/V for the V2 contract.
            y4 = output[:, :4]
            y = y4.mean(dim=1, keepdim=True)
            uv = output[:, 4:6] if output.shape[1] >= 6 else output[:, 1:3]
            uv = torch.nn.functional.interpolate(uv, size=(height, width), mode="bilinear", align_corners=False)
            output = torch.cat([y, uv], dim=1)
        if output.shape[1:] != (3, height, width):
            raise RuntimeError(f"R4 Student output contract failed: {tuple(output.shape)}")
        return output


def sha256_file(path: str | Path) -> str:
    """@shanlan-comment
    Input: readable checkpoint or manifest path.
    Output: lowercase SHA-256 digest; file contents are not changed.
    Key Steps: stream fixed-size blocks into a SHA-256 digest.
    Key Params: block size is 1 MiB to avoid loading checkpoints into memory.
    State/Dependencies: depends only on the local file and hashlib.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_strict_checkpoint(model: nn.Module, checkpoint_path: str | Path, *, map_location="cpu") -> str:
    """@shanlan-comment
    Input: model instance and a checkpoint containing params_ema, params, or state_dict.
    Output: strict model load plus the checkpoint SHA-256; incompatible state raises.
    Key Steps: select the parameter mapping, strip a distributed module prefix,
    perform strict loading, and hash the source file.
    Key Params: map_location controls torch deserialization placement.
    State/Dependencies: mutates model parameters and uses torch checkpoint format.
    """
    path = Path(checkpoint_path)
    state = torch.load(str(path), map_location=map_location)
    if isinstance(state, Mapping):
        for key in ("params_ema", "params", "state_dict"):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must contain a parameter mapping")
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return sha256_file(path)


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentV2Large(BasicTextLineR4StudentV2):
    """Compute-scaled BASIC V2 graph with the unchanged packed-NV21 contract."""

    def __init__(
        self,
        canvas_hw=(96, 320),
        in_channel=12,
        out_channel=12,
        layer_num=4,
        expand_channel=(1.0, 1.5, 2.0, 3.0, 4.0),
        depth_factor=1,
        scale_factor=2,
        with_idt=False,
        act_type="prelu",
        scale=2,
        with_proj=True,
        **kwargs,
    ):
        """@shanlan-comment
        Input: the BASIC V2 fixed canvas and scalable ECBSR encoder/decoder settings.
        Output: initialized generator retaining [B,1,H+H/2,W] -> [B,3,H,W].
        Key Steps: validate channel/canvas/list cardinality, then initialize the
        existing ECBSR U-Net directly without the baseline-size restrictions.
        Key Params: layer_num sets U-Net depth; expand_channel controls stage
        width; depth_factor controls repeated ECB volume; scale_factor controls
        internal expansion while preserving spatial size.
        State/Dependencies: delegates forward behavior to BasicTextLineR4StudentV2.
        """
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        if len(self.canvas_hw) != 2 or any(value <= 0 or value % 32 for value in self.canvas_hw):
            raise ValueError("canvas_hw values must be positive multiples of 32")
        if int(in_channel) != 12 or int(out_channel) != 12:
            raise ValueError("BASIC V2 Large requires in_channel=12 and out_channel=12")
        if int(layer_num) < 1:
            raise ValueError("layer_num must be positive")
        expand = tuple(float(value) for value in expand_channel)
        if len(expand) != int(layer_num) + 1 or any(value <= 0 for value in expand):
            raise ValueError("expand_channel must contain layer_num + 1 positive values")
        depth = [depth_factor] * int(layer_num) if isinstance(depth_factor, (int, float)) else list(depth_factor)
        scale_factors = ([scale_factor] * (int(layer_num) + 1)
                         if isinstance(scale_factor, (int, float)) else list(scale_factor))
        if len(depth) != int(layer_num) or any(float(value) <= 0 for value in depth):
            raise ValueError("depth_factor must contain layer_num positive values")
        if len(scale_factors) != int(layer_num) + 1 or any(float(value) <= 0 for value in scale_factors):
            raise ValueError("scale_factor must contain layer_num + 1 positive values")
        _R4StudentGraph.__init__(
            self,
            in_channel=12,
            out_channel=12,
            layer_num=int(layer_num),
            expand_channel=list(expand),
            depth_factor=depth,
            scale_factor=scale_factors,
            with_idt=with_idt,
            act_type=act_type,
            scale=scale,
            with_proj=with_proj,
            **kwargs,
        )


__all__ = [
    "BasicTextLineR4StudentV2", "BasicTextLineR4StudentV2Large", "R4_STUDENT_EXPAND_CHANNEL", "R4_STUDENT_LAYER_NUM",
    "R4_STUDENT_DEPTH_FACTOR", "R4_STUDENT_SCALE_FACTOR", "load_strict_checkpoint", "sha256_file",
]
