"""Volume-preserving six-channel BASIC V2 Y4UV architecture.

The V4 graph is intentionally isolated from the existing V3 implementation.
It keeps the native Y4UV420 contract at the input, skip connection, and
output, while increasing channel width whenever a spatial downsample is
performed so that ``C * H * W`` does not decrease between encoder stages.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from .ecbsr_Unet_arch_server import HaarLikeToY4, Y4ToHaarLikeWavelet


Y4UV_CHANNELS = 6
V4_CANVAS_HW = (96, 480)
V4_STAGE_CHANNELS = (6, 24, 96, 384)
V4_REFINE_CHANNELS = 16
V4_STAGE_REPEAT = 1
V4_BOTTLENECK_REPEAT = 2


def _coerce_switch(value: Any) -> bool:
    """@shanlan-comment Parse YAML booleans without treating the string ``false`` as true."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"invalid haar_switch value: {value!r}")
    return bool(value)


def _validate_volume_preserving_channels(stage_channels: Iterable[int]) -> tuple[int, int, int, int]:
    """Validate four encoder widths whose downsampling never reduces C*H*W.

    Each encoder transition halves both spatial axes.  Therefore a stage must
    have at least four times the preceding channel count to preserve (or grow)
    tensor volume.  This is an architectural contract, not a loss-side hint.
    """
    channels = tuple(int(value) for value in stage_channels)
    if len(channels) != 4 or any(value <= 0 for value in channels):
        raise ValueError("stage_channels must contain four positive integers")
    if channels[0] < Y4UV_CHANNELS:
        raise ValueError(f"stage_channels[0] must be at least {Y4UV_CHANNELS}")
    for previous, current in zip(channels, channels[1:]):
        if current < 4 * previous:
            raise ValueError(
                "encoder C*H*W would decrease: each downsampled stage must have at least 4x channels "
                f"(got {previous}->{current})"
            )
    return channels  # type: ignore[return-value]


class LitePointwiseBlock(nn.Module):
    """@shanlan-comment Use depthwise plus pointwise convolution for width-expanded stages."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("LitePointwiseBlock supports stride 1 or 2")
        self.depthwise = nn.Conv2d(
            int(in_channels),
            int(in_channels),
            kernel_size=3,
            stride=int(stride),
            padding=1,
            groups=int(in_channels),
            bias=True,
        )
        self.pointwise = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True)
        self.activation = nn.PReLU(int(out_channels))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """@shanlan-comment Apply spatial mixing, channel projection, and a bounded nonlinear activation."""
        return self.activation(self.pointwise(self.depthwise(input_tensor)))


def _repeated_block(channels: int, repeat: int = V4_STAGE_REPEAT) -> nn.Sequential:
    """@shanlan-comment Build a same-width block group with explicit repeat count."""
    if int(repeat) < 1:
        raise ValueError("repeat must be positive")
    return nn.Sequential(*(LitePointwiseBlock(channels, channels) for _ in range(int(repeat))))


class VolumePreservingY4UVBackboneV4(nn.Module):
    """@shanlan-comment Encode and decode six-channel features with constant encoder tensor volume."""

    def __init__(
        self,
        stage_channels: Iterable[int] = V4_STAGE_CHANNELS,
        stage_repeat: int = V4_STAGE_REPEAT,
        bottleneck_repeat: int = V4_BOTTLENECK_REPEAT,
    ) -> None:
        super().__init__()
        channels = _validate_volume_preserving_channels(stage_channels)
        if int(stage_repeat) < 1 or int(bottleneck_repeat) < 1:
            raise ValueError("stage_repeat and bottleneck_repeat must be positive")
        c0, c1, c2, c3 = channels
        self.stage_repeat = int(stage_repeat)
        self.bottleneck_repeat = int(bottleneck_repeat)

        # Large presets may widen the first encoder stage.  Keep the external
        # Y4UV contract at six channels while projecting before the encoder and
        # returning to six channels only after decoding.
        self.input_projection: nn.Module = (
            nn.Identity() if c0 == Y4UV_CHANNELS else nn.Conv2d(Y4UV_CHANNELS, c0, kernel_size=1, bias=True)
        )
        self.output_projection: nn.Module = (
            nn.Identity() if c0 == Y4UV_CHANNELS else nn.Conv2d(c0, Y4UV_CHANNELS, kernel_size=1, bias=True)
        )
        self.stage0 = _repeated_block(c0, self.stage_repeat)
        self.enc1 = nn.Sequential(
            LitePointwiseBlock(c0, c1, stride=2),
            *_repeated_block(c1, self.stage_repeat),
        )
        self.enc2 = nn.Sequential(
            LitePointwiseBlock(c1, c2, stride=2),
            *_repeated_block(c2, self.stage_repeat),
        )
        self.enc3 = nn.Sequential(
            LitePointwiseBlock(c2, c3, stride=2),
            *_repeated_block(c3, self.stage_repeat),
        )
        self.bottleneck = _repeated_block(c3, repeat=self.bottleneck_repeat)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = LitePointwiseBlock(c3, c2)
        self.dec1 = LitePointwiseBlock(c2, c1)
        self.dec0 = LitePointwiseBlock(c1, c0)

    def forward_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """Return the pre-projection highest-resolution decoder feature."""
        stage0 = self.stage0(self.input_projection(input_features))
        stage1 = self.enc1(stage0)
        stage2 = self.enc2(stage1)
        stage3 = self.enc3(stage2)
        bottleneck = self.bottleneck(stage3)

        decoded2 = self.dec2(self.up2(bottleneck)) + stage2
        decoded1 = self.dec1(self.up1(decoded2)) + stage1
        decoded0 = self.dec0(self.up0(decoded1)) + stage0
        return decoded0

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """@shanlan-comment Run constant-volume encoder stages and same-resolution skip decoding."""
        return self.output_projection(self.forward_features(input_features))


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV4(nn.Module):
    """@shanlan-comment Build the isolated volume-preserving Y4UV420 residual model."""

    def __init__(
        self,
        canvas_hw=V4_CANVAS_HW,
        in_channel=Y4UV_CHANNELS,
        out_channel=Y4UV_CHANNELS,
        stage_channels=V4_STAGE_CHANNELS,
        refine_channels=V4_REFINE_CHANNELS,
        stage_repeat=V4_STAGE_REPEAT,
        bottleneck_repeat=V4_BOTTLENECK_REPEAT,
        haar_switch=True,
        learnable_haar=False,
        stroke_prob_enabled=False,
        **kwargs,
    ) -> None:
        super().__init__()
        if len(canvas_hw) != 2:
            raise ValueError("canvas_hw must contain height and width")
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        if any(value <= 0 for value in self.canvas_hw):
            raise ValueError("canvas_hw values must be positive")
        if any(value % 16 for value in self.canvas_hw):
            raise ValueError("V4 canvas_hw values must be positive multiples of 16")
        if int(in_channel) != Y4UV_CHANNELS or int(out_channel) != Y4UV_CHANNELS:
            raise ValueError("Y4UV V4 requires in_channel=out_channel=6")
        channels = _validate_volume_preserving_channels(stage_channels)
        if int(refine_channels) < 1:
            raise ValueError("refine_channels must be positive")
        if int(stage_repeat) < 1 or int(bottleneck_repeat) < 1:
            raise ValueError("stage_repeat and bottleneck_repeat must be positive")

        self.haar_switch = _coerce_switch(haar_switch)
        self.learnable_haar = bool(learnable_haar)
        self.stage_channels = channels
        self.refine_channels = int(refine_channels)
        self.stage_repeat = int(stage_repeat)
        self.bottleneck_repeat = int(bottleneck_repeat)
        self.stroke_prob_enabled = bool(stroke_prob_enabled)

        if self.haar_switch:
            self.y4_to_haar: nn.Module | None = Y4ToHaarLikeWavelet(
                learnable_scale=self.learnable_haar
            )
            self.haar_to_y4: nn.Module | None = HaarLikeToY4(
                learnable_scale=self.learnable_haar
            )
            self.input_transform: nn.Module | None = None
        else:
            self.y4_to_haar = None
            self.haar_to_y4 = None
            # Disabled mode deliberately receives all six channels directly;
            # no Y4/UV split, repeat, or concatenation is performed here.
            self.input_transform = nn.Conv2d(Y4UV_CHANNELS, Y4UV_CHANNELS, kernel_size=3, padding=1)

        self.backbone = VolumePreservingY4UVBackboneV4(
            stage_channels=channels,
            stage_repeat=self.stage_repeat,
            bottleneck_repeat=self.bottleneck_repeat,
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(Y4UV_CHANNELS, self.refine_channels, kernel_size=3, padding=1),
            nn.PReLU(self.refine_channels),
            nn.Conv2d(self.refine_channels, Y4UV_CHANNELS, kernel_size=3, padding=1),
        )
        # Preserve the residual identity at initialization, as in the V3
        # projection, while retaining the complete refinement graph for
        # training and performance export.
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)
        # Training-only 1x1 head.  Four Y4 logits retain the 2x2 luma phase
        # layout and are deterministically unpacked to full-resolution logits.
        self.stroke_prob_head: nn.Module | None = (
            nn.Conv2d(channels[0], 4, kernel_size=1, bias=True) if self.stroke_prob_enabled else None
        )
        if self.stroke_prob_head is not None:
            nn.init.kaiming_normal_(self.stroke_prob_head.weight, nonlinearity="linear")
            nn.init.zeros_(self.stroke_prob_head.bias)

    @property
    def expected_input_hw(self) -> tuple[int, int]:
        """@shanlan-comment Return the half-resolution tensor shape required by the canvas contract."""
        return self.canvas_hw[0] // 2, self.canvas_hw[1] // 2

    @property
    def stage_shapes(self) -> tuple[tuple[int, int, int], ...]:
        """@shanlan-comment Expose the input/stage0 shape and three downsampled encoder shapes."""
        height, width = self.expected_input_hw
        if height % 8 or width % 8:
            raise ValueError("half-resolution canvas must be divisible by 8 for three downsample stages")
        c0, c1, c2, c3 = self.stage_channels
        return (
            (c0, height, width),
            (c1, height // 2, width // 2),
            (c2, height // 4, width // 4),
            (c3, height // 8, width // 8),
        )

    @property
    def stage_volumes(self) -> tuple[int, ...]:
        """@shanlan-comment Return ``C*H*W`` for each exposed input/encoder stage."""
        return tuple(channels * height * width for channels, height, width in self.stage_shapes)

    @property
    def encoder_stage_shapes(self) -> tuple[tuple[int, int, int], ...]:
        """@shanlan-comment Alias encoder metadata for callers that exclude the raw input entry."""
        return self.stage_shapes

    @property
    def encoder_stage_volumes(self) -> tuple[int, ...]:
        """@shanlan-comment Alias stage volume metadata for acceptance checks."""
        return self.stage_volumes

    def _prepare_features(self, input_y4uv: torch.Tensor) -> torch.Tensor:
        """@shanlan-comment Apply either the legacy four-band Haar boundary or direct six-channel convolution."""
        if not self.haar_switch:
            assert self.input_transform is not None
            return self.input_transform(input_y4uv)
        assert self.y4_to_haar is not None
        y4_bands = self.y4_to_haar(input_y4uv[:, :4])
        return torch.cat((y4_bands, input_y4uv[:, 4:]), dim=1)

    def _decode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """@shanlan-comment Convert the four luma residual bands back to native Y4 order before refinement."""
        if not self.haar_switch:
            return residual
        assert self.haar_to_y4 is not None
        y4_residual = self.haar_to_y4(residual[:, :4])
        return torch.cat((y4_residual, residual[:, 4:]), dim=1)

    @staticmethod
    def _unpack_y4_logits(logits_y4: torch.Tensor) -> torch.Tensor:
        if logits_y4.ndim != 4 or logits_y4.shape[1] != 4:
            raise ValueError("stroke logits must be [B,4,H/2,W/2]")
        # Fixed phase rearrangement, written explicitly so the training-only
        # branch contains no PixelShuffle/PixelUnshuffle deployment operator.
        batch, _, height, width = logits_y4.shape
        return logits_y4.reshape(batch, 1, 2, 2, height, width).permute(0, 1, 4, 2, 5, 3).reshape(
            batch, 1, height * 2, width * 2
        )

    def drop_stroke_head(self) -> None:
        """Remove the training-only probability branch before deployment export."""
        self.stroke_prob_head = None
        self.stroke_prob_enabled = False

    def forward(self, input_y4uv: torch.Tensor, return_stroke_logits: bool = False):
        """@shanlan-comment Validate native Y4UV input, decode a residual, refine it, and preserve the skip."""
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
        decoded0 = self.backbone.forward_features(features)
        residual = self._decode_residual(self.backbone.output_projection(decoded0))
        residual = self.refine_head(residual)
        output = input_y4uv + residual
        if output.shape != input_y4uv.shape:
            raise RuntimeError(f"Y4UV V4 output contract failed: {tuple(output.shape)}")
        if not return_stroke_logits:
            return output
        if self.stroke_prob_head is None:
            raise RuntimeError("return_stroke_logits requested but stroke_prob_enabled is false")
        return output, self._unpack_y4_logits(self.stroke_prob_head(decoded0))


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV4Large(BasicTextLineR4StudentY4UVV4):
    """Compute-scaled V4 model retaining the native Y4UV residual contract.

    Unlike the historical NV21 Large model, this class changes only V4 encoder
    width/repetition.  The SDF/R1/bgLF trainer and inference interface remain
    exactly the same.  ``stage_channels`` is validated by the parent class so
    every 2x downsample has sufficient channel expansion for non-decreasing
    encoder tensor volume.
    """

    pass


__all__ = [
    "BasicTextLineR4StudentY4UVV4",
    "BasicTextLineR4StudentY4UVV4Large",
    "LitePointwiseBlock",
    "V4_CANVAS_HW",
    "V4_STAGE_CHANNELS",
    "V4_REFINE_CHANNELS",
    "_validate_volume_preserving_channels",
]
