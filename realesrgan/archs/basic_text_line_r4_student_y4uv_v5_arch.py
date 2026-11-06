"""BASIC V5 native Y4UV420 architecture with frozen phase carries.

The V5 graph is deliberately independent from the V4 implementation.  It
keeps the six-channel Y4UV interface, moves the four phase samples into a
frozen grouped convolution before each encoder ECB, and uses an external
resize followed by a same-resolution re-parameterizable ECB in the decoder.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from basicsr.utils.registry import ARCH_REGISTRY

from .ecbsr_Unet_arch_server import HaarLikeToY4, Y4ToHaarLikeWavelet


Y4UV_CHANNELS = 6
V5_CANVAS_HW = (96, 480)
V5_STAGE_CHANNELS = (8, 32, 128)
V5_LOCAL_CHANNELS = 16
V5_CONTEXT_SCALE_LIMIT = 0.5
# Keep the initial forward graph exactly identity while allowing the context
# terminal to receive a first-step gradient when the bounded gate is zero.
V5_CONTEXT_GRAD_WARMUP = 0.1


def _coerce_switch(value: Any) -> bool:
    """@luna-comment
    Input: value: Any
    Output: bool
    Key Steps: Convert YAML boolean spellings to a strict Python bool.
    Key Params: Explicit arguments and constructor settings control the described operation.
    State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"invalid boolean value: {value!r}")
    return bool(value)


def _device_or_none(tensor: Tensor) -> Optional[torch.device]:
    """@luna-comment
    Input: tensor: Tensor
    Output: Optional[torch.device]
    Key Steps: Return a device object usable by temporary fusion tensors.
    Key Params: Explicit arguments and constructor settings control the described operation.
    State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
    """
    return tensor.device if tensor.is_cuda or tensor.device.type != "cpu" else None


class FixedPhasePack(nn.Conv2d):
    """Frozen four-phase ``C -> 4C`` downsample with exact volume preservation.

    Output channels are ordered per input channel as top-left, top-right,
    bottom-left, bottom-right.  The grouped form is the deployment default;
    ``grouped=False`` keeps the same sparse selector semantics for runtimes
    without multiplier-four grouped convolution support.
    """

    def __init__(self, in_channels: int, grouped: bool = True) -> None:
        """@luna-comment
        Input: Positive input channel count C and grouped selector switch.
        Output: Conv2d module with frozen C-to4C selector weights; kernel2, stride2.
        Key Steps: Zero weights, then select TL/TR/BL/BR per input channel; disable weight gradients.
        Key Params: grouped=True uses C groups; False uses equivalent sparse dense weights.
        State/Dependencies: PyTorch parameters/buffers owned by this module; no external mutable state.
        """
        channels = int(in_channels)
        if channels < 1:
            raise ValueError("in_channels must be positive")
        groups = channels if grouped else 1
        super().__init__(
            channels,
            4 * channels,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=groups,
            bias=False,
        )
        with torch.no_grad():
            self.weight.zero_()
            for channel in range(channels):
                for phase, (row, col) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                    if grouped:
                        self.weight[4 * channel + phase, 0, row, col] = 1.0
                    else:
                        self.weight[4 * channel + phase, channel, row, col] = 1.0
        self.weight.requires_grad_(False)

    def train(self, mode: bool = True) -> "FixedPhasePack":
        """@luna-comment
        Input: mode: bool
        Output: 'FixedPhasePack'
        Key Steps: Keep selector weights frozen while allowing module mode propagation.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        super().train(mode)
        self.weight.requires_grad_(False)
        return self

    @property
    def phase_order(self) -> Tuple[str, str, str, str]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Tuple[str, str, str, str]
        Key Steps: Expose the fixed output-channel phase order for audits.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        return ("top_left", "top_right", "bottom_left", "bottom_right")


class V5RepECB(nn.Module):
    """ECB-compatible same-resolution block with explicit deployment fusion.

    Training/evaluation before ``switch_to_deploy`` execute the same padded
    spatial transforms: dense 3x3, 1x1-to-3x3, Sobel-x, Sobel-y, Laplacian,
    and optional identity.  Deployment replaces all branches with one 3x3
    convolution and keeps PReLU outside the convolution, so fusion parity is
    numerical rather than dependent on a train/eval shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth_multiplier: float = 2.0,
        act_type: str = "prelu",
        with_idt: bool = False,
        zero_init: bool = False,
    ) -> None:
        """@luna-comment
        Input: Positive input/output channels; depth multiplier; prelu or linear activation; identity/zero-init switches.
        Output: Unfused same-resolution module; optional identity only when input/output channels match.
        Key Steps: Create dense, sequential and three edge branches, fixed edge masks and optional PReLU.
        Key Params: zero_init zeros residual branches; depth_multiplier sets sequential intermediate width.
        State/Dependencies: PyTorch parameters/buffers owned by this module; no external mutable state.
        """
        super().__init__()
        in_ch, out_ch = int(in_channels), int(out_channels)
        if in_ch < 1 or out_ch < 1:
            raise ValueError("ECB channel counts must be positive")
        mid_ch = max(1, int(round(out_ch * float(depth_multiplier))))
        if act_type not in {"prelu", "linear"}:
            raise ValueError("V5RepECB supports act_type='prelu' or 'linear'")

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.mid_channels = mid_ch
        self.with_idt = bool(with_idt and in_ch == out_ch)
        self.act_type = act_type
        self.deploy = False

        self.conv3x3: Optional[nn.Conv2d] = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        # Biasless intermediate projections make the padded sequential branch
        # exactly representable by one constant-bias 3x3 at image borders.
        self.conv1x1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.conv1x1_3x3 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=True)

        self.edge_x_pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.edge_y_pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.edge_l_pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.edge_x_scale = nn.Parameter(torch.randn(out_ch, 1, 1, 1) * 1e-3)
        self.edge_y_scale = nn.Parameter(torch.randn(out_ch, 1, 1, 1) * 1e-3)
        self.edge_l_scale = nn.Parameter(torch.randn(out_ch, 1, 1, 1) * 1e-3)
        self.edge_x_bias = nn.Parameter(torch.randn(out_ch) * 1e-3)
        self.edge_y_bias = nn.Parameter(torch.randn(out_ch) * 1e-3)
        self.edge_l_bias = nn.Parameter(torch.randn(out_ch) * 1e-3)
        self.register_buffer("edge_x_mask", self._make_edge_mask("x"), persistent=True)
        self.register_buffer("edge_y_mask", self._make_edge_mask("y"), persistent=True)
        self.register_buffer("edge_l_mask", self._make_edge_mask("laplacian"), persistent=True)

        self.reparam: Optional[nn.Conv2d] = None
        self.act: Optional[nn.PReLU]
        if act_type == "prelu":
            self.act = nn.PReLU(out_ch)
        else:
            self.act = None

        if zero_init:
            self._zero_init_all_branches()

    def _make_edge_mask(self, kind: str) -> Tensor:
        """@luna-comment
        Input: kind: str
        Output: Tensor
        Key Steps: Build the fixed 3x3 spatial kernel used by one edge branch.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        mask = torch.zeros(self.out_channels, 1, 3, 3)
        if kind == "x":
            mask[:, 0, 0, 0] = 1.0
            mask[:, 0, 1, 0] = 2.0
            mask[:, 0, 2, 0] = 1.0
            mask[:, 0, 0, 2] = -1.0
            mask[:, 0, 1, 2] = -2.0
            mask[:, 0, 2, 2] = -1.0
        elif kind == "y":
            mask[:, 0, 0, 0] = 1.0
            mask[:, 0, 0, 1] = 2.0
            mask[:, 0, 0, 2] = 1.0
            mask[:, 0, 2, 0] = -1.0
            mask[:, 0, 2, 1] = -2.0
            mask[:, 0, 2, 2] = -1.0
        elif kind == "laplacian":
            mask[:, 0, 0, 1] = 1.0
            mask[:, 0, 1, 0] = 1.0
            mask[:, 0, 1, 2] = 1.0
            mask[:, 0, 2, 1] = 1.0
            mask[:, 0, 1, 1] = -4.0
        else:
            raise ValueError(f"unknown edge mask: {kind}")
        return mask

    def _zero_init_all_branches(self) -> None:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: None
        Key Steps: Zero a terminal ECB so its residual is exactly zero at initialization.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        modules = (
            self.conv3x3,
            self.conv1x1,
            self.conv1x1_3x3,
            self.edge_x_pw,
            self.edge_y_pw,
            self.edge_l_pw,
        )
        with torch.no_grad():
            for module in modules:
                assert module is not None
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
            self.edge_x_scale.zero_()
            self.edge_y_scale.zero_()
            self.edge_l_scale.zero_()
            self.edge_x_bias.zero_()
            self.edge_y_bias.zero_()
            self.edge_l_bias.zero_()

    def _edge_forward(
        self,
        input_tensor: Tensor,
        pointwise: nn.Conv2d,
        scale: Tensor,
        bias: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """@luna-comment
        Input: input_tensor: Tensor, pointwise: nn.Conv2d, scale: Tensor, bias: Tensor, mask: Tensor
        Output: Tensor
        Key Steps: Apply one learned pointwise projection and fixed spatial edge kernel.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        projected = pointwise(input_tensor)
        return F.conv2d(projected, scale * mask, bias=bias, padding=1, groups=self.out_channels)

    def _forward_branches(self, input_tensor: Tensor) -> Tensor:
        """@luna-comment
        Input: input_tensor: Tensor
        Output: Tensor
        Key Steps: Sum all ECB-compatible training branches before activation.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        assert self.conv3x3 is not None
        output = self.conv3x3(input_tensor)
        output = output + self.conv1x1_3x3(self.conv1x1(input_tensor))
        output = output + self._edge_forward(
            input_tensor, self.edge_x_pw, self.edge_x_scale, self.edge_x_bias, self.edge_x_mask
        )
        output = output + self._edge_forward(
            input_tensor, self.edge_y_pw, self.edge_y_scale, self.edge_y_bias, self.edge_y_mask
        )
        output = output + self._edge_forward(
            input_tensor, self.edge_l_pw, self.edge_l_scale, self.edge_l_bias, self.edge_l_mask
        )
        if self.with_idt:
            output = output + input_tensor
        return output

    def forward(self, input_tensor: Tensor) -> Tensor:
        """@luna-comment
        Input: input_tensor: Tensor
        Output: Tensor
        Key Steps: Run unfused ECB branches or the single fused deployment convolution.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if self.deploy:
            assert self.reparam is not None
            output = self.reparam(input_tensor)
        else:
            output = self._forward_branches(input_tensor)
        if self.act is not None:
            output = self.act(output)
        return output

    def rep_params(self) -> Tuple[Tensor, Tensor]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Tuple[Tensor, Tensor]
        Key Steps: Compose all branch kernels and biases into one padded 3x3 operator.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if self.deploy:
            assert self.reparam is not None
            return self.reparam.weight, self.reparam.bias
        assert self.conv3x3 is not None
        kernel = self.conv3x3.weight.detach().clone()
        bias = self.conv3x3.bias.detach().clone()

        seq_kernel = F.conv2d(
            self.conv1x1_3x3.weight,
            self.conv1x1.weight.permute(1, 0, 2, 3),
        )
        seq_bias = self.conv1x1_3x3.bias
        kernel = kernel + seq_kernel
        bias = bias + seq_bias

        for pointwise, scale, branch_bias, mask in (
            (self.edge_x_pw, self.edge_x_scale, self.edge_x_bias, self.edge_x_mask),
            (self.edge_y_pw, self.edge_y_scale, self.edge_y_bias, self.edge_y_mask),
            (self.edge_l_pw, self.edge_l_scale, self.edge_l_bias, self.edge_l_mask),
        ):
            edge_kernel = pointwise.weight[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
            edge_kernel = edge_kernel * (scale * mask)
            edge_bias = branch_bias
            kernel = kernel + edge_kernel
            bias = bias + edge_bias

        if self.with_idt:
            identity = torch.zeros_like(kernel)
            for channel in range(self.out_channels):
                identity[channel, channel, 1, 1] = 1.0
            kernel = kernel + identity
        return kernel, bias

    def switch_to_deploy(self) -> "V5RepECB":
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: 'V5RepECB'
        Key Steps: Replace training-only ECB branches with one fused 3x3 convolution.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if self.deploy:
            return self
        kernel, bias = self.rep_params()
        fused = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1, bias=True)
        with torch.no_grad():
            fused.weight.copy_(kernel)
            fused.bias.copy_(bias)
        fused = fused.to(device=kernel.device, dtype=kernel.dtype)
        self.reparam = fused
        self.conv3x3 = None
        self.conv1x1 = None  # type: ignore[assignment]
        self.conv1x1_3x3 = None  # type: ignore[assignment]
        self.edge_x_pw = None  # type: ignore[assignment]
        self.edge_y_pw = None  # type: ignore[assignment]
        self.edge_l_pw = None  # type: ignore[assignment]
        self.edge_x_scale = None  # type: ignore[assignment]
        self.edge_y_scale = None  # type: ignore[assignment]
        self.edge_l_scale = None  # type: ignore[assignment]
        self.edge_x_bias = None  # type: ignore[assignment]
        self.edge_y_bias = None  # type: ignore[assignment]
        self.edge_l_bias = None  # type: ignore[assignment]
        self.edge_x_mask = None  # type: ignore[assignment]
        self.edge_y_mask = None  # type: ignore[assignment]
        self.edge_l_mask = None  # type: ignore[assignment]
        self.deploy = True
        return self


class FixedLumaEmbed(nn.Conv2d):
    """Frozen 1x1 embedding of four luma bands into the six-channel feature space."""

    def __init__(self) -> None:
        """@luna-comment
        Input: No explicit arguments.
        Output: Frozen biasless1x1 Conv2d mapping four luma bands into six channels with zero UV residual.
        Key Steps: Zero the matrix, set the first four diagonal entries to one and freeze weights.
        Key Params: Output channel order preserves the four luma bands followed by two zero channels.
        State/Dependencies: PyTorch parameters/buffers owned by this module; no external mutable state.
        """
        super().__init__(4, 6, kernel_size=1, bias=False)
        with torch.no_grad():
            self.weight.zero_()
            for channel in range(4):
                self.weight[channel, channel, 0, 0] = 1.0
        self.weight.requires_grad_(False)

    def train(self, mode: bool = True) -> "FixedLumaEmbed":
        """@luna-comment
        Input: mode: bool
        Output: 'FixedLumaEmbed'
        Key Steps: Keep the fixed channel embedding frozen in every module mode.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        super().train(mode)
        self.weight.requires_grad_(False)
        return self


def _validate_stage_channels(stage_channels: Iterable[int]) -> Tuple[int, int, int]:
    """@luna-comment
    Input: stage_channels: Iterable[int]
    Output: Tuple[int, int, int]
    Key Steps: Validate the two phase-preserving encoder widths.
    Key Params: Explicit arguments and constructor settings control the described operation.
    State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
    """
    channels = tuple(int(value) for value in stage_channels)
    if len(channels) != 3 or any(value < 1 for value in channels):
        raise ValueError("stage_channels must contain three positive integers")
    if channels[0] < 1 or channels[1] < 4 * channels[0] or channels[2] < 4 * channels[1]:
        raise ValueError("each V5 phase downsample requires at least four times the previous width")
    return channels  # type: ignore[return-value]


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV5(nn.Module):
    """Native Y4UV V5 text-line restorer with local and bounded-context residuals."""

    supports_external_skip_input = True

    def __init__(
        self,
        canvas_hw: Tuple[int, int] = V5_CANVAS_HW,
        in_channel: int = Y4UV_CHANNELS,
        out_channel: int = Y4UV_CHANNELS,
        stage_channels: Iterable[int] = V5_STAGE_CHANNELS,
        local_channels: int = V5_LOCAL_CHANNELS,
        stage_repeat: int = 1,
        bottleneck_repeat: int = 2,
        haar_switch: bool = True,
        local_haar_switch: bool | None = None,
        learnable_haar: bool = False,
        grouped_phase_pack: bool = True,
        stroke_prob_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        """@luna-comment
        Input: Canvas dimensions divisible by8; six input/output channels; configured stage/local widths and Haar/auxiliary switches.
        Output: Local/context residual network with six-channel packed Y4UV public interface.
        Key Steps: Validate configuration; initialize fixed boundaries, ECB branches, phase selectors and bounded context scale.
        Key Params: This checkpoint uses256x256 canvas,8/32/128 stages,16 local channels and fixed Haar; stroke head is optional training supervision.
        State/Dependencies: PyTorch parameters/buffers owned by this module; no external mutable state.
        """
        super().__init__()
        if len(canvas_hw) != 2:
            raise ValueError("canvas_hw must contain height and width")
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        if any(value <= 0 or value % 8 for value in self.canvas_hw):
            raise ValueError("V5 canvas dimensions must be positive multiples of 8")
        if int(in_channel) != Y4UV_CHANNELS or int(out_channel) != Y4UV_CHANNELS:
            raise ValueError("V5 requires native six-channel Y4UV input and output")
        channels = _validate_stage_channels(stage_channels)
        if int(local_channels) < 1 or int(stage_repeat) < 1 or int(bottleneck_repeat) < 1:
            raise ValueError("local_channels, stage_repeat, and bottleneck_repeat must be positive")
        if _coerce_switch(learnable_haar):
            raise ValueError("V5 deployment requires fixed Haar boundary coefficients")

        self.stage_channels = channels
        self.local_channels = int(local_channels)
        self.stage_repeat = int(stage_repeat)
        self.bottleneck_repeat = int(bottleneck_repeat)
        self.haar_switch = _coerce_switch(haar_switch)
        self.local_haar_switch = self.haar_switch if local_haar_switch is None else _coerce_switch(local_haar_switch)
        self.learnable_haar = False
        self.grouped_phase_pack = bool(grouped_phase_pack)
        self.stroke_prob_enabled = _coerce_switch(stroke_prob_enabled)

        if self.haar_switch:
            self.y4_to_haar: Optional[nn.Module] = Y4ToHaarLikeWavelet(learnable_scale=False)
            self.haar_to_y4: Optional[nn.Module] = HaarLikeToY4(learnable_scale=False)
            self.input_transform: Optional[nn.Module] = None
        else:
            self.y4_to_haar = None
            self.haar_to_y4 = None
            self.input_transform = nn.Identity()

        c0, c1, c2 = channels
        self.local_pre = V5RepECB(4, self.local_channels, act_type="prelu")
        self.local_terminal = V5RepECB(
            self.local_channels, 4, act_type="linear", zero_init=True
        )
        self.local_embed = FixedLumaEmbed()

        self.stem = V5RepECB(6, c0, act_type="prelu")
        self.phase_pack1 = FixedPhasePack(c0, grouped=self.grouped_phase_pack)
        self.stage1 = nn.ModuleList(
            V5RepECB(c1, c1, act_type="prelu", with_idt=True) for _ in range(self.stage_repeat)
        )
        self.phase_pack2 = FixedPhasePack(c1, grouped=self.grouped_phase_pack)
        self.stage2 = nn.ModuleList(
            V5RepECB(c2, c2, act_type="prelu", with_idt=True) for _ in range(self.bottleneck_repeat)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = V5RepECB(c2, c1, act_type="prelu")
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = V5RepECB(c1, c0, act_type="prelu")
        # Unlike the local terminal, this block must not be zero-initialized:
        # alpha starts at zero, and a zero terminal would deadlock both sides
        # of the multiplicative context gate.  The training forward below adds
        # a straight-through warm-up gradient without changing its value.
        self.context_terminal = V5RepECB(c0, 6, act_type="linear")
        self.context_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_deployed_context_scale", torch.tensor(0.0), persistent=True)
        self._deployment = False

        # Training-only foreground probability branch.  decoded0 is the
        # half-resolution S0 carry, so four logits preserve the native 2x2
        # luma phase layout and can be unpacked without a learned resize.
        self.stroke_prob_head: Optional[nn.Conv2d] = (
            nn.Conv2d(c0, 4, kernel_size=1, bias=True) if self.stroke_prob_enabled else None
        )
        if self.stroke_prob_head is not None:
            nn.init.kaiming_normal_(self.stroke_prob_head.weight, nonlinearity="linear")
            nn.init.zeros_(self.stroke_prob_head.bias)

    @property
    def expected_input_hw(self) -> Tuple[int, int]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Tuple[int, int]
        Key Steps: Return the half-resolution Y4UV tensor geometry for the canvas contract.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        return self.canvas_hw[0] // 2, self.canvas_hw[1] // 2

    @property
    def stage_shapes(self) -> Tuple[Tuple[int, int, int], ...]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Tuple[Tuple[int, int, int], ...]
        Key Steps: Return S0, P1, and P2 channel-height-width shapes.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        height, width = self.expected_input_hw
        if height % 4 or width % 4:
            raise ValueError("V5 half-resolution canvas must be divisible by four")
        c0, c1, c2 = self.stage_channels
        return ((c0, height, width), (c1, height // 2, width // 2), (c2, height // 4, width // 4))

    @property
    def stage_volumes(self) -> Tuple[int, ...]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Tuple[int, ...]
        Key Steps: Return C*H*W for the stem and both frozen phase carries.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        return tuple(channel * height * width for channel, height, width in self.stage_shapes)

    @property
    def encoder_stage_shapes(self) -> Tuple[Tuple[int, int, int], ...]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Tuple[Tuple[int, int, int], ...]
        Key Steps: Alias the exposed V5 encoder stages for acceptance tooling.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        return self.stage_shapes

    @property
    def encoder_stage_volumes(self) -> Tuple[int, ...]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Tuple[int, ...]
        Key Steps: Alias the V5 tensor-volume contract for acceptance tooling.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        return self.stage_volumes

    @property
    def context_scale_value(self) -> float:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: float
        Key Steps: Return the bounded context amplitude used by the active graph.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if self._deployment:
            return float(self._deployed_context_scale.detach().cpu())
        value = V5_CONTEXT_SCALE_LIMIT * torch.tanh(self.context_scale.detach())
        return float(value.cpu())

    @property
    def analytic_receptive_field(self) -> Dict[str, Dict[str, float]]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Dict[str, Dict[str, float]]
        Key Steps: Report analytic feature-cell and original-canvas receptive fields.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        # Feature-cell RF: local 3x3 + terminal 3x3 = 5; context follows
        # stem, pack, ECB, pack, two S2 ECBs, two bilinear resizes, two decoder
        # ECBs, and the terminal 3x3: 40 cells.  Each Y4UV cell is 2x2 canvas
        # pixels at the external 96x480 contract.
        return {
            "local_residual": {"feature_cells": 5.0, "canvas_pixels": 10.0},
            "context_residual": {"feature_cells": 40.0, "canvas_pixels": 80.0},
            "full_output": {"feature_cells": 40.0, "canvas_pixels": 80.0},
        }

    def _prepare_features(self, input_y4uv: Tensor) -> Tensor:
        """@luna-comment
        Input: input_y4uv: Tensor
        Output: Tensor
        Key Steps: Convert native luma bands to fixed Haar bands while preserving UV channels.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if not self.haar_switch:
            assert self.input_transform is not None
            return self.input_transform(input_y4uv)
        assert self.y4_to_haar is not None
        return torch.cat((self.y4_to_haar(input_y4uv[:, :4]), input_y4uv[:, 4:]), dim=1)

    def _decode_luma(self, features: Tensor) -> Tensor:
        """@luna-comment
        Input: features: Tensor
        Output: Tensor
        Key Steps: Convert a six-channel feature residual back to native Y4UV ordering.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if not self.haar_switch:
            return features
        assert self.haar_to_y4 is not None
        return torch.cat((self.haar_to_y4(features[:, :4]), features[:, 4:]), dim=1)

    def _decode_local_luma(self, features: Tensor) -> Tensor:
        """@luna-comment
        Input: features: Tensor
        Output: Tensor
        Key Steps: Decode local residual bands according to the local-only Haar ablation switch.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if not self.local_haar_switch:
            return features
        return self._decode_luma(features)

    @staticmethod
    def _run_blocks(input_tensor: Tensor, blocks: nn.ModuleList) -> Tensor:
        """@luna-comment
        Input: input_tensor: Tensor, blocks: nn.ModuleList
        Output: Tensor
        Key Steps: Run a configured same-resolution ECB sequence.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        output = input_tensor
        for block in blocks:
            output = block(output)
        return output

    def _decode_context(
        self, bottleneck: Tensor, phase_carry: Tensor, stem_carry: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """@luna-comment
        Input: bottleneck: Tensor, phase_carry: Tensor, stem_carry: Tensor
        Output: Tuple[Tensor, Tensor, Tensor, Tensor]
        Key Steps: Decode through external resizes and linear carry-plus-residual fusions.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        up2 = self.up2(bottleneck)
        residual2 = self.dec2(up2)
        decoded1 = phase_carry + residual2
        up1 = self.up1(decoded1)
        residual1 = self.dec1(up1)
        decoded0 = stem_carry + residual1
        return decoded0, decoded1, residual2, residual1

    @staticmethod
    def _unpack_y4_logits(logits_y4: Tensor) -> Tensor:
        """@luna-comment
        Input: logits_y4: Tensor
        Output: Tensor
        Key Steps: Expand four half-resolution luma phases to full-resolution logits.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if logits_y4.ndim != 4 or logits_y4.shape[1] != 4:
            raise ValueError("stroke logits must be [B,4,H/2,W/2]")
        batch, _, height, width = logits_y4.shape
        return logits_y4.reshape(batch, 1, 2, 2, height, width).permute(
            0, 1, 4, 2, 5, 3
        ).reshape(batch, 1, height * 2, width * 2)

    def drop_stroke_head(self) -> None:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: None
        Key Steps: Remove the training-only probability branch before deployment export.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        self.stroke_prob_head = None
        self.stroke_prob_enabled = False

    def forward(
        self,
        input_y4uv: Tensor,
        return_branches: bool = False,
        return_debug: bool = False,
        return_stroke_logits: bool = False,
        skip_input_y4uv: Optional[Tensor] = None,
    ) -> Any:
        """@luna-comment
        Input: BCHW six-channel Y4UV; spatial dimensions are multiples of four;
            optional matching skip tensor and auxiliary-output flags.
        Output: Same-shape residual-restored Y4UV, or requested branch/debug data.
        Key Steps: Validate phases, run local/context branches, add identity skip.
        Key Params: Runtime dimensions need not equal the configured training canvas.
        State/Dependencies: Network weights and training/deployment flags; PyTorch.
        """
        if not isinstance(input_y4uv, Tensor) or input_y4uv.ndim != 4:
            raise ValueError("input must be [B,6,H/2,W/2] Y4UV420")
        if input_y4uv.shape[1] != Y4UV_CHANNELS:
            raise ValueError(f"input must have six channels, got {input_y4uv.shape[1]}")
        # The V5 graph is fully convolutional.  Phase packing twice requires
        # half-resolution dimensions divisible by four (equivalently, an RGB
        # canvas divisible by eight), but does not require the configured
        # training canvas at inference time.
        input_height, input_width = map(int, input_y4uv.shape[-2:])
        if input_height <= 0 or input_width <= 0 or input_height % 4 or input_width % 4:
            raise ValueError(
                "Y4UV half-resolution height and width must be positive multiples of four, "
                f"got {(input_height, input_width)}"
            )
        if skip_input_y4uv is not None:
            if not isinstance(skip_input_y4uv, Tensor) or skip_input_y4uv.shape != input_y4uv.shape:
                raise ValueError("skip_input_y4uv must match input_y4uv shape")
            if skip_input_y4uv.device != input_y4uv.device or skip_input_y4uv.dtype != input_y4uv.dtype:
                raise ValueError("skip_input_y4uv must match input_y4uv device and dtype")

        features = self._prepare_features(input_y4uv)

        local_input = features[:, :4] if self.local_haar_switch else input_y4uv[:, :4]
        local_features = self.local_pre(local_input)
        local_bands = self.local_terminal(local_features)
        local_feature_residual = self.local_embed(local_bands)
        local_residual = self._decode_local_luma(local_feature_residual)

        stem_carry = self.stem(features)
        phase_carry1 = self.phase_pack1(stem_carry)
        mixed1 = self._run_blocks(phase_carry1, self.stage1)
        phase_carry2 = self.phase_pack2(mixed1)
        bottleneck = self._run_blocks(phase_carry2, self.stage2)
        decoded0, decoded1, decoder_residual2, decoder_residual1 = self._decode_context(
            bottleneck, phase_carry1, stem_carry
        )
        context_feature_residual = self.context_terminal(decoded0)
        if self._deployment:
            context_residual = self._decode_luma(context_feature_residual)
        else:
            alpha = V5_CONTEXT_SCALE_LIMIT * torch.tanh(self.context_scale)
            # The second and third terms cancel in the forward pass, but keep
            # a small derivative to the terminal at alpha=0.  This preserves
            # output=input at initialization while both context_scale and
            # context_terminal can learn on the first backward pass.
            if self.training:
                context_feature_residual = (
                    (alpha + V5_CONTEXT_GRAD_WARMUP) * context_feature_residual
                    - V5_CONTEXT_GRAD_WARMUP * context_feature_residual.detach()
                )
            else:
                context_feature_residual = alpha * context_feature_residual
            context_residual = self._decode_luma(context_feature_residual)

        skip = input_y4uv if skip_input_y4uv is None else skip_input_y4uv
        output = skip + local_residual + context_residual
        if output.shape != input_y4uv.shape:
            raise RuntimeError(f"Y4UV V5 output contract failed: {tuple(output.shape)}")

        if return_stroke_logits:
            if self.stroke_prob_head is None or not self.stroke_prob_enabled:
                raise RuntimeError("return_stroke_logits requested but stroke_prob_enabled is false")
            return output, self._unpack_y4_logits(self.stroke_prob_head(decoded0))

        if return_debug:
            return {
                "output": output,
                "input": input_y4uv,
                "skip_input": skip,
                "local_input": local_input,
                "local_bands": local_bands,
                "local_haar_switch": self.local_haar_switch,
                "local_residual": local_residual,
                "context_residual": context_residual,
                "stem_carry": stem_carry,
                "phase_carry1": phase_carry1,
                "phase_carry2": phase_carry2,
                "decoded1": decoded1,
                "decoded0": decoded0,
                "decoder_residual2": decoder_residual2,
                "decoder_residual1": decoder_residual1,
            }
        if return_branches:
            return {
                "input": input_y4uv,
                "skip_input": skip,
                "local_residual": local_residual,
                "context_residual": context_residual,
                "full": output,
            }
        return output

    def switch_to_deploy(self) -> "BasicTextLineR4StudentY4UVV5":
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: 'BasicTextLineR4StudentY4UVV5'
        Key Steps: Fuse every V5 ECB and bake the bounded context amplitude for export.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        if self._deployment:
            return self
        # The probability branch is a training-only objective and must never
        # survive in the deployment module graph or exported state.
        self.drop_stroke_head()
        active_blocks = [
            self.local_pre,
            self.local_terminal,
            self.stem,
            *list(self.stage1),
            *list(self.stage2),
            self.dec2,
            self.dec1,
            self.context_terminal,
        ]
        for block in active_blocks:
            block.switch_to_deploy()
        alpha = V5_CONTEXT_SCALE_LIMIT * torch.tanh(self.context_scale.detach())
        assert self.context_terminal.reparam is not None
        with torch.no_grad():
            self.context_terminal.reparam.weight.mul_(alpha)
            self.context_terminal.reparam.bias.mul_(alpha)
            self._deployed_context_scale.copy_(alpha)
            self.context_scale.requires_grad_(False)
        self._deployment = True
        return self

    def deployment_audit(self) -> Dict[str, Any]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Dict[str, Any]
        Key Steps: Audit that the active module graph contains no unfused or forbidden convolution class.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        unfused = [name for name, module in self.named_modules() if isinstance(module, V5RepECB) and not module.deploy]
        stroke_head = [
            name for name, module in self.named_modules() if name == "stroke_prob_head" or name.startswith("stroke_prob_head.")
        ]
        forbidden_class_names = {"Conv" + "Transpose2d", "Pixel" + "Shuffle", "Pixel" + "Unshuffle"}
        forbidden = [
            name
            for name, module in self.named_modules()
            if module.__class__.__name__ in forbidden_class_names
        ]
        resize_count = sum(isinstance(module, nn.Upsample) for module in self.modules())
        return {
            "passed": not unfused and not forbidden and not stroke_head and self._deployment,
            "unfused_ecb": unfused,
            "forbidden_modules": forbidden,
            "training_only_modules": stroke_head,
            "external_resize_count": resize_count,
            "context_scale_limit": V5_CONTEXT_SCALE_LIMIT,
            "context_scale_value": self.context_scale_value,
        }

    def parameter_count(self) -> int:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: int
        Key Steps: Return the number of trainable and frozen model parameters.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        return sum(parameter.numel() for parameter in self.parameters())

    def estimate_macs(self) -> Dict[str, int]:
        """@luna-comment
        Input: No explicit inputs; uses the module configuration.
        Output: Dict[str, int]
        Key Steps: Return analytic MAC counts for the unfused and fused V5 graphs.
        Key Params: Explicit arguments and constructor settings control the described operation.
        State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
        """
        height, width = self.expected_input_hw

        def conv_macs(in_ch: int, out_ch: int, kernel: int, out_h: int, out_w: int, groups: int = 1) -> int:
            """@luna-comment
            Input: in_ch: int, out_ch: int, kernel: int, out_h: int, out_w: int, groups: int
            Output: int
            Key Steps: conv macs
            Key Params: Explicit arguments and constructor settings control the described operation.
            State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
            """
            return int(out_h * out_w * out_ch * (in_ch // groups) * kernel * kernel)

        def ecb_macs(block: V5RepECB, out_h: int, out_w: int, deployed: bool) -> int:
            """@luna-comment
            Input: block: V5RepECB, out_h: int, out_w: int, deployed: bool
            Output: int
            Key Steps: ecb macs
            Key Params: Explicit arguments and constructor settings control the described operation.
            State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
            """
            if deployed:
                return conv_macs(block.in_channels, block.out_channels, 3, out_h, out_w)
            dense = conv_macs(block.in_channels, block.out_channels, 3, out_h, out_w)
            sequence = conv_macs(block.in_channels, block.mid_channels, 1, out_h, out_w)
            sequence += conv_macs(block.mid_channels, block.out_channels, 3, out_h, out_w)
            edge = 3 * (
                conv_macs(block.in_channels, block.out_channels, 1, out_h, out_w)
                + conv_macs(block.out_channels, block.out_channels, 3, out_h, out_w, groups=block.out_channels)
            )
            return dense + sequence + edge

        c0, c1, c2 = self.stage_channels
        h1, w1 = height // 2, width // 2
        h2, w2 = height // 4, width // 4

        def graph_macs(deployed: bool) -> int:
            """@luna-comment
            Input: deployed: bool
            Output: int
            Key Steps: graph macs
            Key Params: Explicit arguments and constructor settings control the described operation.
            State/Dependencies: PyTorch tensors and module-owned parameters/buffers; no external mutable state.
            """
            total = ecb_macs(self.local_pre, height, width, deployed)
            total += ecb_macs(self.local_terminal, height, width, deployed)
            total += ecb_macs(self.stem, height, width, deployed)
            total += conv_macs(c0, c1, 2, h1, w1, groups=c0)
            total += sum(ecb_macs(block, h1, w1, deployed) for block in self.stage1)
            total += conv_macs(c1, c2, 2, h2, w2, groups=c1)
            total += sum(ecb_macs(block, h2, w2, deployed) for block in self.stage2)
            total += ecb_macs(self.dec2, h1, w1, deployed)
            total += ecb_macs(self.dec1, height, width, deployed)
            total += ecb_macs(self.context_terminal, height, width, deployed)
            return total

        return {"training_branches": graph_macs(False), "deployment_fused": graph_macs(True)}


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV5Large(BasicTextLineR4StudentY4UVV5):
    """Compatibility alias reserved for future V5 capacity variants."""

    pass


__all__ = [
    "BasicTextLineR4StudentY4UVV5",
    "BasicTextLineR4StudentY4UVV5Large",
    "FixedLumaEmbed",
    "FixedPhasePack",
    "V5RepECB",
    "V5_CANVAS_HW",
    "V5_CONTEXT_GRAD_WARMUP",
    "V5_CONTEXT_SCALE_LIMIT",
    "V5_STAGE_CHANNELS",
]
