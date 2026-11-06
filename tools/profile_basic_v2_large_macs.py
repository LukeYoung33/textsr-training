"""Profile executed convolution MACs for fixed-canvas BASIC V2 generators."""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
import torch
from torch import nn
from torch.nn import functional as F

try:
    import easydict as _easydict  # noqa: F401
except ModuleNotFoundError:
    class _EasyDict(dict):
        """Minimal import-time compatibility object for model-only profiling."""

        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    _easydict_stub = types.ModuleType("easydict")
    _easydict_stub.EasyDict = _EasyDict
    sys.modules["easydict"] = _easydict_stub

from realesrgan.archs.basic_text_line_r4_student_v2_arch import (
    BasicTextLineR4StudentV2,
    BasicTextLineR4StudentV2Large,
)


CANVAS_HW = (96, 320)
PACKED_INPUT_SHAPE = (1, 1, 144, 320)
LARGE_PRESETS = {
    "10x": {
        "layer_num": 3,
        "expand_channel": [0.5, 1.0, 2.0, 4.0],
        "depth_factor": [4, 4, 4],
        "scale_factor": [16, 16, 16, 16],
        "measured_macs": 1798208640,
    },
    "20x": {
        "layer_num": 4,
        "expand_channel": [0.5, 1.0, 2.0, 4.0, 8.0],
        "depth_factor": [4, 4, 4, 4],
        "scale_factor": [28, 28, 28, 28, 28],
        "measured_macs": 3736411200,
    },
    "50x": {
        "layer_num": 4,
        "expand_channel": [1.0, 2.0, 4.0, 8.0, 16.0],
        "depth_factor": [4, 4, 4, 4],
        "scale_factor": [17, 17, 17, 17, 17],
        "measured_macs": 9382193280,
    },
}


def _conv_macs(input_tensor: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, groups: int) -> int:
    """@shanlan-comment
    Input: convolution input, OIHW kernel, output tensor, and groups count.
    Output: integer multiply-accumulate count for one executed functional convolution.
    Key Steps: derive one output dot-product width from channels, groups, and kernel;
    multiply it by all output elements including batch.
    Key Params: groups divides input channels; bias and non-convolution operations
    are deliberately excluded to keep the metric MAC-specific.
    State/Dependencies: relies on PyTorch functional-convolution tensor layouts.
    """
    if input_tensor.ndim != 4 or weight.ndim != 4 or output.ndim != 4:
        raise TypeError("convolution output must be a tensor")
    kernel_h, kernel_w = weight.shape[-2:]
    per_output = (input_tensor.shape[1] // groups) * kernel_h * kernel_w
    return int(output.numel() * per_output)


def profile_generator(model: nn.Module, packed_input: torch.Tensor, *, training: bool = True) -> dict[str, int | list[int] | str]:
    """@shanlan-comment
    Input: BASIC V2 generator and exact packed-NV21 tensor [1,1,144,320].
    Output: executed convolution MACs, parameter count, and observed output shape.
    Key Steps: temporarily wrap F.conv2d, execute one selected-mode forward, restore
    the function and model mode even on failure, and validate the output contract.
    Key Params: training=True counts the actual multi-branch training compute; the
    same functional-convolution metric is used for baseline and Large ratios.
    State/Dependencies: temporarily replaces torch.nn.functional.conv2d.
    """
    macs = 0

    original_conv2d = F.conv2d

    def counted_conv2d(*args, **kwargs) -> torch.Tensor:
        input_tensor = kwargs.get("input", args[0] if len(args) > 0 else None)
        weight = kwargs.get("weight", args[1] if len(args) > 1 else None)
        groups = kwargs.get("groups", args[6] if len(args) > 6 else 1)
        if not isinstance(input_tensor, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise TypeError("conv2d profiler requires tensor input and weight")
        nonlocal macs
        output = original_conv2d(*args, **kwargs)
        macs += _conv_macs(input_tensor, weight, output, groups)
        return output

    was_training = model.training
    try:
        F.conv2d = counted_conv2d
        model.train(training)
        with torch.no_grad():
            output = model(packed_input)
    finally:
        F.conv2d = original_conv2d
        model.train(was_training)
    expected_output = (packed_input.shape[0], 3, *CANVAS_HW)
    if tuple(output.shape) != expected_output:
        raise RuntimeError(f"expected output {expected_output}, got {tuple(output.shape)}")
    return {
        "macs": macs,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(packed_input.shape),
        "output_shape": list(output.shape),
        "mode": "train" if training else "eval",
    }


def _positive_number_list(raw: str) -> list[float]:
    """@shanlan-comment
    Input: comma-separated positive numeric command-line value.
    Output: list of positive floats in the original order.
    Key Steps: split, parse, reject empty/non-positive entries, and return values.
    Key Params: commas separate scale or width stages.
    State/Dependencies: no shared state; used by CLI argument parsing.
    """
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a non-empty list of positive numbers")
    return values


def build_large_preset(preset: str) -> BasicTextLineR4StudentV2Large:
    """@shanlan-comment
    Input: one measured Large preset name: 10x, 20x, or 50x.
    Output: uninitialized-weight BASIC V2 Large generator with the selected topology.
    Key Steps: validate the preset name and forward its frozen architecture knobs
    to the scalable adapter with the production canvas.
    Key Params: preset fixes measured training MAC band, U-Net depth, width, and volume.
    State/Dependencies: LARGE_PRESETS is the task-owned single source of model knobs.
    """
    try:
        config = LARGE_PRESETS[preset]
    except KeyError as error:
        raise ValueError(f"unknown BASIC V2 Large preset: {preset}") from error
    return BasicTextLineR4StudentV2Large(canvas_hw=CANVAS_HW, **{
        key: value for key, value in config.items() if key != "measured_macs"
    })


def build_parser() -> argparse.ArgumentParser:
    """@shanlan-comment
    Input: none.
    Output: CLI parser for baseline or configurable Large MAC measurements.
    Key Steps: declare architecture choice and scalable U-Net argument defaults.
    Key Params: values map directly to BasicTextLineR4StudentV2Large construction.
    State/Dependencies: argparse only; parsing does not create a model.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--preset", choices=sorted(LARGE_PRESETS))
    parser.add_argument("--eval", action="store_true", help="profile merged inference compute instead of training compute")
    parser.add_argument("--layer-num", type=int, default=4)
    parser.add_argument("--expand-channel", type=_positive_number_list, default=[1.0, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--depth-factor", type=_positive_number_list, default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--scale-factor", type=_positive_number_list, default=[2.0, 2.0, 2.0, 2.0, 2.0])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """@shanlan-comment
    Input: optional CLI arguments selecting baseline or Large architecture settings.
    Output: one JSON profile to stdout and process status zero on valid contracts.
    Key Steps: parse settings, construct the requested fixed-canvas model, profile
    one production-size forward, and annotate Large configuration metadata.
    Key Params: --layer-num, --expand-channel, --depth-factor, and --scale-factor
    change executed convolution volume without changing the external canvas.
    State/Dependencies: uses CPU PyTorch; no weights, datasets, or files are mutated.
    """
    args = build_parser().parse_args(argv)
    if args.baseline and args.preset:
        raise ValueError("--baseline and --preset are mutually exclusive")
    if args.baseline:
        model = BasicTextLineR4StudentV2(canvas_hw=CANVAS_HW)
        label = "baseline"
    elif args.preset:
        model = build_large_preset(args.preset)
        label = args.preset
    else:
        model = BasicTextLineR4StudentV2Large(
            canvas_hw=CANVAS_HW,
            layer_num=args.layer_num,
            expand_channel=args.expand_channel,
            depth_factor=args.depth_factor,
            scale_factor=args.scale_factor,
        )
        label = "large"
    result = profile_generator(model, torch.zeros(PACKED_INPUT_SHAPE), training=not args.eval)
    result["variant"] = label
    if not args.baseline:
        config = LARGE_PRESETS[args.preset] if args.preset else vars(args)
        result["layer_num"] = config["layer_num"]
        result["expand_channel"] = config["expand_channel"]
        result["depth_factor"] = config["depth_factor"]
        result["scale_factor"] = config["scale_factor"]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
