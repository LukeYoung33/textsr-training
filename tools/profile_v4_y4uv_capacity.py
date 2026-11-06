"""Measure executed training-mode convolution MACs for scalable Y4UV-V4 presets."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import BasicTextLineR4StudentY4UVV4Large


PRESETS = {
    "base": {"stage_channels": [6, 24, 96, 384], "stage_repeat": 1, "bottleneck_repeat": 2, "refine_channels": 16},
    # Matches the fused V5 deployment graph, not its multi-branch training graph.
    "v5deploymatch": {"stage_channels": [8, 32, 128, 512], "stage_repeat": 2, "bottleneck_repeat": 3, "refine_channels": 24},
    "10x": {"stage_channels": [12, 48, 192, 768], "stage_repeat": 4, "bottleneck_repeat": 8, "refine_channels": 32},
    # Same channel schedule as 10x, but one fewer repeated block per stage.
    # This targets V5's 1.259G *training-graph* MAC budget rather than its
    # fused deployment budget.
    "v5trainmatch": {"stage_channels": [12, 48, 192, 768], "stage_repeat": 3, "bottleneck_repeat": 6, "refine_channels": 32},
    "20x": {"stage_channels": [12, 48, 192, 768], "stage_repeat": 8, "bottleneck_repeat": 16, "refine_channels": 32},
    "50x": {"stage_channels": [24, 96, 384, 1536], "stage_repeat": 5, "bottleneck_repeat": 10, "refine_channels": 64},
    "100x": {"stage_channels": [24, 96, 384, 1536], "stage_repeat": 11, "bottleneck_repeat": 22, "refine_channels": 64},
}


@contextmanager
def _count_convolution_macs():
    original = F.conv2d
    counter = {"macs": 0}

    def counted(*args, **kwargs):
        input_tensor = kwargs.get("input", args[0])
        weight = kwargs.get("weight", args[1])
        groups = kwargs.get("groups", args[6] if len(args) > 6 else 1)
        result = original(*args, **kwargs)
        counter["macs"] += int(result.numel() * (input_tensor.shape[1] // groups) * weight.shape[-2] * weight.shape[-1])
        return result

    F.conv2d = counted
    try:
        yield counter
    finally:
        F.conv2d = original


def profile(name: str) -> dict[str, object]:
    spec = PRESETS[name]
    model = BasicTextLineR4StudentY4UVV4Large(canvas_hw=(96, 480), haar_switch=True, **spec).train()
    input_tensor = torch.zeros(1, 6, 48, 240)
    with _count_convolution_macs() as counter, torch.no_grad():
        output = model(input_tensor)
    volumes = [int(channels * height * width) for channels, height, width in model.encoder_stage_shapes]
    return {"preset": name, "spec": spec, "macs": counter["macs"], "params": sum(p.numel() for p in model.parameters()), "output_shape": list(output.shape), "encoder_volumes": volumes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), action="append")
    args = parser.parse_args()
    chosen = args.preset or list(PRESETS)
    result = [profile(name) for name in chosen]
    base = next(item["macs"] for item in result if item["preset"] == "base") if "base" in chosen else profile("base")["macs"]
    for item in result:
        item["mac_ratio_to_base"] = float(item["macs"]) / float(base)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
