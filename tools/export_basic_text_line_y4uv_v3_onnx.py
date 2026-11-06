"""Export the isolated six-channel BASIC V2 Y4UV graph.

The exporter has one input and one output, both ``[B,6,H/2,W/2]``.  It does
not import or alter the legacy YUV444 exporter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from realesrgan.archs.basic_text_line_r4_student_y4uv_v3_arch import BasicTextLineR4StudentY4UVV3


def _load_checkpoint(model: nn.Module, path: Path) -> str:
    """@luna-comment Strictly load a Y4UV checkpoint and return its SHA-256."""
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(str(path), map_location="cpu")
    state = raw
    if isinstance(raw, dict):
        for key in ("params_ema", "params", "state_dict"):
            if isinstance(raw.get(key), dict):
                state = raw[key]
                break
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a parameter mapping")
    model.load_state_dict({str(key).removeprefix("module."): value for key, value in state.items()}, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_y4uv(
    output: Path,
    canvas_hw: tuple[int, int] = (96, 320),
    batch_size: int = 1,
    haar_switch: bool = True,
    checkpoint: Path | None = None,
    opset: int = 17,
) -> dict:
    """@luna-comment Export a fixed-shape Y4UV model and return the shape report."""
    model = BasicTextLineR4StudentY4UVV3(canvas_hw=canvas_hw, haar_switch=haar_switch).eval()
    checkpoint_sha256 = None
    if checkpoint is not None:
        checkpoint_sha256 = _load_checkpoint(model, checkpoint)
    half_hw = (canvas_hw[0] // 2, canvas_hw[1] // 2)
    sample = torch.randn(batch_size, 6, *half_hw)
    with torch.inference_mode():
        reference = model(sample)
    if tuple(reference.shape) != (batch_size, 6, *half_hw):
        raise RuntimeError(f"unexpected output shape: {tuple(reference.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        str(output),
        input_names=["y4uv"],
        output_names=["y4uv"],
        opset_version=int(opset),
        do_constant_folding=True,
        dynamic_axes={"y4uv": {0: "batch"}},
    )
    report = {
        "architecture": "BasicTextLineR4StudentY4UVV3",
        "haar_switch": bool(haar_switch),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "input_contract": ["B", 6, half_hw[0], half_hw[1]],
        "output_contract": ["B", 6, half_hw[0], half_hw[1]],
        "output": str(output),
        "opset": int(opset),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """@luna-comment Parse exporter arguments and print the Y4UV contract report."""
    parser = argparse.ArgumentParser(description="Export BASIC V2 Y4UV V3 to ONNX")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--canvas-h", type=int, default=96)
    parser.add_argument("--canvas-w", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--haar-switch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    report = export_y4uv(
        args.output,
        canvas_hw=(args.canvas_h, args.canvas_w),
        batch_size=args.batch_size,
        haar_switch=args.haar_switch,
        checkpoint=args.checkpoint,
        opset=args.opset,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
