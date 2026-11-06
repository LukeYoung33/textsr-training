"""Export one fixed-batch Y4UV V3 checkpoint without changing legacy exporters.

The existing V3 checkpoint exporter intentionally keeps a dynamic batch axis.
This development utility is separate: it emits a static ``[B,6,48,240]`` graph
for a trained checkpoint, which is useful for a fixed-batch performance runner.
The model architecture and checkpoint loading contract remain the original
``BasicTextLineR4StudentY4UVV3`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from realesrgan.archs.basic_text_line_r4_student_y4uv_v3_arch import (
    BasicTextLineR4StudentY4UVV3,
)


Y4UV_CHANNELS = 6
DEFAULT_CANVAS_HW = (96, 480)
DEFAULT_BATCH_SIZE = 8
DEFAULT_OPSET = 17
DEFAULT_SEED = 20260826


def _checkpoint_state(raw: Any) -> dict[str, Any]:
    """@luna-comment Extract and normalize a BasicSR generator state mapping."""
    state: Any = raw
    if isinstance(raw, dict):
        for key in ("params_ema", "params", "state_dict"):
            if isinstance(raw.get(key), dict):
                state = raw[key]
                break
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a generator parameter mapping")

    normalized: dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        # BasicSR/DDP checkpoints can carry one or more wrapper prefixes.  Do
        # not rename parameter suffixes; strict loading below remains required.
        for prefix in ("module.", "net_g.", "generator."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = value
    return normalized


def _load_checkpoint(model: nn.Module, checkpoint: Path) -> str:
    """@luna-comment Strictly load one V3 checkpoint and return its SHA-256."""
    try:
        raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        raw = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(_checkpoint_state(raw), strict=True)

    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_kwargs(opset: int) -> dict[str, Any]:
    """@luna-comment Build a version-compatible static-axis ONNX argument set."""
    kwargs: dict[str, Any] = {
        "input_names": ["y4uv"],
        "output_names": ["y4uv"],
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    return kwargs


def _shape_from_value_info(value_info: Any) -> list[Any]:
    """@luna-comment Read concrete or symbolic dimensions from ONNX metadata."""
    dimensions: list[Any] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_value:
            dimensions.append(int(dimension.dim_value))
        elif dimension.dim_param:
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append(None)
    return dimensions


def inspect_graph(path: Path) -> dict[str, Any]:
    """@luna-comment Validate the one-input/one-output static graph contract."""
    try:
        import onnx
    except ImportError as exc:
        return {"status": "dependency_unavailable", "dependency": "onnx", "reason": str(exc)}

    graph = onnx.load(str(path)).graph
    if not graph.input or not graph.output:
        raise RuntimeError("ONNX graph has no input or output value info")
    return {
        "status": "validated",
        "input_name": graph.input[0].name,
        "output_name": graph.output[0].name,
        "input_shape": _shape_from_value_info(graph.input[0]),
        "output_shape": _shape_from_value_info(graph.output[0]),
    }


def check_onnxruntime(path: Path, sample: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """@luna-comment Compare ONNX Runtime output with the PyTorch checkpoint output."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        return {
            "status": "dependency_unavailable",
            "dependency": "onnxruntime",
            "reason": str(exc),
        }

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0])
    expected = reference.detach().cpu().numpy()
    if actual.shape != expected.shape:
        raise RuntimeError(f"ONNX Runtime shape mismatch: {actual.shape} vs {expected.shape}")
    error = np.abs(actual - expected)
    max_abs = float(error.max()) if error.size else 0.0
    mean_abs = float(error.mean()) if error.size else 0.0
    result = {
        "status": "passed" if max_abs <= 1e-4 else "failed",
        "output_shape": list(actual.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "tolerance": 1e-4,
    }
    if result["status"] == "failed":
        raise RuntimeError(f"ONNX Runtime parity failed: {result}")
    return result


def export_checkpoint(
    checkpoint: Path,
    output: Path,
    canvas_hw: tuple[int, int] = DEFAULT_CANVAS_HW,
    batch_size: int = DEFAULT_BATCH_SIZE,
    haar_switch: bool = True,
    opset: int = DEFAULT_OPSET,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """@luna-comment Export and validate a static batch graph from a trained V3 checkpoint."""
    canvas = (int(canvas_hw[0]), int(canvas_hw[1]))
    batch = int(batch_size)
    if canvas != DEFAULT_CANVAS_HW:
        raise ValueError(f"this fixed export requires canvas_hw={DEFAULT_CANVAS_HW}, got {canvas}")
    if batch <= 0:
        raise ValueError(f"batch_size must be positive, got {batch}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = BasicTextLineR4StudentY4UVV3(canvas_hw=canvas, haar_switch=bool(haar_switch)).eval()
    checkpoint_sha256 = _load_checkpoint(model, checkpoint)
    half_hw = (canvas[0] // 2, canvas[1] // 2)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sample = torch.randn(batch, Y4UV_CHANNELS, *half_hw, generator=generator)
    with torch.inference_mode():
        reference = model(sample)
    expected_shape = (batch, Y4UV_CHANNELS, *half_hw)
    if tuple(reference.shape) != expected_shape:
        raise RuntimeError(f"PyTorch output shape mismatch: {tuple(reference.shape)} vs {expected_shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (sample,), str(output), **_export_kwargs(opset))

    graph_report = inspect_graph(output)
    if graph_report.get("status") == "validated":
        expected_list = list(expected_shape)
        if graph_report.get("input_shape") != expected_list:
            raise RuntimeError(f"ONNX input shape mismatch: {graph_report.get('input_shape')} vs {expected_list}")
        if graph_report.get("output_shape") != expected_list:
            raise RuntimeError(f"ONNX output shape mismatch: {graph_report.get('output_shape')} vs {expected_list}")
    runtime_report = check_onnxruntime(output, sample, reference)
    report = {
        "status": "passed",
        "architecture": "BasicTextLineR4StudentY4UVV3",
        "weights": "trained_checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "haar_switch": bool(haar_switch),
        "batch_size": batch,
        "canvas_hw": list(canvas),
        "input_contract": list(expected_shape),
        "output_contract": list(expected_shape),
        "static_axes": True,
        "opset": int(opset),
        "output": str(output),
        "onnx_graph": graph_report,
        "onnxruntime": runtime_report,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """@luna-comment Parse checkpoint export arguments and print the validation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas-h", type=int, default=DEFAULT_CANVAS_HW[0])
    parser.add_argument("--canvas-w", type=int, default=DEFAULT_CANVAS_HW[1])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--haar-switch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = export_checkpoint(
        checkpoint=args.checkpoint,
        output=args.output,
        canvas_hw=(args.canvas_h, args.canvas_w),
        batch_size=args.batch_size,
        haar_switch=args.haar_switch,
        opset=args.opset,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
