"""Export a fused deployment ONNX graph from a trained BASIC V2 Y4UV V3 checkpoint.

The training architecture keeps ECB/ECB2 branches and computes their fused
kernel in ``eval()``.  This exporter materializes those kernels once in
PyTorch, replaces every ECB/ECB2 with an ordinary deployment module, and then
exports a fixed ``[B,6,48,240]`` graph.  The Haar boundary is selected at
export time; there is no runtime Haar switch input or Python training branch.
"""

from __future__ import annotations

import argparse
import copy
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
from realesrgan.archs.ecbsr_Unet_arch_server import ECB, ECB2


Y4UV_CHANNELS = 6
DEFAULT_CANVAS_HW = (96, 480)
DEFAULT_BATCH_SIZE = 8
DEFAULT_OPSET = 17
DEFAULT_SEED = 20260826
PARITY_TOLERANCE = 1e-4


def _checkpoint_state(raw: Any) -> dict[str, Any]:
    """@luna-comment
    Input: a checkpoint dictionary with params_ema, params, state_dict or raw tensor entries.
    Output: a normalized generator mapping with known wrapper prefixes removed.
    Key Steps: select the first recognized nested mapping, validate its type, normalize parameter names.
    Key Params: params_ema takes precedence over params and state_dict.
    State/Dependencies: mapping values are reused; no model or checkpoint file is modified.
    """
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
        for prefix in ("module.", "net_g.", "generator."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = value
    return normalized


def _load_checkpoint(model: nn.Module, checkpoint: Path) -> str:
    """@luna-comment
    Input: target module and a trusted checkpoint file path.
    Output: SHA256 of the loaded file; target module parameters are replaced in place.
    Key Steps: load on CPU, normalize state, strict-load the module, stream file bytes into SHA256.
    Key Params: strict loading rejects missing or unexpected parameter keys.
    State/Dependencies: PyTorch deserialization and file reads; legacy trusted-checkpoint semantics preserved.
    """
    try:
        raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(_checkpoint_state(raw), strict=True)

    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _activation(source: nn.Module) -> nn.Module:
    """@luna-comment
    Input: source ECB or ECB2 module.
    Output: Identity for linear mode, otherwise an independent activation copy.
    Key Steps: inspect act_type and deepcopy the existing activation with learned parameters.
    Key Params: linear mode has no activation; PReLU retains its trained slopes.
    State/Dependencies: source activation state is read but not changed.
    """
    if getattr(source, "act_type", "linear") == "linear":
        return nn.Identity()
    return copy.deepcopy(source.act)


def _fused_kernel(source: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """@luna-comment
    Input: source ECB or ECB2 module with learned branch parameters.
    Output: detached cloned merged 3x3 kernel and bias on the source device.
    Key Steps: use the direct kernel in train_repParam mode, otherwise merge branches under no_grad.
    Key Params: train_repParam selects the existing direct-convolution semantics.
    State/Dependencies: source rep_params and current weights; no source parameter mutation.
    """
    if bool(getattr(source, "train_repParam", False)):
        # This mode intentionally bypasses the auxiliary branches in the
        # source implementation, so its deploy equivalent is just conv3x3.
        return source.conv3x3.weight.detach().clone(), source.conv3x3.bias.detach().clone()
    with torch.no_grad():
        weight, bias = source.rep_params()
    return weight.detach().clone(), bias.detach().clone()


class FusedECB(nn.Module):
    """@luna-comment Deployment replacement for one eval-time ECB block."""

    def __init__(self, source: ECB) -> None:
        """@luna-comment
        Input: an existing encoder or same-scale ECB source.
        Output: deployment Conv2d and independent original activation initialized in this module.
        Key Steps: materialize kernel, construct stride-selected convolution, copy weights and activation.
        Key Params: Encoder uses stride2; other supported phases use stride1 with padding1.
        State/Dependencies: source channel/phase settings and branch parameters are read only.
        """
        super().__init__()
        weight, bias = _fused_kernel(source)
        stride = 2 if source.phase == "Encoder" else 1
        self.conv = nn.Conv2d(
            int(source.inp_planes),
            int(source.out_planes),
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)
            self.conv.bias.copy_(bias)
        self.act = _activation(source)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """@luna-comment
        Input: BCHW tensor matching the source channel geometry.
        Output: activated convolution result at the original phase-selected resolution.
        Key Steps: apply the stored fused Conv2d then activation.
        Key Params: stride and padding were fixed from the source at construction.
        State/Dependencies: this module stores fused parameters; source branch modules are not called.
        """
        return self.act(self.conv(input_tensor))


class FusedECB2(nn.Module):
    """@luna-comment Deployment replacement for one decoder ECB2 block."""

    def __init__(self, source: ECB2) -> None:
        """@luna-comment
        Input: an existing decoder ECB2 source.
        Output: deployment upsampler, fused convolution, optional independent skip projection and activation.
        Key Steps: reject unsupported direct mode, materialize kernel and copy projection/activation.
        Key Params: upsample2 uses bilinear align_corners=False; with_proj preserves skip transformation.
        State/Dependencies: source weights and settings are read; generated modules own independent parameters.
        """
        super().__init__()
        if bool(getattr(source, "train_repParam", False)):
            raise ValueError("ECB2 train_repParam=True has no valid decoder deployment path")
        weight, bias = _fused_kernel(source)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(
            int(source.inp_planes),
            int(source.out_planes),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)
            self.conv.bias.copy_(bias)
        self.with_proj = bool(source.with_proj)
        self.proj_conv = copy.deepcopy(source.proj_conv) if self.with_proj else None
        self.act = _activation(source)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """@luna-comment
        Input: decoder BCHW tensor and spatially compatible encoder skip tuple.
        Output: activated decoder feature after skip addition.
        Key Steps: optionally project skip, upsample decoder, convolve, add skip and activate.
        Key Params: with_proj and the fixed2x bilinear upsampling preserve source semantics.
        State/Dependencies: deployment modules and input tensors only; no branch-fusion work per call.
        """
        input_tensor, skip = inputs
        if self.with_proj:
            assert self.proj_conv is not None
            skip = self.proj_conv(skip)
        output = self.conv(self.upsample(input_tensor))
        return self.act(output + skip)


def fuse_reparameterized_modules(model: nn.Module) -> dict[str, int]:
    """@luna-comment
    Input: model tree containing ECB and ECB2 modules.
    Output: counts of replaced ECB/ECB2 blocks; model children are replaced in place.
    Key Steps: recurse through children, replacing decoder blocks first and accumulating replacement counts.
    Key Params: isinstance dispatch preserves ECB2 versus ECB replacement types.
    State/Dependencies: mutates the supplied model; callers copy the model first when preserving training state.
    """
    counts = {"ECB": 0, "ECB2": 0}
    for name, child in list(model.named_children()):
        if isinstance(child, ECB2):
            setattr(model, name, FusedECB2(child))
            counts["ECB2"] += 1
        elif isinstance(child, ECB):
            setattr(model, name, FusedECB(child))
            counts["ECB"] += 1
        else:
            nested = fuse_reparameterized_modules(child)
            counts["ECB"] += nested["ECB"]
            counts["ECB2"] += nested["ECB2"]
    return counts


def _export_kwargs(opset: int) -> dict[str, Any]:
    """@luna-comment
    Input: integer ONNX opset version.
    Output: static-axis ONNX export keyword dictionary.
    Key Steps: set input/output names and constant folding; disable dynamo when supported.
    Key Params: opset controls ONNX operator compatibility; no dynamic_axes are introduced.
    State/Dependencies: torch.onnx.export signature inspection; no model or file mutation.
    """
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
    """@luna-comment Read concrete or symbolic dimensions from one ONNX value-info record."""
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
    """@luna-comment Validate static I/O metadata and summarize deployment graph operators."""
    try:
        import onnx
    except ImportError as exc:
        return {"status": "dependency_unavailable", "dependency": "onnx", "reason": str(exc)}

    graph = onnx.load(str(path)).graph
    if not graph.input or not graph.output:
        raise RuntimeError("ONNX graph has no input or output value info")
    from collections import Counter

    op_counts = Counter(node.op_type for node in graph.node)
    return {
        "status": "validated",
        "input_name": graph.input[0].name,
        "output_name": graph.output[0].name,
        "input_shape": _shape_from_value_info(graph.input[0]),
        "output_shape": _shape_from_value_info(graph.output[0]),
        "node_count": len(graph.node),
        "op_counts": dict(sorted(op_counts.items())),
        "has_if_node": bool(op_counts.get("If", 0)),
        "has_reparam_scatter_nodes": bool(op_counts.get("ScatterND", 0)),
    }


def _runtime_parity(path: Path, sample: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """@luna-comment Compare ONNX Runtime output against the unfused PyTorch eval output."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        return {"status": "dependency_unavailable", "dependency": "onnxruntime", "reason": str(exc)}

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0])
    expected = reference.detach().cpu().numpy()
    if actual.shape != expected.shape:
        raise RuntimeError(f"ONNX Runtime shape mismatch: {actual.shape} vs {expected.shape}")
    error = np.abs(actual - expected)
    max_abs = float(error.max()) if error.size else 0.0
    mean_abs = float(error.mean()) if error.size else 0.0
    result = {
        "status": "passed" if max_abs <= PARITY_TOLERANCE else "failed",
        "output_shape": list(actual.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "tolerance": PARITY_TOLERANCE,
    }
    if result["status"] == "failed":
        raise RuntimeError(f"ONNX Runtime parity failed: {result}")
    return result


def _tensor_parity(reference: torch.Tensor, fused: torch.Tensor) -> dict[str, float | str]:
    """@luna-comment Measure the offline-fused PyTorch graph against the original eval graph."""
    error = (reference - fused).abs()
    max_abs = float(error.max().item()) if error.numel() else 0.0
    mean_abs = float(error.mean().item()) if error.numel() else 0.0
    result: dict[str, float | str] = {
        "status": "passed" if max_abs <= PARITY_TOLERANCE else "failed",
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "tolerance": PARITY_TOLERANCE,
    }
    if result["status"] == "failed":
        raise RuntimeError(f"offline fusion parity failed: {result}")
    return result


def export_deployment(
    checkpoint: Path,
    output: Path,
    canvas_hw: tuple[int, int] = DEFAULT_CANVAS_HW,
    batch_size: int = DEFAULT_BATCH_SIZE,
    haar_switch: bool = True,
    opset: int = DEFAULT_OPSET,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """@luna-comment Fuse, export, and validate one fixed-batch deployment graph."""
    canvas = (int(canvas_hw[0]), int(canvas_hw[1]))
    batch = int(batch_size)
    if canvas != DEFAULT_CANVAS_HW:
        raise ValueError(f"deployment export requires canvas_hw={DEFAULT_CANVAS_HW}, got {canvas}")
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
        original_reference = model(sample)

    fusion_counts = fuse_reparameterized_modules(model)
    model.eval()
    with torch.inference_mode():
        fused_reference = model(sample)
    fusion_parity = _tensor_parity(original_reference, fused_reference)
    expected_shape = (batch, Y4UV_CHANNELS, *half_hw)
    if tuple(fused_reference.shape) != expected_shape:
        raise RuntimeError(f"PyTorch output shape mismatch: {tuple(fused_reference.shape)} vs {expected_shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (sample,), str(output), **_export_kwargs(opset))
    graph_report = inspect_graph(output)
    if graph_report.get("status") == "validated":
        expected_list = list(expected_shape)
        if graph_report.get("input_shape") != expected_list:
            raise RuntimeError(f"ONNX input shape mismatch: {graph_report.get('input_shape')} vs {expected_list}")
        if graph_report.get("output_shape") != expected_list:
            raise RuntimeError(f"ONNX output shape mismatch: {graph_report.get('output_shape')} vs {expected_list}")
        if graph_report.get("has_if_node"):
            raise RuntimeError("deployment graph unexpectedly contains an If node")
        if graph_report.get("has_reparam_scatter_nodes"):
            raise RuntimeError("deployment graph still contains ScatterND reparameterization nodes")

    runtime_report = _runtime_parity(output, sample, original_reference)
    report = {
        "status": "passed",
        "architecture": "BasicTextLineR4StudentY4UVV3",
        "weights": "trained_checkpoint_offline_fused",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "haar_switch": bool(haar_switch),
        "haar_switch_runtime_input": False,
        "reparameterization": "offline_fused_to_conv2d",
        "fused_modules": fusion_counts,
        "batch_size": batch,
        "canvas_hw": list(canvas),
        "input_contract": list(expected_shape),
        "output_contract": list(expected_shape),
        "static_axes": True,
        "opset": int(opset),
        "output": str(output),
        "fusion_pytorch_parity": fusion_parity,
        "onnx_graph": graph_report,
        "onnxruntime": runtime_report,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """@luna-comment Parse deployment exporter arguments and print the validation report."""
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
    report = export_deployment(
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
