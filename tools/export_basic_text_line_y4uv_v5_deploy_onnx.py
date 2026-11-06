"""Export and audit the fused BASIC V5 native Y4UV deployment graph."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import torch

# Make direct ``python tools/...py`` execution resolve the repository package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch import (
    BasicTextLineR4StudentY4UVV5,
)


def _load_checkpoint(model: torch.nn.Module, checkpoint: Optional[str]) -> list[str]:
    """@luna-comment
    Input: model: torch.nn.Module, checkpoint: Optional[str]
    Output: list[str]
    Key Steps: Load a V5 checkpoint while dropping training-only head keys.
    Key Params: Explicit arguments and constructor settings control the described operation.
    State/Dependencies: PyTorch tensors and module parameters; exporter additionally uses ONNX and filesystem output.
    """
    if not checkpoint:
        return []
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint container: {type(state)!r}")
    ignored = sorted(
        str(key) for key in state
        if str(key).startswith("stroke_prob_head.")
    )
    for key in ignored:
        state.pop(key, None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return ignored


def audit_onnx_ops(onnx_path: str | Path) -> Dict[str, Any]:
    """@luna-comment
    Input: onnx_path: str | Path
    Output: Dict[str, Any]
    Key Steps: Reject unsupported learned/deconvolution operators in an exported ONNX graph.
    Key Params: Explicit arguments and constructor settings control the described operation.
    State/Dependencies: PyTorch tensors and module parameters; exporter additionally uses ONNX and filesystem output.
    """
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    operators = sorted({node.op_type for node in graph.node})
    forbidden_names = {"ConvTranspose", "DepthToSpace", "SpaceToDepth"}
    forbidden = sorted(set(operators).intersection(forbidden_names))
    # Fixed Haar conversion may lower to Slice/Concat/Div/Mul/Constant.  The
    # learned deployment body is still restricted to Conv/Resize/PRelu/Add;
    # the audit reports all graph operators so a platform owner can inspect
    # those fixed-boundary nodes explicitly.
    allowed_fixed_boundary = {"Constant", "Identity", "Slice", "Concat", "Div", "Mul", "Sub", "Shape", "Cast"}
    allowed_runtime = {"Conv", "Resize", "PRelu", "Add"}
    unexpected = sorted(set(operators) - allowed_runtime - allowed_fixed_boundary)
    return {
        "operators": operators,
        "forbidden": forbidden,
        "unexpected": unexpected,
        "passed": not forbidden and not unexpected,
    }


def export_v5_deploy(
    output_path: str | Path,
    checkpoint: Optional[str] = None,
    canvas_hw: Iterable[int] = (96, 480),
    opset: int = 18,
) -> Dict[str, Any]:
    """@luna-comment
    Input: output_path: str | Path, checkpoint: Optional[str], canvas_hw: Iterable[int], opset: int
    Output: Dict[str, Any]
    Key Steps: Fuse V5, export a fixed-canvas ONNX graph, and return its operator audit.
    Key Params: Explicit arguments and constructor settings control the described operation.
    State/Dependencies: PyTorch tensors and module parameters; exporter additionally uses ONNX and filesystem output.
    """
    canvas = tuple(int(value) for value in canvas_hw)
    model = BasicTextLineR4StudentY4UVV5(canvas_hw=canvas)
    ignored_training_only = _load_checkpoint(model, checkpoint)
    model.eval()
    model.switch_to_deploy()
    module_audit = model.deployment_audit()
    if not module_audit["passed"]:
        raise RuntimeError(f"module deployment audit failed: {module_audit}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_tensor = torch.zeros(1, 6, canvas[0] // 2, canvas[1] // 2)
    export_kwargs = {
        "opset_version": int(opset),
        "do_constant_folding": True,
        "input_names": ["y4uv_input"],
        "output_names": ["y4uv_output"],
        "dynamic_axes": None,
    }
    try:
        torch.onnx.export(model, input_tensor, str(output), dynamo=False, **export_kwargs)
    except TypeError:
        # PyTorch releases before the dynamo flag keep the same legacy exporter
        # contract but reject the newer keyword.
        torch.onnx.export(model, input_tensor, str(output), **export_kwargs)
    graph_audit = audit_onnx_ops(output)
    if not graph_audit["passed"]:
        raise RuntimeError(f"ONNX deployment audit failed: {graph_audit}")
    return {
        "path": str(output),
        "canvas_hw": list(canvas),
        "ignored_training_only_checkpoint_keys": ignored_training_only,
        "module_audit": module_audit,
        "graph_audit": graph_audit,
        "parameter_count": model.parameter_count(),
        "macs": model.estimate_macs(),
        "analytic_receptive_field": model.analytic_receptive_field,
    }


def main() -> None:
    """@luna-comment
    Input: No explicit inputs; uses the module configuration.
    Output: None
    Key Steps: Parse export arguments and print a machine-readable audit summary.
    Key Params: Explicit arguments and constructor settings control the described operation.
    State/Dependencies: PyTorch tensors and module parameters; exporter additionally uses ONNX and filesystem output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--canvas-h", type=int, default=96)
    parser.add_argument("--canvas-w", type=int, default=480)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    report = export_v5_deploy(
        args.output,
        checkpoint=args.checkpoint,
        canvas_hw=(args.canvas_h, args.canvas_w),
        opset=args.opset,
    )
    import json

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
