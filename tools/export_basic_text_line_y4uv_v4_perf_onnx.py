"""Export static V4 Y4UV graphs for performance measurements.

The exporter is intentionally independent of the existing V3 exporter.  It
does not load a checkpoint or alter a legacy graph; each requested batch gets
one static ``[B,6,48,240]`` input/output ONNX model plus a parity report.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import (
    BasicTextLineR4StudentY4UVV4,
    V4_CANVAS_HW,
)


PERFORMANCE_CANVAS_HW = V4_CANVAS_HW
PERFORMANCE_BATCH_SIZES = (1, 2, 4, 8, 16)
Y4UV_CHANNELS = 6
DEFAULT_OPSET = 17
DEFAULT_SEED = 20260826


def validate_canvas_hw(canvas_hw: Sequence[int]) -> tuple[int, int]:
    """@shanlan-comment Validate the fixed full-resolution canvas for static performance graphs."""
    if len(canvas_hw) != 2:
        raise ValueError("canvas_hw must contain exactly height and width")
    normalized = (int(canvas_hw[0]), int(canvas_hw[1]))
    if normalized != PERFORMANCE_CANVAS_HW:
        raise ValueError(
            f"performance exports require canvas_hw={PERFORMANCE_CANVAS_HW}, got {normalized}"
        )
    if any(value <= 0 or value % 16 for value in normalized):
        raise ValueError("V4 canvas dimensions must be positive multiples of 16")
    return normalized


def validate_batch_sizes(batch_sizes: Iterable[int]) -> tuple[int, ...]:
    """@shanlan-comment Normalize unique positive static batches while preserving request order."""
    result: list[int] = []
    seen: set[int] = set()
    for value in batch_sizes:
        batch = int(value)
        if batch <= 0:
            raise ValueError(f"batch size must be positive, got {batch}")
        if batch not in seen:
            result.append(batch)
            seen.add(batch)
    if not result:
        raise ValueError("at least one batch size is required")
    return tuple(result)


def _onnx_export_kwargs(opset: int) -> dict[str, Any]:
    """@shanlan-comment Build version-compatible torch ONNX arguments without dynamic axes."""
    kwargs: dict[str, Any] = {
        "input_names": ["y4uv"],
        "output_names": ["y4uv"],
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    return kwargs


def _value_info_shape(value_info: Any) -> list[Any]:
    """@shanlan-comment Extract concrete dimensions from one ONNX value-info record."""
    dimensions: list[Any] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_value:
            dimensions.append(int(dimension.dim_value))
        elif dimension.dim_param:
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append(None)
    return dimensions


def inspect_onnx_contract(path: Path) -> dict[str, Any]:
    """@shanlan-comment Inspect static input/output graph metadata when the ONNX package is installed."""
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
        "input_shape": _value_info_shape(graph.input[0]),
        "output_shape": _value_info_shape(graph.output[0]),
    }


def run_onnxruntime_parity(
    path: Path,
    sample: torch.Tensor,
    reference: torch.Tensor,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """@shanlan-comment Compare the static ONNX output against its PyTorch reference."""
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
    outputs = session.run(None, {session.get_inputs()[0].name: sample.numpy()})
    if not outputs:
        raise RuntimeError("ONNX Runtime returned no outputs")
    actual = np.asarray(outputs[0])
    expected = reference.detach().cpu().numpy()
    if actual.shape != expected.shape:
        raise RuntimeError(f"ONNX Runtime shape mismatch: {actual.shape} vs {expected.shape}")
    absolute_error = np.abs(actual - expected)
    max_abs = float(absolute_error.max()) if absolute_error.size else 0.0
    mean_abs = float(absolute_error.mean()) if absolute_error.size else 0.0
    return {
        "status": "passed" if max_abs <= float(tolerance) else "failed",
        "output_shape": list(actual.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "tolerance": float(tolerance),
    }


def export_static_y4uv_v4(
    model: torch.nn.Module,
    output: Path,
    batch_size: int,
    canvas_hw: Sequence[int] = PERFORMANCE_CANVAS_HW,
    opset: int = DEFAULT_OPSET,
    seed: int = DEFAULT_SEED,
    inspect_graph: bool = True,
    check_onnxruntime: bool = True,
) -> dict[str, Any]:
    """@shanlan-comment Export and validate one static random-weight V4 graph."""
    canvas = validate_canvas_hw(canvas_hw)
    batch = validate_batch_sizes((batch_size,))[0]
    half_hw = (canvas[0] // 2, canvas[1] // 2)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + batch)
    sample = torch.randn(batch, Y4UV_CHANNELS, *half_hw, generator=generator)

    model.eval()
    with torch.inference_mode():
        reference = model(sample)
    expected_shape = (batch, Y4UV_CHANNELS, *half_hw)
    if tuple(reference.shape) != expected_shape:
        raise RuntimeError(f"PyTorch output shape mismatch: {tuple(reference.shape)} vs {expected_shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (sample,), str(output), **_onnx_export_kwargs(opset))
    report: dict[str, Any] = {
        "status": "exported",
        "architecture": "BasicTextLineR4StudentY4UVV4",
        "weights": "random_initialized",
        "checkpoint": None,
        "haar_switch": bool(getattr(model, "haar_switch", True)),
        "batch": batch,
        "batch_size": batch,
        "canvas_hw": list(canvas),
        "input_shape": list(expected_shape),
        "output_shape": list(expected_shape),
        "input_contract": list(expected_shape),
        "output_contract": list(expected_shape),
        "onnx_path": str(output),
        "output": str(output),
        "opset": int(opset),
        "seed": int(seed) + batch,
        "static_axes": True,
        "stage_shapes": [list(shape) for shape in getattr(model, "stage_shapes", ())],
        "stage_volumes": list(getattr(model, "stage_volumes", ())),
    }
    if inspect_graph:
        graph_report = inspect_onnx_contract(output)
        report["onnx_graph"] = graph_report
        if graph_report.get("status") == "validated":
            expected_list = list(expected_shape)
            if graph_report.get("input_shape") != expected_list:
                raise RuntimeError(
                    f"ONNX input shape mismatch: {graph_report.get('input_shape')} vs {expected_list}"
                )
            if graph_report.get("output_shape") != expected_list:
                raise RuntimeError(
                    f"ONNX output shape mismatch: {graph_report.get('output_shape')} vs {expected_list}"
                )
    if check_onnxruntime:
        report["onnxruntime"] = run_onnxruntime_parity(output, sample, reference)
        if report["onnxruntime"].get("status") == "failed":
            raise RuntimeError(f"ONNX Runtime parity failed: {report['onnxruntime']}")
    return report


def export_performance_set(
    output_dir: Path,
    batch_sizes: Iterable[int] = PERFORMANCE_BATCH_SIZES,
    canvas_hw: Sequence[int] = PERFORMANCE_CANVAS_HW,
    haar_switch: bool = True,
    opset: int = DEFAULT_OPSET,
    seed: int = DEFAULT_SEED,
    inspect_graph: bool = True,
    check_onnxruntime: bool = True,
) -> dict[str, Any]:
    """@shanlan-comment Export the requested static V4 batch set and persist one manifest."""
    canvas = validate_canvas_hw(canvas_hw)
    batches = validate_batch_sizes(batch_sizes)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    model = BasicTextLineR4StudentY4UVV4(canvas_hw=canvas, haar_switch=haar_switch).eval()
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    suffix = "haar" if haar_switch else "nohaar"
    for batch in batches:
        output = output_dir / f"basic_text_line_y4uv_v4_perf_b{batch}_{suffix}.onnx"
        try:
            reports.append(
                export_static_y4uv_v4(
                    model,
                    output,
                    batch,
                    canvas_hw=canvas,
                    opset=opset,
                    seed=seed,
                    inspect_graph=inspect_graph,
                    check_onnxruntime=check_onnxruntime,
                )
            )
        except Exception as exc:  # pragma: no cover - CLI failure report path
            failures.append({"batch": batch, "onnx_path": str(output), "error": repr(exc)})
            raise

    manifest = {
        "status": "passed" if not failures else "failed",
        "architecture": "BasicTextLineR4StudentY4UVV4",
        "haar_switch": bool(haar_switch),
        "canvas_hw": list(canvas),
        "input_output_contract": "[B,6,48,240]",
        "batch_sizes": list(batches),
        "reports": reports,
        "failures": failures,
        "stage_shapes": [list(shape) for shape in model.stage_shapes],
        "stage_volumes": list(model.stage_volumes),
    }
    manifest_path = output_dir / f"basic_text_line_y4uv_v4_perf_{suffix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """@shanlan-comment Define the isolated V4 performance-export command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".artifacts/basic_text_line_y4uv_v4_perf_20260826"),
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(PERFORMANCE_BATCH_SIZES))
    parser.add_argument("--canvas-hw", nargs=2, type=int, default=list(PERFORMANCE_CANVAS_HW))
    parser.add_argument("--haar-switch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-onnx-inspect", action="store_true")
    parser.add_argument("--skip-onnxruntime", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """@shanlan-comment Run the V4 export set and print a compact JSON manifest summary."""
    args = build_parser().parse_args(argv)
    manifest = export_performance_set(
        output_dir=args.output_dir,
        batch_sizes=args.batch_sizes,
        canvas_hw=args.canvas_hw,
        haar_switch=args.haar_switch,
        opset=args.opset,
        seed=args.seed,
        inspect_graph=not args.skip_onnx_inspect,
        check_onnxruntime=not args.skip_onnxruntime,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
