"""Export static, randomly initialized Y4UV V3 graphs for performance tests.

This module is deliberately separate from the legacy YUV444 exporters and from
the checkpoint-capable Y4UV development exporter.  It emits one static graph
for each requested batch size.  The graph has one six-channel input and one
six-channel output at half resolution; no checkpoint or training state is
loaded.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import torch

# Direct ``python tools/<script>.py`` execution places only ``tools/`` on the
# import path.  Add this script's repository root without depending on the
# caller's environment or changing any legacy module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from realesrgan.archs.basic_text_line_r4_student_y4uv_v3_arch import (
    BasicTextLineR4StudentY4UVV3,
)


PERFORMANCE_CANVAS_HW = (96, 480)
PERFORMANCE_BATCH_SIZES = (1, 2, 4, 8, 16)
Y4UV_CHANNELS = 6
DEFAULT_OPSET = 17
DEFAULT_SEED = 20260825


def validate_canvas_hw(canvas_hw: Sequence[int]) -> tuple[int, int]:
    """@luna-comment Validate and normalize the fixed full-resolution canvas.

    Input:
        A two-element height/width sequence in full-resolution pixels.
    Output:
        An integer ``(height, width)`` tuple accepted by the Y4UV model.
    Key Steps:
        Check the exact performance canvas and require dimensions divisible by
        the architecture's 32-pixel downsampling granularity.
    Key Params:
        ``PERFORMANCE_CANVAS_HW`` is the required 96x480 benchmark canvas.
    State/Dependencies:
        No mutable state; this mirrors the model constructor contract.
    """
    if len(canvas_hw) != 2:
        raise ValueError("canvas_hw must contain exactly height and width")
    normalized = (int(canvas_hw[0]), int(canvas_hw[1]))
    if normalized != PERFORMANCE_CANVAS_HW:
        raise ValueError(
            f"performance exports require canvas_hw={PERFORMANCE_CANVAS_HW}, "
            f"got {normalized}"
        )
    if any(value <= 0 or value % 32 for value in normalized):
        raise ValueError("canvas dimensions must be positive multiples of 32")
    return normalized


def validate_batch_sizes(batch_sizes: Iterable[int]) -> tuple[int, ...]:
    """@luna-comment Normalize unique positive batches while preserving CLI order.

    Input:
        An iterable of requested static batch sizes.
    Output:
        A tuple of positive integers without duplicates.
    Key Steps:
        Convert values to integers, reject non-positive entries, and preserve
        the first occurrence of each value for deterministic output names.
    Key Params:
        The default set is ``(1, 2, 4, 8, 16)``; custom sets are useful for a
        focused local export but the standard command uses all five values.
    State/Dependencies:
        No external state.
    """
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
    """@luna-comment Build version-compatible ONNX arguments with static axes.

    Input:
        An ONNX opset integer.
    Output:
        Keyword arguments for ``torch.onnx.export``; notably no
        ``dynamic_axes`` entry is present.
    Key Steps:
        Set one input/output name, constant folding, and the requested opset.
        Pass ``dynamo=False`` only on PyTorch versions exposing that parameter.
    Key Params:
        Static axes are intentional because benchmark runners need fixed
        ``[B,6,48,240]`` buffers.
    State/Dependencies:
        Depends on the installed PyTorch ONNX exporter signature only.
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


def _value_info_shape(value_info: Any) -> list[Any]:
    """@luna-comment Extract concrete or symbolic dimensions from ONNX metadata.

    Input:
        An ONNX ``ValueInfoProto`` object.
    Output:
        A list containing integer dimensions or symbolic names/``None``.
    Key Steps:
        Read ``dim_value`` first, then symbolic ``dim_param`` for each axis.
    Key Params:
        None.
    State/Dependencies:
        Requires an object implementing ONNX tensor shape metadata.
    """
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
    """@luna-comment Inspect exported graph I/O shapes using ONNX when available.

    Input:
        A generated ONNX file path.
    Output:
        A status record with input/output names and static shape lists, or an
        explicit dependency-unavailable record.
    Key Steps:
        Load the graph with the optional ``onnx`` package and inspect its first
        input and output value-info entries.
    Key Params:
        None.
    State/Dependencies:
        This function is read-only; absence of ``onnx`` is reported, not hidden.
    """
    try:
        import onnx
    except ImportError as exc:
        return {
            "status": "dependency_unavailable",
            "dependency": "onnx",
            "reason": str(exc),
        }

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
    """@luna-comment Compare an ONNX Runtime result with the PyTorch result.

    Input:
        An ONNX path, the exact float32 sample used for export, and its
        PyTorch output reference.
    Output:
        A parity status with output shape and maximum/mean absolute error, or
        a dependency-unavailable record.
    Key Steps:
        Create a CPU inference session, run the named six-channel input, and
        compare the first output against the traced PyTorch tensor.
    Key Params:
        ``tolerance`` is the maximum absolute error accepted for the check.
    State/Dependencies:
        Requires optional ``onnxruntime`` and NumPy; no files are modified.
    """
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


def export_static_y4uv(
    model: torch.nn.Module,
    output: Path,
    batch_size: int,
    canvas_hw: Sequence[int] = PERFORMANCE_CANVAS_HW,
    opset: int = DEFAULT_OPSET,
    seed: int = DEFAULT_SEED,
    inspect_graph: bool = True,
    check_onnxruntime: bool = True,
) -> dict[str, Any]:
    """@luna-comment Export and validate one fixed-batch random-weight graph.

    Input:
        An eval-mode Y4UV model, output path, positive static batch, full
        canvas, ONNX opset, and deterministic sample seed.
    Output:
        A JSON-serializable per-graph report proving tensor shapes and optional
        ONNX/ONNX Runtime checks.
    Key Steps:
        Generate a fixed sample, check PyTorch output, export without dynamic
        axes, inspect graph metadata, and optionally run runtime parity.
    Key Params:
        ``inspect_graph`` and ``check_onnxruntime`` allow dependency-aware unit
        tests while the CLI enables both checks.
    State/Dependencies:
        Writes only ``output``; model parameters are not changed.
    """
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
    export_kwargs = _onnx_export_kwargs(opset)
    torch.onnx.export(model, (sample,), str(output), **export_kwargs)
    report: dict[str, Any] = {
        "status": "exported",
        "architecture": "BasicTextLineR4StudentY4UVV3",
        "weights": "random_initialized",
        "checkpoint": None,
        "haar_switch": bool(getattr(model, "haar_switch", True)),
        "batch_size": batch,
        "canvas_hw": list(canvas),
        "input_contract": list(expected_shape),
        "output_contract": list(expected_shape),
        "output": str(output),
        "opset": int(opset),
        "seed": int(seed) + batch,
        "static_axes": True,
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
    """@luna-comment Export the standard five static performance fixtures.

    Input:
        Output directory, batch sizes, fixed canvas, Haar switch, opset, seed,
        and optional dependency checks.
    Output:
        A manifest containing one report for every requested static graph.
    Key Steps:
        Validate the fixed 96x480 canvas, construct one random model, export
        each batch-specific graph, and persist a manifest beside the graphs.
    Key Params:
        The standard batches are ``1,2,4,8,16`` and the standard input/output
        shape is ``[B,6,48,240]``.
    State/Dependencies:
        Writes ONNX files and one JSON manifest under ``output_dir`` only.
    """
    canvas = validate_canvas_hw(canvas_hw)
    batches = validate_batch_sizes(batch_sizes)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    model = BasicTextLineR4StudentY4UVV3(canvas_hw=canvas, haar_switch=haar_switch).eval()
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    suffix = "haar" if haar_switch else "nohaar"
    for batch in batches:
        output = output_dir / f"basic_text_line_y4uv_v3_perf_b{batch}_{suffix}.onnx"
        try:
            reports.append(
                export_static_y4uv(
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
        except Exception as exc:  # Keep all per-batch evidence in the manifest.
            failures.append({"batch_size": batch, "output": str(output), "error": repr(exc)})

    manifest = {
        "status": "passed" if not failures else "failed",
        "architecture": "BasicTextLineR4StudentY4UVV3",
        "weights": "random_initialized",
        "checkpoint": None,
        "canvas_hw": list(canvas),
        "half_resolution": [canvas[0] // 2, canvas[1] // 2],
        "batch_sizes": list(batches),
        "expected_graph_count": len(batches),
        "haar_switch": bool(haar_switch),
        "opset": int(opset),
        "seed": int(seed),
        "graphs": reports,
        "failures": failures,
    }
    manifest_path = output_dir / f"basic_text_line_y4uv_v3_perf_{suffix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"one or more static exports failed; see {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """@luna-comment Parse performance-export arguments and emit a JSON manifest.

    Input:
        CLI arguments selecting output directory, optional batch subset, Haar
        mode, opset, seed, and dependency-check switches.
    Output:
        Exit code ``0`` with a manifest on success, or ``2`` after writing a
        failure manifest with per-batch evidence.
    Key Steps:
        Resolve the standard five batches by default and delegate to
        ``export_performance_set`` without accepting checkpoint arguments.
    Key Params:
        ``--no-haar-switch`` exports the ordinary-convolution variant; it does
        not change the six-channel static I/O contract.
    State/Dependencies:
        May write ONNX and JSON artifacts under the requested output directory.
    """
    parser = argparse.ArgumentParser(description="Export static Y4UV V3 performance ONNX graphs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=PERFORMANCE_BATCH_SIZES)
    parser.add_argument("--canvas-h", type=int, default=PERFORMANCE_CANVAS_HW[0])
    parser.add_argument("--canvas-w", type=int, default=PERFORMANCE_CANVAS_HW[1])
    parser.add_argument("--haar-switch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-onnx-graph-check", action="store_true")
    parser.add_argument("--skip-onnxruntime", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = export_performance_set(
            args.output_dir,
            batch_sizes=args.batch_sizes,
            canvas_hw=(args.canvas_h, args.canvas_w),
            haar_switch=args.haar_switch,
            opset=args.opset,
            seed=args.seed,
            inspect_graph=not args.skip_onnx_graph_check,
            check_onnxruntime=not args.skip_onnxruntime,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, indent=2))
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
