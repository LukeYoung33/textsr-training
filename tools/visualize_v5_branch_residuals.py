"""Visualize signed full-resolution Y residuals from an existing V5 branch dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv420


def _args() -> argparse.Namespace:
    """@luna-comment
    Input: an existing branch-dump root and a new empty visualization root. Output: parsed visualization arguments.
    Key Steps: require explicit source/output paths. Key Params: percentile fixes one shared signed display scale.
    State/Dependencies: argparse only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-root", type=Path)
    source.add_argument("--subband-root", type=Path)
    source.add_argument("--raw-subband-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--percentile", type=float, default=99.5)
    return parser.parse_args()


def _load_luma_residual(path: Path) -> np.ndarray:
    """@luna-comment
    Input: CHW six-channel raw Y4UV residual NPY file. Output: full-resolution HW signed luma residual.
    Key Steps: restore BCHW, pixel-shuffle the four Y phases, and preserve signed float values. Key Params: no clamp,
    normalization, or RGB decode. State/Dependencies: NumPy/Torch Y4UV contract.
    """
    packed = torch.from_numpy(np.load(path)).unsqueeze(0)
    y, _, _ = y4uv_to_yuv420(packed)
    return y[0, 0].float().numpy()


def _write_signed(path: Path, residual: np.ndarray, scale: float) -> None:
    """@luna-comment
    Input: signed HW luma residual and one positive shared display scale. Output: RGB PNG with neutral gray at zero,
    red positive residual, and blue negative residual. Key Steps: symmetric clipping and channel-specific mapping.
    Key Params: identical scale is used for every branch to preserve visual comparability. State/Dependencies: Pillow.
    """
    value = np.clip(residual / scale, -1.0, 1.0)
    neutral = np.full_like(value, 128.0)
    image = np.stack((neutral + 127.0 * np.maximum(value, 0.0), neutral - 100.0 * np.abs(value),
                      neutral + 127.0 * np.maximum(-value, 0.0)), axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), "RGB").save(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    """@luna-comment
    Input: completed V5 branch dump. Output: local/context/total signed luma maps under a new append-only root.
    Key Steps: load all four cases, calculate a global percentile scale, and render each residual separately plus their sum.
    Key Params: total is local plus context, not a reconstructed RGB output. State/Dependencies: existing dump NPY files only.
    """
    source = args.input_root.resolve()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    report_path = source / "branch_dump_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    cases = [(input_dir.name, model_dir.name, model_dir) for input_dir in source.iterdir() if input_dir.is_dir()
             for model_dir in input_dir.iterdir() if model_dir.is_dir()]
    if not cases:
        raise RuntimeError("no input/model branch directories found")
    values: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for input_name, model_name, directory in cases:
        local = _load_luma_residual(directory / "local_residual.npy")
        context = _load_luma_residual(directory / "context_residual.npy")
        values[(input_name, model_name)] = {"local": local, "context": context, "total": local + context}
    percentile = float(args.percentile)
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    scale = float(np.percentile(np.concatenate([np.abs(value) for row in values.values() for value in row.values()]), percentile))
    scale = max(scale, 1e-8)
    for (input_name, model_name), row in values.items():
        directory = root / input_name / model_name
        directory.mkdir(parents=True, exist_ok=True)
        for branch, value in row.items():
            np.save(directory / f"{branch}_y_residual.npy", value.astype(np.float32))
            _write_signed(directory / f"{branch}_y_signed.png", value, scale)
    report = {"status": "COMPLETE", "source": str(source), "percentile": percentile, "shared_luma_residual_scale": scale,
              "legend": {"gray": "zero", "red": "positive luma residual", "blue": "negative luma residual"},
              "cases": [{"input": input_name, "model": model_name} for input_name, model_name, _ in cases]}
    (root / "visualization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_subbands(args: argparse.Namespace) -> dict[str, object]:
    """@luna-comment
    Input: a completed local-subband dump and a new empty visualization root. Output: common-scale signed LL/LH/HL/HH
    luma contribution maps. Key Steps: load raw full-resolution contribution arrays, compute one global display scale,
    and render without per-image normalization. Key Params: same percentile rule as branch residual visualization.
    State/Dependencies: existing diagnostic NPY files only.
    """
    source = args.subband_root.resolve()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    report_path = source / "subband_dump_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    files = sorted(source.glob("*/*/*_y_contribution.npy"))
    if not files:
        raise RuntimeError("no subband contribution arrays found")
    values = {path: np.load(path) for path in files}
    scale = float(np.percentile(np.concatenate([np.abs(value) for value in values.values()]), float(args.percentile)))
    scale = max(scale, 1e-8)
    for path, value in values.items():
        relative = path.relative_to(source).with_suffix("")
        _write_signed(root / f"{relative}_signed.png", value, scale)
    report = {"status": "COMPLETE", "source": str(source), "percentile": float(args.percentile),
              "shared_luma_contribution_scale": scale,
              "legend": {"gray": "zero", "red": "positive luma contribution", "blue": "negative luma contribution"}}
    (root / "subband_visualization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_raw_subbands(args: argparse.Namespace) -> dict[str, object]:
    """Render native local Haar maps without inverse Haar, phase unpacking, or resize.

    Input arrays are the 48x240 local-grid coefficients exported by the raw
    diagnostic. A separate shared signed scale is calculated for ``input`` and
    ``predicted`` maps so the small predicted residual is not hidden by the
    larger input Haar coefficients; no map receives an individual scale.
    """
    source = args.raw_subband_root.resolve()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    report_path = source / "raw_subband_dump_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    files = sorted(path for path in source.glob("*/*/*_raw.npy") if path.name != "background_selector_raw.npy")
    if not files:
        raise RuntimeError("no native raw subband arrays found")
    values = {path: np.load(path) for path in files}
    if any(value.ndim != 2 for value in values.values()):
        raise ValueError("raw subband arrays must be HW")
    percentile = float(args.percentile)
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    grouped = {
        kind: {path: value for path, value in values.items() if path.name.startswith(f"{kind}_")}
        for kind in ("input", "predicted")
    }
    scales = {
        kind: max(float(np.percentile(np.concatenate([np.abs(value) for value in rows.values()]), percentile)), 1e-8)
        for kind, rows in grouped.items() if rows
    }
    for path, value in values.items():
        kind = "input" if path.name.startswith("input_") else "predicted"
        relative = path.relative_to(source).with_suffix("")
        _write_signed(root / f"{relative}_signed.png", value, scales[kind])
    report = {
        "status": "COMPLETE",
        "source": str(source),
        "percentile": percentile,
        "native_only": True,
        "shared_luma_coefficient_scales": scales,
        "legend": {"gray": "zero", "red": "positive coefficient", "blue": "negative coefficient"},
    }
    (root / "raw_subband_visualization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    arguments = _args()
    outcome = run_raw_subbands(arguments) if arguments.raw_subband_root else (run_subbands(arguments) if arguments.subband_root else run(arguments))
    print(json.dumps(outcome, ensure_ascii=False))
