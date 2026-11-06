"""Export exact CUDA P17 preprocessing stages for hollow-stroke root-cause debugging."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.lenovo_p17_preprocess import preprocess_lq_rgb  # noqa: E402
from realesrgan.lenovo_p17_torch_gpu import preprocess_p17_torch_gpu_with_debug  # noqa: E402
from tools.infer_basic_text_line_full_image import load_lq_preprocess_config  # noqa: E402


VARIANT_OVERRIDES = {
    "A": {},
    "B": {"p17_wiener_global_clip_enabled": True},
    "C": {"p17_wiener_antiring_direction_enabled": True},
    "D": {"p17_wiener_antiring_envelope_enabled": True},
    "E": {"p17_wiener_antiring_direction_enabled": True, "p17_wiener_antiring_envelope_enabled": True},
    "F": {"lambda": 0.04},
    "G": {"p17_wiener_bypass_enabled": True},
}


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    """@luna-comment
    Input: Output path and float RGB HWC image in [0, 1].
    Output: Writes an 8-bit PNG in BGR file encoding; creates the parent directory.
    Key Steps: Clips and rounds RGB before OpenCV color conversion and validates write.
    Key Params: No algorithm parameters.
    State/Dependencies: OpenCV filesystem output only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.clip(rgb * 255.0 + .5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(path)


def _write_scalar(path: Path, value: np.ndarray, *, signed_scale: float | None = None) -> None:
    """@luna-comment
    Input: Output path, float HW map, and optional signed visualization scale in luma units.
    Output: Writes an 8-bit grayscale map or a BGR signed heatmap PNG.
    Key Steps: Normalizes unsigned maps to [0,1]; signed maps map zero to mid-gray
    before applying a color map so positive and negative correction are distinguishable.
    Key Params: signed_scale is the absolute luma delta represented by full scale.
    State/Dependencies: OpenCV filesystem output only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if signed_scale is None:
        encoded = np.clip(value * 255.0 + .5, 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(path), encoded):
            raise IOError(path)
        return
    encoded = np.clip((value / signed_scale * .5 + .5) * 255.0 + .5, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO)
    if not cv2.imwrite(str(path), heat):
        raise IOError(path)


def _choose_points(score: np.ndarray, count: int, min_distance: int) -> list[tuple[int, int]]:
    """@luna-comment
    Input: HW nonnegative diagnostic score map, requested count, and Euclidean exclusion radius.
    Output: Descending peak coordinates as (x,y) with spatially separated neighborhoods.
    Key Steps: Repeatedly takes the maximum then masks its circular neighborhood.
    Key Params: min_distance prevents repeated profiles over one glyph defect.
    State/Dependencies: NumPy/OpenCV only; does not change P17 data.
    """
    selected: list[tuple[int, int]] = []
    remaining = score.copy()
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(remaining)), remaining.shape)
        if remaining[y, x] <= 0:
            break
        selected.append((int(x), int(y)))
        cv2.circle(remaining, (int(x), int(y)), min_distance, 0.0, thickness=-1)
    return selected


def _sample_normal_profiles(
    output: Path, input_y: np.ndarray, stages: dict[str, np.ndarray], points: list[tuple[int, int]],
) -> None:
    """@luna-comment
    Input: Output directory, input luma, named HW stage maps, and sampled defect points.
    Output: Per-point CSV and PNG profiles along the local input-luma gradient normal.
    Key Steps: Estimates local Sobel normal, bilinearly samples each stage over +/-40 px,
    and writes named curves for first-stage attribution.
    Key Params: The fixed 40 px half-length captures both sides of narrow text strokes.
    State/Dependencies: OpenCV and Matplotlib only; profiles are diagnostic artifacts.
    """
    grad_x = cv2.Sobel(input_y, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(input_y, cv2.CV_32F, 0, 1, ksize=3)
    distance = np.arange(-40., 41., 1., dtype=np.float32)
    height, width = input_y.shape
    for number, (x, y) in enumerate(points, 1):
        nx, ny = float(grad_x[y, x]), float(grad_y[y, x])
        norm = max((nx * nx + ny * ny) ** .5, 1.e-6)
        nx, ny = nx / norm, ny / norm
        map_x = np.clip(x + distance * nx, 0, width - 1).reshape(1, -1)
        map_y = np.clip(y + distance * ny, 0, height - 1).reshape(1, -1)
        input_samples = cv2.remap(input_y.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)[0]
        sampled = {name: cv2.remap(value.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)[0] for name, value in stages.items()}
        with (output / f"profile_{number:02d}_normal.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["distance_px", "input_y", *stages.keys()])
            for index, d in enumerate(distance):
                writer.writerow([float(d), float(input_samples[index]), *[float(values[index]) for values in sampled.values()]])
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(distance, input_samples, "--", color="0.35", label="input_y")
        for name, values in sampled.items():
            axis.plot(distance, values, label=name)
        axis.set(title=f"P{number} normal profile at ({x}, {y}), n=({nx:.3f},{ny:.3f})", xlabel="normal distance (px)", ylabel="luma / delta")
        axis.grid(alpha=.25)
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(output / f"profile_{number:02d}_normal.png", dpi=180)
        plt.close(figure)


def main() -> int:
    """@luna-comment
    Input: One RGB image, enabled frozen P17 YAML, output directory, and optional ROI.
    Output: Exact CUDA P17 stage images, patch table, normal profiles, and validation JSON.
    Key Steps: Runs regular and debug P17 paths, rejects any output mismatch, exports
    all retained tensors, selects high-impact points, then serializes Otsu patch evidence.
    Key Params: --points, --point, and --roi restrict only diagnostic profile selection;
    --variant selects an explicit A/B override and neither changes a frozen default.
    State/Dependencies: Requires a CUDA device and the repository P17 GPU implementation;
    does not load a restoration checkpoint or invoke a restoration model.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"),
                        help="export one exact normal profile at this image-space coordinate")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--variant", choices=sorted(VARIANT_OVERRIDES), default="A",
                        help="Frozen A, global-clip B, direction C, envelope D, composed E, or lambda control F.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P17 stage diagnostic requires CUDA")
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).cuda()
    params = load_lq_preprocess_config(args.config)
    params.update(VARIANT_OVERRIDES[args.variant])
    if not params["enabled"] or params["implementation"] != "torch_gpu":
        raise ValueError("stage diagnostic requires enabled torch_gpu P17 config")
    with torch.no_grad():
        regular = preprocess_lq_rgb(rgb, params)
        debug_output, debug = preprocess_p17_torch_gpu_with_debug(rgb, params)
    difference = float((regular.float() - debug_output.float()).abs().max().item())
    if difference != 0.0:
        raise AssertionError(f"debug path differs from regular P17 output: {difference}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    arrays = {name: value[0, 0].detach().float().cpu().numpy() for name, value in debug.items() if isinstance(value, torch.Tensor) and value.ndim == 4}
    rgb_out = debug_output[0].permute(1, 2, 0).detach().float().cpu().numpy()
    _write_rgb(output / "00_input_rgb.png", rgb_np)
    _write_scalar(output / "01_input_y.png", arrays["input_y"])
    _write_scalar(output / "02_text_mask.png", arrays["text_mask"])
    _write_scalar(output / "03_deblurred_y.png", arrays["deblurred_y"])
    _write_scalar(output / "04_deblur_delta_signed_x16.png", arrays["deblur_delta"], signed_scale=1. / 16.)
    _write_scalar(output / "04a_raw_wiener_local_y.png", arrays["raw_wiener_local_y"])
    _write_scalar(output / "04b_direction_rejected_delta_signed_x16.png", arrays["direction_rejected_delta"], signed_scale=1. / 16.)
    _write_scalar(output / "04c_envelope_rejected_delta_signed_x16.png", arrays["envelope_rejected_delta"], signed_scale=1. / 16.)
    _write_scalar(output / "04d_global_clip_delta_signed_x16.png", arrays["global_clip_delta"], signed_scale=1. / 16.)
    _write_scalar(output / "04e_safe_deblurred_y.png", arrays["safe_deblurred_y"])
    _write_scalar(output / "05_regional_residual_sum_signed_x16.png", arrays["regional_residual_sum"], signed_scale=1. / 16.)
    _write_scalar(output / "06_regional_weight_sum.png", arrays["regional_weight_sum"] / max(float(arrays["regional_weight_sum"].max()), 1.e-6))
    _write_scalar(output / "07_fused_correction_signed_x16.png", arrays["fused_correction"], signed_scale=1. / 16.)
    _write_scalar(output / "08_write_weight.png", arrays["write_weight"])
    _write_scalar(output / "09_weighted_correction_signed_x16.png", arrays["weighted_correction"], signed_scale=1. / 16.)
    _write_scalar(output / "10_corrected_y.png", arrays["corrected_y"])
    _write_scalar(output / "11_requested_y_delta_signed_x16.png", arrays["requested_y_delta"], signed_scale=1. / 16.)
    _write_scalar(output / "12_applied_y_delta_signed_x16.png", arrays["applied_y_delta"], signed_scale=1. / 16.)
    _write_scalar(output / "13_output_y.png", arrays["output_y"])
    _write_rgb(output / "14_output_rgb.png", rgb_out)
    score = np.abs(arrays["weighted_correction"])
    if args.roi is not None:
        x0, y0, x1, y1 = args.roi
        allowed = np.zeros_like(score, dtype=bool)
        allowed[max(0, y0):min(score.shape[0], y1), max(0, x0):min(score.shape[1], x1)] = True
        score[~allowed] = 0.0
    points = [(int(args.point[0]), int(args.point[1]))] if args.point is not None else _choose_points(score, args.points, min_distance=96)
    profile_stages = {
        "deblurred_y": arrays["deblurred_y"],
        "fused_correction": arrays["fused_correction"],
        "weighted_correction": arrays["weighted_correction"],
        "corrected_y": arrays["corrected_y"],
        "output_y": arrays["output_y"],
    }
    _sample_normal_profiles(output, arrays["input_y"], profile_stages, points)
    patch_xyxy = debug["patch_xyxy"].detach().cpu().numpy()
    patch_low = debug["patch_low"][0].detach().float().cpu().numpy()
    patch_high = debug["patch_high"][0].detach().float().cpu().numpy()
    patch_valid = debug["patch_valid"][0].detach().cpu().numpy()
    with (output / "regional_otsu_patches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left", "top", "right", "bottom", "low_y", "high_y", "valid"])
        for box, low, high, valid in zip(patch_xyxy, patch_low, patch_high, patch_valid):
            writer.writerow([*map(int, box), float(low), float(high), bool(valid)])
    contributors: list[dict[str, object]] = []
    for number, (x, y) in enumerate(points, 1):
        covered = np.where((patch_xyxy[:, 0] <= x) & (x < patch_xyxy[:, 2]) & (patch_xyxy[:, 1] <= y) & (y < patch_xyxy[:, 3]))[0]
        contributors.append({"point": number, "xy": [x, y], "patch_indices": covered.astype(int).tolist()})
    report = {
        "image": str(args.image),
        "config": str(args.config),
        "variant": args.variant,
        "variant_overrides": VARIANT_OVERRIDES[args.variant],
        "restoration_model_loaded": False,
        "max_abs_debug_vs_regular": difference,
        "sigma": debug["sigma"].detach().float().cpu().tolist(),
        "profile_points_xy": points,
        "patch_count": int(len(patch_xyxy)),
        "direction_rejected_pixels": int(np.count_nonzero(np.abs(arrays["direction_rejected_delta"]) >= 1.e-8)),
        "envelope_rejected_pixels": int(np.count_nonzero(np.abs(arrays["envelope_rejected_delta"]) >= 1.e-8)),
        "global_clip_rejected_pixels": int(np.count_nonzero(np.abs(arrays["global_clip_delta"]) >= 1.e-8)),
        "contributors": contributors,
        "stage_files": [item.name for item in sorted(output.glob("*.png"))],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
