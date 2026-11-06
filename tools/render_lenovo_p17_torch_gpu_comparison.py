"""Render a real-image comparison between frozen CPU P17 and the opt-in CUDA candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from realesrgan.lenovo_p17_preprocess import preprocess_lq_rgb
from realesrgan.lenovo_p17_preprocess import normalize_lq_preprocess_config
from realesrgan.lenovo_p17_stage_a_exact import _p10_segment_text, _p14_kernel_ab, _rgb_to_luma
from realesrgan.lenovo_p17_torch_gpu import _estimate_sigma, _regional_otsu_lut, _segment_text, _wiener_deblur


def _sha256(path: Path) -> str:
    """@luna-comment Return the SHA-256 of one source or input file for renderer provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    """@luna-comment Load one 8-bit image as normalized RGB and reject unreadable or non-RGB inputs."""
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _write_rgb(path: Path, value: np.ndarray) -> None:
    """@luna-comment Save normalized RGB as a deterministic 8-bit PNG comparison artifact."""
    clipped = np.clip(value * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(clipped, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write {path}")


def main() -> None:
    """@luna-comment Produce LQ, frozen CPU, CUDA candidate, and amplified error panels for one supplied real LQ."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("the comparison renderer requires a CUDA device")
    rgb = _read_rgb(args.input)
    source = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    config = {"enabled": True, "lambda": 0.03, "alpha": 1.0}
    cpu = preprocess_lq_rgb(source, {**config, "implementation": "cpu_exact"})
    gpu = preprocess_lq_rgb(source.to(args.device), {**config, "implementation": "torch_gpu"}).cpu()
    lq_rgb, cpu_rgb, gpu_rgb = (item[0].permute(1, 2, 0).numpy() for item in (source, cpu, gpu))
    difference = np.abs(gpu_rgb - cpu_rgb)
    cpu_config = normalize_lq_preprocess_config(config)
    _, cpu_meta = _p14_kernel_ab(rgb, cpu_config)
    cpu_mask, _ = _p10_segment_text(_rgb_to_luma(rgb), cpu_config)
    gpu_y = (0.299 * source.to(args.device)[:, 0:1] + 0.587 * source.to(args.device)[:, 1:2] + 0.114 * source.to(args.device)[:, 2:3]).clamp(0.0, 1.0)
    gpu_mask, _ = _segment_text(gpu_y, cpu_config)
    gpu_sigma = _estimate_sigma(gpu_y, gpu_mask, cpu_config)
    cpu_mask_tensor = torch.from_numpy(cpu_mask.astype(np.bool_)).to(args.device).unsqueeze(0).unsqueeze(0)
    gpu_sigma_with_cpu_mask = _estimate_sigma(gpu_y, cpu_mask_tensor, cpu_config)
    gpu_deblur = _wiener_deblur(gpu_y, gpu_mask, gpu_sigma, float(cpu_config["lambda"]))
    gpu_corrected = _regional_otsu_lut(gpu_deblur, gpu_mask, float(cpu_config["alpha"]), cpu_config)
    cpu_stage = cpu_meta["_p14_stage_outputs"][0]
    cpu_deblur = np.asarray(cpu_stage["deblur_y"], dtype=np.float32)
    cpu_final_y = np.asarray(cpu_stage["b_stages"][0]["y"], dtype=np.float32)
    gpu_mask_np = gpu_mask[0, 0].cpu().numpy().astype(bool)
    mask_intersection = np.logical_and(cpu_mask, gpu_mask_np).sum()
    mask_union = np.logical_or(cpu_mask, gpu_mask_np).sum()
    cpu_sigma_tensor = torch.full_like(gpu_sigma, float(cpu_meta["p14_kernel_sigma"]))
    gpu_deblur_cpu_sigma = _wiener_deblur(gpu_y, gpu_mask, cpu_sigma_tensor, float(cpu_config["lambda"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(args.output_dir / "00_lq.png", lq_rgb)
    _write_rgb(args.output_dir / "01_cpu_exact.png", cpu_rgb)
    _write_rgb(args.output_dir / "02_torch_gpu.png", gpu_rgb)
    _write_rgb(args.output_dir / "03_gpu_minus_cpu_x2.png", np.clip(difference * 2.0, 0.0, 1.0))
    _write_rgb(args.output_dir / "04_comparison.png", np.concatenate((lq_rgb, cpu_rgb, gpu_rgb, np.clip(difference * 2.0, 0.0, 1.0)), axis=1))
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "input": str(args.input), "input_sha256": _sha256(args.input), "shape": list(source.shape),
        "gpu_module_sha256": _sha256(Path(__import__("realesrgan.lenovo_p17_torch_gpu", fromlist=["__file__"]).__file__)),
        "cpu_reference_sha256": _sha256(Path(__import__("realesrgan.lenovo_p17_stage_a_exact", fromlist=["__file__"]).__file__)),
        "normalized_config": cpu_config, "cpu_gpu_mae": float(difference.mean()),
        "cpu_gpu_p99_abs_error": float(np.quantile(difference, 0.99)),
        "cpu_mask_fraction": float(cpu_mask.mean()), "gpu_mask_fraction": float(gpu_mask_np.mean()),
        "mask_iou": float(mask_intersection / max(mask_union, 1)),
        "cpu_kernel_sigma": cpu_meta.get("p14_kernel_sigma"), "gpu_kernel_sigma": float(gpu_sigma[0].item()),
        "gpu_kernel_sigma_with_cpu_mask": float(gpu_sigma_with_cpu_mask[0].item()),
        "deblur_y_mae": float(np.mean(np.abs(cpu_deblur - gpu_deblur[0, 0].cpu().numpy()))),
        "deblur_y_mae_with_cpu_sigma": float(np.mean(np.abs(cpu_deblur - gpu_deblur_cpu_sigma[0, 0].cpu().numpy()))),
        "post_lut_y_mae": float(np.mean(np.abs(cpu_final_y - gpu_corrected[0, 0].cpu().numpy()))),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
