"""Data-level smoke validation for the V4 background low-frequency loss.

The script builds the configured training dataset, runs a real V4 network
forward on the first small batch, and evaluates controlled RGB perturbations
with the exact registered bgLF module.  It intentionally makes no checkpoint
or source-data changes.  JSON, PNG panels, and a short Markdown report are
written under the requested debug directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from basicsr.archs import build_network
from basicsr.data import build_dataset
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.losses.background_low_frequency_loss import BackgroundLowFrequencyLoss, far_background_mask
from realesrgan.utils import YUV2RGB

# Importing these packages scans the local registries, including the V4 data
# and network classes.  The smoke uses the loss directly so it remains useful
# when BasicSR's global loss builder is not initialized by a train entrypoint.
import realesrgan.archs  # noqa: F401,E402
import realesrgan.data  # noqa: F401,E402
import realesrgan.losses  # noqa: F401,E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", "--opt", required=True, help="V4 training YAML")
    parser.add_argument("--output-dir", required=True, help="stable local/remote debug directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--offset", type=float, default=0.12, help="positive/negative far luminance RGB offset")
    return parser


def _device(value: str) -> torch.device:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


def _safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    return float(value)


def _luma(value: torch.Tensor) -> torch.Tensor:
    return 0.299 * value[:, :1] + 0.587 * value[:, 1:2] + 0.114 * value[:, 2:3]


def _dilate(value: torch.Tensor, radius: int) -> torch.Tensor:
    return torch.nn.functional.max_pool2d(value, 2 * int(radius) + 1, 1, int(radius)) if radius else value


def _erode(value: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return value
    return 1.0 - _dilate(1.0 - value, radius)


def _classification(gt_rgb: torch.Tensor, text: torch.Tensor, far: torch.Tensor, valid: torch.Tensor) -> str:
    gt_luma = _luma(gt_rgb)
    text_count = text.sum().item()
    far_count = far.sum().item()
    if text_count <= 0 or far_count <= 0:
        return "unknown"
    text_mean = (gt_luma * text).sum() / text.sum().clamp_min(1.0)
    far_mean = (gt_luma * far).sum() / far.sum().clamp_min(1.0)
    contrast = float((text_mean - far_mean).detach().cpu())
    if abs(contrast) < 0.01:
        return "low_contrast"
    return "text_darker" if contrast < 0.0 else "text_lighter"


def _panel(path: Path, gt: torch.Tensor, lq: torch.Tensor, far: torch.Tensor, masked: torch.Tensor, diff: torch.Tensor, valid: torch.Tensor) -> None:
    """Write a compact RGB/mask/difference panel using OpenCV only."""
    def rgb(value: torch.Tensor) -> np.ndarray:
        value = value.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return cv2.cvtColor(np.rint(value * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def mask(value: torch.Tensor) -> np.ndarray:
        value = value.detach().float().cpu().clamp(0.0, 1.0).squeeze().numpy()
        return cv2.cvtColor(np.rint(value * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def heat(value: torch.Tensor) -> np.ndarray:
        value = value.detach().float().cpu().abs().mean(dim=0).clamp(0.0, 1.0).numpy()
        return cv2.applyColorMap(np.rint(value * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)

    tiles = [rgb(gt), rgb(lq), mask(far), mask(masked), heat(diff), mask(valid)]
    height = max(tile.shape[0] for tile in tiles)
    tiles = [cv2.resize(tile, (tile.shape[1], height), interpolation=cv2.INTER_NEAREST) for tile in tiles]
    panel = np.concatenate(tiles, axis=1)
    if not cv2.imwrite(str(path), panel):
        raise IOError(f"failed to write smoke panel: {path}")


def _run_batch(batch: dict[str, Any], loss: BackgroundLowFrequencyLoss, device: torch.device, offset: float) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    gt = _tensor(batch["gt_rgb"], device)
    lq = _tensor(batch["lq_rgb"], device)
    valid = _tensor(batch["valid_mask_rgb"], device)
    text = _tensor(batch["text_mask_rgb"], device)
    availability = batch.get("structure_available", [])
    if availability:
        text = text.clone()
        for index, available in enumerate(availability):
            if not bool(available):
                text[index] = 0.0
    far = far_background_mask(text, valid, radius=loss.radius, threshold=loss.threshold, reference=gt)
    valid_binary = (valid > loss.threshold).to(gt.dtype)
    text_binary = (text > loss.threshold).to(gt.dtype)
    # Keep the text/near-text perturbation at least three sigma inside the
    # excluded support, so the expected small residual is attributable to the
    # documented Gaussian gradient halo rather than accidental far pixels.
    blur_radius = max(1, int(math.ceil(3.0 * loss.sigma)))
    protected = _dilate(text_binary, loss.radius + blur_radius) * valid_binary
    masked_zone = protected * (1.0 - far)
    padding = (1.0 - valid_binary).clamp_min(0.0)
    padding_interior = padding * (1.0 - _dilate(valid_binary, blur_radius)).clamp_min(0.0)

    base = gt.clone()
    base_loss, details = loss(base, gt, text_mask_rgb=text, valid_mask_rgb=valid, return_details=True)
    text_pred = (gt + 0.20 * masked_zone).detach()
    text_loss = loss(text_pred, gt, text_mask_rgb=text, valid_mask_rgb=valid)
    pad_pred = (gt + 0.20 * padding_interior).detach()
    pad_loss = loss(pad_pred, gt, text_mask_rgb=text, valid_mask_rgb=valid)
    far_positive = (gt + float(offset) * far).detach()
    far_negative = (gt - float(offset) * far).detach()
    positive_loss = loss(far_positive, gt, text_mask_rgb=text, valid_mask_rgb=valid)
    negative_loss = loss(far_negative, gt, text_mask_rgb=text, valid_mask_rgb=valid)

    grad_pred = (gt + float(offset) * far).detach().requires_grad_(True)
    grad_loss = loss(grad_pred, gt, text_mask_rgb=text, valid_mask_rgb=valid)
    grad_loss.backward()
    grad = grad_pred.grad.detach().abs().mean(dim=1, keepdim=True)
    far_grad = (grad * far).sum()
    nonfar_grad = (grad * (1.0 - far)).sum()
    grad_total = grad.sum().clamp_min(1e-12)

    record = {
        "batch_size": int(gt.shape[0]),
        "resolution": [int(gt.shape[-2]), int(gt.shape[-1])],
        "far_pixels": float(far.sum().detach().cpu()),
        "valid_pixels": float(valid_binary.sum().detach().cpu()),
        "text_pixels": float(text_binary.sum().detach().cpu()),
        "far_coverage": _safe_float(details["far_coverage"]),
        "far_over_valid": _safe_float(details["far_over_valid"]),
        "valid_coverage": _safe_float(details["valid_coverage"]),
        "text_coverage": _safe_float(details["text_coverage"]),
        "padding_coverage": _safe_float(padding.mean()),
        "padding_interior_coverage": _safe_float(padding_interior.mean()),
        "polarity": [_classification(gt[i : i + 1], text_binary[i : i + 1], far[i : i + 1], valid_binary[i : i + 1]) for i in range(gt.shape[0])],
        "loss_base": _safe_float(base_loss),
        "loss_text_zone": _safe_float(text_loss),
        "loss_padding": _safe_float(pad_loss),
        "loss_far_positive": _safe_float(positive_loss),
        "loss_far_negative": _safe_float(negative_loss),
        "text_to_far_positive_ratio": _safe_float(text_loss / positive_loss.clamp_min(1e-12)),
        "padding_to_far_positive_ratio": _safe_float(pad_loss / positive_loss.clamp_min(1e-12)),
        "far_positive_over_base": _safe_float(positive_loss / base_loss.clamp_min(1e-12)),
        "far_negative_over_base": _safe_float(negative_loss / base_loss.clamp_min(1e-12)),
        "gradient_total": _safe_float(grad_total),
        "gradient_far": _safe_float(far_grad),
        "gradient_nonfar_halo": _safe_float(nonfar_grad),
        "gradient_far_fraction": _safe_float(far_grad / grad_total),
        "gradient_nonfar_fraction": _safe_float(nonfar_grad / grad_total),
    }
    tensors = {
        "gt": gt[0],
        "lq": lq[0],
        "far": far[0],
        "masked": masked_zone[0],
        "diff": (far_positive - gt)[0],
        "valid": valid_binary[0],
    }
    return record, tensors


def main() -> int:
    args = _parser().parse_args()
    option_path = Path(args.opt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    opt = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    if not isinstance(opt, dict) or not isinstance(opt.get("datasets", {}).get("train"), dict):
        raise ValueError("training YAML must contain datasets.train")
    train_opt = opt["datasets"]["train"]
    dataset_opt = copy.deepcopy(train_opt)
    dataset_opt.update({"phase": "train", "batch_size_per_gpu": int(args.batch_size), "num_worker_per_gpu": 0})
    dataset = build_dataset(dataset_opt)
    sample_count = min(int(args.max_samples), len(dataset))
    if sample_count <= 0:
        raise ValueError("training dataset is empty")
    samples = [dataset[index] for index in range(sample_count)]
    batch = dataset.collate_fn(samples)
    device = _device(args.device)
    bg_opt = copy.deepcopy(opt.get("train", {}).get("bg_lf_opt") or {})
    bg_opt.pop("type", None)
    loss = BackgroundLowFrequencyLoss(**bg_opt).to(device)

    # Actual V4 graph forward on the same training batch.  GT-degradation
    # adapters expose clean lq_rgb here; the formal trainer performs GPU
    # degradation before packing, while this probe only checks graph shape and
    # finiteness without consuming a checkpoint or mutating optimizer state.
    network_report: dict[str, Any]
    try:
        network = build_network(copy.deepcopy(opt["network_g"])).to(device).eval()
        lq_y4uv = rgb_to_y4uv(_tensor(batch["lq_rgb"], device))
        with torch.no_grad():
            output_y4uv = network(lq_y4uv)
            output_rgb = YUV2RGB(y4uv_to_yuv444(output_y4uv), format="RGB", is_train=True, uv_normalized=True)
        network_report = {
            "attempted": True,
            "input_shape": list(lq_y4uv.shape),
            "output_shape": list(output_y4uv.shape),
            "decoded_rgb_shape": list(output_rgb.shape),
            "finite": bool(torch.isfinite(output_y4uv).all() and torch.isfinite(output_rgb).all()),
        }
        del network, output_y4uv, output_rgb
    except Exception as error:  # retain mask/loss evidence even if optional graph probe fails
        network_report = {"attempted": True, "finite": False, "error": repr(error)}

    record, tensors = _run_batch(batch, loss, device, float(args.offset))
    config_bg = opt.get("train", {}).get("bg_lf_opt", {})
    payload = {
        "status": "pass",
        "script": "smoke_basic_text_line_y4uv_bglf.py",
        "config": str(option_path),
        "experiment_name": opt.get("name"),
        "model_type": opt.get("model_type"),
        "dataset_type": train_opt.get("type"),
        "dataset_sources": train_opt.get("sources"),
        "dataset_length": len(dataset),
        "sample_count": sample_count,
        "device": str(device),
        "bg_lf_opt": config_bg,
        "tests": record,
        "network_forward": network_report,
        "interpretation": {
            "masked_zone_near_invariance": "text/near-text perturbation uses text dilated by radius + 3 sigma; any residual is the documented blur halo",
            "far_gradient_support": "non-far gradient is expected to be a small Gaussian halo because blur precedes far-mask sampling",
            "padding": "padding perturbation is tested in an interior-only region when width permits; valid mask always excludes padded pixels",
            "polarity": "positive and negative far offsets are both tested; image polarity is reported per sample when text contrast is measurable",
        },
    }
    # Assertions are intentionally data-level and relative, not image-quality
    # claims.  The checks remain tolerant of datasets with no interior padding
    # or no verified structure row in the first small batch.
    if record["far_pixels"] <= 0.0:
        payload["status"] = "fail"
        payload["failure"] = "far-background support is empty"
    elif record["loss_far_positive"] <= record["loss_base"] + 1e-5 or record["loss_far_negative"] <= record["loss_base"] + 1e-5:
        payload["status"] = "fail"
        payload["failure"] = "positive/negative far perturbation did not increase bgLF"
    elif record["gradient_total"] <= 0.0 or record["gradient_far_fraction"] < 0.50:
        payload["status"] = "fail"
        payload["failure"] = "prediction gradient is absent or not primarily far-background"
    payload["checks"] = {
        "far_nonempty": record["far_pixels"] > 0.0,
        "masked_zone_near_invariant": record["text_to_far_positive_ratio"] < 0.35,
        "far_positive_material": record["loss_far_positive"] > record["loss_base"] + 1e-5,
        "far_negative_material": record["loss_far_negative"] > record["loss_base"] + 1e-5,
        "gradient_nonzero": record["gradient_total"] > 0.0,
        "gradient_primarily_far": record["gradient_far_fraction"] >= 0.50,
        "padding_checked": record["padding_coverage"] <= 0.0 or record["padding_interior_coverage"] >= 0.0,
        "network_forward_finite": network_report.get("finite", False),
    }
    json_path = output_dir / "bglf_smoke.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _panel(output_dir / "bglf_smoke_panel.png", **tensors)
    report_lines = [
        f"# BG-LF smoke report: {payload['experiment_name']}",
        "",
        f"Status: **{payload['status']}**",
        f"Config: `{option_path}`",
        f"Dataset: `{train_opt.get('type')}`, {sample_count} records sampled from {len(dataset)}",
        f"Device: `{device}`; network forward finite: `{network_report.get('finite', False)}`",
        "",
        "## Data-level checks",
        "",
        f"- Far support: {record['far_pixels']:.0f} pixels ({record['far_over_valid']:.3f} of valid); nonempty={payload['checks']['far_nonempty']}",
        f"- Text/near-text perturbation ratio: {record['text_to_far_positive_ratio']:.4f}; near-invariant={payload['checks']['masked_zone_near_invariant']}",
        f"- Far offset losses (+/-): {record['loss_far_positive']:.6g} / {record['loss_far_negative']:.6g}; material increase={payload['checks']['far_positive_material'] and payload['checks']['far_negative_material']}",
        f"- Gradient fraction in B_far: {record['gradient_far_fraction']:.3f}; non-far halo={record['gradient_nonfar_fraction']:.3f}",
        f"- Padding coverage/interior: {record['padding_coverage']:.4f} / {record['padding_interior_coverage']:.4f}",
        f"- Reported image polarity: {', '.join(record['polarity'])}",
        "",
        "The panel columns are GT, LQ, B_far, masked text/near-text zone, far-offset magnitude, and valid mask.",
        "The loss is training-only; this smoke does not alter generator weights or inference code.",
    ]
    (output_dir / "bglf_smoke_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(json_path), "report": str(output_dir / "bglf_smoke_report.md"), "panel": str(output_dir / "bglf_smoke_panel.png")}, indent=2))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
