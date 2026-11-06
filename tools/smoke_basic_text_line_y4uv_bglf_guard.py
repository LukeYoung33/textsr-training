"""Smoke-test the strict R1 BG-LF guard on a real training batch.

The smoke uses the configured dataset and V4 model, then checks the exact
guarded loss support and one trainer optimization step.  It is deliberately a
plumbing test: it makes no image-quality claim and does not write a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from basicsr.archs import build_network
from basicsr.data import build_dataset
from basicsr.models import build_model
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.losses.background_low_frequency_loss import (
    BackgroundLowFrequencyGuardLoss,
    far_background_mask,
)
from realesrgan.utils import YUV2RGB

# Import registry scanners before building a dataset/network/model.
import realesrgan.archs  # noqa: F401,E402
import realesrgan.data  # noqa: F401,E402
import realesrgan.losses  # noqa: F401,E402
import realesrgan.models  # noqa: F401,E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", "--opt", required=True, help="R1 guard training YAML")
    parser.add_argument("--output-dir", required=True, help="stable local/remote smoke directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--offset", type=float, default=0.12)
    parser.add_argument("--skip-trainer-step", action="store_true")
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
    return (
        torch.nn.functional.max_pool2d(value, 2 * int(radius) + 1, 1, int(radius))
        if radius
        else value
    )


def _panel(path: Path, gt: torch.Tensor, lq: torch.Tensor, far: torch.Tensor, protected: torch.Tensor, diff: torch.Tensor, valid: torch.Tensor) -> None:
    """Write GT/LQ/support/protected/error/valid tiles for human inspection."""

    def rgb(value: torch.Tensor) -> np.ndarray:
        value = value.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return cv2.cvtColor(np.rint(value * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def mask(value: torch.Tensor) -> np.ndarray:
        value = value.detach().float().cpu().clamp(0.0, 1.0).squeeze().numpy()
        return cv2.cvtColor(np.rint(value * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def heat(value: torch.Tensor) -> np.ndarray:
        value = value.detach().float().cpu().abs().mean(dim=0).clamp(0.0, 1.0).numpy()
        return cv2.applyColorMap(np.rint(value * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)

    tiles = [rgb(gt), rgb(lq), mask(far), mask(protected), heat(diff), mask(valid)]
    height = max(tile.shape[0] for tile in tiles)
    tiles = [cv2.resize(tile, (tile.shape[1], height), interpolation=cv2.INTER_NEAREST) for tile in tiles]
    if not cv2.imwrite(str(path), np.concatenate(tiles, axis=1)):
        raise IOError(f"failed to write smoke panel: {path}")


def _remap_local_cp_path(opt: dict[str, Any]) -> None:
    """Use the repository-local CRNN file when a Linux training path is absent."""
    cp_opt = opt.get("train", {}).get("cp_opt")
    if not isinstance(cp_opt, dict):
        return
    path = Path(str(cp_opt.get("recognizer_path", "")))
    if path.is_file():
        return
    candidate = Path(__file__).resolve().parents[1] / "weights" / path.name
    if candidate.is_file():
        cp_opt["recognizer_path"] = str(candidate)


def _build_batch(opt: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    """Build one deterministic collated batch from the configured train root."""
    train_opt = copy.deepcopy(opt["datasets"]["train"])
    train_opt.update({"phase": "train", "batch_size_per_gpu": int(args.batch_size), "num_worker_per_gpu": 0})
    dataset = build_dataset(train_opt)
    sample_count = min(int(args.max_samples), len(dataset))
    if sample_count <= 0:
        raise ValueError("training dataset is empty")
    samples = [dataset[index] for index in range(sample_count)]
    return dataset.collate_fn(samples), dataset


def _guard_probe(batch: dict[str, Any], loss: BackgroundLowFrequencyGuardLoss, device: torch.device, offset: float) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Measure guarded support, perturbation polarity, and derivative support."""
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
    protected = _dilate(text_binary, loss.radius) * valid_binary
    padding = (1.0 - valid_binary).clamp_min(0.0)

    base, details = loss(gt, gt, text_mask_rgb=text, valid_mask_rgb=valid, return_details=True)
    positive = loss(gt + float(offset) * far, gt, text_mask_rgb=text, valid_mask_rgb=valid)
    negative = loss(gt - float(offset) * far, gt, text_mask_rgb=text, valid_mask_rgb=valid)

    prediction = (gt + float(offset) * far).detach().requires_grad_(True)
    grad_value = loss(prediction, gt, text_mask_rgb=text, valid_mask_rgb=valid)
    grad_value.backward()
    gradient = prediction.grad.detach().abs().mean(dim=1, keepdim=True)
    gradient_total = gradient.sum().clamp_min(1e-12)
    far_gradient = (gradient * far).sum()
    protected_gradient = (gradient * protected).sum()
    padding_values = gradient[padding > 0.0]
    padding_max = padding_values.max() if padding_values.numel() else gradient.new_zeros(())
    record = {
        "batch_size": int(gt.shape[0]),
        "resolution": [int(gt.shape[-2]), int(gt.shape[-1])],
        "far_pixels": _safe_float(far.sum()),
        "valid_pixels": _safe_float(valid_binary.sum()),
        "text_pixels": _safe_float(text_binary.sum()),
        "far_coverage": _safe_float(details["far_coverage"]),
        "far_over_valid": _safe_float(details["far_over_valid"]),
        "valid_coverage": _safe_float(details["valid_coverage"]),
        "text_coverage": _safe_float(details["text_coverage"]),
        "padding_coverage": _safe_float(padding.mean()),
        "loss_base": _safe_float(base),
        "loss_far_positive": _safe_float(positive),
        "loss_far_negative": _safe_float(negative),
        "far_positive_delta": _safe_float(positive - base),
        "far_negative_delta": _safe_float(negative - base),
        "gradient_total": _safe_float(gradient_total),
        "gradient_far_fraction": _safe_float(far_gradient / gradient_total),
        "gradient_text_near_fraction": _safe_float(protected_gradient / gradient_total),
        "gradient_text_near_max": _safe_float(gradient[protected > 0.0].max() if (protected > 0.0).any() else gradient.new_zeros(())),
        "gradient_padding_max": _safe_float(padding_max),
        "gradient_padding_sum": _safe_float((gradient * padding).sum()),
        "guard_normalizer_min": _safe_float(details["guard_normalizer_min"]),
        "guard_normalizer_mean": _safe_float(details["guard_normalizer_mean"]),
    }
    tensors = {
        "gt": gt[0],
        "lq": lq[0],
        "far": far[0],
        "protected": protected[0],
        "diff": (float(offset) * far)[0],
        "valid": valid_binary[0],
    }
    return record, tensors


def _trainer_probe(opt: dict[str, Any], batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Run exactly one V4 trainer optimization step with all R1 auxiliaries on."""
    trainer_opt = copy.deepcopy(opt)
    trainer_opt.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1})
    trainer_opt["num_gpu"] = 0 if device.type != "cuda" else 1
    trainer_opt["root_path"] = str(Path(__file__).resolve().parents[1])
    trainer_opt["path"] = copy.deepcopy(trainer_opt.get("path", {}))
    trainer_opt["path"].setdefault("models", str(Path(trainer_opt["root_path"]) / ".tmp_guard_smoke_models"))
    trainer_opt["path"].setdefault("training_states", str(Path(trainer_opt["root_path"]) / ".tmp_guard_smoke_states"))
    trainer_opt["path"].setdefault("log", str(Path(trainer_opt["root_path"]) / ".tmp_guard_smoke_logs"))
    _remap_local_cp_path(trainer_opt)
    # Avoid Real-ESRGAN degradation in the one-step probe; use the already
    # collated paired RGB tensors while retaining the exact trainer class and
    # all R1 native/CP/recall/guard terms.
    step_batch = {key: value for key, value in batch.items() if key not in {"kernel1", "kernel2", "sinc_kernel"}}
    model = build_model(trainer_opt)
    model.feed_data(step_batch)
    model.optimize_parameters(current_iter=7000)
    logs = {}
    for key, value in model.get_current_log().items():
        logs[key] = _safe_float(value)
    required = ("l_native", "l_cp_raw", "l_cp", "l_missing", "l_false", "l_bg_lf_raw", "l_bg_lf_weighted", "bg_lf_far_coverage", "bg_lf_guard_active", "l_total")
    missing = [key for key in required if key not in logs]
    if missing:
        raise RuntimeError(f"trainer guard step missing logs: {missing}")
    finite = all(math.isfinite(value) for value in logs.values())
    output = model.output.detach()
    output_rgb = YUV2RGB(y4uv_to_yuv444(output), format="RGB", is_train=True, uv_normalized=True)
    return {
        "attempted": True,
        "finite": bool(finite and torch.isfinite(output).all() and torch.isfinite(output_rgb).all()),
        "current_iter": 7000,
        "logs": logs,
        "output_shape": list(output.shape),
        "decoded_rgb_shape": list(output_rgb.shape),
        "missing_logs": missing,
    }


def main() -> int:
    args = _parser().parse_args()
    option_path = Path(args.opt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    opt = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    if not isinstance(opt, dict) or not isinstance(opt.get("datasets", {}).get("train"), dict):
        raise ValueError("training YAML must contain datasets.train")
    train_opt = opt["datasets"]["train"]
    expected_root = "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2"
    sources = train_opt.get("sources", [])
    if sources != [{"root": expected_root, "weight": 1.0}]:
        raise ValueError(f"R1 guard smoke requires exactly one V2 source, got {sources!r}")
    bg_opt = copy.deepcopy(opt.get("train", {}).get("bg_lf_opt") or {})
    if bg_opt.pop("type", None) != "BackgroundLowFrequencyGuardLoss":
        raise ValueError("smoke config must select BackgroundLowFrequencyGuardLoss")
    device = _device(args.device)
    loss = BackgroundLowFrequencyGuardLoss(**bg_opt).to(device)
    batch, dataset = _build_batch(opt, args)
    record, tensors = _guard_probe(batch, loss, device, float(args.offset))

    network = None
    trainer_report = {"attempted": False, "skipped": bool(args.skip_trainer_step)}
    if not args.skip_trainer_step:
        trainer_report = _trainer_probe(opt, batch, device)
        # The trainer's output is the authoritative one-step network forward.
        network = {"output_shape": trainer_report["output_shape"], "decoded_rgb_shape": trainer_report["decoded_rgb_shape"], "finite": trainer_report["finite"]}
    else:
        probe_network = build_network(copy.deepcopy(opt["network_g"])).to(device).eval()
        lq_y4uv = rgb_to_y4uv(_tensor(batch["lq_rgb"], device))
        with torch.no_grad():
            output_y4uv = probe_network(lq_y4uv)
            output_rgb = YUV2RGB(y4uv_to_yuv444(output_y4uv), format="RGB", is_train=True, uv_normalized=True)
        network = {"output_shape": list(output_y4uv.shape), "decoded_rgb_shape": list(output_rgb.shape), "finite": bool(torch.isfinite(output_y4uv).all() and torch.isfinite(output_rgb).all())}

    checks = {
        "v2_only_source": sources == [{"root": expected_root, "weight": 1.0}],
        "far_nonempty": record["far_pixels"] > 0.0,
        "text_near_gradient_share_lt_1e4": record["gradient_text_near_fraction"] < 1.0e-4,
        "padding_gradient_zero": record["gradient_padding_max"] == 0.0 and record["gradient_padding_sum"] == 0.0,
        "far_positive_material": record["far_positive_delta"] > 1.0e-5,
        "far_negative_material": record["far_negative_delta"] > 1.0e-5,
        "guard_normalizer_finite": math.isfinite(record["guard_normalizer_min"]) and math.isfinite(record["guard_normalizer_mean"]),
        "network_forward_finite": bool(network["finite"]),
        "trainer_step_finite": bool(trainer_report.get("finite", False)) if not args.skip_trainer_step else True,
        "trainer_guard_log": bool(trainer_report.get("logs", {}).get("bg_lf_guard_active", 0.0) == 1.0) if not args.skip_trainer_step else True,
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "script": "smoke_basic_text_line_y4uv_bglf_guard.py",
        "config": str(option_path),
        "experiment_name": opt.get("name"),
        "dataset_type": train_opt.get("type"),
        "dataset_sources": sources,
        "dataset_length": len(dataset),
        "sample_count": int(record["batch_size"]),
        "device": str(device),
        "bg_lf_opt": opt.get("train", {}).get("bg_lf_opt", {}),
        "tests": record,
        "network_forward": network,
        "trainer_step": trainer_report,
        "checks": checks,
        "interpretation": {
            "guard_support": "B = V * (1 - dilate(M, 7)); masks are GT-derived and detached",
            "guard_formula": "residual = Gaussian(B * (Ypred - Ygt), sigma=4) / (Gaussian(B) + eps); L = mean_B(abs(residual))",
            "gradient_firewall": "prediction error is masked before convolution and final residual is masked by B; smooth kernel cannot enter text/near-text/padding",
            "metric_tier": "plumbing smoke only; no image-quality result",
        },
    }
    json_path = output_dir / "bglf_guard_smoke.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _panel(output_dir / "bglf_guard_smoke_panel.png", **tensors)
    report = [
        f"# BG-LF guard smoke report: {payload['experiment_name']}",
        "",
        f"Status: **{payload['status']}**",
        f"Config: `{option_path}`",
        f"Dataset: `{train_opt.get('type')}`, {record['batch_size']} records sampled from {len(dataset)}",
        f"Device: `{device}`; trainer step finite: `{trainer_report.get('finite', False)}`",
        "",
        "## Guard checks",
        "",
        f"- B support: {record['far_pixels']:.0f} pixels ({record['far_over_valid']:.4f} of valid); nonempty={checks['far_nonempty']}",
        f"- Text/near-text gradient share: {record['gradient_text_near_fraction']:.3e}; pass={checks['text_near_gradient_share_lt_1e4']}",
        f"- Padding gradient max/sum: {record['gradient_padding_max']:.3e} / {record['gradient_padding_sum']:.3e}; pass={checks['padding_gradient_zero']}",
        f"- Far offset deltas (+/-): {record['far_positive_delta']:.6g} / {record['far_negative_delta']:.6g}; material={checks['far_positive_material'] and checks['far_negative_material']}",
        f"- Trainer raw/weighted/coverage: {trainer_report.get('logs', {}).get('l_bg_lf_raw', float('nan')):.6g} / {trainer_report.get('logs', {}).get('l_bg_lf_weighted', float('nan')):.6g} / {trainer_report.get('logs', {}).get('bg_lf_far_coverage', float('nan')):.6g}",
        f"- Trainer guard active: {trainer_report.get('logs', {}).get('bg_lf_guard_active', float('nan'))}; all finite={checks['trainer_step_finite']}",
        "",
        "The panel columns are GT, LQ, B, dilated text/near-text protection, far-offset magnitude, and valid mask.",
        "This is a plumbing smoke only; it does not establish visual quality.",
    ]
    (output_dir / "bglf_guard_smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(json_path), "report": str(output_dir / "bglf_guard_smoke_report.md"), "panel": str(output_dir / "bglf_guard_smoke_panel.png")}, indent=2))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
