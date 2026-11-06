"""Frozen, no-update gradient audit for the R1-only recall-aware training terms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from basicsr.data import build_dataset
from basicsr.losses import build_loss
from basicsr.archs import build_network
from realesrgan.archs.basic_text_line_r4_student_v2_arch import load_strict_checkpoint
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444, yuv420_mask_to_y4uv
from realesrgan.lenovo_degradation import LenovoDegradationPipeline
from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses
from realesrgan.losses import basic_text_line_y4uv_v3_losses as _y4uv_native_loss_registration
from realesrgan.utils import YUV2RGB


def _args() -> argparse.Namespace:
    """@luna-comment Parse explicit frozen-checkpoint and append-only audit inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--c0-checkpoint", required=True, type=Path)
    parser.add_argument("--r1-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """@luna-comment Return a streamed SHA-256 digest for frozen checkpoint provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kernel_metrics(gradient: torch.Tensor) -> dict[str, float]:
    """@luna-comment Summarize 3x3 gradient magnitude by center, horizontal, vertical, and corner taps."""
    absolute = gradient.detach().abs()
    if absolute.ndim != 4 or absolute.shape[-2:] != (3, 3):
        return {}
    left_right = torch.cat((absolute[:, :, :, 0], absolute[:, :, :, 2]), dim=-1).mean()
    up_down = torch.cat((absolute[:, :, 0, :], absolute[:, :, 2, :]), dim=-1).mean()
    corners = torch.stack((absolute[:, :, 0, 0], absolute[:, :, 0, 2], absolute[:, :, 2, 0], absolute[:, :, 2, 2])).mean()
    center = absolute[:, :, 1, 1].mean()
    return {
        "tap_left_right_abs_mean": float(left_right),
        "tap_up_down_abs_mean": float(up_down),
        "tap_corner_abs_mean": float(corners),
        "tap_center_abs_mean": float(center),
        "left_right_to_up_down": float(left_right / up_down.clamp_min(1e-20)),
    }


def _grad_metrics(gradient: torch.Tensor | None, parameter: torch.Tensor) -> dict[str, float]:
    """@luna-comment Return normalized magnitude and tap metrics for one loss-to-parameter gradient."""
    if gradient is None:
        return {"grad_present": 0.0, "grad_l2": 0.0, "grad_abs_mean": 0.0, "grad_abs_max": 0.0, "grad_l2_over_param_l2": 0.0}
    result = {
        "grad_present": 1.0,
        "grad_l2": float(torch.linalg.vector_norm(gradient)),
        "grad_abs_mean": float(gradient.detach().abs().mean()),
        "grad_abs_max": float(gradient.detach().abs().max()),
        "grad_l2_over_param_l2": float(torch.linalg.vector_norm(gradient) / torch.linalg.vector_norm(parameter.detach()).clamp_min(1e-20)),
    }
    result.update(_kernel_metrics(gradient))
    return result


def _collect_batch(dataset: Any, indices: list[int]) -> dict[str, Any]:
    """@luna-comment Read only indexed real training rows and use the dataset's native collate contract."""
    if not indices:
        raise ValueError("at least one real training dataset index is required")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError(f"indices must be within [0, {len(dataset) - 1}]")
    rows = [dataset[index] for index in indices]
    return dataset.collate_fn(rows) if hasattr(dataset, "collate_fn") else {key: torch.stack([row[key] for row in rows]) for key in rows[0] if isinstance(rows[0][key], torch.Tensor)}


def _tracked_parameters(network: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """@luna-comment Select only R1-localized stage2[1] branches plus downstream controls for gradient reporting."""
    block = network.stage2[1]
    result = {
        "stage2_1_b3_weight": block.conv3x3.weight,
        "stage2_1_b3_bias": block.conv3x3.bias,
        "stage2_1_b13_pw_weight": block.conv1x1.weight,
        "stage2_1_b13_spatial_weight": block.conv1x1_3x3.weight,
        "stage2_1_b13_spatial_bias": block.conv1x1_3x3.bias,
        "stage2_1_edge_x_pw": block.edge_x_pw.weight,
        "stage2_1_edge_y_pw": block.edge_y_pw.weight,
        "stage2_1_edge_l_pw": block.edge_l_pw.weight,
        "stage2_1_prelu": block.act.weight,
        "dec2_weight": network.dec2.conv3x3.weight,
        "dec1_weight": network.dec1.conv3x3.weight,
        "context_terminal_weight": network.context_terminal.conv3x3.weight,
    }
    if any(value is None for value in result.values()):
        raise RuntimeError("expected unfused V5RepECB parameters are missing")
    return result  # type: ignore[return-value]


def _audit_one_network(network: torch.nn.Module, batch: dict[str, Any], pipeline: LenovoDegradationPipeline, seed: int, device: torch.device, train_opt: dict[str, Any], native_loss: torch.nn.Module) -> dict[str, Any]:
    """@luna-comment Run no-update per-loss autograd on one frozen checkpoint and one real masked training batch."""
    gt_rgb = batch["gt_rgb"].to(device).float().clamp(0.0, 1.0)
    valid_rgb = batch["valid_mask_rgb"].to(device).float()
    text_rgb = batch["text_mask_rgb"].to(device).float()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    with torch.no_grad():
        lq_rgb, degradation_metadata = pipeline(gt_rgb, generator)
    lq, gt = rgb_to_y4uv(lq_rgb), rgb_to_y4uv(gt_rgb)
    valid, text = yuv420_mask_to_y4uv(valid_rgb), yuv420_mask_to_y4uv(text_rgb)
    output = network(lq)
    native = native_loss(output, gt, valid_mask=valid, text_mask=text)
    output_rgb = YUV2RGB(y4uv_to_yuv444(output), format="RGB", is_train=True, uv_normalized=True)
    missing, false, details = recall_aware_structure_losses(output_rgb, gt_rgb, text_rgb, valid_rgb, **train_opt.get("recall_aware_opt", {}))
    missing_weight = float(train_opt.get("recall_aware_weight_missing", 0.0))
    false_weight = float(train_opt.get("recall_aware_weight_false", 0.0))
    losses = {
        "native": native,
        "missing_raw": missing,
        "false_weighted": false_weight * false,
        "r1_extra": missing_weight * missing + false_weight * false,
        "native_plus_r1_extra": native + missing_weight * missing + false_weight * false,
    }
    tracked = _tracked_parameters(network)
    parameters = tuple(tracked.values())
    result: dict[str, Any] = {
        "loss_values": {name: float(value.detach()) for name, value in losses.items()},
        "weights": {"missing": missing_weight, "false": false_weight},
        "recall_active_ratio": float(details["active"].float().mean()),
        "gradients": {},
        "degradation_metadata": degradation_metadata,
    }
    gradient_vectors: dict[str, dict[str, torch.Tensor | None]] = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        result["gradients"][name] = {key: _grad_metrics(gradient, tracked[key]) for key, gradient in zip(tracked, gradients)}
        gradient_vectors[name] = {key: gradient.detach() if gradient is not None else None for key, gradient in zip(tracked, gradients)}
    result["gradient_cosine"] = {}
    for left, right in (("native", "r1_extra"), ("missing_raw", "false_weighted")):
        key = f"{left}__vs__{right}"
        result["gradient_cosine"][key] = {}
        for parameter_name in tracked:
            first, second = gradient_vectors[left][parameter_name], gradient_vectors[right][parameter_name]
            if first is None or second is None:
                result["gradient_cosine"][key][parameter_name] = None
            else:
                result["gradient_cosine"][key][parameter_name] = float(
                    torch.dot(first.reshape(-1), second.reshape(-1)) /
                    (torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)).clamp_min(1e-20)
                )
    result["batch"] = {
        "line_id": [str(value) for value in batch.get("line_id", [])],
        "lq_path": [str(value) for value in batch.get("lq_path", [])],
        "gt_path": [str(value) for value in batch.get("gt_path", [])],
        "text_mask_coverage": [float(value) for value in text_rgb.mean((1, 2, 3)).detach().cpu()],
    }
    return result


def main() -> None:
    """@luna-comment Execute append-only C0/R1 no-update gradient audit using real training masks and synthetic LQ."""
    args = _args()
    for path in (args.config, args.c0_checkpoint, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("training YAML must be a mapping")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = build_dataset(document["datasets"]["train"])
    batch = _collect_batch(dataset, indices)
    pipeline = LenovoDegradationPipeline(document["lenovo_degradation"])
    native_loss = build_loss(document["train"]["y4uv_native_opt"]).to(device)
    root.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "COMPLETE",
        "contract": "real train rows and masks; synthetic Lenovo LQ; frozen checkpoints; torch.autograd.grad only; no optimizer, backward, or parameter update",
        "config": str(args.config), "indices": indices, "seed": int(args.seed), "models": {},
    }
    for name, checkpoint in {"c0": args.c0_checkpoint, "r1": args.r1_checkpoint}.items():
        network = build_network(document["network_g"]).to(device)
        digest = load_strict_checkpoint(network, checkpoint, map_location="cpu")
        network.eval()
        result = _audit_one_network(network, batch, pipeline, int(args.seed), device, document["train"], native_loss)
        report["models"][name] = {"checkpoint_sha256": digest, **result}
    (root / "r1_stage2_gradient_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
