"""No-update virtual-step probe for output-space directionality of R1 stage2[1] gradients."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from basicsr.archs import build_network
from basicsr.losses import build_loss
from realesrgan.archs.basic_text_line_r4_student_v2_arch import load_strict_checkpoint
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444, yuv420_mask_to_y4uv
from realesrgan.losses import basic_text_line_y4uv_v3_losses as _y4uv_native_loss_registration
from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses
from realesrgan.lenovo_degradation import LenovoDegradationPipeline
from realesrgan.utils import YUV2RGB
from tools.audit_lenovo_v5_r1_stage2_gradients import _collect_batch
from basicsr.data import build_dataset


def _args() -> argparse.Namespace:
    """@luna-comment Parse a strictly append-only frozen-R1 virtual-step direction probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--r1-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--relative-step", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _direction_scores(value: torch.Tensor, rgb: bool) -> dict[str, float]:
    """@luna-comment Measure horizontal-line, vertical-line, and diagonal Fourier-sector energy of one output-space delta."""
    array = value.detach().float().cpu().numpy()
    if rgb:
        array = 0.299 * array[:, 0] + 0.587 * array[:, 1] + 0.114 * array[:, 2]
    else:
        array = array.mean(axis=1)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(array, axes=(-2, -1)), axes=(-2, -1))) ** 2
    height, width = spectrum.shape[-2:]
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(height)), np.fft.fftshift(np.fft.fftfreq(width)), indexing="ij")
    non_dc = (np.abs(fx) + np.abs(fy)) > 0
    horizontal = (np.abs(fy) > 2.0 * np.abs(fx)) & non_dc
    vertical = (np.abs(fx) > 2.0 * np.abs(fy)) & non_dc
    diagonal = non_dc & ~(horizontal | vertical)
    total = spectrum[..., non_dc].mean()
    return {
        "horizontal_line_sector_ratio": float(spectrum[..., horizontal].mean() / max(total, 1e-20)),
        "vertical_line_sector_ratio": float(spectrum[..., vertical].mean() / max(total, 1e-20)),
        "diagonal_sector_ratio": float(spectrum[..., diagonal].mean() / max(total, 1e-20)),
        "delta_rms": float(np.sqrt(np.mean(array ** 2))),
    }


def _forward_debug(network: torch.nn.Module, lq: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """@luna-comment Capture the actual stage2[1] output alongside standard debug tensors without altering forward code."""
    captured: list[torch.Tensor] = []
    hook = network.stage2[1].register_forward_hook(lambda _m, _i, output: captured.append(output.detach()))
    try:
        debug = network(lq, return_debug=True)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one stage2[1] hook output, got {len(captured)}")
    return debug, captured[0]


def _selection_mask(name: str, parameter: torch.Tensor, group: str) -> torch.Tensor | None:
    """@luna-comment Return a parameter-space mask for one explicitly named stage2[1] virtual-update group."""
    if group == "all_stage2_1":
        return torch.ones_like(parameter) if name.startswith("stage2.1.") else None
    if group == "b13_pw" and name == "stage2.1.conv1x1.weight":
        return torch.ones_like(parameter)
    if group.startswith("b3_") and name == "stage2.1.conv3x3.weight":
        mask = torch.zeros_like(parameter)
    elif group.startswith("b13_spatial_") and name == "stage2.1.conv1x1_3x3.weight":
        mask = torch.zeros_like(parameter)
    else:
        return None
    if group.endswith("ud"):
        mask[:, :, 0, :] = 1.0; mask[:, :, 2, :] = 1.0
    elif group.endswith("lr"):
        mask[:, :, :, 0] = 1.0; mask[:, :, :, 2] = 1.0
    else:
        raise ValueError(f"unknown virtual update group: {group}")
    return mask


def _losses(network: torch.nn.Module, lq: torch.Tensor, gt: torch.Tensor, gt_rgb: torch.Tensor, valid: torch.Tensor, text: torch.Tensor, valid_rgb: torch.Tensor, text_rgb: torch.Tensor, native_loss: torch.nn.Module, train_opt: dict[str, Any]) -> dict[str, torch.Tensor]:
    """@luna-comment Recreate only native and R1-extra losses needed for a no-update local-direction experiment."""
    output = network(lq)
    native = native_loss(output, gt, valid_mask=valid, text_mask=text)
    rgb = YUV2RGB(y4uv_to_yuv444(output), format="RGB", is_train=True, uv_normalized=True)
    missing, false, _ = recall_aware_structure_losses(rgb, gt_rgb, text_rgb, valid_rgb, **train_opt.get("recall_aware_opt", {}))
    return {"native": native, "r1_extra": float(train_opt["recall_aware_weight_missing"]) * missing + float(train_opt["recall_aware_weight_false"]) * false}


def main() -> None:
    """@luna-comment Run virtual SGD steps only in memory and report direct output-space orientation changes."""
    args = _args()
    for path in (args.config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    indices = [int(item) for item in args.indices.split(",") if item.strip()]
    dataset = build_dataset(document["datasets"]["train"])
    batch = _collect_batch(dataset, indices)
    gt_rgb = batch["gt_rgb"].to(device).float().clamp(0.0, 1.0)
    valid_rgb, text_rgb = batch["valid_mask_rgb"].to(device).float(), batch["text_mask_rgb"].to(device).float()
    generator = torch.Generator(device=device); generator.manual_seed(args.seed)
    with torch.no_grad():
        lq_rgb, metadata = LenovoDegradationPipeline(document["lenovo_degradation"])(gt_rgb, generator)
    lq, gt = rgb_to_y4uv(lq_rgb), rgb_to_y4uv(gt_rgb)
    valid, text = yuv420_mask_to_y4uv(valid_rgb), yuv420_mask_to_y4uv(text_rgb)
    network = build_network(document["network_g"]).to(device)
    digest = load_strict_checkpoint(network, args.r1_checkpoint, map_location="cpu")
    network.eval()
    native_loss = build_loss(document["train"]["y4uv_native_opt"]).to(device)
    baseline_debug, baseline_z = _forward_debug(network, lq)
    named = dict(network.named_parameters())
    groups = ("all_stage2_1", "b3_ud", "b3_lr", "b13_pw", "b13_spatial_ud", "b13_spatial_lr")
    report: dict[str, Any] = {"status": "COMPLETE", "contract": "temporary in-memory SGD direction only; no optimizer, checkpoint write, or persistent parameter mutation", "checkpoint_sha256": digest, "indices": indices, "seed": args.seed, "relative_step": args.relative_step, "degradation_metadata": metadata, "runs": {}}
    for loss_name in ("native", "r1_extra"):
        losses = _losses(network, lq, gt, gt_rgb, valid, text, valid_rgb, text_rgb, native_loss, document["train"])
        for group in groups:
            selected = [(name, parameter, _selection_mask(name, parameter, group)) for name, parameter in named.items()]
            selected = [(name, parameter, mask) for name, parameter, mask in selected if mask is not None]
            parameters = [parameter for _, parameter, _ in selected]
            gradients = torch.autograd.grad(losses[loss_name], parameters, retain_graph=True)
            clone = copy.deepcopy(network)
            clone_named = dict(clone.named_parameters())
            with torch.no_grad():
                for (name, parameter, mask), gradient in zip(selected, gradients):
                    scale = float(args.relative_step) * torch.linalg.vector_norm(parameter).item() / max(torch.linalg.vector_norm(gradient * mask).item(), 1e-20)
                    clone_named[name].add_(-scale * gradient * mask)
            updated_debug, updated_z = _forward_debug(clone, lq)
            output_delta = YUV2RGB(y4uv_to_yuv444(updated_debug["output"]), format="RGB", is_train=True, uv_normalized=True) - YUV2RGB(y4uv_to_yuv444(baseline_debug["output"]), format="RGB", is_train=True, uv_normalized=True)
            report["runs"][f"{loss_name}__{group}"] = {
                "loss_value": float(losses[loss_name].detach()),
                "stage2_delta": _direction_scores(updated_z - baseline_z, rgb=False),
                "context_delta": _direction_scores(updated_debug["context_residual"] - baseline_debug["context_residual"], rgb=False),
                "output_delta": _direction_scores(output_delta, rgb=True),
            }
    root.mkdir(parents=True)
    (root / "r1_virtual_direction_probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
