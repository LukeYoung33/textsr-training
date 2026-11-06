"""D1/D2 diagnostic for the fixed BASIC V4 Y4UV+Haar+CP model.

This program never updates parameters.  It loads the production V4 structure and
checkpoint, runs the supplied 96x480 LQ/GT sample, then writes:
  * D1 controlled-output loss responses for native Y4UV L1 and the frozen CP;
  * D2 separate output-gradient maps and region statistics for those losses.

The image triplet is intentionally used only as an input/GT diagnostic fixture.
Its text mask is an Otsu mask from GT, therefore its spatial statistics are a
diagnostic proxy rather than a replacement for polygon-mask validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import BasicTextLineR4StudentY4UVV4
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444, yuv420_mask_to_y4uv
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss
from realesrgan.utils import YUV2RGB


def _read_rgb(path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def _write_rgb(path: Path, value: torch.Tensor) -> None:
    image = value.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(path), cv2.cvtColor((image * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))


def _normal_map(value: torch.Tensor) -> np.ndarray:
    image = value.detach().float().cpu().numpy()
    high = float(np.percentile(image, 99.0))
    if high <= 1e-12:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip(image / high * 255.0, 0, 255).astype(np.uint8)


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a parameter mapping")
    model.load_state_dict({str(k).removeprefix("module."): v for k, v in state.items()}, strict=True)


def _gt_text_mask(gt_rgb: torch.Tensor) -> torch.Tensor:
    gray = cv2.cvtColor((gt_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = (gray <= threshold).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(gt_rgb.device)


def _to_rgb(y4uv: torch.Tensor) -> torch.Tensor:
    return YUV2RGB(y4uv_to_yuv444(y4uv), format="RGB", is_train=True, uv_normalized=True)


def _native_from_rgb(rgb: torch.Tensor) -> torch.Tensor:
    return rgb_to_y4uv(rgb)


def _cp_loss(cp: CRNNTextLineTileContentLoss, y4uv: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
    metadata = {"valid_hw": [[96, 480]], "orientation": ["HORIZONTAL"], "cp_eligible": [True]}
    return cp(_to_rgb(y4uv), gt_rgb, metadata)


def _region_summary(gradient_y: torch.Tensor, gt_rgb: torch.Tensor, text_mask: torch.Tensor) -> dict[str, float]:
    gray = gt_rgb.mean(1, keepdim=True)
    low = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
    high_frequency = (gray - low).abs()
    hf_threshold = torch.quantile(high_frequency[text_mask.bool()], 0.75)
    hf = (high_frequency >= hf_threshold) & text_mask.bool()
    edge = F.max_pool2d(text_mask, 3, 1, 1) - F.max_pool2d(1.0 - text_mask, 3, 1, 1) < 0.5
    edge = edge & text_mask.bool()
    total = gradient_y.sum().clamp_min(1e-12)
    return {
        "text_fraction": float((gradient_y * text_mask).sum().item() / total.item()),
        "high_frequency_text_fraction": float(gradient_y[hf].sum().item() / total.item()),
        "text_edge_fraction": float(gradient_y[edge].sum().item() / total.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--recognizer", required=True, type=Path)
    parser.add_argument("--alphabet", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    lq_rgb = _read_rgb(args.sample_dir / "lq_source_rgb.png", device)
    gt_rgb = _read_rgb(args.sample_dir / "gt_source_rgb.png", device)
    if tuple(lq_rgb.shape[-2:]) != (96, 480) or tuple(gt_rgb.shape[-2:]) != (96, 480):
        raise ValueError("D1/D2 fixture must be 96x480 RGB")
    text_full = _gt_text_mask(gt_rgb)
    text_y4uv = yuv420_mask_to_y4uv(text_full)
    valid_y4uv = torch.ones_like(text_y4uv)

    model = BasicTextLineR4StudentY4UVV4().to(device).eval()
    _load_checkpoint(model, args.checkpoint)
    native = Y4UVNativeMaskedL1LossV3().to(device)
    cp = CRNNTextLineTileContentLoss(
        recognizer_path=str(args.recognizer), alphabet_path=str(args.alphabet),
        layer_weights=(1.4, 1.4, 1.4, 0.4, 0.4), micro_batch_size=1,
    ).to(device)

    with torch.no_grad():
        pred_y4uv = model(_native_from_rgb(lq_rgb))
    _write_rgb(args.out / "model_pred_rgb.png", _to_rgb(pred_y4uv))

    base_rgb = _to_rgb(pred_y4uv).detach()
    mask3 = text_full.expand_as(base_rgb)
    kernel = torch.ones((1, 1, 3, 3), device=device)
    mask_np = text_full[0, 0].cpu().numpy().astype(np.uint8)
    boundary = cv2.dilate(mask_np, np.ones((3, 3), np.uint8), 1) - cv2.erode(mask_np, np.ones((3, 3), np.uint8), 1)
    boundary = torch.from_numpy(boundary.astype(np.float32)).view(1, 1, 96, 480).to(device).expand_as(base_rgb)
    variants = {
        "base": base_rgb,
        "text_darken": (base_rgb - 0.10 * mask3).clamp(0, 1),
        "blur_sigma1": F.avg_pool2d(F.pad(base_rgb, (1, 1, 1, 1), mode="reflect"), 3, 1),
        "shift_right_1px": F.pad(base_rgb[..., :, :-1], (1, 0, 0, 0), mode="replicate"),
        "boundary_lighten": (base_rgb + 0.18 * boundary).clamp(0, 1),
    }
    d1: dict[str, dict[str, float]] = {}
    for name, rgb in variants.items():
        candidate = _native_from_rgb(rgb)
        with torch.no_grad():
            d1[name] = {
                "native_l1": float(native(candidate, _native_from_rgb(gt_rgb), valid_mask=valid_y4uv, text_mask=text_y4uv).item()),
                "cp": float(_cp_loss(cp, candidate, gt_rgb).item()),
            }
        _write_rgb(args.out / f"d1_{name}.png", rgb)

    d2: dict[str, dict[str, float]] = {}
    for loss_name in ("native_l1", "cp"):
        output = model(_native_from_rgb(lq_rgb))
        output.retain_grad()
        if loss_name == "native_l1":
            value = native(output, _native_from_rgb(gt_rgb), valid_mask=valid_y4uv, text_mask=text_y4uv)
        else:
            value = _cp_loss(cp, output, gt_rgb)
        value.backward()
        grad_y = F.pixel_shuffle(output.grad[:, :4].abs(), 2).sum(1, keepdim=True)
        heat = cv2.applyColorMap(_normal_map(grad_y[0, 0]), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(args.out / f"d2_{loss_name}_grad_y.png"), heat)
        d2[loss_name] = {"loss": float(value.detach().item()), **_region_summary(grad_y, gt_rgb, text_full)}
        model.zero_grad(set_to_none=True)

    result = {
        "model": "BasicTextLineR4StudentY4UVV4", "canvas_hw": [96, 480],
        "haar_switch": True, "checkpoint": str(args.checkpoint), "d1": d1, "d2": d2,
        "mask_note": "GT Otsu text mask proxy; verify again with the manifest polygon mask before changing training.",
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
