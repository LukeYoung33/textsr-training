"""Measure native-L1 and CP gradient allocation in real stroke-mask regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from diagnose_v4_real_structure_masks import (
    CANVAS_HW, load_model, prepare_mask, prepare_rgb, read_rgb, region_masks, to_rgb,
)
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, yuv420_mask_to_y4uv
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss


def gradient_regions(gradient_y: torch.Tensor, regions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    value = gradient_y[0, 0].detach().cpu().numpy()
    total = max(float(value.sum()), 1e-12)
    global_mean = max(float(value.mean()), 1e-12)
    result = {}
    for name, region in regions.items():
        if not region.any():
            continue
        mean = float(value[region].mean())
        result[name] = {
            "pixels": int(region.sum()),
            "gradient_fraction": float(value[region].sum() / total),
            "mean_gradient": mean,
            "density_vs_global": float(mean / global_mean),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    native_loss = Y4UVNativeMaskedL1LossV3().to(device)
    cp_loss = CRNNTextLineTileContentLoss(
        recognizer_path=str(args.recognizer), alphabet_path=str(args.alphabet),
        layer_weights=(1.4, 1.4, 1.4, 0.4, 0.4), micro_batch_size=1,
    ).to(device)
    all_results = {}
    for stem in args.ids:
        lq, valid_width = prepare_rgb(read_rgb(args.root / "lq" / args.split / f"{stem}.png"))
        gt, gt_width = prepare_rgb(read_rgb(args.root / "gt" / args.split / f"{stem}.png"))
        if valid_width != gt_width:
            raise RuntimeError(f"LQ/GT geometry differs for {stem}")
        strokes = prepare_mask(args.root / "structure" / args.split / f"{stem}.png", valid_width)
        valid = np.zeros(CANVAS_HW, dtype=np.float32)
        valid[:, :valid_width] = 1.0
        regions = region_masks(strokes, valid_width)
        lq_native = rgb_to_y4uv(torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).to(device))
        gt_native = rgb_to_y4uv(torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device))
        text_native = yuv420_mask_to_y4uv(
            torch.from_numpy(strokes.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
        )
        valid_native = yuv420_mask_to_y4uv(torch.from_numpy(valid).to(device).unsqueeze(0).unsqueeze(0))
        outputs = {}
        for name in ("native_l1", "cp"):
            output = model(lq_native)
            output.retain_grad()
            if name == "native_l1":
                loss = native_loss(output, gt_native, valid_mask=valid_native, text_mask=text_native)
            else:
                predicted_rgb = to_rgb(output)
                predicted_rgb_t = torch.from_numpy(predicted_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                # Recreate a differentiable RGB decode; to_rgb is only for the non-grad image path.
                from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
                from realesrgan.utils import YUV2RGB
                predicted_rgb_t = YUV2RGB(y4uv_to_yuv444(output), format="RGB", is_train=True, uv_normalized=True)
                meta = {"valid_hw": [[96, valid_width]], "orientation": ["HORIZONTAL"], "cp_eligible": [True]}
                loss = cp_loss(predicted_rgb_t, torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device), meta)
            loss.backward()
            grad_y = F.pixel_shuffle(output.grad[:, :4].abs(), 2).sum(1, keepdim=True)
            outputs[name] = {"loss": float(loss.detach().item()), "regions": gradient_regions(grad_y, regions)}
            model.zero_grad(set_to_none=True)
        all_results[stem] = {"valid_width": valid_width, "losses": outputs}
    (args.out / "gradient_summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
