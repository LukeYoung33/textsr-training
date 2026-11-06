"""D4: compare loss sensitivity to real mask-defined text-structure failures.

No network parameters are updated.  Controlled corruptions are applied to GT
tiles so each loss can be tested against isolated missing-stroke, gap-filling,
boundary and global-tone errors under the exact V4 canvas geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from diagnose_v4_real_structure_masks import CANVAS_HW, prepare_mask, prepare_rgb, read_rgb
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444, yuv420_mask_to_y4uv
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss
from realesrgan.utils import YUV2RGB


def masks(strokes: np.ndarray, valid_width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    inner = cv2.erode(strokes, kernel, iterations=1).astype(bool)
    boundary = strokes.astype(bool) & ~inner
    outer = cv2.dilate(strokes, kernel, iterations=3).astype(bool) & ~strokes.astype(bool)
    valid = np.zeros(CANVAS_HW, dtype=bool)
    valid[:, :valid_width] = True
    weight = np.full(CANVAS_HW, 0.25, dtype=np.float32)
    weight[strokes.astype(bool)] = 1.0
    weight[boundary] = 5.0
    weight[outer] = 6.0
    weight[~valid] = 0.0
    return (
        torch.from_numpy(inner & valid),
        torch.from_numpy(outer & valid),
        torch.from_numpy(weight),
    )


def struct_hf_loss(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mask-weighted directional Scharr error on luminance, normalized by weight mass."""
    pred_y = 0.299 * pred_rgb[:, :1] + 0.587 * pred_rgb[:, 1:2] + 0.114 * pred_rgb[:, 2:3]
    gt_y = 0.299 * gt_rgb[:, :1] + 0.587 * gt_rgb[:, 1:2] + 0.114 * gt_rgb[:, 2:3]
    kx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], device=pred_rgb.device).view(1, 1, 3, 3) / 16.0
    ky = kx.transpose(-1, -2)
    wx = weight.to(pred_rgb.device).view(1, 1, *CANVAS_HW)
    diff_x = F.conv2d(pred_y, kx, padding=1) - F.conv2d(gt_y, kx, padding=1)
    diff_y = F.conv2d(pred_y, ky, padding=1) - F.conv2d(gt_y, ky, padding=1)
    charbonnier = torch.sqrt(diff_x.square() + diff_y.square() + 1e-6)
    return (charbonnier * wx).sum() / wx.sum().clamp_min(1.0)


def rgb_from_y4uv(value: torch.Tensor) -> torch.Tensor:
    return YUV2RGB(y4uv_to_yuv444(value), format="RGB", is_train=True, uv_normalized=True)


def make_variants(gt: torch.Tensor, inner: torch.Tensor, outer: torch.Tensor, valid_width: int) -> dict[str, torch.Tensor]:
    valid = torch.zeros((1, 1, *CANVAS_HW), device=gt.device)
    valid[..., :valid_width] = 1.0
    inner = inner.to(gt.device).view(1, 1, *CANVAS_HW).float()
    outer = outer.to(gt.device).view(1, 1, *CANVAS_HW).float()
    # These signs mimic the observed failures: pale/missing strokes and dark filled gaps.
    return {
        "base_gt": gt,
        "erase_stroke_core": (gt * (1.0 - inner) + (gt + 0.42 * (1.0 - gt)) * inner).clamp(0, 1),
        "fill_near_stroke_gaps": (gt * (1.0 - outer) + (gt * 0.38) * outer).clamp(0, 1),
        "boundary_shift_blur": F.avg_pool2d(F.pad(gt, (1, 1, 1, 1), mode="reflect"), 3, 1),
        "global_darken": (gt - 0.10 * valid).clamp(0, 1),
    }


def write_sheet(path: Path, variants: dict[str, torch.Tensor]) -> None:
    rows = []
    for name, value in variants.items():
        image = value[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        panel = cv2.cvtColor((image * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(panel, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1, cv2.LINE_AA)
        rows.append(panel)
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    native = Y4UVNativeMaskedL1LossV3().to(device)
    cp = CRNNTextLineTileContentLoss(
        recognizer_path=str(args.recognizer), alphabet_path=str(args.alphabet),
        layer_weights=(1.4, 1.4, 1.4, 0.4, 0.4), micro_batch_size=1,
    ).to(device)
    report = {}
    for stem in args.ids:
        gt_np, valid_width = prepare_rgb(read_rgb(args.root / "gt" / args.split / f"{stem}.png"))
        strokes = prepare_mask(args.root / "structure" / args.split / f"{stem}.png", valid_width)
        inner, outer, weight = masks(strokes, valid_width)
        gt = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_native = rgb_to_y4uv(gt)
        valid = np.zeros(CANVAS_HW, dtype=np.float32)
        valid[:, :valid_width] = 1.0
        valid_native = yuv420_mask_to_y4uv(torch.from_numpy(valid).to(device).unsqueeze(0).unsqueeze(0))
        text_native = yuv420_mask_to_y4uv(torch.from_numpy(strokes.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0))
        variants = make_variants(gt, inner, outer, valid_width)
        scores = {}
        for name, candidate in variants.items():
            candidate_native = rgb_to_y4uv(candidate)
            metadata = {"valid_hw": [[96, valid_width]], "orientation": ["HORIZONTAL"], "cp_eligible": [True]}
            with torch.no_grad():
                scores[name] = {
                    "native_l1": float(native(candidate_native, gt_native, valid_mask=valid_native, text_mask=text_native).item()),
                    "cp": float(cp(candidate, gt, metadata).item()),
                    "struct_hf": float(struct_hf_loss(candidate, gt, weight).item()),
                }
        report[stem] = {"valid_width": valid_width, "scores": scores}
        write_sheet(args.out / f"{stem}_variants.png", variants)
    (args.out / "d4_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
