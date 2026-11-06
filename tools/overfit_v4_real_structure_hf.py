"""Paired three-sample V4 overfit: CP control versus CP plus structure-HF loss."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_v4_d4_structure_hf_sensitivity import masks, rgb_from_y4uv, struct_hf_loss
from diagnose_v4_real_structure_masks import (
    CANVAS_HW, load_model, metrics, read_rgb, region_masks, write_montage,
)
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, yuv420_mask_to_y4uv
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss


def prepare_triplet(lq: np.ndarray, gt: np.ndarray, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Apply identical V4 resize and a mask-energy-selected common crop to a real pair."""
    if lq.shape[:2] != gt.shape[:2]:
        raise RuntimeError(f"LQ/GT source geometry differs: {lq.shape[:2]} vs {gt.shape[:2]}")
    source_h, source_w = lq.shape[:2]
    resized_w = max(2, int(round(source_w * CANVAS_HW[0] / source_h)))
    resized_w += resized_w % 2
    lq = cv2.resize(lq, (resized_w, CANVAS_HW[0]), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt, (resized_w, CANVAS_HW[0]), interpolation=cv2.INTER_AREA)
    source_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if source_mask is None:
        raise FileNotFoundError(mask_path)
    if source_mask.shape != (source_h, source_w):
        raise RuntimeError(f"mask/source geometry differs: {source_mask.shape} vs {(source_h, source_w)}")
    mask = cv2.resize((source_mask > 127).astype(np.uint8), (resized_w, CANVAS_HW[0]), interpolation=cv2.INTER_NEAREST)
    if resized_w <= CANVAS_HW[1]:
        pad = CANVAS_HW[1] - resized_w
        return (
            np.pad(lq, ((0, 0), (0, pad), (0, 0)), mode="edge"),
            np.pad(gt, ((0, 0), (0, pad), (0, 0)), mode="edge"),
            np.pad(mask, ((0, 0), (0, pad))),
            resized_w,
            0,
        )
    # Select one even-aligned common window by stroke+edge energy, then apply
    # exactly that crop to LQ, GT, and mask. This is a data selection rule only.
    edge = cv2.Canny(mask * 255, 50, 150).astype(np.float32) / 255.0
    energy = mask.astype(np.float32) + edge
    scores = np.convolve(energy.sum(axis=0), np.ones(CANVAS_HW[1], dtype=np.float32), mode="valid")
    crop_x = int(np.argmax(scores))
    crop_x -= crop_x % 2
    crop_x = min(crop_x, resized_w - CANVAS_HW[1])
    crop_x -= crop_x % 2
    return lq[:, crop_x:crop_x + 480], gt[:, crop_x:crop_x + 480], mask[:, crop_x:crop_x + 480], 480, crop_x


def build_batch(root: Path, split: str, stems: list[str], device: torch.device):
    lqs, gts, strokes, widths, crop_xs, weights = [], [], [], [], [], []
    for stem in stems:
        lq, gt, mask, width, crop_x = prepare_triplet(
            read_rgb(root / "lq" / split / f"{stem}.png"),
            read_rgb(root / "gt" / split / f"{stem}.png"),
            root / "structure" / split / f"{stem}.png",
        )
        _, _, weight = masks(mask, width)
        lqs.append(torch.from_numpy(lq).permute(2, 0, 1))
        gts.append(torch.from_numpy(gt).permute(2, 0, 1))
        strokes.append(mask)
        widths.append(width)
        crop_xs.append(crop_x)
        weights.append(weight)
    lq_rgb = torch.stack(lqs).to(device)
    gt_rgb = torch.stack(gts).to(device)
    valid = torch.zeros((len(stems), 1, *CANVAS_HW), device=device)
    for index, width in enumerate(widths):
        valid[index, :, :, :width] = 1.0
    text = torch.from_numpy(np.stack(strokes).astype(np.float32)).to(device).unsqueeze(1)
    return {
        "lq_rgb": lq_rgb, "gt_rgb": gt_rgb, "lq": rgb_to_y4uv(lq_rgb), "gt": rgb_to_y4uv(gt_rgb),
        "valid": yuv420_mask_to_y4uv(valid), "text": yuv420_mask_to_y4uv(text),
        "strokes": strokes, "widths": widths, "crop_xs": crop_xs, "weights": torch.stack(weights).to(device),
    }


def structure_loss_batch(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.stack([struct_hf_loss(pred[i:i + 1], gt[i:i + 1], weights[i]) for i in range(pred.shape[0])]).mean()


def evaluate(model, batch, stems, out: Path, tag: str):
    model.eval()
    with torch.no_grad():
        # Match the production output boundary for metrics and image serialization.
        # Do not clamp inside the optimization loop, where it would change gradients.
        pred = rgb_from_y4uv(model(batch["lq"])).clamp(0.0, 1.0)
    result = {}
    for i, stem in enumerate(stems):
        lq = batch["lq_rgb"][i].permute(1, 2, 0).cpu().numpy()
        gt = batch["gt_rgb"][i].permute(1, 2, 0).cpu().numpy()
        image = pred[i].permute(1, 2, 0).cpu().numpy()
        regions = region_masks(batch["strokes"][i], batch["widths"][i])
        result[stem] = metrics(lq, image, gt, regions)
        write_montage(out / f"{tag}_{stem}_montage.png", lq, image, gt, batch["strokes"][i], batch["widths"][i])
    model.train()
    return result


def run_arm(name, initial_state, batch, stems, args, native, cp, device):
    model = load_model(args.checkpoint, device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    metadata = {"valid_hw": [[96, width] for width in batch["widths"]], "orientation": ["HORIZONTAL"] * len(stems), "cp_eligible": [True] * len(stems)}
    history = []
    for step in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["lq"])
        l_native = native(output, batch["gt"], valid_mask=batch["valid"], text_mask=batch["text"])
        pred_rgb = rgb_from_y4uv(output)
        l_cp_raw = cp(pred_rgb, batch["gt_rgb"], metadata)
        l_struct_raw = structure_loss_batch(pred_rgb, batch["gt_rgb"], batch["weights"])
        total = l_native + args.cp_weight * l_cp_raw
        if name == "cp_plus_struct_hf":
            total = total + args.struct_weight * l_struct_raw
        total.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.iterations:
            entry = {
                "iter": step, "native": float(l_native.detach()), "cp_raw": float(l_cp_raw.detach()),
                "cp_weighted": float((args.cp_weight * l_cp_raw).detach()), "struct_raw": float(l_struct_raw.detach()),
                "struct_weighted": float((args.struct_weight * l_struct_raw).detach()) if name == "cp_plus_struct_hf" else 0.0,
                "total": float(total.detach()),
            }
            history.append(entry)
            print(json.dumps({name: entry}), flush=True)
    arm_out = args.out / name
    arm_out.mkdir(parents=True, exist_ok=True)
    torch.save({"params": model.state_dict()}, arm_out / "net_g_final.pth")
    return {"history": history, "final_regions": evaluate(model, batch, stems, arm_out, "final")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cp-weight", type=float, default=5e-4)
    parser.add_argument("--struct-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(20260831)
    np.random.seed(20260831)
    batch = build_batch(args.root, args.split, args.ids, device)
    seed_model = load_model(args.checkpoint, device)
    initial_state = copy.deepcopy(seed_model.state_dict())
    del seed_model
    native = Y4UVNativeMaskedL1LossV3().to(device)
    cp = CRNNTextLineTileContentLoss(
        recognizer_path=str(args.recognizer), alphabet_path=str(args.alphabet),
        layer_weights=(1.4, 1.4, 1.4, 0.4, 0.4), micro_batch_size=3,
    ).to(device)
    report = {"setup": {"architecture": "BasicTextLineR4StudentY4UVV4", "haar_switch": True, "checkpoint": str(args.checkpoint), "ids": args.ids, "iterations": args.iterations, "lr": args.lr, "cp_weight": args.cp_weight, "struct_weight": args.struct_weight, "crop_x_after_resize": batch["crop_xs"], "crop_policy": "same mask-energy-selected 480px crop for LQ, GT, and structure mask when resized line is wider than canvas"}}
    report["cp_only"] = run_arm("cp_only", initial_state, batch, args.ids, args, native, cp, device)
    report["cp_plus_struct_hf"] = run_arm("cp_plus_struct_hf", initial_state, batch, args.ids, args, native, cp, device)
    (args.out / "overfit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
