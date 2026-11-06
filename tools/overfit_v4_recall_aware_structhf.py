"""Train one V4 train32 overfit arm with edge, missing-stroke and false-stroke terms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diagnose_v4_d4_structure_hf_sensitivity import rgb_from_y4uv, struct_hf_loss
from diagnose_v4_real_structure_masks import load_model
from overfit_v4_real_structure_hf import build_batch, evaluate, structure_loss_batch
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss
from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cp-weight", type=float, default=5e-4)
    parser.add_argument("--struct-weight", type=float, default=1.0)
    parser.add_argument("--missing-weight", type=float, default=1.0)
    parser.add_argument("--false-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    batch = build_batch(args.root, args.split, args.ids, device)
    # build_batch retains numpy masks for diagnostics; build full-resolution masks for the new RGB-domain loss.
    text_full = torch.from_numpy(__import__("numpy").stack(batch["strokes"]).astype("float32")).to(device).unsqueeze(1)
    valid_full = torch.zeros_like(text_full)
    for index, width in enumerate(batch["widths"]):
        valid_full[index, :, :, :width] = 1.0
    model = load_model(args.checkpoint, device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    native = Y4UVNativeMaskedL1LossV3().to(device)
    cp = CRNNTextLineTileContentLoss(
        recognizer_path=str(args.recognizer), alphabet_path=str(args.alphabet),
        layer_weights=(1.4, 1.4, 1.4, 0.4, 0.4), micro_batch_size=8,
    ).to(device)
    metadata = {"valid_hw": [[96, width] for width in batch["widths"]], "orientation": ["HORIZONTAL"] * len(args.ids), "cp_eligible": [True] * len(args.ids)}
    history = []
    for step in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["lq"])
        pred_rgb = rgb_from_y4uv(output)
        l_native = native(output, batch["gt"], valid_mask=batch["valid"], text_mask=batch["text"])
        l_cp_raw = cp(pred_rgb, batch["gt_rgb"], metadata)
        l_struct = structure_loss_batch(pred_rgb, batch["gt_rgb"], batch["weights"])
        l_missing, l_false, _ = recall_aware_structure_losses(pred_rgb, batch["gt_rgb"], text_full, valid_full)
        total = l_native + args.cp_weight * l_cp_raw + args.struct_weight * l_struct + args.missing_weight * l_missing + args.false_weight * l_false
        total.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.iterations:
            item = {"iter": step, "native": float(l_native.detach()), "cp_weighted": float((args.cp_weight * l_cp_raw).detach()), "struct_weighted": float((args.struct_weight * l_struct).detach()), "missing_weighted": float((args.missing_weight * l_missing).detach()), "false_weighted": float((args.false_weight * l_false).detach()), "total": float(total.detach())}
            history.append(item)
            print(json.dumps(item), flush=True)
    torch.save({"params": model.state_dict()}, args.out / "net_g_final.pth")
    report = {"setup": {"ids": args.ids, "iterations": args.iterations, "cp_weight": args.cp_weight, "struct_weight": args.struct_weight, "missing_weight": args.missing_weight, "false_weight": args.false_weight}, "history": history, "final_regions": evaluate(model, batch, args.ids, args.out, "final")}
    (args.out / "overfit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
