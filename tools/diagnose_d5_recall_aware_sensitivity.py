"""D5: verify recall-aware terms distinguish missing strokes from gap filling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diagnose_v4_d4_structure_hf_sensitivity import make_variants, masks
from diagnose_v4_real_structure_masks import CANVAS_HW, read_rgb
from overfit_v4_real_structure_hf import prepare_triplet
from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    report = {}
    for stem in args.ids:
        _, gt_np, strokes, valid_width, _ = prepare_triplet(
            read_rgb(args.root / "lq" / args.split / f"{stem}.png"),
            read_rgb(args.root / "gt" / args.split / f"{stem}.png"),
            args.root / "structure" / args.split / f"{stem}.png",
        )
        inner, outer, _ = masks(strokes, valid_width)
        gt = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).to(device)
        valid = torch.zeros((1, 1, *CANVAS_HW), device=device)
        valid[..., :valid_width] = 1.0
        text = torch.from_numpy(strokes.astype("float32")).view(1, 1, *CANVAS_HW).to(device)
        variants = make_variants(gt, inner, outer, valid_width)
        scores = {}
        for name, candidate in variants.items():
            missing, false, _ = recall_aware_structure_losses(candidate, gt, text, valid)
            scores[name] = {"missing_stroke": float(missing), "false_stroke": float(false), "sum": float((missing + false))}
        report[stem] = {"valid_width": valid_width, "scores": scores}
    (args.out / "d5_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
