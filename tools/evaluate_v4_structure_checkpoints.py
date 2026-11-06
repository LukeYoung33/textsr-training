"""Evaluate saved V4 checkpoints on a mask-aligned real_finetune_v2 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diagnose_v4_real_structure_masks import load_model
from overfit_v4_real_structure_hf import build_batch, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="Repeat as name=path, for example cp_only=/tmp/net.pth")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    batch = build_batch(args.root, args.split, args.ids, device)
    report = {"ids": args.ids, "split": args.split, "crop_x_after_resize": batch["crop_xs"], "models": {}}
    for item in args.checkpoint:
        name, raw_path = str(item).split("=", 1)
        model = load_model(Path(raw_path), device)
        arm_out = args.out / name
        arm_out.mkdir(exist_ok=True)
        report["models"][name] = evaluate(model, batch, args.ids, arm_out, "final")
    (args.out / "evaluation_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
