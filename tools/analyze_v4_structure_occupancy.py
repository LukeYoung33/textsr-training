"""Diagnose whether a structure loss trades stroke recall for false-positive suppression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_v4_real_structure_masks import load_model
from diagnose_v4_d4_structure_hf_sensitivity import rgb_from_y4uv
from overfit_v4_real_structure_hf import build_batch


def ink_mask(rgb: np.ndarray, valid_width: int) -> np.ndarray:
    y = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    source = (y[:, :valid_width] * 255).astype(np.uint8)
    threshold, _ = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result = np.zeros_like(y, dtype=bool)
    result[:, :valid_width] = y[:, :valid_width] <= threshold / 255.0
    return result


def stats(pred: np.ndarray, gt: np.ndarray, valid_width: int) -> dict[str, float]:
    valid = np.zeros_like(gt, dtype=bool)
    valid[:, :valid_width] = True
    truth = gt.astype(bool) & valid
    observed = pred & valid
    tp = int((truth & observed).sum())
    fn = int((truth & ~observed).sum())
    fp = int((~truth & observed & valid).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"tp": tp, "fn": fn, "fp": fp, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def overlay(path: Path, rgb: np.ndarray, truth: np.ndarray, prediction: np.ndarray, width: int, zoom_first_char: bool) -> None:
    panel = (np.clip(rgb, 0, 1) * 255 + .5).astype(np.uint8)
    # true-positive white stays image; FN red, FP cyan.
    fn = truth & ~prediction
    fp = ~truth & prediction
    panel[fn] = (255, 0, 0)
    panel[fp] = (0, 255, 255)
    panel = panel[:, :width]
    if zoom_first_char:
        panel = panel[:, :min(90, width)]
        panel = cv2.resize(panel, (panel.shape[1] * 4, panel.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    batch = build_batch(args.root, args.split, args.ids, device)
    report = {"models": {}}
    for value in args.checkpoint:
        name, raw_path = str(value).split("=", 1)
        model = load_model(Path(raw_path), device).eval()
        with torch.no_grad():
            pred = rgb_from_y4uv(model(batch["lq"])).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        per_sample = {}
        for index, stem in enumerate(args.ids):
            observed = ink_mask(pred[index], batch["widths"][index])
            truth = batch["strokes"][index].astype(bool)
            per_sample[stem] = stats(observed, truth, batch["widths"][index])
            if stem == "002652":
                overlay(args.out / f"{name}_{stem}_overlay.png", pred[index], truth, observed, batch["widths"][index], False)
                overlay(args.out / f"{name}_{stem}_firstchar_zoom.png", pred[index], truth, observed, batch["widths"][index], True)
        report["models"][name] = per_sample
    for name, per_sample in report["models"].items():
        report["models"][name]["_mean"] = {key: float(np.mean([v[key] for v in per_sample.values()])) for key in ("precision", "recall", "f1")}
    (args.out / "occupancy_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
