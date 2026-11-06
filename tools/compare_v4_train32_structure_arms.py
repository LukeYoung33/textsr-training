"""Produce a common, post-hoc train32 report for all V4 structure-loss arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from analyze_v4_structure_occupancy import ink_mask, stats
from diagnose_v4_d4_structure_hf_sensitivity import rgb_from_y4uv
from diagnose_v4_real_structure_masks import load_model, region_masks
from overfit_v4_real_structure_hf import build_batch
from overfit_v4_explicit_structure import ExplicitStructureV4


def load_params(module: nn.Module, path: Path) -> None:
    raw = torch.load(path, map_location="cpu")
    module.load_state_dict(raw.get("params", raw), strict=True)


def scharr_luma(image: np.ndarray) -> np.ndarray:
    y = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    dx = cv2.Scharr(y, cv2.CV_32F, 1, 0) / 16.0
    dy = cv2.Scharr(y, cv2.CV_32F, 0, 1) / 16.0
    return np.sqrt(dx * dx + dy * dy)


def quantiles(value: np.ndarray) -> dict[str, float]:
    if value.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    return {key: float(np.quantile(value, q)) for key, q in (("p50", .5), ("p90", .9), ("p99", .99))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--v4-init", type=Path, required=True); parser.add_argument("--cp-only", type=Path, required=True)
    parser.add_argument("--struct-hf", type=Path, required=True); parser.add_argument("--recall-aware", type=Path, required=True)
    parser.add_argument("--tversky", type=Path, required=True); parser.add_argument("--textgestalt", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ragan-vgg", type=Path, default=None, help="optional native+CP+VGG+RaGAN generator checkpoint")
    parser.add_argument("--split", default="train"); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    batch = build_batch(args.root, args.split, args.ids, device)
    base_state = load_model(args.v4_init, device).state_dict()

    models: dict[str, nn.Module | None] = {"lq_input": None}
    for name, path in (("v4_cp_init", args.v4_init), ("cp_only", args.cp_only), ("struct_hf", args.struct_hf), ("recall_aware", args.recall_aware)):
        model = load_model(path, device).eval(); models[name] = model
    if args.ragan_vgg is not None:
        model = load_model(args.ragan_vgg, device).eval(); models["cp_vgg_ragan"] = model
    for name, path, mode in (("tversky", args.tversky, "concat"), ("textgestalt_attention", args.textgestalt, "attention"), ("oracle_mask_injected", args.oracle, "concat")):
        base = load_model(args.v4_init, device); base.load_state_dict(base_state, strict=True)
        model = ExplicitStructureV4(base, mode).to(device).eval(); load_params(model, path); models[name] = model

    predictions: dict[str, np.ndarray] = {"lq_input": batch["lq_rgb"].permute(0, 2, 3, 1).cpu().numpy()}
    text = torch.from_numpy(np.stack(batch["strokes"]).astype("float32")).to(device).unsqueeze(1)
    with torch.no_grad():
        for name, model in models.items():
            if model is None:
                continue
            if name == "oracle_mask_injected":
                output, _ = model(batch["lq"], text)
            elif name in {"tversky", "textgestalt_attention"}:
                output, _ = model(batch["lq"])
            else:
                output = model(batch["lq"])
            predictions[name] = rgb_from_y4uv(output).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    report: dict[str, object] = {"scope": {"split": args.split, "ids": args.ids, "note": "All V4 arms use the identical train32 resize/crop and valid region. LQ is the input baseline; v4_cp_init is the supplied CP-trained V4 checkpoint before train32 overfit."}, "models": {}}
    gt_all = batch["gt_rgb"].permute(0, 2, 3, 1).cpu().numpy()
    for name, values in predictions.items():
        region_values: dict[str, list[float]] = {key: [] for key in ("all_valid", "stroke_interior", "stroke_boundary", "near_stroke_gap", "far_background")}
        gradient_values: dict[str, list[float]] = {key: [] for key in region_values}
        pred_boundary_grad, gt_boundary_grad, grad_error = [], [], []
        occupancy = []
        per_sample = {}
        for index, stem in enumerate(args.ids):
            width = batch["widths"][index]; pred = values[index, :, :width]; gt = gt_all[index, :, :width]
            y_pred = cv2.cvtColor(pred.astype(np.float32), cv2.COLOR_RGB2GRAY); y_gt = cv2.cvtColor(gt.astype(np.float32), cv2.COLOR_RGB2GRAY)
            grad_pred, grad_gt = scharr_luma(pred), scharr_luma(gt)
            regions = region_masks(batch["strokes"][index], width)
            regions["all_valid"] = np.ones((96, width), dtype=bool)
            row = {}
            for region, mask in regions.items():
                mask = mask[:, :width].astype(bool)
                mae = float(np.abs(y_pred - y_gt)[mask].mean()); gmae = float(np.abs(grad_pred - grad_gt)[mask].mean())
                region_values[region].append(mae); gradient_values[region].append(gmae); row[region] = {"y_mae": mae, "gradient_mae": gmae}
                if region == "stroke_boundary":
                    pred_boundary_grad.append(grad_pred[mask]); gt_boundary_grad.append(grad_gt[mask]); grad_error.append(np.abs(grad_pred - grad_gt)[mask])
            # ``stats`` owns the valid-width crop; preserve the padded canvas
            # here so its masks retain their common 96 x 480 geometry.
            occ = stats(ink_mask(values[index], width), batch["strokes"][index], width); occupancy.append(occ); per_sample[stem] = row | {"occupancy": occ}
        report["models"][name] = {"macro_regions": {region: {"y_mae": float(np.mean(region_values[region])), "gradient_mae": float(np.mean(gradient_values[region]))} for region in region_values}, "occupancy_macro": {key: float(np.mean([item[key] for item in occupancy])) for key in ("precision", "recall", "f1")}, "boundary_gradient_distribution": {"pred": quantiles(np.concatenate(pred_boundary_grad)), "gt": quantiles(np.concatenate(gt_boundary_grad)), "abs_error": quantiles(np.concatenate(grad_error))}, "per_sample": per_sample}

    # A compact visual strip for the decisive weak-stroke sample.  Values are
    # source tensors, not previously rendered montage panels.
    sample = args.ids.index("002652") if "002652" in args.ids else 0; width = batch["widths"][sample]
    rows = []
    for name in predictions:
        image = (predictions[name][sample, :, :width].clip(0, 1) * 255 + .5).astype(np.uint8)
        panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR); cv2.putText(panel, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 255), 1, cv2.LINE_AA); rows.append(panel)
    gt_panel = cv2.cvtColor((gt_all[sample, :, :width].clip(0, 1) * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR); cv2.putText(gt_panel, "GT", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 255), 1, cv2.LINE_AA); rows.append(gt_panel)
    cv2.imwrite(str(args.out / "002652_all_arms.png"), np.vstack(rows))
    (args.out / "comparison_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
