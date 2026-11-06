"""Aggregate loss-like reconstruction errors and stroke_metrics over existing paired V4 validation dumps.

This intentionally reads already-frozen LQ/GT/output validation artifacts.  It
does not write any restoration images: only CSV/JSON scalar records and metric
curve plots.  ``masked_l1_rgb``/``masked_l1_y`` are common output-space
validation losses (not each arm's non-comparable weighted training total).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("c0", "r1", "r1_bglf", "sdf", "core", "overshoot", "width", "cldice")
STROKE = ("width_cv", "width_ink_r", "ink_unif", "halo_bright", "halo_dark", "edge_rise", "eff_bw")


def _image(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value.astype(np.float32) / 255.0


def _scalar(output: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    weight = valid[..., None]
    denom = float(weight.sum() * 3.0)
    if denom <= 0:
        raise ValueError("empty valid mask")
    diff = output - gt
    l1_rgb = float((np.abs(diff) * weight).sum() / denom)
    mse_rgb = float((np.square(diff) * weight).sum() / denom)
    y_diff = 0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2]
    y_denom = float(valid.sum())
    l1_y = float((np.abs(y_diff) * valid).sum() / max(y_denom, 1.0))
    mse_y = float((np.square(y_diff) * valid).sum() / max(y_denom, 1.0))
    return {"masked_l1_rgb": l1_rgb, "masked_l1_y": l1_y,
            "psnr_rgb": float(99.0 if mse_rgb <= 1e-12 else -10.0 * math.log10(mse_rgb)),
            "psnr_y": float(99.0 if mse_y <= 1e-12 else -10.0 * math.log10(mse_y))}


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    labels = [str(row["arm"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for key, axis, title, better in (
        ("masked_l1_rgb", axes[0, 0], "masked RGB L1 (lower is better)", "low"),
        ("masked_l1_y", axes[0, 1], "masked Y L1 (lower is better)", "low"),
        ("psnr_rgb", axes[1, 0], "masked RGB PSNR (higher is better)", "high"),
        ("psnr_y", axes[1, 1], "masked Y PSNR (higher is better)", "high"),
    ):
        values = [float(row["loss_metrics"][key]) for row in rows]
        axis.plot(x, values, marker="o", linewidth=1.8)
        axis.set_title(title); axis.set_xticks(x, labels, rotation=25); axis.grid(alpha=0.25)
        for xx, yy in zip(x, values): axis.annotate(f"{yy:.4g}", (xx, yy), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    fig.savefig(output, dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    plotted = ("width_cv", "ink_unif", "halo_dark", "edge_rise")
    for key, axis in zip(plotted, axes.flat):
        values = [float(row["stroke_metrics"][key]["paired_delta_mean"]) for row in rows]
        axis.axhline(0.0, color="black", linewidth=0.8); axis.plot(x, values, marker="o", linewidth=1.8)
        axis.set_title(f"stroke_metrics: output - GT, {key}")
        axis.set_xticks(x, labels, rotation=25); axis.grid(alpha=0.25)
        for xx, yy in zip(x, values): axis.annotate(f"{yy:.3g}", (xx, yy), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    fig.savefig(output.with_name("stroke_metrics_delta_curve.png"), dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for arm in ARMS:
        root = args.validation_root / arm
        samples = []
        for output in sorted(root.glob("*/*/output.png")):
            sample = output.parent
            samples.append({"sample": str(sample.relative_to(root)), **_scalar(_image(output), _image(sample / "gt.png"), _mask(sample / "valid_mask.png"))})
        if not samples:
            raise RuntimeError(f"{arm}: no validation outputs under {root}")
        with (args.output_root / f"{arm}_per_sample_losses.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
        stroke = json.loads((root / "stroke_metrics_summary.json").read_text(encoding="utf-8"))["metrics"]
        summary_rows.append({"arm": arm, "sample_count": len(samples),
                             "loss_metrics": {key: _mean(samples, key) for key in ("masked_l1_rgb", "masked_l1_y", "psnr_rgb", "psnr_y")},
                             "stroke_metrics": stroke})
    result = {"status": "COMPLETE", "contract": "same frozen 747-sample real paired validation outputs; no result images written", "arms": summary_rows}
    (args.output_root / "comparison_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["arm", "sample_count", "masked_l1_rgb", "masked_l1_y", "psnr_rgb", "psnr_y"] + [f"delta_{key}" for key in STROKE]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in summary_rows:
            writer.writerow({"arm": row["arm"], "sample_count": row["sample_count"], **row["loss_metrics"], **{f"delta_{key}": row["stroke_metrics"][key]["paired_delta_mean"] for key in STROKE}})
    _plot(summary_rows, args.output_root / "validation_loss_curve.png")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output_root), "arms": [row["arm"] for row in summary_rows]}))


if __name__ == "__main__":
    main()
