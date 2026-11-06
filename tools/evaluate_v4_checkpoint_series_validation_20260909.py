"""Evaluate every numeric V4 checkpoint on frozen paired validation, without image dumps.

The reported loss is a shared valid-mask RGB/Y L1, deliberately separate from
each arm's differently weighted training total. Stroke values call the supplied
``textsr/stroke_metrics.py`` implementation directly on every valid crop.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realesrgan.data  # noqa: F401
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.utils import YUV2RGB
from tools.basic_v2_eval_network import build_eval_network
from tools.dump_basic_v2_validation_rgb_pairs import _load_dataset, _nv21_to_rgb, _yuv444_to_rgb

CHECKPOINT = re.compile(r"net_g_(\d+)\.pth$")
STROKE = ("width_cv", "width_ink_r", "ink_unif", "halo_bright", "halo_dark", "edge_rise", "eff_bw")


def _load_stroke(path: Path):
    spec = importlib.util.spec_from_file_location("stroke_metrics", path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _checkpoints(directory: Path) -> list[tuple[int, Path]]:
    values = []
    for path in directory.glob("net_g_*.pth"):
        match = CHECKPOINT.match(path.name)
        if match: values.append((int(match.group(1)), path))
    if not values: raise FileNotFoundError(f"no numeric generator checkpoints in {directory}")
    return sorted(values)


def _mean(values: list[float]) -> float:
    kept = [value for value in values if math.isfinite(value)]
    return float(np.mean(kept)) if kept else float("nan")


def _median(values: list[float]) -> float:
    kept = [value for value in values if math.isfinite(value)]
    return float(np.median(kept)) if kept else float("nan")


def _evaluate(network, dataset, device, stroke) -> dict[str, float]:
    scalars: dict[str, list[float]] = {key: [] for key in ("masked_l1_rgb", "masked_l1_y", "psnr_rgb", "psnr_y", *STROKE)}
    stroke_raw: dict[str, dict[str, list[float]]] = {
        key: {"output": [], "gt": []} for key in STROKE
    }
    network.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            lq_rgb = _nv21_to_rgb(item["lq"])
            packed = rgb_to_y4uv(torch.from_numpy(np.ascontiguousarray(lq_rgb)).permute(2, 0, 1).unsqueeze(0)).to(device)
            output = _yuv444_to_rgb(y4uv_to_yuv444(network(packed)))
            gt = _yuv444_to_rgb(item["gt"])
            valid = item["valid_mask"][0].numpy() > 0.5
            diff = output - gt
            scalars["masked_l1_rgb"].append(float(np.abs(diff[valid]).mean()))
            scalars["psnr_rgb"].append(float(-10.0 * np.log10(max(np.square(diff[valid]).mean(), 1e-12))))
            ydiff = 0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2]
            scalars["masked_l1_y"].append(float(np.abs(ydiff[valid]).mean()))
            scalars["psnr_y"].append(float(-10.0 * np.log10(max(np.square(ydiff[valid]).mean(), 1e-12))))
            ys, xs = np.where(valid)
            if len(ys):
                crop = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                out_gray = cv2.cvtColor(np.clip(np.rint(output * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                gt_gray = cv2.cvtColor(np.clip(np.rint(gt * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                measured_o, measured_g = stroke.measure(out_gray[crop]), stroke.measure(gt_gray[crop])
                for key in STROKE:
                    output_value, gt_value = float(measured_o[key]), float(measured_g[key])
                    scalars[key].append(output_value - gt_value)
                    stroke_raw[key]["output"].append(output_value)
                    stroke_raw[key]["gt"].append(gt_value)
            if (index + 1) % 100 == 0 or index + 1 == len(dataset): print(f"  samples {index + 1}/{len(dataset)}", flush=True)
    summary = {key: _mean(values) for key, values in scalars.items()}
    summary["stroke_summary"] = {
        key: {
            "valid_samples": len([value for value in values["output"] if math.isfinite(value)]),
            "gt_median": _median(values["gt"]),
            "output_median": _median(values["output"]),
            "paired_delta_median": _median([
                output - gt for output, gt in zip(values["output"], values["gt"])
            ]),
        }
        for key, values in stroke_raw.items()
    }
    return summary


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    x = [int(row["iteration"]) / 1000 for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, key, title in zip(axes.flat, ("masked_l1_rgb", "masked_l1_y", "psnr_rgb", "psnr_y"), ("RGB L1 ↓", "Y L1 ↓", "RGB PSNR ↑", "Y PSNR ↑")):
        axis.plot(x, [float(row[key]) for row in rows], marker="o"); axis.set(title=title, xlabel="iteration (K)"); axis.grid(alpha=.25)
    fig.savefig(output, dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, key in zip(axes.flat, ("width_cv", "ink_unif", "halo_dark", "edge_rise")):
        axis.axhline(0, color="black", linewidth=.7); axis.plot(x, [float(row[key]) for row in rows], marker="o"); axis.set(title=f"stroke output−GT: {key}", xlabel="iteration (K)"); axis.grid(alpha=.25)
    fig.savefig(output.with_name("stroke_metrics_curve.png"), dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--stroke-metrics", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", default="", help="optional comma-separated numeric checkpoint iterations")
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    device, dataset, stroke = torch.device(args.device), _load_dataset(args.config), _load_stroke(args.stroke_metrics)
    rows = []
    checkpoints = _checkpoints(args.checkpoints_dir)
    if args.iterations:
        requested = {int(value) for value in args.iterations.split(",") if value.strip()}
        checkpoints = [item for item in checkpoints if item[0] in requested]
        if not checkpoints: raise ValueError("none of --iterations found in checkpoints-dir")
    for iteration, checkpoint in checkpoints:
        print(f"checkpoint {iteration}: {checkpoint}", flush=True)
        network, digest = build_eval_network(checkpoint, device, args.config)
        values = _evaluate(network, dataset, device, stroke)
        rows.append({"iteration": iteration, "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "samples": len(dataset), **values})
        with (args.output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        del network; torch.cuda.empty_cache()
    report = {"status": "COMPLETE", "config": str(args.config), "samples": len(dataset), "metric_contract": "shared valid-mask RGB/Y L1 & PSNR; stroke values are output-minus-GT means from supplied stroke_metrics.py", "rows": rows}
    (args.output_root / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot(rows, args.output_root / "loss_curve.png")
    print(json.dumps({"status": "COMPLETE", "checkpoints": len(rows), "output": str(args.output_root)}))


if __name__ == "__main__": main()
