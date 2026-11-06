"""Evaluate one V4 arm's log-selected checkpoint and retain grouped RGB visuals.

Each fixed-validation GT has three deterministic LQ variants.  This evaluator
keeps one GT canvas and vertically concatenates the matching LQ and restored
RGB canvases, preserving their manifest order.  It also applies the supplied
stroke-metrics implementation to the same valid crop used by scalar metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realesrgan.data  # noqa: F401
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from tools.basic_v2_eval_network import build_eval_network
from tools.dump_basic_v2_validation_rgb_pairs import _load_dataset, _write_rgb, _yuv444_to_rgb

STROKE = ("width_cv", "width_ink_r", "ink_unif", "halo_bright", "halo_dark", "edge_rise", "eff_bw")
LOG_VALIDATION = re.compile(
    r"Validation .*? iter (?P<iteration>\d+): l_native=(?P<native>[\deE+-.]+)"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--stroke-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _tool(path: Path):
    spec = importlib.util.spec_from_file_location("stroke_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _finite_median(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.median(values)) if values else float("nan")


def _selected_checkpoint(log: Path, checkpoints_dir: Path) -> tuple[int, float, Path]:
    candidates: list[tuple[float, int, Path]] = []
    for match in LOG_VALIDATION.finditer(log.read_text(encoding="utf-8", errors="replace")):
        iteration = int(match.group("iteration"))
        checkpoint = checkpoints_dir / f"net_g_{iteration}.pth"
        if checkpoint.is_file():
            candidates.append((float(match.group("native")), iteration, checkpoint))
    if not candidates:
        raise RuntimeError(f"no logged validation checkpoint found for {log}")
    native, iteration, checkpoint = min(candidates, key=lambda value: (value[0], value[1]))
    return iteration, native, checkpoint


def _group_key(gt_path: str, index: int) -> str:
    digest = hashlib.sha1(gt_path.encode("utf-8")).hexdigest()[:10]
    return f"{index:03d}_{Path(gt_path).stem}_{digest}"


def _dataset_rgb(item: dict, key: str) -> np.ndarray:
    """Use preserved dataset RGB rather than decoding packed NV21."""
    value = item[key]
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"{key} must be an RGB CHW tensor")
    return value.detach().cpu().float().permute(1, 2, 0).numpy().copy()


def _metrics(output: np.ndarray, gt: np.ndarray, valid: np.ndarray, stroke) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    diff = output - gt
    rgb_l1 = float(np.abs(diff[valid]).mean())
    rgb_psnr = float(-10.0 * np.log10(max(float(np.square(diff[valid]).mean()), 1e-12)))
    y_diff = 0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2]
    y_l1 = float(np.abs(y_diff[valid]).mean())
    y_psnr = float(-10.0 * np.log10(max(float(np.square(y_diff[valid]).mean()), 1e-12)))
    ys, xs = np.where(valid)
    if not len(ys):
        raise RuntimeError("validation sample has no valid pixels")
    crop = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    out_gray = cv2.cvtColor(np.clip(np.rint(output * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gt_gray = cv2.cvtColor(np.clip(np.rint(gt * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    output_stroke, gt_stroke = stroke.measure(out_gray[crop]), stroke.measure(gt_gray[crop])
    scalar = {"masked_l1_rgb": rgb_l1, "masked_l1_y": y_l1, "psnr_rgb": rgb_psnr, "psnr_y": y_psnr}
    return scalar, {key: float(output_stroke[key]) for key in STROKE}, {key: float(gt_stroke[key]) for key in STROKE}


def main() -> None:
    args = _args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    iteration, logged_native, checkpoint = _selected_checkpoint(args.log, args.checkpoints_dir)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset, stroke = _load_dataset(args.config), _tool(args.stroke_metrics)
    network, digest = build_eval_network(checkpoint, device, args.config)
    scalars: dict[str, list[float]] = {key: [] for key in ("masked_l1_rgb", "masked_l1_y", "psnr_rgb", "psnr_y")}
    stroke_values = {key: {"output": [], "gt": [], "delta": []} for key in STROKE}
    groups: dict[str, dict[str, object]] = {}
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            lq = _dataset_rgb(item, "lq_rgb")
            gt = _dataset_rgb(item, "gt_rgb")
            packed = rgb_to_y4uv(torch.from_numpy(np.ascontiguousarray(lq)).permute(2, 0, 1).unsqueeze(0)).to(device)
            output = _yuv444_to_rgb(y4uv_to_yuv444(network(packed)))
            valid = item["valid_mask"][0].numpy() > 0.5
            scalar, output_stroke, gt_stroke = _metrics(output, gt, valid, stroke)
            for key, value in scalar.items():
                scalars[key].append(value)
            for key in STROKE:
                stroke_values[key]["output"].append(output_stroke[key])
                stroke_values[key]["gt"].append(gt_stroke[key])
                stroke_values[key]["delta"].append(output_stroke[key] - gt_stroke[key])
            gt_path = str(item["gt_path"])
            group = groups.setdefault(gt_path, {"gt": gt, "lq": [], "output": [], "items": []})
            group["lq"].append(lq)
            group["output"].append(output)
            group["items"].append({"index": index, "line_id": str(item["line_id"]), "lq_path": str(item["lq_path"])})
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"[{args.arm}] samples {index + 1}/{len(dataset)}", flush=True)
    visual_root = args.output_root / "visuals"
    visual_records = []
    for group_index, (gt_path, group) in enumerate(groups.items()):
        lq, output = group["lq"], group["output"]
        if len(lq) != 3 or len(output) != 3:
            raise RuntimeError(f"{gt_path}: expected exactly three LQ variants, got {len(lq)}")
        directory = visual_root / _group_key(gt_path, group_index)
        _write_rgb(directory / "gt.png", group["gt"])
        _write_rgb(directory / "lq_vertical.png", np.concatenate(lq, axis=0))
        _write_rgb(directory / "output_vertical.png", np.concatenate(output, axis=0))
        metadata = {"gt_path": gt_path, "variant_order": group["items"], "layout": {"gt": "96x480", "lq_vertical": "three 96x480 RGB canvases, top to bottom", "output_vertical": "three 96x480 RGB canvases, same order as LQ"}}
        (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        visual_records.append({"directory": str(directory.relative_to(args.output_root)), **metadata})
    stroke_summary = {
        key: {"valid_samples": len([value for value in values["output"] if math.isfinite(value)]), "gt_median": _finite_median(values["gt"]), "output_median": _finite_median(values["output"]), "paired_delta_mean": _finite_mean(values["delta"]), "paired_delta_median": _finite_median(values["delta"])}
        for key, values in stroke_values.items()
    }
    report = {"status": "COMPLETE", "arm": args.arm, "samples": len(dataset), "gt_groups": len(groups), "checkpoint_selection": {"rule": "lowest logged l_native among checkpoints present on disk", "iteration": iteration, "logged_l_native": logged_native, "checkpoint": str(checkpoint), "checkpoint_sha256": digest}, "metrics": {key: _finite_mean(values) for key, values in scalars.items()}, "stroke_metrics": stroke_summary, "visual_layout": "one GT plus three vertically concatenated LQs and outputs per group", "input_color_contract": {"lq": "dataset lq_rgb -> rgb_to_y4uv -> network", "gt": "dataset gt_rgb", "replaces": "packed lq -> OpenCV COLOR_YUV2RGB_NV21"}, "visual_records": visual_records}
    (args.output_root / "best_checkpoint_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "arm": args.arm, "iteration": iteration, "samples": len(dataset), "gt_groups": len(groups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
