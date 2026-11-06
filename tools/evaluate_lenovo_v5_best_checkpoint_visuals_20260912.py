"""Evaluate a V5 checkpoint on the fixed Lenovo validation set.

The output layout follows the V4 visual evaluator: one directory per GT line
contains one GT canvas and vertically concatenated LQ/output variants.  V5
also writes the valid mask (both one-canvas and vertical forms) and records
raw per-image stroke measurements before reporting their medians.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realesrgan.data  # noqa: F401
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from tools.basic_v2_eval_network import build_eval_network
from tools.dump_basic_v2_validation_rgb_pairs import (
    _load_dataset,
    _write_rgb,
    _yuv444_to_rgb,
)


STROKE = (
    "width_cv",
    "width_ink_r",
    "ink_unif",
    "halo_bright",
    "halo_dark",
    "edge_rise",
    "eff_bw",
)


def _empty_stroke_measurement() -> dict[str, float]:
    """Return a comparable missing-value record for a degenerate line crop."""
    return {key: float("nan") for key in STROKE}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stroke-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _tool(path: Path):
    spec = importlib.util.spec_from_file_location("textsr_stroke_metrics_v5", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "measure"):
        raise AttributeError(f"stroke metrics module has no measure(): {path}")
    return module


def _finite(values: list[float], reducer: str) -> float:
    kept = [float(value) for value in values if math.isfinite(float(value))]
    if not kept:
        return float("nan")
    return float(np.median(kept) if reducer == "median" else np.mean(kept))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_key(gt_path: str, index: int) -> str:
    digest = hashlib.sha1(gt_path.encode("utf-8")).hexdigest()[:10]
    # OCR record IDs contain ``::``; keep visual directories portable to
    # Windows when results are copied back from Remote58.
    stem = Path(gt_path).stem.replace(":", "_")
    return f"{index:03d}_{stem}_{digest}"


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.rint(np.asarray(mask, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), value):
        raise IOError(f"failed to write valid mask: {path}")


def _dataset_rgb(item: dict[str, Any], key: str) -> np.ndarray:
    """Use the dataset's authoritative RGB tensor, never a packed-NV21 decoder."""
    value = item[key]
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"{key} must be an RGB CHW tensor")
    return value.detach().cpu().float().permute(1, 2, 0).numpy().copy()


def _measure(
    output: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    stroke: Any,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """@luna-comment Measure one valid crop and retain raw output/GT stroke values."""
    if valid.ndim != 2:
        raise ValueError(f"valid mask must be HW, got {valid.shape}")
    ys, xs = np.where(valid)
    if not len(ys):
        raise RuntimeError("validation sample has no valid pixels")
    diff = output - gt
    rgb_l1 = float(np.abs(diff[valid]).mean())
    rgb_mse = float(np.square(diff[valid]).mean())
    y_diff = 0.299 * diff[..., 0] + 0.587 * diff[..., 1] + 0.114 * diff[..., 2]
    y_l1 = float(np.abs(y_diff[valid]).mean())
    y_mse = float(np.square(y_diff[valid]).mean())
    crop = np.s_[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    out_gray = cv2.cvtColor(
        np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    gt_gray = cv2.cvtColor(
        np.clip(np.rint(gt * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    # A valid OCR crop may contain a separator, punctuation, or a saturated
    # line whose Otsu mask has no paper/ink pixels.  Pixel metrics remain
    # meaningful; stroke descriptors for that individual crop are missing.
    try:
        output_stroke = stroke.measure(out_gray[crop])
    except (IndexError, ValueError):
        output_stroke = _empty_stroke_measurement()
    try:
        gt_stroke = stroke.measure(gt_gray[crop])
    except (IndexError, ValueError):
        gt_stroke = _empty_stroke_measurement()
    scalar = {
        "masked_l1_rgb": rgb_l1,
        "masked_l1_y": y_l1,
        "psnr_rgb": float(-10.0 * np.log10(max(rgb_mse, 1e-12))),
        "psnr_y": float(-10.0 * np.log10(max(y_mse, 1e-12))),
    }
    output_values = {key: float(output_stroke[key]) for key in STROKE}
    gt_values = {key: float(gt_stroke[key]) for key in STROKE}
    return scalar, output_values, gt_values


def evaluate(
    arm: str,
    config: Path,
    checkpoint: Path,
    stroke_metrics: Path,
    output_root: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """@luna-comment Run fixed V5 inference, grouped visual export, and median stroke metrics."""
    config = config.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    stroke_metrics = stroke_metrics.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not stroke_metrics.is_file():
        raise FileNotFoundError(stroke_metrics)

    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    dataset = _load_dataset(config)
    stroke = _tool(stroke_metrics)
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    scalar_values: dict[str, list[float]] = {
        key: [] for key in ("masked_l1_rgb", "masked_l1_y", "psnr_rgb", "psnr_y")
    }
    stroke_values = {
        key: {"output": [], "gt": [], "delta": []} for key in STROKE
    }
    rows: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            # Packed NV21 was formed with a full-range RGB2YUV conversion;
            # OpenCV's NV21 decoder has a different video-range contract.
            lq = _dataset_rgb(item, "lq_rgb")
            gt = _dataset_rgb(item, "gt_rgb")
            packed = rgb_to_y4uv(
                torch.from_numpy(np.ascontiguousarray(lq))
                .permute(2, 0, 1)
                .unsqueeze(0)
            ).to(device)
            output = _yuv444_to_rgb(y4uv_to_yuv444(network(packed)))
            valid = item["valid_mask"][0].numpy() > 0.5
            scalar, output_stroke, gt_stroke = _measure(output, gt, valid, stroke)
            row: dict[str, Any] = {
                "index": index,
                "line_id": str(item["line_id"]),
                "gt_path": str(item["gt_path"]),
                "lq_path": str(item["lq_path"]),
                **scalar,
            }
            for key in STROKE:
                delta = output_stroke[key] - gt_stroke[key]
                row.update(
                    {
                        f"output_{key}": output_stroke[key],
                        f"gt_{key}": gt_stroke[key],
                        f"delta_{key}": delta,
                    }
                )
                stroke_values[key]["output"].append(output_stroke[key])
                stroke_values[key]["gt"].append(gt_stroke[key])
                stroke_values[key]["delta"].append(delta)
            for key, value in scalar.items():
                scalar_values[key].append(value)
            rows.append(row)
            # Legacy fixed validation contains three LQ variants for each GT.
            # P17 expands each full page into unique OCR-line records, so a
            # page path no longer identifies a visual group.
            group_key = (
                str(item["line_id"])
                if bool(getattr(dataset, "ocr_line_crop", False))
                else str(item["gt_path"])
            )
            group = groups.setdefault(
                group_key,
                {"gt": gt, "lq": [], "output": [], "valid": [], "items": []},
            )
            group["lq"].append(lq)
            group["output"].append(output)
            group["valid"].append(valid.astype(np.float32))
            group["items"].append(
                {
                    "index": index,
                    "line_id": str(item["line_id"]),
                    "gt_path": str(item["gt_path"]),
                    "lq_path": str(item["lq_path"]),
                }
            )
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"[{arm}] samples {index + 1}/{len(dataset)}", flush=True)

    visual_root = output_root / "visuals"
    visual_records: list[dict[str, Any]] = []
    for group_index, (group_key, group) in enumerate(groups.items()):
        lq = group["lq"]
        output = group["output"]
        valid = group["valid"]
        variant_count = len(lq)
        if not variant_count or len(output) != variant_count or len(valid) != variant_count:
            raise RuntimeError(f"{group_key}: inconsistent LQ/output/valid variants")
        directory = visual_root / _group_key(group_key, group_index)
        _write_rgb(directory / "gt.png", group["gt"])
        _write_rgb(directory / "lq_vertical.png", np.concatenate(lq, axis=0))
        _write_rgb(directory / "output_vertical.png", np.concatenate(output, axis=0))
        _write_mask(directory / "valid_mask.png", valid[0])
        _write_mask(directory / "valid_mask_vertical.png", np.concatenate(valid, axis=0))
        metadata = {
            "gt_path": str(group["items"][0]["gt_path"]),
            "variant_order": group["items"],
            "layout": {
                "gt": "96x480 RGB canvas",
                "lq_vertical": f"{variant_count} 96x480 RGB canvases, top to bottom",
                "output_vertical": f"{variant_count} 96x480 RGB canvases, same order as LQ",
                "valid_mask": "96x480 one-canvas mask for the first variant",
                "valid_mask_vertical": f"{variant_count} 96x480 masks, same order as LQ",
            },
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        visual_records.append(
            {"directory": str(directory.relative_to(output_root)), **metadata}
        )

    per_sample = output_root / "stroke_metrics_per_sample.jsonl"
    with per_sample.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stroke_summary = {
        key: {
            "valid_samples": len(
                [value for value in values["output"] if math.isfinite(float(value))]
            ),
            "gt_median": _finite(values["gt"], "median"),
            "output_median": _finite(values["output"], "median"),
            "paired_delta_median": _finite(values["delta"], "median"),
        }
        for key, values in stroke_values.items()
    }
    stroke_report = {
        "status": "COMPLETE",
        "arm": arm,
        "sample_count": len(rows),
        "aggregation": "median of per-image raw values; valid crop applied per image",
        "metrics": stroke_summary,
        "per_sample": str(per_sample),
    }
    (output_root / "stroke_metrics_summary.json").write_text(
        json.dumps(stroke_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "COMPLETE",
        "arm": arm,
        "samples": len(rows),
        "gt_groups": len(groups),
        "checkpoint_selection": {
            "rule": "explicit post-training checkpoint",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "metrics_mean": {
            key: _finite(values, "mean") for key, values in scalar_values.items()
        },
        "metrics_median": {
            key: _finite(values, "median") for key, values in scalar_values.items()
        },
        "stroke_metrics": stroke_summary,
        "visual_layout": "one GT plus all vertically concatenated LQ/output/valid variants per group",
        "input_color_contract": {
            "lq": "dataset lq_rgb -> rgb_to_y4uv -> network",
            "gt": "dataset gt_rgb",
            "replaces": "packed lq -> OpenCV COLOR_YUV2RGB_NV21",
        },
        "visual_records": visual_records,
        "config": str(config),
        "config_sha256": _sha256(config),
        "stroke_metrics_tool": str(stroke_metrics),
    }
    (output_root / "best_checkpoint_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """@luna-comment Parse the fixed V5 evaluation contract and print completion evidence."""
    args = _args()
    report = evaluate(
        args.arm,
        args.config,
        args.checkpoint,
        args.stroke_metrics,
        args.output_root,
        args.device,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": report["arm"],
                "samples": report["samples"],
                "gt_groups": report["gt_groups"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
