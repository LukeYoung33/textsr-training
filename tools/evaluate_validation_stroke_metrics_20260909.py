"""Paired validation self-check using the supplied TextSR stroke_metrics tool.

This is deliberately descriptive rather than a fidelity metric: it reports the
toolkit's image-level stroke-shape measurements for GT and restoration on the
same valid crop, plus paired output-minus-GT deltas.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np


METRICS = ("width_cv", "width_ink_r", "ink_unif", "halo_bright", "halo_dark", "edge_rise", "eff_bw")


def _tool(path: Path):
    spec = importlib.util.spec_from_file_location("textsr_stroke_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gray(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value.astype(np.float32)


def _valid_crop(valid: np.ndarray) -> tuple[slice, slice] | None:
    ys, xs = np.where(valid > 127)
    if not len(ys):
        return None
    return slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)


def _mean(values: list[float]) -> float | None:
    kept = [value for value in values if math.isfinite(value)]
    return float(np.mean(kept)) if kept else None


def _median(values: list[float]) -> float | None:
    kept = [value for value in values if math.isfinite(value)]
    return float(np.median(kept)) if kept else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--stroke-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = _tool(args.stroke_metrics)
    rows: list[dict[str, object]] = []
    for sample in sorted(args.validation_root.glob("*/*")):
        if not sample.is_dir():
            continue
        output, gt, valid = sample / "output.png", sample / "gt.png", sample / "valid_mask.png"
        if not (output.is_file() and gt.is_file() and valid.is_file()):
            continue
        crop = _valid_crop(_gray(valid))
        if crop is None:
            continue
        result = {"sample": str(sample.relative_to(args.validation_root))}
        for prefix, image in (("gt", _gray(gt)[crop]), ("output", _gray(output)[crop])):
            measured = metrics.measure(image)
            for key in METRICS:
                result[f"{prefix}_{key}"] = float(measured[key])
        for key in METRICS:
            result[f"delta_{key}"] = float(result[f"output_{key}"]) - float(result[f"gt_{key}"])
        rows.append(result)
    if not rows:
        raise RuntimeError(f"no complete validation samples under {args.validation_root}")
    summary: dict[str, object] = {"status": "COMPLETE", "sample_count": len(rows), "metrics": {}}
    for key in METRICS:
        summary["metrics"][key] = {
            "gt_mean": _mean([float(row[f"gt_{key}"]) for row in rows]),
            "output_mean": _mean([float(row[f"output_{key}"]) for row in rows]),
            "paired_delta_mean": _mean([float(row[f"delta_{key}"]) for row in rows]),
            "paired_delta_median": _median([float(row[f"delta_{key}"]) for row in rows]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
