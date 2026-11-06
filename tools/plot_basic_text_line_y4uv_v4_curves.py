"""Plot the native/reference loss curves of the four V4 training runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRAIN_ITER_RE = re.compile(r"iter:\s*([\d,]+)")
TRAIN_VALUE_RE = re.compile(
    r"\b(l_native|l_reference|l_compare_abs|l_selected|l_total):\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
VAL_RE = re.compile(
    r"Validation .*? iter\s+([\d,]+):\s+"
    r"l_native=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"l_reference=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"l_compare_abs=([-+]?\d+(?:\.\d+)?(?:[eE][-]?\d+)?)"
)


def parse_log(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """@shanlan-comment Parse finite V4 training and validation metrics from one BasicSR log."""
    train: dict[str, list[tuple[int, float]]] = {
        key: [] for key in ("l_native", "l_reference", "l_compare_abs", "l_selected", "l_total")
    }
    val: dict[str, list[tuple[int, float]]] = {
        key: [] for key in ("l_native", "l_reference", "l_compare_abs")
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        iteration = TRAIN_ITER_RE.search(line)
        if iteration:
            step = int(iteration.group(1).replace(",", ""))
            for key, value in TRAIN_VALUE_RE.findall(line):
                train[key].append((step, float(value)))
        validation = VAL_RE.search(line)
        if validation:
            step = int(validation.group(1).replace(",", ""))
            for key, value in zip(("l_native", "l_reference", "l_compare_abs"), validation.groups()[1:]):
                val[key].append((step, float(value)))
    train_arrays = {
        key: np.asarray(sorted(values), dtype=np.float64).reshape(-1, 2)
        for key, values in train.items()
        if values
    }
    val_arrays = {
        key: np.asarray(sorted(values), dtype=np.float64).reshape(-1, 2)
        for key, values in val.items()
        if values
    }
    return train_arrays, val_arrays


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """@shanlan-comment Apply an edge-padded moving average without changing curve length."""
    if window <= 1 or len(values) <= 1:
        return values
    width = min(int(window), len(values))
    kernel = np.ones(width, dtype=np.float64) / width
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def run_name(path: Path) -> str:
    """@shanlan-comment Convert a long V4 log filename into a compact plot legend label."""
    name = path.name.lower()
    dataset = "0812" if "20260812" in name else "0820"
    haar = "haar" if "_haar_" in name else "nohaar"
    return f"{haar}_{dataset}"


def write_summary(path: Path, parsed: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]]) -> None:
    """@shanlan-comment Persist latest train values and every validation point for audit-friendly review."""
    rows: list[dict[str, object]] = []
    for label, train, val in parsed:
        latest_step = int(train["l_total"][-1, 0]) if "l_total" in train else 0
        row: dict[str, object] = {
            "run": label,
            "latest_train_iter": latest_step,
            "latest_l_native": float(train["l_native"][-1, 1]) if "l_native" in train else "",
            "latest_l_reference": float(train["l_reference"][-1, 1]) if "l_reference" in train else "",
            "latest_l_total": float(train["l_total"][-1, 1]) if "l_total" in train else "",
        }
        for key in ("l_native", "l_reference", "l_compare_abs"):
            for step, value in val.get(key, np.empty((0, 2))):
                row[f"val_{key}_{int(step)}"] = float(value)
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(parsed: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]], output: Path,
                smooth_window: int) -> None:
    """@shanlan-comment Draw synchronized train and eval panels for all four V4 runs."""
    colors = {"haar_0812": "#1f77b4", "nohaar_0812": "#ff7f0e", "haar_0820": "#2ca02c", "nohaar_0820": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    panels = (("l_native", "Native loss"), ("l_reference", "Reference loss"), ("l_total", "Total loss"))
    for ax, (metric, title) in zip(axes.flat[:3], panels):
        for label, train, _ in parsed:
            if metric not in train:
                continue
            steps = train[metric][:, 0]
            values = train[metric][:, 1]
            color = colors.get(label, None)
            ax.plot(steps, values, color=color, alpha=0.10, linewidth=0.5)
            ax.plot(steps, moving_average(values, smooth_window), color=color, linewidth=1.7, label=label)
        ax.set_title(f"Training {title} (moving average {smooth_window})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="x", style="plain")
    ax = axes.flat[3]
    for label, _, val in parsed:
        color = colors.get(label, None)
        for metric, style in (("l_native", "o-"), ("l_reference", "s--"), ("l_compare_abs", "^:")):
            if metric not in val:
                continue
            ax.plot(val[metric][:, 0], val[metric][:, 1], style, color=color, linewidth=1.4,
                    markersize=4, label=f"{label} {metric}")
    ax.set_title("Validation losses")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    ax.ticklabel_format(axis="x", style="plain")
    fig.suptitle("BASIC V2 Y4UV V4 CP training curves", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """@shanlan-comment Parse CLI logs, render the curve image, and write the metric summary."""
    parser = argparse.ArgumentParser(description="Plot BASIC V2 Y4UV V4 training curves")
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()
    parsed = []
    for log in args.logs:
        train, val = parse_log(log)
        if not train or not val:
            raise RuntimeError(f"log has no V4 train/eval records: {log}")
        parsed.append((run_name(log), train, val))
    plot_curves(parsed, args.output, args.smooth_window)
    write_summary(args.summary, parsed)
    print(f"saved={args.output}")
    print(f"summary={args.summary}")
    for label, train, val in parsed:
        print(label, "train_iter", int(train["l_total"][-1, 0]), "eval_points", len(val.get("l_native", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
