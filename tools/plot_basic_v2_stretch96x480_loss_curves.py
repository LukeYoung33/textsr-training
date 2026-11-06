"""Plot training and offline validation losses for BASIC V2 runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ITER_RE = re.compile(r"\biter:\s*([\d,]+)")
VALUE_RE = re.compile(r"\b(l_base|l_cp|l_total):\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.IGNORECASE)


def parse_training_log(path: Path) -> dict[str, np.ndarray]:
    """Parse finite BASIC V2 loss values keyed by training iteration."""
    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITER_RE.search(line)
        if not match:
            continue
        iteration = int(match.group(1).replace(",", ""))
        values = {key.lower(): float(value) for key, value in VALUE_RE.findall(line)}
        if "l_total" in values:
            rows[iteration] = values
    if not rows:
        raise RuntimeError(f"no BASIC V2 losses found in {path}")
    columns = {"iteration": [], "l_base": [], "l_cp": [], "l_total": []}
    for iteration in sorted(rows):
        columns["iteration"].append(float(iteration))
        for key in ("l_base", "l_cp", "l_total"):
            columns[key].append(float(rows[iteration].get(key, np.nan)))
    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}


def read_validation_csv(path: Path) -> dict[str, np.ndarray]:
    """Read the auditable offline validation curve CSV."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["iteration"]))
    if not rows:
        raise RuntimeError(f"validation curve is empty: {path}")
    return {
        "iteration": np.asarray([int(row["iteration"]) for row in rows], dtype=np.float64),
        "val_l_base": np.asarray([float(row["val_l_base"]) for row in rows], dtype=np.float64),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth finite values with edge padding while preserving curve length."""
    if window <= 1 or len(values) <= 1:
        return values.copy()
    width = min(int(window), len(values))
    kernel = np.ones(width, dtype=np.float64) / width
    left = width // 2
    right = width - 1 - left
    return np.convolve(np.pad(values, (left, right), mode="edge"), kernel, mode="valid")


def write_training_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    """Persist every parsed training point used in the plots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "l_base", "l_cp", "l_total", "cp_fraction"])
        for index in range(len(data["iteration"])):
            total = data["l_total"][index]
            cp = data["l_cp"][index]
            writer.writerow([
                int(data["iteration"][index]), data["l_base"][index], cp, total,
                cp / total if np.isfinite(total) and total > 0 else "",
            ])


def main() -> int:
    """Render per-run and comparison plots plus machine-readable summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--validation-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()
    if not (len(args.labels) == len(args.logs) == len(args.validation_csvs)):
        raise ValueError("labels, logs, and validation-csvs must have equal lengths")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    parsed = []
    summary: dict[str, object] = {
        "validation_definition": "offline BoxAwareMaskedL1LossV2 l_base on a fixed source-balanced validation subset",
        "training_smooth_window_points": args.smooth_window,
        "runs": {},
    }
    for label, log, validation_csv in zip(args.labels, args.logs, args.validation_csvs):
        train = parse_training_log(log)
        validation = read_validation_csv(validation_csv)
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        write_training_csv(output / f"{slug}_training_loss.csv", train)
        parsed.append((label, slug, train, validation))
        summary["runs"][label] = {
            "last_iteration": int(train["iteration"][-1]),
            "last_train_l_base": float(train["l_base"][-1]),
            "last_train_l_cp": float(train["l_cp"][-1]),
            "last_train_l_total": float(train["l_total"][-1]),
            "validation_points": int(len(validation["iteration"])),
            "last_val_l_base": float(validation["val_l_base"][-1]),
            "best_val_l_base": float(np.min(validation["val_l_base"])),
        }

    fig, axes = plt.subplots(len(parsed), 2, figsize=(15, 4 * len(parsed)), squeeze=False, constrained_layout=True)
    for row, (label, _, train, validation) in enumerate(parsed):
        iterations = train["iteration"]
        left, right = axes[row]
        left.plot(iterations, train["l_total"], color="#1f77b4", alpha=0.08, linewidth=0.5)
        left.plot(iterations, moving_average(train["l_total"], args.smooth_window), color="#1f77b4", label="train l_total")
        left.plot(iterations, moving_average(train["l_base"], args.smooth_window), color="#2ca02c", label="train l_base")
        left.plot(iterations, moving_average(train["l_cp"], args.smooth_window), color="#d62728", label="train weighted l_cp")
        left.set_title(f"{label}: training losses")
        left.set_xlabel("Iteration")
        left.set_ylabel("Loss")
        left.grid(alpha=0.2)
        left.legend(fontsize=8)

        right.plot(iterations, moving_average(train["l_base"], args.smooth_window), color="#2ca02c", alpha=0.7, label="train l_base (smoothed)")
        right.plot(validation["iteration"], validation["val_l_base"], "o-", color="#9467bd", markersize=3, label="offline val l_base")
        right.set_title(f"{label}: train vs validation reconstruction loss")
        right.set_xlabel("Iteration")
        right.set_ylabel("Box-aware YUV L1")
        right.grid(alpha=0.2)
        right.legend(fontsize=8)
    fig.suptitle("BASIC V2 direct-resize 96x480 loss curves", fontsize=15)
    fig.savefig(output / "all_models_train_validation_loss_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(parsed)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for color, (label, _, train, validation) in zip(colors, parsed):
        axes[0].plot(train["iteration"], moving_average(train["l_total"], args.smooth_window), color=color, label=label)
        axes[1].plot(validation["iteration"], validation["val_l_base"], "o-", color=color, markersize=3, label=label)
    axes[0].set_title("Smoothed training total loss")
    axes[1].set_title("Offline validation reconstruction loss")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(output / "all_models_loss_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    (output / "loss_curve_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
