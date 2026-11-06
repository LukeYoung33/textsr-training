"""Plot comparable training and validation loss curves from Basic V5 log files."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN = re.compile(r"iter:\s*([\d,]+).*?l_native:\s*([\deE+.-]+).*?l_cp:\s*([\deE+.-]+).*?l_total:\s*([\deE+.-]+)")
VAL = re.compile(r"Validation .*? iter\s+(\d+): l_native=([\deE+.-]+) l_reference=([\deE+.-]+)")


def smooth(values: np.ndarray, width: int = 31) -> np.ndarray:
    if len(values) < width:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    half = width // 2
    return np.convolve(np.pad(values, (half, half), mode="edge"), kernel, mode="valid")


def parse(path: Path):
    train, val = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        hit = TRAIN.search(line)
        if hit:
            train.append(tuple(float(v.replace(",", "")) for v in hit.groups()))
        hit = VAL.search(line)
        if hit:
            val.append(tuple(float(v) for v in hit.groups()))
    return np.asarray(train), np.asarray(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = {"base": "base", "gtdeg": "simulated degradation", "large10": "10x", "large20": "20x", "large50": "50x"}
    files = sorted(args.logs.glob("*.log"))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    summary = ["model,train_points,last_iter,last_total,last_native,last_cp,val_iter,last_val_native,last_val_reference"]
    for path in files:
        key = path.stem
        label = labels.get(key, key)
        train, val = parse(path)
        if not len(train):
            continue
        x = train[:, 0] / 1000.0
        axes[0, 0].plot(x, smooth(train[:, 3]), label=label, linewidth=1.4)
        axes[0, 1].plot(x, smooth(train[:, 1]), label=label, linewidth=1.4)
        axes[1, 0].plot(x, smooth(train[:, 2]), label=label, linewidth=1.4)
        if len(val):
            axes[1, 1].plot(val[:, 0] / 1000.0, val[:, 1], marker="o", label=f"{label}: native", linewidth=1.3)
            axes[1, 1].plot(val[:, 0] / 1000.0, val[:, 2], marker="x", linestyle="--", label=f"{label}: reference", linewidth=1.0)
            last_val = val[-1]
        else:
            last_val = [np.nan, np.nan, np.nan]
        last = train[-1]
        summary.append(f"{label},{len(train)},{int(last[0])},{last[3]:.7g},{last[1]:.7g},{last[2]:.7g},{int(last_val[0]) if np.isfinite(last_val[0]) else ''},{last_val[1]:.7g},{last_val[2]:.7g}")
    titles = ("Training total loss (31-point moving average)", "Training native Y4UV loss (31-point moving average)", "Training CP loss (31-point moving average)", "Validation loss")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("iteration (k)")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    args.output.with_suffix(".csv").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
