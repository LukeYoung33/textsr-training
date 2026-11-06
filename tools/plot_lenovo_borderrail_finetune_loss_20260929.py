"""Render smoothed training-loss trends for the eight whiteboard finetunes."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_continuous_20260928")
OUTPUT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_finetune_loss_20260929")
ITER_RE = re.compile(r"\biter:\s*([\d,]+),\s*lr:")


def values(log: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    iterations, losses = [], []
    value_re = re.compile(rf"\b{re.escape(key)}:\s*([-+\deE.]+)")
    for line in log.open(encoding="utf-8", errors="replace"):
        iteration = ITER_RE.search(line)
        value = value_re.search(line)
        if iteration and value:
            iterations.append(int(iteration.group(1).replace(",", "")))
            losses.append(float(value.group(1)))
    return np.asarray(iterations), np.asarray(losses)


def smooth_by_bin(x: np.ndarray, y: np.ndarray, width: int = 5000) -> tuple[list[float], list[float]]:
    centers, medians = [], []
    for start in range(0, int(x.max()) + 1, width):
        sample = y[(x >= start) & (x < start + width)]
        if len(sample) >= 5:
            centers.append(start + width / 2)
            medians.append(float(np.median(sample)))
    return centers, medians


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    arms = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
    for column, height in enumerate((96, 64)):
        for arm in arms:
            log = ROOT / f"finetune_{arm}_h{height}_attempt1.log"
            for row, key in enumerate(("l_total", "l_native")):
                x, y = values(log, key)
                if not len(x):
                    continue
                sx, sy = smooth_by_bin(x, y)
                axes[row, column].plot(sx, sy, marker="o", markersize=2, label=arm)
                axes[row, column].set_title(f"h{height} {key} (5K-bin median)")
                axes[row, column].grid(alpha=0.25)
                axes[row, column].set_xlim(0, 100000)
                axes[row, column].set_xlabel("iteration")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    path = OUTPUT / "finetune_loss_curves.png"
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == "__main__":
    main()
