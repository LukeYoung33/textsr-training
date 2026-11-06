"""Summarize finetune loss logs by iteration window for convergence review."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_continuous_20260928")
ITER_RE = re.compile(r"\biter:\s*([\d,]+),\s*lr:")
VALUE_RE = re.compile(r"\b(l_total|l_native|l_cp|l_uv_strong|l_bg_lf|l_sdf_boundary|l_missing|l_false):\s*([-+\deE.]+)")
WINDOWS = ((10000, 20000), (30000, 40000), (50000, 60000), (60000, 70000), (70000, 80000), (80000, 90000), (90000, 100000))


def main() -> None:
    results = {}
    for log in sorted(ROOT.glob("finetune_*_attempt1.log")):
        records = []
        nonfinite = []
        for line in log.open(encoding="utf-8", errors="replace"):
            match = ITER_RE.search(line)
            if match is None:
                continue
            iteration = int(match.group(1).replace(",", ""))
            metrics = {key: float(value) for key, value in VALUE_RE.findall(line)}
            if "l_total" not in metrics:
                continue
            if any(not math.isfinite(value) for value in metrics.values()):
                nonfinite.append(iteration)
            records.append((iteration, metrics))
        windows = []
        for lo, hi in WINDOWS:
            chunk = [metrics for iteration, metrics in records if lo <= iteration < hi]
            if not chunk:
                continue
            medians = {key: round(statistics.median(row[key] for row in chunk if key in row), 6)
                       for key in ("l_total", "l_native", "l_cp", "l_uv_strong", "l_bg_lf", "l_sdf_boundary", "l_missing", "l_false")
                       if any(key in row for row in chunk)}
            p10 = sorted(row["l_total"] for row in chunk)[max(0, int(0.1 * len(chunk)) - 1)]
            p90 = sorted(row["l_total"] for row in chunk)[min(len(chunk) - 1, int(0.9 * len(chunk)))]
            windows.append({"iter": f"{lo}-{hi}", "points": len(chunk), "median": medians,
                            "l_total_p10_p90": [round(p10, 6), round(p90, 6)]})
        results[log.stem] = {"latest_logged_iter": records[-1][0] if records else None,
                             "nonfinite_iters": nonfinite, "windows": windows}
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
