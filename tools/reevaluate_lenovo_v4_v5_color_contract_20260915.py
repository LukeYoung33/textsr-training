"""Re-evaluate the reported Lenovo V4/V5 checkpoints with direct RGB inputs.

This is an append-only correction pass.  It writes a new result tree and never
modifies the historical NV21-decoded evaluations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path("/home/guang.yang/workspace/Real-ESRGAN")
PYTHON = Path("/home/guang.yang/realSR/bin/python")
EXPERIMENTS = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments")
OUTPUT_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_color_contract_fixed_20260915")
STROKE = REPO / "tools/stroke_metrics_reference_20260908.py"

ARMS = ("c0", "r1", "rb", "s1", "core", "over", "width", "cldice")
CAPACITY = (
    "v5deploymatch_c0", "v5deploymatch_r1", "10x_c0", "10x_r1",
    "50x_c0", "50x_r1", "100x_c0", "100x_r1",
)


def tasks() -> list[dict[str, str]]:
    records = []
    for arm in ARMS:
        name = f"lenovo_v4_{arm}_250k_20260911"
        records.append({
            "family": "v4_eight_arm", "arm": arm,
            "config": str(REPO / "options/train_bucket_3/lenovo_v4_cp_strong_20260911" / f"train_{name}.yml"),
            "checkpoint": str(EXPERIMENTS / "lenovo_v4_cp_strong_20260911" / name / "models/net_g_250000.pth"),
        })
    for arm in ARMS:
        name = f"lenovo_v5_{arm}_250k_20260912"
        records.append({
            "family": "v5_eight_arm", "arm": arm,
            "config": str(REPO / "options/train_bucket_3/lenovo_v5_cp_strong_20260912" / f"train_{name}.yml"),
            "checkpoint": str(EXPERIMENTS / "lenovo_v5_cp_strong_20260912" / name / "models/net_g_250000.pth"),
        })
    for arm in CAPACITY:
        name = f"lenovo_v4cap_{arm}_250k_20260913"
        records.append({
            "family": "v4_capacity", "arm": arm,
            "config": str(REPO / "options/train_bucket_3/lenovo_v4_capacity_vs_v5_20260913" / f"train_{name}.yml"),
            "checkpoint": str(EXPERIMENTS / "lenovo_v4_capacity_vs_v5_20260913" / name / "models/net_g_250000.pth"),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--task-index", type=int, action="append")
    args = parser.parse_args()
    selected = tasks()
    if args.task_index:
        selected = [selected[index] for index in args.task_index]
    for record in selected:
        output = OUTPUT_ROOT / record["family"] / record["arm"]
        output.mkdir(parents=True, exist_ok=True)
        marker = output / "best_checkpoint_evaluation.json"
        if marker.is_file():
            print(f"[skip] {record['family']}/{record['arm']}", flush=True)
            continue
        command = [
            str(PYTHON), "tools/evaluate_lenovo_v5_best_checkpoint_visuals_20260912.py",
            "--arm", f"{record['family']}_{record['arm']}",
            "--config", record["config"], "--checkpoint", record["checkpoint"],
            "--stroke-metrics", str(STROKE), "--output-root", str(output), "--device", "cuda",
        ]
        print(json.dumps({"status": "START", **record, "output": str(output)}, ensure_ascii=False), flush=True)
        subprocess.run(command, cwd=REPO, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1"})
        print(json.dumps({"status": "COMPLETE", **record, "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
