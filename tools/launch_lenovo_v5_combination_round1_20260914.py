"""Launch the eight V5 round-one combination runs as two persistent GPU queues."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO = Path("/home/guang.yang/workspace/Real-ESRGAN")
PYTHON = Path("/home/guang.yang/realSR/bin/python")
CONFIG_ROOT = REPO / "options/train_bucket_3/lenovo_v5_combination_round1_20260914"
EXPERIMENTS_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_v5_combination_round1_20260914")
FORMAL_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/formal/lenovo_v5_combination_round1_20260914")
STROKE_METRICS = REPO / "tools/stroke_metrics_reference_20260908.py"
QUEUES = {"2": ["g0_r1", "g2_r1_p", "g4_r1_p_cldice", "g6_r1_p_core"], "3": ["g1_rb", "g3_rb_p", "g5_rb_p_cldice", "g7_rb_p_core"]}


def _command(gpu: str, arm: str) -> str:
    config = CONFIG_ROOT / f"train_lenovo_v5_{arm}_250k_20260914.yml"
    experiment = EXPERIMENTS_ROOT / f"lenovo_v5_{arm}_250k_20260914"
    log = FORMAL_ROOT / arm / "train.log"
    return " ".join((
        f"mkdir -p {log.parent}",
        "&&",
        f"cd {REPO}",
        "&&",
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH={REPO} PYTHONUNBUFFERED=1 REAL_ESRGAN_DEFER_FINAL_VALIDATION=1 "
        f"{PYTHON} -u realesrgan/train.py -opt {config} --launcher none > {log} 2>&1",
        "&&",
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH={REPO} {PYTHON} tools/run_lenovo_v5_post_training_eval.py "
        f"--arm {arm} --config {config} --experiment-root {experiment} --stroke-metrics {STROKE_METRICS} "
        f"--device cuda >> {log} 2>&1",
    ))


def main() -> int:
    FORMAL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = FORMAL_ROOT / "launch_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing duplicate launch: {manifest_path}")
    records = []
    for gpu, arms in QUEUES.items():
        command = " && ".join(_command(gpu, arm) for arm in arms)
        log_path = FORMAL_ROOT / f"queue_gpu{gpu}.log"
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                ["bash", "-lc", command],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        records.append({"gpu": gpu, "arms": arms, "wrapper_pid": process.pid, "queue_log": str(log_path)})
    manifest_path.write_text(json.dumps({"status": "launched", "queues": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "launched", "queues": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
