"""Run the shape-aligned two-route 33-image check for eight 50K checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/guang.yang/realSR/bin/python")
RUN_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_tile_shapealigned_20260929")
EXPERIMENT_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_8arm_mojito_finetune_20260928")
CONFIG_ROOT = REPO / "options/train_bucket_3/lenovo_borderrail_8arm_20260928/finetune"
ARMS = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
HEIGHTS = (96, 64)
ROUTES = ("large_tile", "overlap_tile")
STATUS_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_status(status: dict) -> None:
    with STATUS_LOCK:
        status["updated_at"] = now()
        target = RUN_ROOT / "status.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)


def run_one(arm: str, height: int, gpu: int, status: dict) -> bool:
    name = f"{arm}_h{height}"
    record = status["jobs"][name]
    checkpoint = EXPERIMENT_ROOT / f"lenovo_v5_r1_bglf_uv_{arm}_h{height}_mojito_ft100k_20260928/models/net_g_50000.pth"
    config = CONFIG_ROOT / f"train_{arm}_h{height}_ft100k_20260928.yml"
    output_root = RUN_ROOT / name
    for item in (checkpoint, config):
        if not item.is_file() or item.stat().st_size < 1024:
            record.update(state="failed", error=f"missing input: {item}")
            save_status(status)
            return False
    command = [str(PYTHON), "-u", str(REPO / "tools/render_lenovo_borderrail_ft50k_whiteboard33_20260929.py"),
               "--arm", arm, "--height", str(height), "--checkpoint", str(checkpoint),
               "--config", str(config), "--output-root", str(output_root), "--device", "cuda:0",
               "--routes", *ROUTES]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2")
    for attempt in range(1, 4):
        log_path = RUN_ROOT / f"{name}_attempt{attempt}.log"
        record.update(state="running", attempt=attempt, gpu=gpu, checkpoint=str(checkpoint),
                      config=str(config), output_root=str(output_root), log=str(log_path), started_at=now())
        save_status(status)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = process.pid
            save_status(status)
            return_code = process.wait()
        summary_path = output_root / "summary.json"
        if return_code == 0 and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = summary.get("rows", [])
            if summary.get("status") == "complete" and summary.get("completed") == 66 and len(rows) == 66:
                missing = [row["output"] for row in rows if not Path(row["output"]).is_file()]
                missing += [row["preview"] for row in rows if not Path(row["preview"]).is_file()]
                missing += [row["output_full_jpg"] for row in rows if not Path(row["output_full_jpg"]).is_file()]
                if not missing:
                    record.update(state="complete", finished_at=now(), completed=66, error=None)
                    save_status(status)
                    return True
                record["error"] = f"missing rendered artifacts: {missing[:3]}"
            else:
                record["error"] = "summary did not contain 66 complete results"
        else:
            record["error"] = f"renderer exited {return_code}"
        record.update(state="retry_wait", exit_code=return_code)
        save_status(status)
        time.sleep(min(20 * attempt, 60))
    record["state"] = "failed"
    save_status(status)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    if args.output_root != RUN_ROOT:
        raise ValueError("custom output root is not supported by this fixed experiment launcher")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = RUN_ROOT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {
        "created_at": now(), "stage": "inference", "jobs": {}}
    for arm in ARMS:
        for height in HEIGHTS:
            status["jobs"].setdefault(f"{arm}_h{height}", {"state": "pending"})
    save_status(status)

    def gpu_queue(gpu: int, arm: str) -> bool:
        return all(run_one(arm, height, gpu, status) for height in HEIGHTS)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(gpu_queue, gpu, arm) for gpu, arm in enumerate(ARMS)]
        passed = all(future.result() for future in as_completed(futures))
    status["stage"] = "complete" if passed else "failed"
    save_status(status)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
