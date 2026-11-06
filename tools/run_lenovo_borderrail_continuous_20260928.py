"""Run eight border-rail trainings, then their eight whiteboard finetunes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/guang.yang/realSR/bin/python")
MAIN_CONFIGS = REPO / "options/train_bucket_3/lenovo_borderrail_8arm_20260928"
FINETUNE_CONFIGS = MAIN_CONFIGS / "finetune"
RUN_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_continuous_20260928")
ARMS = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
HEIGHTS = (96, 64)
MAX_ATTEMPTS = 4
STATUS_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_status(status: dict) -> None:
    with STATUS_LOCK:
        status["updated_at"] = now()
        destination = RUN_ROOT / "status.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)


def job_specs(stage: str) -> list[dict]:
    specs = []
    for gpu, arm in enumerate(ARMS):
        for height in HEIGHTS:
            suffix = "250k" if stage == "main" else "ft100k"
            directory = MAIN_CONFIGS if stage == "main" else FINETUNE_CONFIGS
            config_path = directory / f"train_{arm}_h{height}_{suffix}_20260928.yml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            experiment = Path(config["path"]["experiments_root"]) / config["name"]
            specs.append({"id": f"{stage}_{arm}_h{height}", "stage": stage, "gpu": gpu,
                          "config": str(config_path), "experiment": str(experiment),
                          "total_iter": int(config["train"]["total_iter"]),
                          "pretrain": config["path"]["pretrain_network_g"]})
    return specs


def checkpoint_valid(spec: dict) -> bool:
    root = Path(spec["experiment"])
    latest = root / "models/net_g_latest.pth"
    numbered = root / f"models/net_g_{spec['total_iter']}.pth"
    state = root / f"training_states/{spec['total_iter']}.state"
    return all(path.is_file() and path.stat().st_size > 1024 for path in (latest, numbered, state))


def newest_resume(spec: dict) -> Path | None:
    root = Path(spec["experiment"])
    candidates = []
    for state in (root / "training_states").glob("*.state"):
        if not state.stem.isdecimal():
            continue
        iteration = int(state.stem)
        model = root / f"models/net_g_{iteration}.pth"
        if iteration < spec["total_iter"] and state.stat().st_size > 1024 and model.is_file() and model.stat().st_size > 1024:
            candidates.append((iteration, state))
    return max(candidates)[1] if candidates else None


def run_job(spec: dict, status: dict) -> bool:
    name = spec["id"]
    record = status["jobs"][name]
    if checkpoint_valid(spec):
        record.update(state="complete", completed_at=now())
        save_status(status)
        return True
    pretrain = Path(spec["pretrain"])
    if not pretrain.is_file() or pretrain.stat().st_size <= 1024:
        record.update(state="failed", error=f"pretrain checkpoint unavailable: {pretrain}")
        save_status(status)
        return False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        resume = newest_resume(spec)
        config_path = Path(spec["config"])
        if resume is not None:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["path"]["resume_state"] = str(resume)
            config_path = RUN_ROOT / f"{name}_resume_{attempt}.yml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        log_path = RUN_ROOT / f"{name}_attempt{attempt}.log"
        command = [str(PYTHON), "-u", "realesrgan/train.py", "-opt", str(config_path), "--launcher", "none"]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(spec["gpu"]), PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1",
                   REAL_ESRGAN_DEFER_FINAL_VALIDATION="1", OMP_NUM_THREADS="2")
        record.update(state="running", attempt=attempt, resume_state=str(resume) if resume else None,
                      started_at=now(), log=str(log_path), gpu=spec["gpu"])
        save_status(status)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"COMMAND: {' '.join(command)}\nGPU: {spec['gpu']}\nRESUME: {resume}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = process.pid
            save_status(status)
            code = process.wait()
        record["exit_code"] = code
        if code == 0 and checkpoint_valid(spec):
            record.update(state="complete", completed_at=now())
            save_status(status)
            return True
        record.update(state="retry_wait", last_error=f"exit={code}, complete_checkpoint={checkpoint_valid(spec)}")
        save_status(status)
        time.sleep(min(60 * attempt, 180))
    record.update(state="failed", error=f"{MAX_ATTEMPTS} attempts exhausted")
    save_status(status)
    return False


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / "supervisor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A supervisor is already running", flush=True)
            return 2
        specs = job_specs("main") + job_specs("finetune")
        status_path = RUN_ROOT / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            status = {"created_at": now(), "stage": "main", "jobs": {}}
        for spec in specs:
            status["jobs"].setdefault(spec["id"], {"state": "pending", "experiment": spec["experiment"],
                                                      "config": spec["config"], "gpu": spec["gpu"]})
        save_status(status)
        for stage in ("main", "finetune"):
            status["stage"] = stage
            save_status(status)
            stage_specs = [spec for spec in specs if spec["stage"] == stage]
            if stage == "finetune" and not all(checkpoint_valid(spec) for spec in specs if spec["stage"] == "main"):
                status["stage"] = "blocked_main_checkpoint"
                save_status(status)
                return 1
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = [pool.submit(run_job, spec, status) for spec in stage_specs]
                success = all(future.result() for future in as_completed(results))
            if not success:
                status["stage"] = f"failed_{stage}"
                save_status(status)
                return 1
        status["stage"] = "complete"
        save_status(status)
        return 0


if __name__ == "__main__":
    sys.exit(main())
