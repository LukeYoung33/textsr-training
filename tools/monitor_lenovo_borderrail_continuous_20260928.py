"""Persist two hours of quarter-hour training progress snapshots."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_continuous_20260928")
OUTPUT = ROOT / "two_hour_monitor_20260928.jsonl"
INTERVAL_SECONDS = 900
CHECKS = 9  # Start plus eight quarter-hour reports.


def latest_iteration(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    with log_path.open("rb") as handle:
        handle.seek(max(0, log_path.stat().st_size - 65536))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = re.findall(r"iter:\s*([\d,]+),\s*lr:", tail)
    return int(matches[-1].replace(",", "")) if matches else None


def snapshot() -> dict:
    status = json.loads((ROOT / "status.json").read_text(encoding="utf-8"))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    jobs = {}
    for name, record in status["jobs"].items():
        pid = record.get("pid")
        alive = bool(pid and Path(f"/proc/{pid}").exists())
        log_path = Path(record["log"]) if record.get("log") else None
        jobs[name] = {"state": record["state"], "attempt": record.get("attempt"), "pid": pid,
                      "pid_alive": alive, "iter": latest_iteration(log_path) if log_path else None,
                      "log": str(log_path) if log_path else None,
                      "last_error": record.get("last_error") if record["state"] in ("failed", "retry_wait") else None}
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage": status["stage"], "gpu": gpu.stdout.splitlines(), "jobs": jobs}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    next_check = time.monotonic()
    for _ in range(CHECKS):
        while time.monotonic() < next_check:
            time.sleep(min(60, next_check - time.monotonic()))
        report = snapshot()
        with OUTPUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        next_check += INTERVAL_SECONDS


if __name__ == "__main__":
    main()
