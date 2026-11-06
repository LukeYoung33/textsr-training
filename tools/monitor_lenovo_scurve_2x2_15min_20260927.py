"""Check the four Lenovo 2x2 training jobs every 15 minutes for two hours."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


RUNS = {"scurve_h96": 0, "scurve_h64": 1, "raw_h96": 2, "raw_h64": 3}
ITER_RE = re.compile(r"epoch:\s*\d+, iter:\s*([\d,]+), lr:")
LOSS_RE = re.compile(r"l_total:\s*([^\s]+)")
ERROR_RE = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:")


def gpu_state() -> dict[int, dict[str, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    states = {}
    for line in result.stdout.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        states[index] = {"memory_mib": memory, "utilization_pct": utilization}
    return states


def inspect_run(root: Path, tag: str, gpu: int, previous_iter: int | None,
                previous_log_bytes: int, gpus: dict[int, dict[str, int]]) -> dict:
    launch = root / "launch_logs"
    pid = int((launch / f"{tag}.pid").read_text().strip())
    cmdline = Path(f"/proc/{pid}/cmdline")
    alive = cmdline.is_file() and f"train_{tag}_250k_20260927.yml" in cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    log_path = launch / f"{tag}.log"
    log_bytes = log_path.read_bytes()
    text = log_bytes.decode("utf-8", errors="replace")
    iterations = ITER_RE.findall(text)
    iteration = int(iterations[-1].replace(",", "")) if iterations else None
    train_lines = [line for line in text.splitlines() if ITER_RE.search(line)]
    loss_match = LOSS_RE.search(train_lines[-1]) if train_lines else None
    loss = float(loss_match.group(1)) if loss_match else None
    new_text = log_bytes[min(previous_log_bytes, len(log_bytes)):].decode("utf-8", errors="replace")
    errors = ERROR_RE.findall(new_text)
    status = "healthy"
    if not alive:
        status = "dead"
    elif errors:
        status = "error"
    elif loss is not None and not math.isfinite(loss):
        status = "nonfinite_loss"
    elif iteration is None or (previous_iter is not None and iteration <= previous_iter):
        status = "stalled"
    elif gpu not in gpus:
        status = "gpu_missing"
    return {
        "tag": tag, "gpu": gpu, "pid": pid, "alive": alive,
        "iteration": iteration, "iteration_delta": None if previous_iter is None or iteration is None else iteration - previous_iter,
        "loss_total": loss, "log_bytes": len(log_bytes), "new_error_count": len(errors),
        "gpu_state": gpus.get(gpu), "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--duration-seconds", type=int, default=7200)
    parser.add_argument("--output-prefix", default="monitor_15min_2h")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.duration_seconds < 0 or args.duration_seconds % args.interval_seconds:
        parser.error("duration must be a nonnegative multiple of the positive interval")
    output = args.root / "launch_logs" / f"{args.output_prefix}.jsonl"
    summary_path = args.root / "launch_logs" / f"{args.output_prefix}_summary.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"monitor output already exists: {output}")
    previous_iterations: dict[str, int | None] = {tag: None for tag in RUNS}
    previous_bytes = {tag: 0 for tag in RUNS}
    all_healthy = True
    started = time.monotonic()
    checks = args.duration_seconds // args.interval_seconds + 1
    with output.open("x", encoding="utf-8") as handle:
        for check_index in range(checks):
            time.sleep(max(0, started + check_index * args.interval_seconds - time.monotonic()))
            try:
                gpus = gpu_state()
                rows = [inspect_run(args.root, tag, gpu, previous_iterations[tag], previous_bytes[tag], gpus)
                        for tag, gpu in RUNS.items()]
                for row in rows:
                    previous_iterations[row["tag"]] = row["iteration"]
                    previous_bytes[row["tag"]] = row["log_bytes"]
                all_healthy &= all(row["status"] == "healthy" for row in rows)
                record = {"check": check_index, "time": datetime.now().astimezone().isoformat(), "runs": rows}
            except Exception as exc:
                all_healthy = False
                record = {"check": check_index, "time": datetime.now().astimezone().isoformat(),
                          "monitor_error": f"{type(exc).__name__}: {exc}"}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {"completed_at": datetime.now().astimezone().isoformat(), "checks": checks,
               "interval_seconds": args.interval_seconds, "duration_seconds": args.duration_seconds,
               "all_checks_healthy": all_healthy, "records": str(output)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
