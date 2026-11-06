"""Watch a training directory and dump every 5K checkpoint on HX text lines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _stable(path: Path, wait_seconds: float = 2.0) -> bool:
    """Avoid reading a checkpoint while the trainer is still writing it."""
    try:
        first = (path.stat().st_size, path.stat().st_mtime_ns)
        time.sleep(wait_seconds)
        second = (path.stat().st_size, path.stat().st_mtime_ns)
        return first == second and first[0] > 0
    except FileNotFoundError:
        return False


def _write_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def watch(args) -> int:
    experiment_root = args.experiment_root
    models = experiment_root / "models"
    output_root = experiment_root / "visualizations"
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = experiment_root / "hx_dump_watcher_status.json"
    dump_script = args.dump_script
    status = {
        "experiment_root": str(experiment_root),
        "manifest": str(args.manifest),
        "max_iter": int(args.max_iter),
        "checkpoint_every": int(args.checkpoint_every),
        "device": args.device,
        "completed": [],
        "failures": {},
        "state": "watching",
    }
    _write_status(status_path, status)

    while len(status["completed"]) < int(args.max_iter // args.checkpoint_every):
        progress = False
        for iteration in range(args.checkpoint_every, args.max_iter + 1, args.checkpoint_every):
            if iteration in status["completed"]:
                continue
            checkpoint = models / f"net_g_{iteration}.pth"
            if not checkpoint.is_file() or not _stable(checkpoint, args.stable_wait):
                continue
            summary = output_root / f"iter_{iteration:09d}" / "summary.json"
            if summary.is_file():
                status["completed"].append(iteration)
                progress = True
                _write_status(status_path, status)
                continue

            command = [
                sys.executable,
                str(dump_script),
                "--manifest", str(args.manifest),
                "--manifest-summary", str(args.manifest_summary) if args.manifest_summary else "",
                "--checkpoint", str(checkpoint),
                "--output-root", str(output_root),
                "--iteration", str(iteration),
                "--device", args.device,
            ]
            if not args.manifest_summary:
                command = [item for item in command if item != "--manifest-summary" and item != ""]
            try:
                subprocess.run(command, cwd=str(args.repo_root), check=True)
            except Exception:
                status["failures"][str(iteration)] = traceback.format_exc()
                _write_status(status_path, status)
                continue
            status["completed"].append(iteration)
            status["completed"].sort()
            progress = True
            _write_status(status_path, status)

        if len(status["completed"]) >= int(args.max_iter // args.checkpoint_every):
            break
        if not progress:
            time.sleep(args.poll_seconds)

    status["state"] = "complete"
    _write_status(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, default=None)
    parser.add_argument("--dump-script", type=Path, default=repo_root / "tools" / "dump_hx_basic_text_lines_v3.py")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iter", type=int, default=250000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stable-wait", type=float, default=2.0)
    args = parser.parse_args(argv)
    return watch(args)


if __name__ == "__main__":
    raise SystemExit(main())
