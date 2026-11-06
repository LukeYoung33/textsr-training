"""Run the eight matched 250K validation visualizations across four GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/guang.yang/realSR/bin/python")
ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_pretrain250k_val_fullbox_complete_20260930")
ARMS = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
LOCK = Lock()


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    status_path = ROOT / "status.json"
    status = {"stage": "running", "parents_per_dataset": "all", "expected_parents_per_job": 206, "jobs": {}}

    def save() -> None:
        with LOCK:
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, status_path)

    def worker(gpu: int, arm: str) -> bool:
        passed = True
        for height in (96, 64):
            name = f"{arm}_h{height}"
            output_root = ROOT / name
            summary_path = output_root / "summary.json"
            if summary_path.is_file() and json.loads(summary_path.read_text(encoding="utf-8")).get("completed") == 206:
                status["jobs"][name] = {"state": "complete", "gpu": gpu, "completed": 206}
                save()
                continue
            command = [str(PYTHON), "-u", str(REPO / "tools/evaluate_lenovo_borderrail_pretrain_visuals_20260929.py"),
                       "--arm", arm, "--height", str(height), "--parents-per-dataset", "0",
                       "--output-root", str(output_root)]
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO), OMP_NUM_THREADS="2")
            log_path = ROOT / f"{name}.log"
            status["jobs"][name] = {"state": "running", "gpu": gpu, "log": str(log_path)}
            save()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            complete = result.returncode == 0 and summary_path.is_file()
            if complete:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                complete = summary.get("completed") == summary.get("expected") == 206
            status["jobs"][name] = {"state": "complete" if complete else "failed", "gpu": gpu,
                                    "exit_code": result.returncode, "completed": 206 if complete else None,
                                    "log": str(log_path), "output_root": str(output_root)}
            save()
            passed &= complete
        return passed

    save()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda item: worker(*item), enumerate(ARMS)))
    status["stage"] = "complete" if all(results) else "failed"
    save()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
