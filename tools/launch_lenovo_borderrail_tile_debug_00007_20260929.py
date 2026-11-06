"""Launch two gtfit_v6 tile-debug renders for the 00007 whiteboard image."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/guang.yang/realSR/bin/python")
RUN_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929/debug_00007_gtfit_v6_shapealigned_20260929")
EXPERIMENT_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_8arm_mojito_finetune_20260928")
CONFIG_ROOT = REPO / "options/train_bucket_3/lenovo_borderrail_8arm_20260928/finetune"


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for gpu, height in enumerate((96, 64)):
        checkpoint = EXPERIMENT_ROOT / f"lenovo_v5_r1_bglf_uv_gtfit_v6_borderrail_h{height}_mojito_ft100k_20260928/models/net_g_50000.pth"
        config = CONFIG_ROOT / f"train_gtfit_v6_borderrail_h{height}_ft100k_20260928.yml"
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(checkpoint if not checkpoint.is_file() else config)
        output_root = RUN_ROOT / f"h{height}"
        log_path = RUN_ROOT / f"h{height}.log"
        command = [str(PYTHON), "-u", str(REPO / "tools/render_lenovo_borderrail_ft50k_whiteboard33_20260929.py"),
                   "--arm", "gtfit_v6_borderrail", "--height", str(height),
                   "--checkpoint", str(checkpoint), "--config", str(config),
                   "--output-root", str(output_root), "--device", "cuda:0",
                   "--sample-prefix", "mojito_4160_00007_ss_IMG_20260611_061821228",
                   "--routes", "large_tile", "overlap_tile", "--tile-debug"]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        records.append({"height": height, "gpu": gpu, "pid": process.pid, "output_root": str(output_root), "log": str(log_path)})
    (RUN_ROOT / "launcher_status.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
