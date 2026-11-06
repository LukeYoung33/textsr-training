"""Generate and launch concurrent V5 native-Y4UV V4 training variants."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path("/home/guang.yang/workspace/Real-ESRGAN")
BASE_CONFIG = REPO / "options/train_bucket_3/train_basic_text_line_y4uv_v4_cached_v5_cp_haar_20260901.yml"
PYTHON = Path("/home/guang.yang/ViT_env/bin/python")
RUN_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/unattended/basic_v5_y4uv_v4_20260901")

# Repeats only deepen the V4 volume-preserving backbone.  Input/output,
# Y4UV layout, masks, CP loss and the native crop dataset remain identical.
RECIPES = {
    "base": {"gpu": "0", "stage_repeat": 1, "bottleneck_repeat": 2, "batch": 16},
    "gtdeg": {"gpu": "0", "stage_repeat": 1, "bottleneck_repeat": 2, "batch": 16,
              "simulated_degradation": {"enabled": True, "downscale": 0.72, "blur_sigma": 1.15, "noise_std": 0.012}},
    "large10": {"gpu": "1", "stage_repeat": 8, "bottleneck_repeat": 16, "batch": 8},
    "large20": {"gpu": "2", "stage_repeat": 16, "bottleneck_repeat": 32, "batch": 4},
    "large50": {"gpu": "3", "stage_repeat": 40, "bottleneck_repeat": 80, "batch": 1},
}


def _config(key: str) -> dict:
    recipe = RECIPES[key]
    value = copy.deepcopy(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")))
    value["name"] = f"basic_v5_y4uv_v4_{key}_cp_20260901"
    value["network_g"]["stage_repeat"] = recipe["stage_repeat"]
    value["network_g"]["bottleneck_repeat"] = recipe["bottleneck_repeat"]
    for split in ("train", "val"):
        value["datasets"][split]["batch_size_per_gpu"] = recipe["batch"] if split == "train" else 1
        value["datasets"][split].pop("simulated_degradation", None)
    if "simulated_degradation" in recipe:
        value["datasets"]["train"]["simulated_degradation"] = recipe["simulated_degradation"]
    value["dist_params"]["port"] = 29561 + list(RECIPES).index(key)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", choices=sorted(RECIPES), default=list(RECIPES))
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not (Path("/home/guang.yang/data/basic_v5_0901_crop_v1/COMPLETE").is_file()):
        raise RuntimeError("V5 crop cache is not COMPLETE")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    launched = []
    for key in args.runs:
        recipe = RECIPES[key]
        run_dir = RUN_ROOT / key
        run_dir.mkdir(exist_ok=True)
        config_path = run_dir / "train.yml"
        config_path.write_text(yaml.safe_dump(_config(key), sort_keys=False), encoding="utf-8")
        record = {"key": key, "gpu": recipe["gpu"], "config": str(config_path), "recipe": recipe}
        if args.launch:
            log_path = run_dir / "train.log"
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [str(PYTHON), "realesrgan/train.py", "-opt", str(config_path), "--launcher", "none"],
                    cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": recipe["gpu"], "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,
                )
            (run_dir / "pid").write_text(f"{process.pid}\n", encoding="utf-8")
            record["pid"] = process.pid
        launched.append(record)
    (RUN_ROOT / "launch_manifest.json").write_text(json.dumps(launched, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(launched, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
