"""Create and optionally launch three controlled V4 Real-ESRGAN degradation ablations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path("/home/guang.yang/workspace/Real-ESRGAN")
BASE_CONFIG = REPO / "options/train_bucket_3/train_basic_v2_y4uv_v4_gtdeg_real_finetune_96x480_cp45_20260902.yml"
PYTHON = Path("/home/guang.yang/ViT_env/bin/python")
RUN_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/unattended/v4_gtdeg_ablation_20260902")

# Only degradation strength varies.  Architecture, 50/50 roots, equal-weight
# base reconstruction, seed, optimizer and CP weight remain identical.
RECIPES = {
    "standard": {"gpu": "0", "updates": {}},
    "text_safe": {"gpu": "1", "updates": {
        "resize_prob": [0.1, 0.7, 0.2], "resize_range": [0.5, 1.2],
        "second_blur_prob": 0.4, "resize_prob2": [0.15, 0.55, 0.30],
        "resize_range2": [0.65, 1.1], "noise_range": [1, 12],
        "poisson_scale_range": [0.05, 1.5], "jpeg_range": [55, 95],
        "noise_range2": [1, 8], "poisson_scale_range2": [0.05, 1.2],
        "jpeg_range2": [60, 95], "sinc_prob": 0.05, "sinc_prob2": 0.05,
        "final_sinc_prob": 0.2, "blur_sigma": [0.2, 1.5], "blur_sigma2": [0.2, 0.8],
    }},
    "light": {"gpu": "2", "updates": {
        "resize_prob": [0.0, 0.55, 0.45], "resize_range": [0.75, 1.1],
        "second_blur_prob": 0.2, "resize_prob2": [0.05, 0.45, 0.50],
        "resize_range2": [0.8, 1.05], "noise_range": [1, 6],
        "poisson_scale_range": [0.05, 0.8], "jpeg_range": [70, 100],
        "noise_range2": [1, 4], "poisson_scale_range2": [0.05, 0.6],
        "jpeg_range2": [75, 100], "sinc_prob": 0.0, "sinc_prob2": 0.0,
        "final_sinc_prob": 0.0, "blur_sigma": [0.2, 0.9], "blur_sigma2": [0.2, 0.5],
    }},
}


def make_config(name: str, resume: bool) -> dict:
    recipe = RECIPES[name]
    config = copy.deepcopy(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")))
    config["name"] = f"v4_y4uv_gtdeg_{name}_20k_cp45_20260902"
    config.update(recipe["updates"])
    config["train"]["total_iter"] = 20_000
    # Dataset validation is intentionally disabled: the decision criterion is
    # the held-out HX first-eight inference set, not these paired val rows.
    config["val"]["val_freq"] = 999_999
    config["logger"]["save_checkpoint_freq"] = 10_000
    config["dist_params"]["port"] = 29571 + list(RECIPES).index(name)
    if resume:
        config["path"]["resume_state"] = str(
            Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments")
            / config["name"] / "training_states" / "10000.state"
        )
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    launched = []
    for name, recipe in RECIPES.items():
        run_dir = RUN_ROOT / name
        run_dir.mkdir(exist_ok=True)
        config_path = run_dir / "train.yml"
        config_path.write_text(yaml.safe_dump(make_config(name, args.resume), sort_keys=False), encoding="utf-8")
        record = {"name": name, "gpu": recipe["gpu"], "config": str(config_path)}
        if args.launch:
            with (run_dir / "train.log").open("ab", buffering=0) as log:
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
