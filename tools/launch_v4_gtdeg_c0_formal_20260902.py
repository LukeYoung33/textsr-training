"""Freeze and launch the two six-channel V4 CP-only formal GT-deg arms."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path("/home/guang.yang/workspace/Real-ESRGAN")
PYTHON = Path("/home/guang.yang/ViT_env/bin/python")
LEGACY = REPO / "options/train_bucket_3/train_basic_v2_gtdeg_real_finetune_96x480_20260831.yml"
V4 = REPO / "options/train_bucket_3/train_basic_v2_y4uv_v4_gtdeg_real_finetune_96x480_cp45_20260902.yml"
RUN_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/formal/v4_gtdeg_c0_20260902")
EXPERIMENT_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments")


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_for(name: str, cp_weight: float, port: int) -> dict:
    legacy, config = load(LEGACY), copy.deepcopy(load(V4))
    config["name"] = name
    # This exactly adopts the legacy 12-channel GT-deg data and degradation
    # contract; only the generator representation and loss adapter are native
    # six-channel Y4UV V4.
    for key in ("high_order_degradation", "gt_usm", "queue_size", "resize_prob", "resize_range",
                "gaussian_noise_prob", "noise_range", "poisson_scale_range", "gray_noise_prob", "jpeg_range",
                "second_blur_prob", "resize_prob2", "resize_range2", "gaussian_noise_prob2", "noise_range2",
                "poisson_scale_range2", "gray_noise_prob2", "jpeg_range2"):
        config[key] = copy.deepcopy(legacy[key])
    config["manual_seed"] = legacy["manual_seed"]
    config["datasets"] = copy.deepcopy(legacy["datasets"])
    config["datasets"]["train"]["name"] = f"{name}_train"
    config["datasets"]["val"]["name"] = f"{name}_val"
    config["datasets"]["train"]["batch_size_per_gpu"] = 4
    config["path"] = {"pretrain_network_g": None, "resume_state": None, "strict_load_g": False,
                      "experiments_root": str(EXPERIMENT_ROOT)}
    config["train"]["total_iter"] = 250_000
    config["train"]["cp_weight"] = float(cp_weight)
    config["val"] = {"val_freq": 25_000, "save_img": False, "metrics": {}}
    config["logger"]["save_checkpoint_freq"] = 25_000
    config["dist_params"]["port"] = int(port)
    return config


def main() -> int:
    arms = (
        ("c0_legacycp", "0", config_for("v4_gtdeg_c0_legacycp5e4_250k_20260902", 5e-4, 29591)),
        ("c0_cp45", "1", config_for("v4_gtdeg_c0_cp45_250k_20260902", 1.8e-4, 29592)),
    )
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, gpu, config in arms:
        run_dir = RUN_ROOT / label
        run_dir.mkdir(exist_ok=True)
        path = run_dir / "train.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        with (run_dir / "train.log").open("xb") as log:
            process = subprocess.Popen([str(PYTHON), "realesrgan/train.py", "-opt", str(path), "--launcher", "none"],
                cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
        (run_dir / "pid").write_text(f"{process.pid}\n", encoding="utf-8")
        manifest.append({"arm": label, "gpu": gpu, "pid": process.pid, "config": str(path), "experiment": config["name"]})
    (RUN_ROOT / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
