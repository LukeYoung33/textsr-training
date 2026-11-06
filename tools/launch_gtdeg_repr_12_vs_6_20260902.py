"""Launch the controlled legacy-12 versus native-6 GT-degradation experiment.

Only the representation/model adapter changes between runs.  The source data,
fixed 96x480 canvas and all standard Real-ESRGAN two-stage degradation fields
are copied from the historical 20260831 GT-degradation YAML.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path("/home/guang.yang/workspace/Real-ESRGAN")
PYTHON = Path("/home/guang.yang/ViT_env/bin/python")
LEGACY_CONFIG = REPO / "options/train_bucket_3/train_basic_v2_gtdeg_real_finetune_96x480_20260831.yml"
V4_CONFIG = REPO / "options/train_bucket_3/train_basic_v2_y4uv_v4_gtdeg_real_finetune_96x480_cp45_20260902.yml"
RUN_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/unattended/gtdeg_repr_12_vs_6_20260902")
EXPERIMENT_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_legacy_12() -> dict:
    config = copy.deepcopy(_load(LEGACY_CONFIG))
    config["name"] = "gtdeg_repr12_legacy_10k_20260902"
    config["path"]["experiments_root"] = str(EXPERIMENT_ROOT)
    config["path"]["resume_state"] = None
    config["train"]["total_iter"] = 10_000
    config["datasets"]["train"]["batch_size_per_gpu"] = 4
    config["val"]["val_freq"] = 999_999
    config["logger"]["save_checkpoint_freq"] = 10_000
    config["dist_params"]["port"] = 29581
    return config


def make_native_6() -> dict:
    legacy = _load(LEGACY_CONFIG)
    config = copy.deepcopy(_load(V4_CONFIG))
    config["name"] = "gtdeg_repr6_y4uv_v4_10k_20260902"
    # Copy every top-level standard Real-ESRGAN degradation knob verbatim from
    # the historical GT-deg run; this is the controlled experimental variable.
    degradation_keys = (
        "high_order_degradation", "gt_usm", "queue_size", "resize_prob", "resize_range",
        "gaussian_noise_prob", "noise_range", "poisson_scale_range", "gray_noise_prob",
        "jpeg_range", "second_blur_prob", "resize_prob2", "resize_range2",
        "gaussian_noise_prob2", "noise_range2", "poisson_scale_range2", "gray_noise_prob2",
        "jpeg_range2",
    )
    for key in degradation_keys:
        config[key] = copy.deepcopy(legacy[key])
    config["manual_seed"] = legacy["manual_seed"]
    config["path"]["experiments_root"] = str(EXPERIMENT_ROOT)
    config["path"]["pretrain_network_g"] = None
    config["path"]["resume_state"] = None
    config["train"]["total_iter"] = 10_000
    config["train"]["cp_weight"] = legacy["train"]["cp_weight"]
    config["datasets"]["train"] = copy.deepcopy(legacy["datasets"]["train"])
    config["datasets"]["train"]["name"] = "gtdeg_repr6_y4uv_v4_train"
    config["datasets"]["train"]["batch_size_per_gpu"] = 4
    config["datasets"]["val"] = copy.deepcopy(legacy["datasets"]["val"])
    config["datasets"]["val"]["name"] = "gtdeg_repr6_y4uv_v4_val"
    config["val"]["val_freq"] = 999_999
    config["logger"]["save_checkpoint_freq"] = 10_000
    config["dist_params"]["port"] = 29582
    return config


def main() -> int:
    runs = (("legacy12", "0", make_legacy_12()), ("native6", "1", make_native_6()))
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, gpu, config in runs:
        run_dir = RUN_ROOT / label
        run_dir.mkdir(exist_ok=True)
        config_path = run_dir / "train.yml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        with (run_dir / "train.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(
                [str(PYTHON), "realesrgan/train.py", "-opt", str(config_path), "--launcher", "none"],
                cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
        (run_dir / "pid").write_text(f"{process.pid}\n", encoding="utf-8")
        manifest.append({"label": label, "gpu": gpu, "pid": process.pid, "config": str(config_path), "name": config["name"]})
    (RUN_ROOT / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
