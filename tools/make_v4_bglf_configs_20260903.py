"""Materialize the paired C0+bgLF and R1+bgLF formal YAMLs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OPTIONS = ROOT / "options" / "train_bucket_3"
LEGACY = OPTIONS / "train_basic_v2_gtdeg_real_finetune_96x480_20260831.yml"
V4 = OPTIONS / "train_basic_v2_y4uv_v4_gtdeg_real_finetune_96x480_cp45_20260902.yml"
EXPERIMENTS = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments"
DATA_ROOT = "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets"


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping YAML: {path}")
    return value


def _bg(base: dict, *, name: str, weight: float, port: int) -> dict:
    config = copy.deepcopy(base)
    config["name"] = name
    config["train"]["bg_lf_weight"] = float(weight)
    config["train"]["bg_lf_opt"] = {
        "type": "BackgroundLowFrequencyLoss",
        "radius": 7,
        "sigma": 4.0,
        "channel": "luma",
        "threshold": 0.5,
    }
    config["dist_params"]["port"] = int(port)
    return config


def make_c0() -> dict:
    legacy, v4 = _load(LEGACY), _load(V4)
    config = copy.deepcopy(v4)
    config["name"] = "v4_gtdeg_c0_cp45_bglf_250k_20260903"
    for key in (
        "high_order_degradation", "gt_usm", "queue_size", "resize_prob", "resize_range",
        "gaussian_noise_prob", "noise_range", "poisson_scale_range", "gray_noise_prob", "jpeg_range",
        "second_blur_prob", "resize_prob2", "resize_range2", "gaussian_noise_prob2", "noise_range2",
        "poisson_scale_range2", "gray_noise_prob2", "jpeg_range2",
    ):
        config[key] = copy.deepcopy(legacy[key])
    config["manual_seed"] = legacy["manual_seed"]
    config["datasets"] = copy.deepcopy(legacy["datasets"])
    config["datasets"]["train"]["name"] = config["name"] + "_train"
    config["datasets"]["val"]["name"] = config["name"] + "_val"
    config["datasets"]["train"]["batch_size_per_gpu"] = 4
    config["path"] = {
        "pretrain_network_g": None,
        "resume_state": None,
        "strict_load_g": False,
        "experiments_root": EXPERIMENTS,
    }
    config["train"]["total_iter"] = 250000
    config["train"]["cp_weight"] = 1.8e-4
    config["val"] = {"val_freq": 25000, "save_img": False, "metrics": {}}
    config["logger"]["save_checkpoint_freq"] = 25000
    return _bg(config, name=config["name"], weight=1.0, port=29701)


def make_r1() -> dict:
    config = _load(V4)
    config["name"] = "v4_r1_recall_bglf_v2only_250k_20260903"
    config["manual_seed"] = 20260828
    config["path"] = {
        "pretrain_network_g": None,
        "resume_state": None,
        "strict_load_g": False,
        "experiments_root": EXPERIMENTS,
    }
    config["train"]["total_iter"] = 250000
    config["train"]["cp_weight"] = 5e-4
    config["train"].update({
        "recall_aware_weight_missing": 1.0,
        "recall_aware_weight_false": 0.25,
        "recall_aware_opt": {
            "ring_radius": 3,
            "margin": 0.03,
            "temperature": 0.01,
            "min_polarity_contrast": 0.03,
        },
    })
    v2 = {"root": f"{DATA_ROOT}/real_finetune_v2", "weight": 1.0}
    config["datasets"]["train"].update({
        "name": config["name"] + "_train",
        "sources": [v2],
        "batch_size_per_gpu": 4,
    })
    config["datasets"]["val"].update({"name": config["name"] + "_val", "sources": [v2]})
    config["val"] = {"val_freq": 25000, "save_img": False, "metrics": {}}
    config["logger"]["save_checkpoint_freq"] = 25000
    return _bg(config, name=config["name"], weight=7.0, port=29702)


def make_r1_calibrated() -> dict:
    """R1+bgLF restart after live-log calibration of the initial lambda=7 run."""
    config = make_r1()
    name = "v4_r1_recall_bglf_v2only_250k_20260903_lambda3"
    config["name"] = name
    config["datasets"]["train"]["name"] = name + "_train"
    config["datasets"]["val"]["name"] = name + "_val"
    config["train"]["bg_lf_weight"] = 3.0
    config["dist_params"]["port"] = 29703
    return config


def main() -> None:
    outputs = {
        "train_v4_gtdeg_c0_cp45_bglf_250k_20260903.yml": make_c0(),
        "train_v4_r1_recall_bglf_v2only_250k_20260903.yml": make_r1(),
        "train_v4_r1_recall_bglf_v2only_250k_20260903_lambda3.yml": make_r1_calibrated(),
    }
    for filename, config in outputs.items():
        (OPTIONS / filename).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(OPTIONS / filename)


if __name__ == "__main__":
    main()
