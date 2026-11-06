"""Materialize the V5 R1/RB probability-head combination matrix."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "options" / "train_bucket_3" / "lenovo_v5_cp_strong_20260912"
DEFAULT_OUTPUT = ROOT / "options" / "train_bucket_3" / "lenovo_v5_combination_round1_20260914"
EXPERIMENTS_ROOT = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_v5_combination_round1_20260914"

ARMS = {
    "g0_r1": {"base": "r1", "prob": False, "cldice": False, "core": False},
    "g1_rb": {"base": "rb", "prob": False, "cldice": False, "core": False},
    "g2_r1_p": {"base": "r1", "prob": True, "cldice": False, "core": False},
    "g3_rb_p": {"base": "rb", "prob": True, "cldice": False, "core": False},
    "g4_r1_p_cldice": {"base": "r1", "prob": True, "cldice": True, "core": False},
    "g5_rb_p_cldice": {"base": "rb", "prob": True, "cldice": True, "core": False},
    "g6_r1_p_core": {"base": "r1", "prob": True, "cldice": False, "core": True},
    "g7_rb_p_core": {"base": "rb", "prob": True, "cldice": False, "core": True},
}


def _load_source(base: str) -> dict:
    return yaml.safe_load((SOURCE / f"train_lenovo_v5_{base}_250k_20260912.yml").read_text(encoding="utf-8"))


def _configure(name: str, recipe: dict, port: int) -> dict:
    config = deepcopy(_load_source(recipe["base"]))
    train, network = config["train"], config["network_g"]
    config["name"] = f"lenovo_v5_{name}_250k_20260914"
    config["path"]["experiments_root"] = EXPERIMENTS_ROOT
    config["dist_params"]["port"] = int(port)
    for split in ("train", "val"):
        config["datasets"][split]["name"] = f"{config['name']}_{split}"
    network["stroke_prob_enabled"] = bool(recipe["prob"])
    train["stroke_prob_enabled"] = bool(recipe["prob"])
    train["stroke_prob_weight"] = 0.10 if recipe["prob"] else 0.0
    train["stroke_prob_opt"] = {
        "type": "StrokeProbabilityBCELoss", "mask_threshold": 0.5, "loss_weight": 1.0
    } if recipe["prob"] else None
    train["stroke_core_weight"] = 0.05 if recipe["core"] else 0.0
    train["stroke_core_opt"] = {
        "type": "StrokeCoreLoss", "max_skeleton_iterations": 96,
        "skeleton_threshold": 0.5, "loss_weight": 1.0,
    } if recipe["core"] else None
    train["cldice_weight"] = 0.02 if recipe["cldice"] else 0.0
    train["cldice_opt"] = {
        "type": "SoftCLDiceLoss", "alpha": 0.4, "skeleton_iterations": 5,
        "max_skeleton_iterations": 96, "tube_radius": 0, "anchor_radius": 3,
        "min_contrast": 0.01, "ink_temperature": 0.05, "loss_weight": 1.0,
    } if recipe["cldice"] else None
    for key in ("stroke_overshoot_weight", "stroke_width_weight", "sdf_boundary_weight"):
        train[key] = 0.0
    for key in ("stroke_overshoot_opt", "stroke_width_opt", "sdf_boundary_opt"):
        train[key] = None
    return config


def _validate(config: dict, recipe: dict) -> list[str]:
    train, network = config["train"], config["network_g"]
    expected_bglf = recipe["base"] == "rb"
    expected_recall = recipe["base"] in {"r1", "rb"}
    expected = {
        "recall_aware_weight_missing": 1.0 if expected_recall else 0.0,
        "recall_aware_weight_false": 0.25 if expected_recall else 0.0,
        "bg_lf_weight": 2.0 if expected_bglf else 0.0,
        "stroke_prob_weight": 0.10 if recipe["prob"] else 0.0,
        "stroke_core_weight": 0.05 if recipe["core"] else 0.0,
        "cldice_weight": 0.02 if recipe["cldice"] else 0.0,
        "stroke_overshoot_weight": 0.0,
        "stroke_width_weight": 0.0,
        "sdf_boundary_weight": 0.0,
    }
    errors = [f"{key}={train.get(key)} expected={value}" for key, value in expected.items() if float(train.get(key, -1.0)) != value]
    if bool(network.get("stroke_prob_enabled")) != bool(recipe["prob"]):
        errors.append("network probability switch mismatch")
    if bool(train.get("stroke_prob_enabled")) != bool(recipe["prob"]):
        errors.append("train probability switch mismatch")
    if recipe["prob"] != (train.get("stroke_prob_opt") is not None):
        errors.append("probability loss mismatch")
    return errors


def build(output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    failures: dict[str, list[str]] = {}
    for index, (name, recipe) in enumerate(ARMS.items()):
        config = _configure(name, recipe, 30220 + index)
        errors = _validate(config, recipe)
        if errors:
            failures[name] = errors
            continue
        path = output / f"train_lenovo_v5_{name}_250k_20260914.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(path)
    if failures:
        raise RuntimeError(f"invalid combination configs: {failures}")
    contract = {
        "contract": "Lenovo V5 round-1 R1/RB probability-head combination matrix",
        "arms": ARMS,
        "experiments_root": EXPERIMENTS_ROOT,
        "training": "from scratch, 250K, fixed Lenovo validation and post-training stroke/visual evaluation",
        "configs": [str(path) for path in paths],
    }
    (output / "combination_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print("\n".join(str(path) for path in build(args.output_dir)))


if __name__ == "__main__":
    main()
