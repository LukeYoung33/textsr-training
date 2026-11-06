"""Validate the Lenovo-degraded V5 eight-arm loss-comparison configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from validate_lenovo_v4_experiment_contract import ARMS, EXPECTED, FORBIDDEN, _keys


def validate(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if set(_keys(config)) & FORBIDDEN:
        errors.append("legacy Real-ESRGAN degradation keys found")
    if config.get("model_type") != "LenovoTextLineDegradationModel":
        errors.append("wrong model type")
    if config.get("network_g", {}).get("type") != "BasicTextLineR4StudentY4UVV5":
        errors.append("wrong V5 network type")
    if config.get("network_g", {}).get("stage_channels") != [8, 32, 128]:
        errors.append("wrong PhaseRepECB stage widths")
    if config.get("datasets", {}).get("train", {}).get("type") != "LenovoCleanTextLineDataset":
        errors.append("wrong clean-GT train dataset")
    train = config.get("train", {})
    if train.get("total_iter") != 250000:
        errors.append("total_iter must be 250000")
    if config.get("val", {}).get("val_freq") != 25000:
        errors.append("validation cadence must be 25K")
    arm = next((value for value in ARMS if f"_{value}_250k_" in path.name), None)
    if arm is None:
        return errors + ["unrecognized arm"]
    enabled = EXPECTED[arm]
    probability_enabled = bool(enabled.get("prob"))
    if config.get("network_g", {}).get("stroke_prob_enabled") is not probability_enabled:
        errors.append("V5 network stroke-probability switch mismatch")
    if train.get("stroke_prob_enabled") is not probability_enabled:
        errors.append("V5 stroke-probability training switch mismatch")
    if float(train.get("stroke_prob_weight", 0.0)) != (0.10 if probability_enabled else 0.0):
        errors.append("V5 stroke-probability auxiliary weight mismatch")
    if probability_enabled != (train.get("stroke_prob_opt") is not None):
        errors.append("V5 stroke-probability loss configuration mismatch")
    expected_weights = {
        "recall_aware_weight_missing": 1.0 if enabled.get("recall") else 0.0,
        "bg_lf_weight": 2.0 if enabled.get("bglf") else 0.0,
        "sdf_boundary_weight": 0.0856286676 if enabled.get("sdf") else 0.0,
        "stroke_core_weight": 0.05 if enabled.get("core") else 0.0,
        "stroke_overshoot_weight": 0.02 if enabled.get("overshoot") else 0.0,
        "stroke_width_weight": 0.02 if enabled.get("width") else 0.0,
        "cldice_weight": 0.02 if enabled.get("cldice") else 0.0,
    }
    for key, expected in expected_weights.items():
        if float(train.get(key, -1.0)) != expected:
            errors.append(f"{key} mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    files = [args.config_dir / f"train_lenovo_v5_{arm}_250k_20260912.yml" for arm in ARMS]
    missing = [str(path) for path in files if not path.is_file()]
    failures = {str(path): validate(path) for path in files if path.is_file() and validate(path)}
    if missing or failures:
        raise SystemExit(f"invalid Lenovo V5 contract; missing={missing}; failures={failures}")
    print("Lenovo V5 eight-arm static contract: PASS")


if __name__ == "__main__":
    main()
