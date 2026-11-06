"""Validate the V4 c0/r1 capacity comparison contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from profile_v4_y4uv_capacity import PRESETS


ARMS = ("c0", "r1")
CAPACITIES = ("v5deploymatch", "10x", "50x", "100x")


def validate(path: Path, capacity: str, arm: str) -> list[str]:
    option = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    network = option.get("network_g", {})
    if network.get("type") != "BasicTextLineR4StudentY4UVV4Large":
        errors.append("wrong network type")
    for key, expected in PRESETS[capacity].items():
        if network.get(key) != expected:
            errors.append(f"network {key} mismatch")
    if option.get("model_type") != "LenovoTextLineDegradationModel":
        errors.append("wrong training model")
    if option.get("train", {}).get("total_iter") != 250000:
        errors.append("total_iter must be 250000")
    if option.get("val", {}).get("val_freq") != 25000:
        errors.append("validation cadence must be 25K")
    if option.get("datasets", {}).get("val", {}).get("type") != "BasicTextLineRemote58Dataset":
        errors.append("wrong fixed validation dataset")
    if arm == "c0" and option.get("train", {}).get("recall_aware_weight_missing") != 0.0:
        errors.append("c0 recall loss must be disabled")
    if arm == "r1" and option.get("train", {}).get("recall_aware_weight_missing") != 1.0:
        errors.append("r1 recall loss must be enabled")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True, type=Path)
    args = parser.parse_args()
    failures = {}
    for capacity in CAPACITIES:
        for arm in ARMS:
            path = args.config_dir / f"train_lenovo_v4cap_{capacity}_{arm}_250k_20260913.yml"
            if not path.is_file():
                failures[str(path)] = ["missing"]
            elif errors := validate(path, capacity, arm):
                failures[str(path)] = errors
    if failures:
        raise SystemExit(f"invalid V4 capacity comparison contract: {failures}")
    print("V4 capacity vs V5 static contract: PASS")


if __name__ == "__main__":
    main()
