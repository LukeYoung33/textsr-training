"""Static contract validation for the Lenovo V4 eight-arm experiment configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


FORBIDDEN = {"high_order_degradation", "gt_usm", "queue_size", "resize_prob", "resize_range",
             "gaussian_noise_prob", "noise_range", "poisson_scale_range", "gray_noise_prob", "jpeg_range",
             "second_blur_prob", "resize_prob2", "resize_range2", "gaussian_noise_prob2", "noise_range2",
             "poisson_scale_range2", "gray_noise_prob2", "jpeg_range2", "blur_kernel_size", "kernel_list",
             "kernel_prob", "sinc_prob", "blur_sigma", "betag_range", "betap_range", "blur_kernel_size2",
             "kernel_list2", "kernel_prob2", "sinc_prob2", "blur_sigma2", "betag_range2", "betap_range2",
             "final_sinc_prob", "kernel1", "kernel2", "sinc_kernel"}
ARMS = ("c0", "r1", "rb", "s1", "core", "over", "width", "cldice")
EXPECTED = {
    "c0": {}, "r1": {"recall": True}, "rb": {"recall": True, "bglf": True}, "s1": {"sdf": True},
    "core": {"prob": True, "core": True}, "over": {"prob": True, "overshoot": True},
    "width": {"prob": True, "width": True}, "cldice": {"prob": True, "cldice": True},
}


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def validate(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = []
    forbidden = sorted(set(_keys(config)) & FORBIDDEN)
    if forbidden: errors.append("legacy degradation keys: " + ", ".join(forbidden))
    if config.get("model_type") != "LenovoTextLineDegradationModel": errors.append("wrong model_type")
    if config.get("train", {}).get("model_type") != "LenovoTextLineDegradationModel": errors.append("wrong train.model_type")
    if config.get("datasets", {}).get("train", {}).get("type") != "LenovoCleanTextLineDataset": errors.append("wrong clean train dataset")
    degradation = config.get("lenovo_degradation", {})
    composition = degradation.get("composition", {})
    capture = composition.get("capture_weights", {})
    if set(capture) != {"c1", "c2", "c1_c2", "zoom_c2"}: errors.append("incomplete capture composition")
    elif abs(sum(float(value) for value in capture.values()) - 1.0) > 1e-9: errors.append("capture weights do not sum to one")
    post = composition.get("post_probabilities", {})
    if set(post) != {"c3", "c4", "c5", "c7"} or any(not 0 <= float(value) <= 1 for value in post.values()): errors.append("invalid post composition")
    if int(composition.get("max_post_branches", -1)) != 2: errors.append("post composition must allow up to two branches")
    if not 0 <= float(composition.get("secondary_capture_probability", -1)) <= 1: errors.append("invalid C8 probability")
    if float(degradation.get("c6", {}).get("weight", -1)) != 0.0 or degradation.get("c6", {}).get("sampleable") is not False: errors.append("C6 must be pinned at zero")
    if config.get("train", {}).get("total_iter") != 250000: errors.append("total_iter must be 250000")
    if config.get("val", {}).get("val_freq") != 25000 or config.get("logger", {}).get("save_checkpoint_freq") != 25000: errors.append("25K monitoring cadence missing")
    arm = next((value for value in ARMS if f"_{value}_250k_" in path.name), None)
    if arm is None:
        errors.append("unrecognized arm name")
        return errors
    expected = EXPECTED[arm]
    train = config["train"]
    enabled_prob = bool(expected.get("prob"))
    if config["network_g"].get("stroke_prob_enabled") != enabled_prob or train.get("stroke_prob_enabled") != enabled_prob:
        errors.append("probability-head mismatch")
    if float(train.get("stroke_prob_weight", -1)) != (0.10 if enabled_prob else 0.0): errors.append("probability BCE mismatch")
    checks = {"recall": ("recall_aware_weight_missing", 1.0), "bglf": ("bg_lf_weight", 2.0),
              "sdf": ("sdf_boundary_weight", 0.0856286676), "core": ("stroke_core_weight", 0.05),
              "overshoot": ("stroke_overshoot_weight", 0.02), "width": ("stroke_width_weight", 0.02),
              "cldice": ("cldice_weight", 0.02)}
    for feature, (key, on_value) in checks.items():
        if float(train.get(key, -1)) != (on_value if expected.get(feature) else 0.0):
            errors.append(f"{key} mismatch")
    if float(train.get("recall_aware_weight_false", -1)) != (0.25 if expected.get("recall") else 0.0):
        errors.append("recall_aware_weight_false mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for arm in ARMS:
        matches = sorted(args.config_dir.glob(f"train_lenovo_v4_{arm}_250k_*.yml"))
        if len(matches) == 1:
            paths.append(matches[0])
        elif not matches:
            paths.append(args.config_dir / f"train_lenovo_v4_{arm}_250k_MISSING.yml")
        else:
            raise SystemExit(f"ambiguous config for arm {arm}: {matches}")
    missing = [str(path) for path in paths if not path.is_file()]
    detail = {str(path): validate(path) for path in paths if path.is_file()}
    failures = {path: errors for path, errors in detail.items() if errors}
    ports = [yaml.safe_load(path.read_text(encoding="utf-8"))["dist_params"]["port"] for path in paths if path.is_file()]
    if len(ports) != len(set(ports)):
        failures["ports"] = ["NCCL ports must be unique"]
    if missing or failures:
        detail = {path: errors for path, errors in detail.items() if errors}
        raise SystemExit("invalid Lenovo contract; missing=" + repr(missing) + "; errors=" + repr(detail))
    print("Lenovo V4 eight-arm static contract: PASS (composed degradation runtime smoke pending)")


if __name__ == "__main__":
    main()
