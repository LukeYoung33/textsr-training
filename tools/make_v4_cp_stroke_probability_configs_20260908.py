"""Generate the eight locked V4 CP-baseline ablation configs."""
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "options/train_bucket_3/train_v4_r1_recall_bglf_guard_v2only_250k_20260904.yml"
OUT = ROOT / "options/train_bucket_3"
STAMP = "20260908"

ARMS = {
    "c0": {},
    "r1": {"recall": True},
    "r1_bglf": {"recall": True, "bglf": True},
    "sdf": {"sdf": True},
    "core": {"prob": True, "core": True},
    "overshoot": {"prob": True, "overshoot": True},
    "width": {"prob": True, "width": True},
    "cldice": {"prob": True, "cldice": True},
}


def main():
    base = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    for index, (arm, enabled) in enumerate(ARMS.items()):
        cfg = deepcopy(base)
        name = f"v4_cp_{arm}_probhead_fromstart_v2only_250k_{STAMP}"
        cfg["name"] = name
        cfg["network_g"]["stroke_prob_enabled"] = bool(enabled.get("prob", False))
        train = cfg["train"]
        train["aux_start_iter"] = 2000
        train["aux_ramp_iters"] = 5000
        train["recall_aware_weight_missing"] = 1.0 if enabled.get("recall") else 0.0
        train["recall_aware_weight_false"] = 0.25 if enabled.get("recall") else 0.0
        train["bg_lf_weight"] = 2.0 if enabled.get("bglf") else 0.0
        train["bg_lf_opt"] = train["bg_lf_opt"] if enabled.get("bglf") else None
        train["sdf_boundary_weight"] = 0.0856286676 if enabled.get("sdf") else 0.0
        train["sdf_boundary_opt"] = {
            "type": "GTSDFBoundaryLoss", "radius": 3, "sdf_temperature": 1.0, "ink_temperature": 0.05,
            "anchor_radius": 3, "min_contrast": 0.03, "mask_threshold": 0.5, "profile_loss": "bce",
            "linear_boundary_weight": 0.0, "boundary_width": 1, "loss_weight": 1.0,
        } if enabled.get("sdf") else None
        train["stroke_prob_enabled"] = bool(enabled.get("prob", False))
        train["stroke_prob_weight"] = 0.10 if enabled.get("prob") else 0.0
        train["stroke_prob_opt"] = {"type": "StrokeProbabilityBCELoss", "mask_threshold": 0.5, "loss_weight": 1.0} if enabled.get("prob") else None
        train["stroke_core_weight"] = 0.05 if enabled.get("core") else 0.0
        train["stroke_overshoot_weight"] = 0.02 if enabled.get("overshoot") else 0.0
        train["cldice_weight"] = 0.02 if enabled.get("cldice") else 0.0
        train["stroke_width_weight"] = 0.02 if enabled.get("width") else 0.0
        train["skel_fbeta_weight"] = 0.0
        train["skel_fbeta_opt"] = None
        train["stroke_core_opt"] = {"type": "StrokeCoreLoss", "max_skeleton_iterations": 96, "skeleton_threshold": 0.5, "loss_weight": 1.0} if enabled.get("core") else None
        train["stroke_overshoot_opt"] = {"type": "OvershootLoss", "kernel_size": 5, "tol": 2.0 / 255.0, "mask_radius": 2, "mask_mode": "text_dilate", "channel": "luma", "loss_weight": 1.0} if enabled.get("overshoot") else None
        train["cldice_opt"] = {"type": "SoftCLDiceLoss", "alpha": 0.4, "skeleton_iterations": 5, "max_skeleton_iterations": 96, "tube_radius": 0, "anchor_radius": 3, "min_contrast": 0.01, "ink_temperature": 0.05, "loss_weight": 1.0} if enabled.get("cldice") else None
        train["stroke_width_opt"] = {"type": "StrokeWidthLoss", "max_radius": 8, "anchor_radius": 3, "min_contrast": 0.01, "ink_temperature": 0.05, "max_skeleton_iterations": 96, "tube_radius": 0, "loss_weight": 1.0} if enabled.get("width") else None
        cfg["dist_params"] = {"backend": "nccl", "port": 29920 + index}
        (OUT / f"train_{name}.yml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
