"""Create matched from-scratch Y4UV-V4 SDF capacity-training configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from profile_v4_y4uv_capacity import PRESETS


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "options/train_bucket_3/train_v4_r1_recall_bglf_guard_v2only_250k_20260904.yml"
SDF_WEIGHTS = {"base": 0.0856286676, "10x": 0.0879214960, "20x": 0.0505949383, "50x": 0.0863648007}


def sdf_opt() -> dict:
    return {
        "type": "GTSDFBoundaryLoss", "radius": 3, "sdf_temperature": 1.0,
        "ink_temperature": 0.05, "anchor_radius": 3, "min_contrast": 0.03,
        "mask_threshold": 0.5, "profile_loss": "bce", "linear_boundary_weight": 0.0,
        "boundary_width": 1, "loss_weight": 1.0,
    }


def main() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    for index, label in enumerate(("base", "10x", "20x", "50x")):
        option = copy.deepcopy(base)
        name = f"v4_r1_guard_sdf_fromstart_{label}_250k_20260906"
        spec = PRESETS[label]
        option["name"] = name
        option["path"]["pretrain_network_g"] = None
        option["path"]["resume_state"] = None
        option["path"]["strict_load_g"] = False
        option["network_g"].update({"type": "BasicTextLineR4StudentY4UVV4Large"} | spec)
        option["train"]["sdf_boundary_weight"] = SDF_WEIGHTS[label]
        option["train"]["sdf_boundary_opt"] = sdf_opt()
        option["datasets"]["train"]["name"] = name + "_train"
        option["datasets"]["val"]["name"] = name + "_val"
        option["dist_params"]["port"] = 29740 + index
        path = ROOT / "options/train_bucket_3" / f"train_{name}.yml"
        path.write_text(yaml.safe_dump(option, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
