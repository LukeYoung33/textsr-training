"""Materialize the four matched R1+guarded-bgLF continuation configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "options/train_bucket_3/train_v4_r1_recall_bglf_guard_v2only_250k_20260904.yml"
STATE = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/v4_r1_recall_bglf_guard_v2only_250k_20260904/training_states/250000.state"
DATE = "20260906"


def _losses() -> tuple[dict, dict]:
    sdf = {
        "type": "GTSDFBoundaryLoss", "radius": 3, "sdf_temperature": 1.0,
        "ink_temperature": 0.05, "anchor_radius": 3, "min_contrast": 0.03,
        "mask_threshold": 0.5, "profile_loss": "bce", "linear_boundary_weight": 0.0,
        "boundary_width": 1, "loss_weight": 1.0,
    }
    profile = {
        "type": "BoundaryNormalProfileLoss", "radius": 3, "samples": 7,
        "tolerance_pixels": 0.75, "tolerance_samples": 3,
        "profile_weight": 1.0, "difference_weight": 1.0, "charbonnier": True,
        "charbonnier_eps": 0.001, "contrast_eps": 0.001,
        "softmin_temperature": 0.05, "threshold": 0.5, "edge_width": 1,
        "coherence_kernel": 3, "min_coherence": 0.35, "loss_weight": 1.0,
    }
    return sdf, profile


def main() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    sdf, profile = _losses()
    arms = {
        "e0_control": {},
        # Calibrated on a real verified V2 batch to 5% of the existing R1+
        # guarded-bgLF output-gradient norm.  In E3 each term is 2.5% so
        # the interaction arm has the same total auxiliary budget.
        "e1_sdf": {"sdf_boundary_weight": 0.0791870321, "sdf_boundary_opt": sdf},
        "e2_profile": {"normal_profile_weight": 0.0079522591, "normal_profile_opt": profile},
        "e3_sdf_profile": {
            "sdf_boundary_weight": 0.0395935160, "sdf_boundary_opt": sdf,
            "normal_profile_weight": 0.0039761295, "normal_profile_opt": profile,
        },
    }
    for index, (arm, changes) in enumerate(arms.items()):
        option = copy.deepcopy(base)
        option["name"] = f"v4_r1_guard_{arm}_continue50k_{DATE}"
        option["path"]["resume_state"] = STATE
        option["train"]["total_iter"] = 300000
        option["train"].update(changes)
        option["dist_params"]["port"] = 29720 + index
        option["datasets"]["train"]["name"] = option["name"] + "_train"
        option["datasets"]["val"]["name"] = option["name"] + "_val"
        output = ROOT / "options/train_bucket_3" / f"train_{option['name']}.yml"
        output.write_text(yaml.safe_dump(option, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
