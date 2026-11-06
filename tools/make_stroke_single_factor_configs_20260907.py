"""Create the four from-scratch single-factor stroke-loss experiments.

Every arm is cloned from the current V5 R1 + guarded-bgLF + SDF baseline.
The only positive new factor is one of core/overshoot/clDice/width; the
other new factors are explicitly present with zero weight so trainer smoke
can verify that no accidental stacking occurs.  The auxiliary schedule is
kept identical across arms and uses the requested 2k/10k warm-up/ramp.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "options/train_bucket_3/train_v5_r1_guard_sdf_fromstart_v2only_base_250k_20260907.yml"
OUT = ROOT / "options/train_bucket_3"
DATE = "20260907"


STROKE_OPTS = {
    "stroke_core_opt": {
        "type": "StrokeCoreLoss",
        "max_skeleton_iterations": 96,
        "skeleton_threshold": 0.5,
        "loss_weight": 1.0,
    },
    "stroke_overshoot_opt": {
        "type": "OvershootLoss",
        "kernel_size": 5,
        "tol": 2.0 / 255.0,
        "mask_radius": 2,
        "mask_mode": "text_dilate",
        "mask_threshold": 0.5,
        "channel": "luma",
        "loss_weight": 1.0,
    },
    "cldice_opt": {
        "type": "SoftCLDiceLoss",
        "skeleton_iterations": 6,
        "alpha": 0.4,
        "anchor_radius": 3,
        "min_contrast": 0.01,
        "ink_temperature": 0.05,
        "mask_threshold": 0.5,
        "max_skeleton_iterations": 96,
        "tube_radius": 0,
        "loss_weight": 1.0,
    },
    "stroke_width_opt": {
        "type": "StrokeWidthLoss",
        "max_radius": 8,
        "anchor_radius": 3,
        "min_contrast": 0.01,
        "ink_temperature": 0.05,
        "mask_threshold": 0.5,
        "max_skeleton_iterations": 96,
        "tube_radius": 0,
        "loss_weight": 1.0,
    },
}


ARMS = (
    ("core", 0.05, 0.0, 0.0, 0.0, 29751),
    ("overshoot", 0.0, 0.02, 0.0, 0.0, 29752),
    ("cldice", 0.0, 0.0, 0.02, 0.0, 29753),
    ("width", 0.0, 0.0, 0.0, 0.02, 29754),
)


def _name(label: str) -> str:
    return f"v5_r1_guard_sdf_stroke_{label}_fromstart_v2only_base_250k_{DATE}"


def _make(label: str, core: float, overshoot: float, cldice: float, width: float, port: int) -> tuple[Path, dict]:
    option = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    name = _name(label)
    option["name"] = name
    option["path"] = copy.deepcopy(option.get("path", {}))
    option["path"].update({"pretrain_network_g": None, "resume_state": None, "strict_load_g": False})
    option["train"] = copy.deepcopy(option["train"])
    option["train"]["aux_start_iter"] = 2000
    option["train"]["aux_ramp_iters"] = 10000
    # Explicit zero gates make the single-factor contract auditable in the
    # YAML and ensure the model emits zero logs for inactive alternatives.
    option["train"].update(
        {
            "stroke_core_weight": float(core),
            "stroke_overshoot_weight": float(overshoot),
            "cldice_weight": float(cldice),
            "stroke_width_weight": float(width),
            "skel_fbeta_weight": 0.0,
            "skel_fbeta_opt": None,
        }
    )
    option["train"].update(copy.deepcopy(STROKE_OPTS))
    option["datasets"]["train"]["name"] = name + "_train"
    option["datasets"]["val"]["name"] = name + "_val"
    option["dist_params"]["port"] = int(port)
    path = OUT / f"train_{name}.yml"
    return path, option


def main() -> None:
    for arm in ARMS:
        path, option = _make(*arm)
        path.write_text(yaml.safe_dump(option, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
