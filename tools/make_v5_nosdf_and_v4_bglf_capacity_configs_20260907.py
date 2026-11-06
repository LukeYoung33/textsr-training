"""Materialize the paired V5 no-SDF ablation and V4 R1+guarded-bgLF capacity arms.

All arms are from scratch and retain the V2-only data, seed, degradation,
optimizer, scheduler, CP, R1, and guarded-bgLF contracts.  The V4 arms omit
SDF by construction; V5 no-SDF removes only the SDF term from the active V5
run.  ``v5trainmatch`` is deliberately calibrated against V5's *training*
graph MAC count (not V5's fused deployment graph).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from profile_v4_y4uv_capacity import PRESETS


ROOT = Path(__file__).resolve().parents[1]
V5_BASE = ROOT / "options/train_bucket_3/train_v5_r1_guard_sdf_fromstart_v2only_base_250k_20260907.yml"
V4_BASE = ROOT / "options/train_bucket_3/train_v4_r1_recall_bglf_guard_v2only_250k_20260904.yml"
DATE = "20260907"


def _set_name(option: dict, name: str, port: int) -> None:
    option["name"] = name
    option["path"].update({"pretrain_network_g": None, "resume_state": None, "strict_load_g": False})
    option["datasets"]["train"]["name"] = name + "_train"
    option["datasets"]["val"]["name"] = name + "_val"
    option["dist_params"]["port"] = int(port)


def make_v5_nosdf() -> tuple[Path, dict]:
    option = yaml.safe_load(V5_BASE.read_text(encoding="utf-8"))
    name = f"v5_r1_guard_bglf_nosdf_fromstart_v2only_base_250k_{DATE}"
    _set_name(option, name, 29742)
    option["train"]["sdf_boundary_weight"] = 0.0
    option["train"]["sdf_boundary_opt"] = None
    return ROOT / "options/train_bucket_3" / f"train_{name}.yml", option


def make_v4_capacity(label: str, port: int) -> tuple[Path, dict]:
    option = yaml.safe_load(V4_BASE.read_text(encoding="utf-8"))
    name = f"v4_r1_guard_bglf_fromstart_{label}_250k_{DATE}"
    _set_name(option, name, port)
    option["network_g"].update({"type": "BasicTextLineR4StudentY4UVV4Large"} | copy.deepcopy(PRESETS[label]))
    # Explicitly prohibit an accidental inherited boundary term.
    option["train"]["sdf_boundary_weight"] = 0.0
    option["train"]["sdf_boundary_opt"] = None
    return ROOT / "options/train_bucket_3" / f"train_{name}.yml", option


def main() -> None:
    planned = [make_v5_nosdf()]
    planned.extend(make_v4_capacity(label, 29743 + index) for index, label in enumerate(("base", "20x", "50x", "v5trainmatch")))
    for path, option in planned:
        path.write_text(yaml.safe_dump(option, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
