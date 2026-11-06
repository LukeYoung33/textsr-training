"""Build matched V4 capacity experiments for the final V5 comparison."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

from profile_v4_y4uv_capacity import PRESETS, profile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "options" / "train_bucket_3" / "lenovo_v4_cp_strong_20260911"
OUT = ROOT / "options" / "train_bucket_3" / "lenovo_v4_capacity_vs_v5_20260913"
EXPERIMENTS_ROOT = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_v4_capacity_vs_v5_20260913"
DATE = "20260913"
ARMS = ("c0", "r1")
CAPACITIES = ("v5deploymatch", "10x", "50x", "100x")


def _name(capacity: str, arm: str) -> str:
    return f"lenovo_v4cap_{capacity}_{arm}_250k_{DATE}"


def _make(capacity: str, arm: str, port: int) -> tuple[Path, dict]:
    source = SOURCE / f"train_lenovo_v4_{arm}_250k_20260911.yml"
    option = deepcopy(yaml.safe_load(source.read_text(encoding="utf-8")))
    name = _name(capacity, arm)
    option["name"] = name
    option["path"].update({
        "pretrain_network_g": None,
        "resume_state": None,
        "strict_load_g": False,
        "experiments_root": EXPERIMENTS_ROOT,
    })
    option["network_g"].update({"type": "BasicTextLineR4StudentY4UVV4Large"} | deepcopy(PRESETS[capacity]))
    option["network_g"]["stroke_prob_enabled"] = False
    option["train"]["stroke_prob_enabled"] = False
    option["train"]["stroke_prob_weight"] = 0.0
    option["train"]["stroke_prob_opt"] = None
    option["datasets"]["train"]["name"] = f"{name}_train"
    option["datasets"]["val"]["name"] = f"{name}_fixed_lenovo_val"
    option["dist_params"]["port"] = int(port)
    return OUT / f"train_{name}.yml", option


def build_configs(output: Path = OUT) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    capacity_profiles = {capacity: profile(capacity) for capacity in ("base", *CAPACITIES)}
    base_macs = capacity_profiles["base"]["macs"]
    for index, capacity in enumerate(CAPACITIES):
        for offset, arm in enumerate(ARMS):
            path, option = _make(capacity, arm, 30100 + index * len(ARMS) + offset)
            path.write_text(yaml.safe_dump(option, sort_keys=False, allow_unicode=True), encoding="utf-8")
            paths.append(path)
    manifest = {
        "contract": "V4 capacity scaling against V5 deployment compute",
        "arms": list(ARMS),
        "capacities": {
            name: {
                "spec": item["spec"],
                "macs": item["macs"],
                "params": item["params"],
                "ratio_to_v4_base": item["macs"] / base_macs,
            }
            for name, item in capacity_profiles.items()
        },
        "v5_deployment_macs": 395550720,
        "v5_ratio_to_v4_base": 395550720 / base_macs,
        "training_degradation": "lenovo_degradation_v2_composed; C6 pinned",
        "final_evaluation": "fixed 128-GT / 384-LQ validation, visuals and stroke_metrics",
    }
    (output / "capacity_contract.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    print("\n".join(str(path) for path in build_configs()))
