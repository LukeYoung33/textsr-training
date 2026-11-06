"""Materialize the Lenovo-degraded V5 eight-arm 250K loss comparison contract.

V5 retains the V4 loss-arm names and direct losses.  Its optional training-only
stroke-probability head preserves the native six-channel deployment contract.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

import make_lenovo_v4_experiment_configs_20260910 as v4


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "options" / "train_bucket_3" / "lenovo_v5_cp_strong_20260912"
STAMP = "20260912"
EXPERIMENTS_ROOT = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_v5_cp_strong_20260912"
ARMS = v4.ARMS


def _configure_v5(config: dict, probability_enabled: bool) -> None:
    config["model_type"] = "LenovoTextLineDegradationModel"
    config["path"]["experiments_root"] = EXPERIMENTS_ROOT
    config["network_g"] = {
        "type": "BasicTextLineR4StudentY4UVV5",
        "canvas_hw": [96, 480],
        "in_channel": 6,
        "out_channel": 6,
        "stage_channels": [8, 32, 128],
        "local_channels": 16,
        "stage_repeat": 1,
        "bottleneck_repeat": 2,
        "haar_switch": True,
        "learnable_haar": False,
        "grouped_phase_pack": True,
    }
    train = config["train"]
    train["model_type"] = "LenovoTextLineDegradationModel"
    # Preserve the V4 ablation contract: only geometry arms whose direct loss
    # consumes a probability field train the auxiliary V5 head.
    config["network_g"]["stroke_prob_enabled"] = bool(probability_enabled)
    train["stroke_prob_enabled"] = bool(probability_enabled)
    train["stroke_prob_weight"] = 0.10 if probability_enabled else 0.0
    train["stroke_prob_opt"] = {
        "type": "StrokeProbabilityBCELoss",
        "mask_threshold": 0.5,
        "loss_weight": 1.0,
    } if probability_enabled else None


def build_configs(output: Path = OUT) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, (arm, enabled) in enumerate(ARMS.items()):
        name = f"lenovo_v5_{arm}_250k_{STAMP}"
        config = deepcopy(v4._common(name, 30040 + index))
        v4._set_arm(config, enabled)
        _configure_v5(config, bool(enabled.get("prob")))
        path = output / f"train_{name}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(path)
    manifest = {
        "contract": "Lenovo V5 PhaseRepECB eight-arm 250K",
        "training_degradation": "lenovo_degradation_v2_composed; C6 pinned",
        "stroke_probability": "V5 training-only 2x2 phase-unpacked head enabled for the four probability-based geometry arms and dropped for deployment",
        "launch": [
            {
                "arm": arm,
                "config": str(path),
                "experiment": f"lenovo_v5_{arm}_250k_{STAMP}",
                "command": f"python realesrgan/train.py -opt {path} --launcher none",
            }
            for (arm, _), path in zip(ARMS.items(), paths)
        ],
        "health": {"checkpoint_and_validation_every": 25000},
        "final_evaluation": {
            "at_iteration": 250000,
            "required": ["fixed_validation_visuals", "stroke_metrics", "fixed_HX", "OCR"],
        },
    }
    (output / "launch_and_evaluation_contract.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths


if __name__ == "__main__":
    print("\n".join(str(path) for path in build_configs()))
