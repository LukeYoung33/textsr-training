"""Create eight 100K whiteboard finetunes paired with border-rail runs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from make_lenovo_borderrail_8arm_configs_20260928 import ARMS, EXPERIMENT_ROOT, OUTPUT


FINETUNE_ROOT = "/AI_DNS/guang.yang/TextSR/data/Lenovo_mojito_whiteboard_v4_v5_v6_ocr_matrix_20260928/gtfit_v6"
FINETUNE_EXPERIMENT_ROOT = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_8arm_mojito_finetune_20260928"
FINETUNE_OUTPUT = OUTPUT / "finetune"


def main() -> None:
    FINETUNE_OUTPUT.mkdir(parents=True, exist_ok=True)
    for arm_index, arm in enumerate(ARMS):
        for height_index, height in enumerate((96, 64)):
            source = OUTPUT / f"train_{arm}_h{height}_250k_20260928.yml"
            config = copy.deepcopy(yaml.safe_load(source.read_text(encoding="utf-8")))
            main_name = config["name"]
            name = f"lenovo_v5_r1_bglf_uv_{arm}_h{height}_mojito_ft100k_20260928"
            config["name"] = name
            config["path"]["pretrain_network_g"] = f"{EXPERIMENT_ROOT}/{main_name}/models/net_g_latest.pth"
            config["path"]["resume_state"] = None
            config["path"]["strict_load_g"] = True
            config["path"]["experiments_root"] = FINETUNE_EXPERIMENT_ROOT
            config["train"]["total_iter"] = 100000
            config["train"]["scheduler"]["milestones"] = [30000, 60000, 90000]
            config["logger"]["save_checkpoint_freq"] = 25000
            config["dist_params"]["port"] = 31300 + 2 * arm_index + height_index
            train = config["datasets"]["train"]
            train["name"] = f"{name}_train"
            train["sources"] = [{
                "name": "mojito_whiteboard_gtfit_v6",
                "root": FINETUNE_ROOT,
                "manifest": f"{FINETUNE_ROOT}/manifests/train.jsonl",
                "split": "train",
                "weight": 405,
            }]
            train["epoch_size"] = 405
            config["datasets"].pop("val", None)
            config.pop("val", None)
            destination = FINETUNE_OUTPUT / f"train_{arm}_h{height}_ft100k_20260928.yml"
            destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
            print(destination)


if __name__ == "__main__":
    main()
