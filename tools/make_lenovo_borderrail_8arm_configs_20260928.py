"""Prepare matched V5 R1/bgLF/UV configs for four border-rail arms at two heights."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "options/train_bucket_3/lenovo_mild_scurve_2x2_20260927/train_scurve_h96_250k_20260927.yml"
OUTPUT = ROOT / "options/train_bucket_3/lenovo_borderrail_8arm_20260928"
EXPERIMENT_ROOT = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_8arm_20260928"
DATASETS = (
    ("lenovo260926", "/AI_DNS/guang.yang/TextSR/data/Lenovo_data_260926/"
     "training_20260927_batch3_validcrop_v4_brightfix_v5_v6_ocr_matrix_20260928", 5510, 1028),
    ("lenovo260920", "/AI_DNS/guang.yang/TextSR/data/Lenovo_data_260920/"
     "training_20260920_ocrfix_v4_formula_groups_v5_v6_ocr_matrix_20260928", 57200, 2245),
)
ARMS = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")


def main() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for arm_index, arm in enumerate(ARMS):
        for height_index, height in enumerate((96, 64)):
            config = copy.deepcopy(base)
            name = f"lenovo_v5_r1_bglf_uv_{arm}_h{height}_250k_20260928"
            config["name"] = name
            config["network_g"]["canvas_hw"] = [height, 480]
            config["path"]["experiments_root"] = EXPERIMENT_ROOT
            config["path"]["resume_state"] = None
            config["dist_params"]["port"] = 31200 + 2 * arm_index + height_index
            config["high_order_degradation"] = False
            config["lq_preprocess"]["enabled"] = False
            for split in ("train", "val"):
                opt = config["datasets"][split]
                opt["name"] = f"{name}_{split}"
                opt["canvas_hw"] = [height, 480]
                opt["ocr_line_crop"] = True
                opt["text_line_crop"] = False
                opt["sources"] = [
                    {"name": source_name, "root": f"{dataset_root}/{arm}",
                     "manifest": f"{dataset_root}/{arm}/manifests/{split}.jsonl",
                     "split": split, "weight": train_count if split == "train" else val_count}
                    for source_name, dataset_root, train_count, val_count in DATASETS
                ]
                opt["epoch_size"] = sum(train_count if split == "train" else val_count
                                        for _, _, train_count, val_count in DATASETS)
            output = OUTPUT / f"train_{arm}_h{height}_250k_20260928.yml"
            output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
            print(output)


if __name__ == "__main__":
    main()
