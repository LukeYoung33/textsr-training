"""Generate four matched V5 paired-LQ training options from the R1 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT_260926 = "/home/guang.yang/data/lenovo_260926_validcrop_v4_all3_mild_scurve_v1_20260927"
ROOT_20260920 = "/home/guang.yang/data/lenovo_textsr_20260920_final_validcrop_v4_mild_scurve_v1_lines"
EXPERIMENT_ROOT = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_mild_scurve_2x2_20260927"
MANIFEST_ROOT = f"{EXPERIMENT_ROOT}/input_manifests"
PRETRAIN = (
    "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_v5_p17_effect_iteration_20260920/"
    "lenovo_v5_p17_original_r1_bglf_uv_250k_20260920/models/net_g_250000.pth"
)
COUNTS = {"train": (1558, 43541), "val": (250, 1691)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("scurve", "raw"):
        for height in (96, 64):
            config = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
            name = f"lenovo_v5_r1_bglf_uv_{mode}_h{height}_250k_20260927"
            config["name"] = name
            config["high_order_degradation"] = False
            config["lq_preprocess"]["enabled"] = False
            config["network_g"]["canvas_hw"] = [height, 480]
            config["path"]["pretrain_network_g"] = PRETRAIN
            config["path"]["resume_state"] = None
            config["path"]["experiments_root"] = EXPERIMENT_ROOT
            config["train"]["total_iter"] = 250000
            config["train"]["scheduler"]["milestones"] = [75000, 150000, 225000]
            config["datasets"] = {}
            for split in ("train", "val"):
                source_opt = dict(base["datasets"]["train"])
                source_opt["name"] = f"{name}_{split}"
                source_opt["type"] = "BasicTextLineRemote58Dataset"
                source_opt["split"] = split
                source_opt["canvas_hw"] = [height, 480]
                source_opt["ocr_line_crop"] = True
                source_opt["text_line_crop"] = False
                source_opt["text_line_vertical_context"] = 0.12
                source_opt["text_line_horizontal_context"] = 0.03
                source_opt["text_line_boundary_jitter"] = 0.05
                source_opt.pop("ocr_line_crop_cache_root", None)
                source_opt.pop("decoded_source_cache_size", None)
                source_opt.pop("source_grouped_sampling", None)
                for key in (
                    "blur_kernel_size", "kernel_list", "kernel_prob", "sinc_prob", "blur_sigma",
                    "betag_range", "betap_range", "blur_kernel_size2", "kernel_list2",
                    "kernel_prob2", "sinc_prob2", "blur_sigma2", "betag_range2",
                    "betap_range2", "final_sinc_prob", "train_homography_prob",
                    "train_homography_jitter", "use_shuffle", "dataset_enlarge_ratio",
                ):
                    if split == "val" or key in (
                        "blur_kernel_size", "kernel_list", "kernel_prob", "sinc_prob", "blur_sigma",
                        "betag_range", "betap_range", "blur_kernel_size2", "kernel_list2",
                        "kernel_prob2", "sinc_prob2", "blur_sigma2", "betag_range2",
                        "betap_range2", "final_sinc_prob",
                    ):
                        source_opt.pop(key, None)
                counts = COUNTS[split]
                source_opt["epoch_size"] = sum(counts)
                source_opt["sources"] = [
                    {"name": "lenovo260926", "root": ROOT_260926,
                     "manifest": f"{MANIFEST_ROOT}/lenovo260926_{split}_{mode}.jsonl",
                     "split": split, "weight": counts[0]},
                    {"name": "lenovo20260920", "root": ROOT_20260920,
                     "manifest": f"{MANIFEST_ROOT}/lenovo20260920_{split}_{mode}.jsonl",
                     "split": split, "weight": counts[1]},
                ]
                if split == "val":
                    source_opt["batch_size_per_gpu"] = 1
                    source_opt["num_worker_per_gpu"] = 0
                config["datasets"][split] = source_opt
            config["val"] = {"val_freq": 25000, "save_img": False, "metrics": {}}
            config["logger"]["save_checkpoint_freq"] = 25000
            config["dist_params"]["port"] = 31100 + (0 if mode == "scurve" else 20) + height
            output = args.output_dir / f"train_{mode}_h{height}_250k_20260927.yml"
            output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
            print(output)


if __name__ == "__main__":
    main()
