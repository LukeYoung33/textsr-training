"""Materialize the signed Lenovo-degradation V4 eight-arm 250K contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "options" / "train_bucket_3" / "lenovo_v4_cp_strong_20260911"
STAMP = "20260911"
EXPERIMENTS_ROOT = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_v4_cp_strong_20260911"
DATA_ROOT = "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2"
FIXED_VALIDATION_ROOT = "/AI_DNS/guang.yang/TextSR/data/lenovo_validation_aspect_2to5_stronger_c5_priority_128_20260911"

ARMS = {
    "c0": {},
    "r1": {"recall": True},
    "rb": {"recall": True, "bglf": True},
    "s1": {"sdf": True},
    "core": {"prob": True, "core": True},
    "over": {"prob": True, "overshoot": True},
    "width": {"prob": True, "width": True},
    "cldice": {"prob": True, "cldice": True},
}


def _degradation_contract() -> dict:
    return {
        "schema_version": "lenovo_degradation_v2_composed",
        # Every maximum is 125% of the approved strong-validation maximum.
        "c1": {"gaussian_sigma": [0.35, 2.6875], "motion_length": [3, 20],
               "defocus_radius": [1, 8], "zoom_source_scale": [0.20, 0.75]},
        "c2": {"psf_sigma": [0.82, 2.375], "sample_scale": [0.225, 0.72], "phase_px": 0.975},
        "c3": {'version': 'c3_multiscale_edge_protected_v1', 'strength': [6.0, 17.5], 'sigmas': [1.5, 4.0, 9.0], 'scale': 0.5, 'rho': -0.75},
        "c4": {"uv_blur_sigma": [1.5, 11.25], "midfreq_strength": [0.0, 3.5]},
        "c5": {"usm_sigma": [1.4, 7.5], "usm_amount": [0.25, 1.6875], "jpeg_quality": [60, 96]},
        "c7": {"shift_x": [1.0, 4.25], "shift_y": [0.5, 2.25], "weight": [0.12, 0.425]},
        "c8": {"display_scale": [0.175, 0.75], "display_jpeg_quality": [72, 96],
               "camera_blur_sigma": [0.35, 1.8125], "final_jpeg_quality": [65, 94],
               "perspective_fraction": [0.0, 0.01125]},
        "binding_status": "fixed_validation_and_runtime_smoke_pending",
        "composition": {
            "capture_weights": {"c1": 0.0, "c2": 0.16, "c1_c2": 0.54, "zoom_c2": 0.30},
            "post_probabilities": {"c3": 0.30, "c4": 0.26, "c5": 0.85, "c7": 0.30},
            "max_post_branches": 2,
            "secondary_capture_probability": 0.42,
        },
        "c6": {"status": "pinned", "weight": 0.0, "sampleable": False},
        "output_contract": {"input": "clean_gt_rgb", "output": "same_size_rgb_lq_and_metadata",
                            "rng": "per_sample_torch_generator"},
    }


def _common(name: str, port: int) -> dict:
    return {
        "name": name,
        "model_type": "LenovoTextLineDegradationModel",
        "scale": 1, "num_gpu": 1, "manual_seed": 20260828, "color": "y4uv420",
        "lenovo_degradation": _degradation_contract(),
        "network_g": {"type": "BasicTextLineR4StudentY4UVV4", "canvas_hw": [96, 480],
                      "in_channel": 6, "out_channel": 6, "stage_channels": [6, 24, 96, 384],
                      "refine_channels": 16, "stage_repeat": 1, "bottleneck_repeat": 2,
                      "haar_switch": True, "learnable_haar": False, "stroke_prob_enabled": False},
        "path": {"pretrain_network_g": None, "resume_state": None, "strict_load_g": False,
                 "experiments_root": EXPERIMENTS_ROOT},
        "train": {"model_type": "LenovoTextLineDegradationModel", "ema_decay": 0,
                  "optim_g": {"type": "Adam", "lr": 0.0001, "betas": [0.9, 0.99]},
                  "scheduler": {"type": "MultiStepLR", "milestones": [75000, 150000, 225000], "gamma": 0.75},
                  "total_iter": 250000, "loss_mode": "native", "train_reference_loss": False,
                  "validation_reference_loss": True, "aux_start_iter": 2000, "aux_ramp_iters": 5000,
                  "cp_weight": 0.0005, "grad_weight": 0.0,
                  "y4uv_native_opt": {"type": "Y4UVNativeMaskedL1LossV3", "loss_weight": 1.0,
                                      "fg_weight": 1.0, "bg_weight": 1.0,
                                      "channel_weight": [1.0, 1.0, 1.0, 1.0, 0.25, 0.25]},
                  "yuv444_reference_opt": {"type": "YUV444ReferenceMaskedL1LossV3", "loss_weight": 1.0,
                                            "fg_weight": 1.0, "bg_weight": 1.0,
                                            "channel_weight": [1.0, 0.25, 0.25]},
                  "cp_opt": {"type": "CRNNTextLineTileContentLoss",
                             "recognizer_path": "/home/guang.yang/workspace/Real-ESRGAN/weights/scene_base_CRNN.pth",
                             "alphabet_path": "/home/guang.yang/workspace/Real-ESRGAN/realesrgan/metrics/benchmark.txt",
                             "crop_height": 32, "min_width": 16, "max_width": 1024,
                             "min_aspect_ratio": 0.5, "max_aspect_ratio": 32.0, "micro_batch_size": 16,
                             "layer_weights": [1.4, 1.4, 1.4, 0.4, 0.4], "loss_weight": 1.0},
                  "recall_aware_weight_missing": 0.0, "recall_aware_weight_false": 0.0,
                  "recall_aware_opt": {"ring_radius": 3, "margin": 0.03, "temperature": 0.01,
                                       "min_polarity_contrast": 0.03},
                  "bg_lf_weight": 0.0, "bg_lf_opt": None, "sdf_boundary_weight": 0.0,
                  "sdf_boundary_opt": None, "stroke_prob_enabled": False, "stroke_prob_weight": 0.0,
                  "stroke_prob_opt": None, "stroke_core_weight": 0.0, "stroke_overshoot_weight": 0.0,
                  "cldice_weight": 0.0, "stroke_width_weight": 0.0, "skel_fbeta_weight": 0.0,
                  "skel_fbeta_opt": None, "stroke_core_opt": None, "stroke_overshoot_opt": None,
                  "cldice_opt": None, "stroke_width_opt": None},
        "datasets": {"train": {"name": f"{name}_train", "type": "LenovoCleanTextLineDataset",
                                "schema": "lenovo_clean_gt_v1", "split": "train",
                                "sources": [{"root": DATA_ROOT, "weight": 1.0}], "canvas_hw": [96, 480],
                                "resize_mode": "stretch", "batch_size_per_gpu": 4,
                                "num_worker_per_gpu": 4, "dataset_enlarge_ratio": 1, "use_shuffle": True},
                     "val": {"name": f"{name}_fixed_lenovo_val", "type": "BasicTextLineRemote58Dataset", "split": "val",
                             "sources": [{"root": FIXED_VALIDATION_ROOT, "manifest": "validation_pairs.jsonl", "weight": 1.0}],
                             "canvas_hw": [96, 480], "resize_mode": "aspect_fit", "max_upscale": 1.0,
                             "batch_size_per_gpu": 4, "num_worker_per_gpu": 2}},
        "val": {"val_freq": 25000, "save_img": False, "metrics": {}},
        "logger": {"print_freq": 100, "save_checkpoint_freq": 25000, "use_tb_logger": True},
        "dist_params": {"backend": "nccl", "port": port},
    }


def _set_arm(config: dict, enabled: dict) -> None:
    train = config["train"]
    config["network_g"]["stroke_prob_enabled"] = bool(enabled.get("prob"))
    train["recall_aware_weight_missing"] = 1.0 if enabled.get("recall") else 0.0
    train["recall_aware_weight_false"] = 0.25 if enabled.get("recall") else 0.0
    train["bg_lf_weight"] = 2.0 if enabled.get("bglf") else 0.0
    train["bg_lf_opt"] = {"type": "BackgroundLowFrequencyGuardLoss", "radius": 7, "sigma": 4.0,
                          "channel": "luma", "threshold": 0.5, "eps": 1e-6} if enabled.get("bglf") else None
    train["sdf_boundary_weight"] = 0.0856286676 if enabled.get("sdf") else 0.0
    train["sdf_boundary_opt"] = {"type": "GTSDFBoundaryLoss", "radius": 3, "sdf_temperature": 1.0,
                                 "ink_temperature": 0.05, "anchor_radius": 3, "min_contrast": 0.03,
                                 "mask_threshold": 0.5, "profile_loss": "bce", "linear_boundary_weight": 0.0,
                                 "boundary_width": 1, "loss_weight": 1.0} if enabled.get("sdf") else None
    train["stroke_prob_enabled"] = bool(enabled.get("prob"))
    train["stroke_prob_weight"] = 0.10 if enabled.get("prob") else 0.0
    train["stroke_prob_opt"] = {"type": "StrokeProbabilityBCELoss", "mask_threshold": 0.5, "loss_weight": 1.0} if enabled.get("prob") else None
    train["stroke_core_weight"] = 0.05 if enabled.get("core") else 0.0
    train["stroke_overshoot_weight"] = 0.02 if enabled.get("overshoot") else 0.0
    train["stroke_width_weight"] = 0.02 if enabled.get("width") else 0.0
    train["cldice_weight"] = 0.02 if enabled.get("cldice") else 0.0
    train["stroke_core_opt"] = {"type": "StrokeCoreLoss", "max_skeleton_iterations": 96,
                                "skeleton_threshold": 0.5, "loss_weight": 1.0} if enabled.get("core") else None
    train["stroke_overshoot_opt"] = {"type": "OvershootLoss", "kernel_size": 5, "tol": 2.0 / 255.0,
                                     "mask_radius": 2, "mask_mode": "text_dilate", "channel": "luma",
                                     "loss_weight": 1.0} if enabled.get("overshoot") else None
    train["stroke_width_opt"] = {"type": "StrokeWidthLoss", "max_radius": 8, "anchor_radius": 3,
                                 "min_contrast": 0.01, "ink_temperature": 0.05,
                                 "max_skeleton_iterations": 96, "tube_radius": 0, "loss_weight": 1.0} if enabled.get("width") else None
    train["cldice_opt"] = {"type": "SoftCLDiceLoss", "alpha": 0.4, "skeleton_iterations": 5,
                           "max_skeleton_iterations": 96, "tube_radius": 0, "anchor_radius": 3,
                           "min_contrast": 0.01, "ink_temperature": 0.05, "loss_weight": 1.0} if enabled.get("cldice") else None


def build_configs(output: Path = OUT) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (arm, enabled) in enumerate(ARMS.items()):
        name = f"lenovo_v4_{arm}_250k_{STAMP}"
        config = _common(name, 29940 + index)
        _set_arm(config, enabled)
        path = output / f"train_{name}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(path)
    manifest = {"contract": "Lenovo V4 CP eight-arm 250K", "binding_status": "composed_pipeline_runtime_smoke_pending",
                "launch": [{"arm": arm, "config": str(path), "experiment": f"lenovo_v4_{arm}_250k_{STAMP}",
                            "command": f"python realesrgan/train.py -opt {path} --launcher none"}
                           for (arm, _), path in zip(ARMS.items(), paths)],
                "health": {"checkpoint_and_validation_every": 25000, "no_intermediate_selection": True,
                           "stop_conditions": ["NaN", "shape_contract_failure", "branch_sampling_anomaly"]},
                "final_evaluation": {"at_iteration": 250000, "required": ["fixed_real_validation", "OCR", "fixed_HX", "per_image_structure_metrics", "visual_review"]}}
    (output / "launch_and_evaluation_contract.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    print("\n".join(str(path) for path in build_configs()))
