"""Calibrate SDF/profile weights by gradients with respect to V4 output."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from basicsr.data import build_dataset
from basicsr.models import build_model
from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses
from realesrgan.utils import YUV2RGB

import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
import realesrgan.losses  # noqa: F401
import realesrgan.models  # noqa: F401


def _norm(value: torch.Tensor, output: torch.Tensor) -> float:
    grad = torch.autograd.grad(value, output, retain_graph=True)[0]
    return float(grad.detach().float().square().sum().sqrt().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", required=True)
    parser.add_argument("--target-single", type=float, default=0.05)
    parser.add_argument("--target-combined-each", type=float, default=0.025)
    args = parser.parse_args()
    opt = yaml.safe_load(Path(args.opt).read_text(encoding="utf-8"))
    ds_opt = copy.deepcopy(opt["datasets"]["train"])
    ds_opt.update({"phase": "train", "batch_size_per_gpu": 1, "num_worker_per_gpu": 0})
    dataset = build_dataset(ds_opt)
    sample = next(item for item in (dataset[i] for i in range(len(dataset))) if item.get("structure_available"))
    batch = dataset.collate_fn([sample])
    runtime = copy.deepcopy(opt)
    runtime.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1, "num_gpu": 1})
    runtime["root_path"] = str(Path(__file__).resolve().parents[1])
    runtime["path"] = copy.deepcopy(runtime["path"])
    runtime["path"]["resume_state"] = None
    model = build_model(runtime)
    model.feed_data(batch)
    output = model.net_g(model.lq)
    native = model.cri_y4uv_native(output, model.gt, valid_mask=model.valid_mask, text_mask=model.text_mask)
    rgb = YUV2RGB(y4uv_to_yuv444(output), format="RGB", is_train=True, uv_normalized=True)
    cp_input = model.segment_metadata if bool(getattr(model.cri_cp, "expects_segment_metadata", False)) else model._build_cp_annotations(model.segment_metadata, output.shape[0])
    cp = model.cri_cp(rgb, model.gt_rgb, cp_input) * float(opt["train"]["cp_weight"])
    text, valid = model._structure_masks()
    missing, false, _ = recall_aware_structure_losses(rgb, model.gt_rgb, text, valid, **opt["train"]["recall_aware_opt"])
    recall = missing + 0.25 * false
    bg, _ = model.cri_bg_lf(rgb, model.gt_rgb, text_mask_rgb=text, valid_mask_rgb=valid, return_details=True)
    baseline = native + cp + recall + 2.0 * bg
    raw_sdf = model.cri_sdf_boundary(rgb, model.gt_rgb, text, valid) if model.cri_sdf_boundary is not None else None
    raw_profile = model.cri_normal_profile(rgb, model.gt_rgb, text, valid) if model.cri_normal_profile is not None else None
    base_norm = _norm(baseline, output)
    sdf_norm = _norm(raw_sdf, output) if raw_sdf is not None else None
    profile_norm = _norm(raw_profile, output) if raw_profile is not None else None
    result = {
        "base_output_grad_norm": base_norm, "sdf_raw_output_grad_norm": sdf_norm,
        "profile_raw_output_grad_norm": profile_norm,
        "sdf_weight_single": (args.target_single * base_norm / sdf_norm) if sdf_norm else None,
        "profile_weight_single": (args.target_single * base_norm / profile_norm) if profile_norm else None,
        "sdf_weight_combined": (args.target_combined_each * base_norm / sdf_norm) if sdf_norm else None,
        "profile_weight_combined": (args.target_combined_each * base_norm / profile_norm) if profile_norm else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
