"""One actual V4 trainer step for SDF/profile loss plumbing validation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import yaml

from basicsr.data import build_dataset
from basicsr.models import build_model

import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
import realesrgan.losses  # noqa: F401
import realesrgan.models  # noqa: F401


def _scalar(value: object) -> float:
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--current-iter", type=int, default=260000)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    option_path = Path(args.opt).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    train = opt["train"]
    has_sdf = float(train.get("sdf_boundary_weight", 0.0)) > 0.0
    has_profile = float(train.get("normal_profile_weight", 0.0)) > 0.0
    if not (has_sdf or has_profile):
        raise ValueError("smoke requires at least one geometry loss")
    dataset_opt = copy.deepcopy(opt["datasets"]["train"])
    dataset_opt.update({"phase": "train", "batch_size_per_gpu": args.batch_size, "num_worker_per_gpu": 0})
    dataset = build_dataset(dataset_opt)
    samples = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample.get("structure_available"):
            samples.append(sample)
        if len(samples) == args.batch_size:
            break
    if len(samples) != args.batch_size:
        raise RuntimeError("insufficient verified-structure samples for requested smoke batch")
    batch = dataset.collate_fn(samples)
    runtime = copy.deepcopy(opt)
    runtime.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1, "num_gpu": 1})
    runtime["root_path"] = str(Path(__file__).resolve().parents[1])
    runtime["path"] = copy.deepcopy(runtime["path"])
    runtime["path"].update({"resume_state": None, "models": str(out_dir / "models"), "training_states": str(out_dir / "states"), "log": str(out_dir / "logs")})
    model = build_model(runtime)
    model.feed_data(batch)
    model.optimize_parameters(args.current_iter)
    logs = {key: _scalar(value) for key, value in model.get_current_log().items()}
    required = ["l_native", "l_cp", "l_missing", "l_false", "l_bg_lf", "l_total"]
    if has_sdf:
        required += ["l_sdf_boundary_raw", "l_sdf_boundary", "sdf_boundary_active_pixels", "sdf_boundary_active_ratio"]
    if has_profile:
        required += ["l_normal_profile_raw", "l_normal_profile", "normal_profile_boundary_weight", "normal_profile_valid_fraction"]
    missing = [key for key in required if key not in logs]
    finite = all(math.isfinite(value) for value in logs.values())
    grad_sq = sum((p.grad.detach().float().square().sum() for p in model.net_g.parameters() if p.grad is not None), model.output.new_zeros(()))
    payload = {"status": "pass" if finite and not missing and float(grad_sq) > 0.0 else "fail", "config": str(option_path), "current_iter": args.current_iter, "has_sdf": has_sdf, "has_profile": has_profile, "gradient_l2": float(grad_sq.sqrt().cpu()), "missing": missing, "logs": logs}
    path = out_dir / "boundary_geometry_trainer_smoke.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
