"""Run one real trainer step for a single stroke-loss experiment.

This smoke is intentionally a plumbing check.  It builds the configured
V2-only dataset and model, selects a verified structure sample, runs one
    optimizer step after the configured auxiliary ramp, and checks that exactly one of
the four new stroke losses is positive while the other three are zero.  It
does not claim image-quality improvement and never resumes or saves a formal
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from basicsr.data import build_dataset
from basicsr.models import build_model

import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
import realesrgan.losses  # noqa: F401
import realesrgan.models  # noqa: F401


NEW = {
    "core": ("l_stroke_core", "stroke_core_scale", "stroke_core_weight"),
    "overshoot": ("l_stroke_overshoot", "stroke_overshoot_scale", "stroke_overshoot_weight"),
    "cldice": ("l_cldice", "cldice_scale", "cldice_weight"),
    "width": ("l_stroke_width", "stroke_width_scale", "stroke_width_weight"),
}


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def _grad_l2(model: Any) -> float:
    value = None
    for parameter in model.net_g.parameters():
        if parameter.grad is None:
            continue
        term = parameter.grad.detach().float().square().sum()
        value = term if value is None else value + term
    return 0.0 if value is None else float(value.sqrt().cpu().item())


def _build_batch(option: dict[str, Any]) -> tuple[dict[str, Any], Any, list[int]]:
    expected = "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2"
    dataset_opt = copy.deepcopy(option["datasets"]["train"])
    if dataset_opt.get("sources") != [{"root": expected, "weight": 1.0}]:
        raise ValueError(f"smoke requires the exact V2-only source, got {dataset_opt.get('sources')!r}")
    dataset_opt.update({"phase": "train", "batch_size_per_gpu": 1, "num_worker_per_gpu": 0})
    dataset = build_dataset(dataset_opt)
    selected: list[int] = []
    samples = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if bool(sample.get("structure_available", False)):
            selected.append(index)
            samples.append(sample)
            break
    if len(samples) != 1:
        raise RuntimeError("could not find one verified structure sample")
    return dataset.collate_fn(samples), dataset, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--current-iter", type=int, default=12000)
    args = parser.parse_args()

    option_path = Path(args.opt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    option = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    label = next((name for name, (_, _, key) in NEW.items() if float(option["train"].get(key, 0.0)) > 0.0), None)
    positive = [name for name, (_, _, key) in NEW.items() if float(option["train"].get(key, 0.0)) > 0.0]
    if label is not None and positive != [label]:
        raise ValueError(f"expected one positive new factor, got {positive!r}")
    if label is None:
        name = str(option.get("name", "")).lower()
        train = option["train"]
        if float(train.get("sdf_boundary_weight", 0.0)) > 0.0:
            label = "sdf"
        elif float(train.get("bg_lf_weight", 0.0)) > 0.0:
            label = "r1_bglf"
        elif float(train.get("recall_aware_weight_missing", 0.0)) > 0.0 or float(train.get("recall_aware_weight_false", 0.0)) > 0.0:
            label = "r1"
        elif "c0" in name:
            label = "c0"
        else:
            raise ValueError("could not classify legacy C0/R1/R1+bgLF/SDF smoke config")
    if int(option["train"].get("aux_start_iter", 0)) != 2000 or int(option["train"].get("aux_ramp_iters", 0)) != 5000:
        raise ValueError("smoke requires the common 2000/5000 auxiliary schedule")
    batch, dataset, selected = _build_batch(option)
    device = torch.device(args.device)
    runtime = copy.deepcopy(option)
    runtime.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1, "num_gpu": 1})
    runtime["root_path"] = str(Path(__file__).resolve().parents[1])
    runtime["path"] = copy.deepcopy(runtime.get("path", {}))
    runtime["path"].update({
        "resume_state": None,
        "models": str(output_dir / "models"),
        "training_states": str(output_dir / "states"),
        "log": str(output_dir / "logs"),
    })
    model = build_model(runtime)
    model.feed_data(batch)
    model.optimize_parameters(int(args.current_iter))
    logs = {key: _float(value) for key, value in model.get_current_log().items()}
    # Inactive criteria do not emit a log entry.  Treat an absent entry as the
    # intended numerical zero, rather than allowing a NaN bookkeeping value to
    # turn a valid smoke into a false failure.
    loss_keys = {name: logs.get(values[0], 0.0) for name, values in NEW.items()}
    scale_keys = {name: logs.get(values[1], 0.0) for name, values in NEW.items()}
    finite = all(math.isfinite(value) for value in logs.values()) and bool(torch.isfinite(model.output).all())
    inactive_zero = all(abs(loss_keys[name]) <= 1.0e-12 for name in NEW if name != label)
    if label in NEW:
        active_positive = loss_keys[label] > 0.0 and scale_keys[label] > 0.0
    elif label == "c0":
        active_positive = all(abs(logs.get(key, 0.0)) <= 1.0e-12 for key in ("l_missing", "l_false", "l_bg_lf", "l_sdf_boundary"))
    elif label == "r1":
        active_positive = logs.get("l_missing", 0.0) > 0.0 and logs.get("l_false", 0.0) > 0.0 and abs(logs.get("l_bg_lf", 0.0)) <= 1.0e-12
    elif label == "r1_bglf":
        active_positive = logs.get("l_missing", 0.0) > 0.0 and logs.get("l_false", 0.0) > 0.0 and logs.get("l_bg_lf", 0.0) > 0.0 and logs.get("bg_lf_guard_active", 0.0) == 1.0
    else:  # SDF
        active_positive = logs.get("l_sdf_boundary", 0.0) > 0.0 and abs(logs.get("l_missing", 0.0)) <= 1.0e-12 and abs(logs.get("l_false", 0.0)) <= 1.0e-12
    probability_enabled = bool(option["train"].get("stroke_prob_enabled", False))
    probability_positive = (not probability_enabled) or (
        logs.get("l_stroke_prob", 0.0) > 0.0 and logs.get("stroke_prob_iou", float("nan")) >= 0.0
    )
    checks = {
        "finite": finite,
        "one_factor_positive": (positive == [label]) if label in NEW else (positive == []),
        "active_loss_positive_after_ramp": active_positive,
        "probability_branch_positive_when_enabled": probability_positive,
        "inactive_new_losses_zero": inactive_zero,
        "nonzero_generator_gradient": _grad_l2(model) > 0.0,
        "structure_sample": bool(batch.get("structure_available", [False])[0]),
        "no_resume": runtime["path"].get("resume_state") is None,
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "script": "smoke_stroke_single_factor_trainer.py",
        "config": str(option_path),
        "experiment_name": option.get("name"),
        "label": label,
        "current_iter": int(args.current_iter),
        "device": str(device),
        "dataset_length": len(dataset),
        "selected_indices": selected,
        "generator_gradient_l2": _grad_l2(model),
        "new_loss_values": loss_keys,
        "new_scales": scale_keys,
        "checks": checks,
        "logs": logs,
        "interpretation": "trainer plumbing only; no image-quality claim",
    }
    path = output_dir / "stroke_single_factor_trainer_smoke.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
