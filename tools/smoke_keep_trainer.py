"""Run the final Keep-enabled V4 trainer one-step smoke.

The probe uses an actual V2 batch and the configured GT-degradation path. It
does one optimizer step at post-ramp iteration 7000, verifies all native/CP/
Recall/bgLF_guard/Keep logs, and records a nonzero generator gradient. It is
plumbing evidence only and never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from basicsr.data import build_dataset
from basicsr.models import build_model

# Import registry scanners before constructing the configured trainer.
import realesrgan.archs  # noqa: F401,E402
import realesrgan.data  # noqa: F401,E402
import realesrgan.losses  # noqa: F401,E402
import realesrgan.models  # noqa: F401,E402


def _parser() -> argparse.ArgumentParser:
    """Build the trainer smoke command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", "--opt", required=True, help="Keep-enabled V2 training YAML")
    parser.add_argument("--output-dir", required=True, help="JSON report directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--current-iter", type=int, default=7000)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="optional exact dataset indices, e.g. 0 103 for both polarities",
    )
    return parser


def _device(value: str) -> torch.device:
    """Resolve a smoke device without changing the training YAML."""

    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _float(value: Any) -> float:
    """Convert a scalar tensor or number into a JSON-safe float."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def _remap_local_cp_path(opt: dict[str, Any]) -> None:
    """Use a repository-local CRNN checkpoint only when the configured path is absent."""

    cp_opt = opt.get("train", {}).get("cp_opt")
    if not isinstance(cp_opt, dict):
        return
    path = Path(str(cp_opt.get("recognizer_path", "")))
    if path.is_file():
        return
    candidate = Path(__file__).resolve().parents[1] / "weights" / path.name
    if candidate.is_file():
        cp_opt["recognizer_path"] = str(candidate)


def _build_batch(
    opt: dict[str, Any],
    batch_size: int,
    max_samples: int,
    indices: list[int] | None,
) -> tuple[dict[str, Any], Any, list[int]]:
    """Build a collated batch from exactly the configured V2 train source."""

    train_opt = copy.deepcopy(opt["datasets"]["train"])
    expected = "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2"
    sources = train_opt.get("sources", [])
    if sources != [{"root": expected, "weight": 1.0}]:
        raise ValueError(f"Keep trainer smoke requires exactly one V2 source, got {sources!r}")
    train_opt.update({"phase": "train", "batch_size_per_gpu": int(batch_size), "num_worker_per_gpu": 0})
    dataset = build_dataset(train_opt)
    wanted = max(1, int(max_samples))
    candidates = [int(value) for value in indices] if indices else list(range(len(dataset)))
    samples = []
    selected_indices = []
    for index in candidates:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"dataset index out of range: {index}")
        sample = dataset[index]
        if bool(sample.get("structure_available", False)):
            samples.append(sample)
            selected_indices.append(index)
        if len(samples) >= wanted and not indices:
            break
    if len(samples) < wanted:
        raise RuntimeError(f"found {len(samples)} verified rows; need {wanted}")
    return dataset.collate_fn(samples), dataset, selected_indices


def _runtime_options(opt: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Prepare isolated BasicSR paths for this one-step probe."""

    runtime = copy.deepcopy(opt)
    runtime.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1})
    runtime["num_gpu"] = 1 if device.type == "cuda" else 0
    runtime["root_path"] = str(Path(__file__).resolve().parents[1])
    runtime["path"] = copy.deepcopy(runtime.get("path", {}))
    runtime["path"].setdefault("models", str(Path(runtime["root_path"]) / ".tmp_keep_trainer_models"))
    runtime["path"].setdefault("training_states", str(Path(runtime["root_path"]) / ".tmp_keep_trainer_states"))
    runtime["path"].setdefault("log", str(Path(runtime["root_path"]) / ".tmp_keep_trainer_logs"))
    _remap_local_cp_path(runtime)
    return runtime


def _generator_gradient_l2(model: Any) -> float:
    """Return the post-step L2 norm of available generator parameter gradients."""

    total = None
    for parameter in model.net_g.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt().cpu().item())


def main() -> int:
    """Run one final calibrated trainer step and write the evidence report."""

    args = _parser().parse_args()
    option_path = Path(args.opt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    opt = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    if not isinstance(opt, dict):
        raise ValueError("training YAML must decode to a mapping")
    train_opt = opt.get("train", {})
    keep_opt = copy.deepcopy(train_opt.get("keep_opt") or train_opt.get("keep_loss_opt") or {})
    if keep_opt.pop("type", None) != "KeepLoss":
        raise ValueError("trainer smoke config must select KeepLoss")
    expected = "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2"
    sources = opt.get("datasets", {}).get("train", {}).get("sources", [])
    if sources != [{"root": expected, "weight": 1.0}]:
        raise ValueError(f"trainer smoke requires exactly one V2 source, got {sources!r}")
    seed = int(opt.get("manual_seed", 0))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    batch, dataset, selected_indices = _build_batch(opt, args.batch_size, args.max_samples, args.indices)
    model = build_model(_runtime_options(opt, device))
    model.feed_data(batch)
    model.optimize_parameters(current_iter=int(args.current_iter))
    logs = {name: _float(value) for name, value in model.get_current_log().items()}
    output = model.output.detach()
    required = (
        "l_native",
        "l_cp_raw",
        "l_cp",
        "l_missing",
        "l_false",
        "l_bg_lf_raw",
        "l_bg_lf_weighted",
        "bg_lf_guard_active",
        "l_keep_raw",
        "l_keep_weighted",
        "l_keep",
        "keep_scale",
        "keep_support_count",
        "keep_visible_to_lost_rate",
        "keep_thin_recall",
        "l_total",
    )
    missing = [name for name in required if name not in logs]
    finite = all(math.isfinite(value) for value in logs.values()) and bool(torch.isfinite(output).all())
    gradient_l2 = _generator_gradient_l2(model)
    checks = {
        "v2_only_source": sources == [{"root": expected, "weight": 1.0}],
        "actual_trainer_step": True,
        "finite_loss_and_output": bool(finite),
        "nonzero_generator_gradient": gradient_l2 > 0.0,
        "required_logs_present": not missing,
        "bg_lf_guard_active": logs.get("bg_lf_guard_active", 0.0) == 1.0,
        "keep_scale_positive": logs.get("keep_scale", 0.0) > 0.0,
        "keep_support_nonempty": logs.get("keep_support_count", 0.0) > 0.0,
        "keep_raw_finite": math.isfinite(logs.get("l_keep_raw", float("nan"))),
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "script": "smoke_keep_trainer.py",
        "config": str(option_path),
        "experiment_name": opt.get("name"),
        "device": str(device),
        "seed": seed,
        "current_iter": int(args.current_iter),
        "dataset": {
            "type": opt["datasets"]["train"].get("type"),
            "length": len(dataset),
            "sample_count": len(selected_indices),
            "selected_indices": selected_indices,
            "sources": sources,
        },
        "keep_opt": opt["train"].get("keep_opt", opt["train"].get("keep_loss_opt", {})),
        "output_shape": list(output.shape),
        "decoded_rgb_shape": [int(output.shape[0]), 3, int(output.shape[-2] * 2), int(output.shape[-1] * 2)],
        "generator_gradient_l2": gradient_l2,
        "logs": logs,
        "missing_logs": missing,
        "checks": checks,
        "interpretation": {
            "trainer_path": "actual model.feed_data(batch) with kernels, then optimize_parameters(current_iter)",
            "keep_support": "detached GT skeleton/tube plus GT/LQ polarity-consistent signed ring contrast",
            "no_model_branch": True,
            "quality_claim": False,
        },
    }
    path = output_dir / "keep_trainer_smoke.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(path), "gradient_l2": gradient_l2, "missing_logs": missing}, indent=2))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
