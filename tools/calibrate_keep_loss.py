"""Calibrate Keep's generator-output gradient against native+CP on V2.

This probe builds the configured V4 trainer, feeds one real V2 batch through
the same GT-degradation path, and measures per-sample L2 norms with respect to
the six-channel network output. It never calls ``optimize_parameters`` and
does not update a checkpoint or optimizer. The selected weight is the nearest
candidate to the midpoint of the required median ratio interval [0.10, 0.25].
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
from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
from realesrgan.utils import YUV2RGB

# Populate all registries before dataset/model construction.
import realesrgan.archs  # noqa: F401,E402
import realesrgan.data  # noqa: F401,E402
import realesrgan.losses  # noqa: F401,E402
import realesrgan.models  # noqa: F401,E402


def _parser() -> argparse.ArgumentParser:
    """Build the calibration command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", "--opt", required=True, help="Keep-enabled V2 training YAML")
    parser.add_argument("--output-dir", required=True, help="JSON report directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--current-iter", type=int, default=7000)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="optional exact dataset indices, e.g. 0 103 1 104 for both polarities",
    )
    return parser


def _device(value: str) -> torch.device:
    """Resolve a calibration device without changing the training YAML."""

    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _float(value: Any) -> float:
    """Convert a scalar tensor or number to a JSON-safe float."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def _remap_local_cp_path(opt: dict[str, Any]) -> None:
    """Use a repository-local CRNN file only when the configured path is absent."""

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
    """Build a bounded collated batch from the exact configured V2 root."""

    train_opt = copy.deepcopy(opt["datasets"]["train"])
    expected = "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2"
    sources = train_opt.get("sources", [])
    if sources != [{"root": expected, "weight": 1.0}]:
        raise ValueError(f"Keep calibration requires exactly one V2 source, got {sources!r}")
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
    """Prepare isolated BasicSR runtime paths for model construction."""

    runtime = copy.deepcopy(opt)
    runtime.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1})
    runtime["num_gpu"] = 1 if device.type == "cuda" else 0
    runtime["root_path"] = str(Path(__file__).resolve().parents[1])
    runtime["path"] = copy.deepcopy(runtime.get("path", {}))
    runtime["path"].setdefault("models", str(Path(runtime["root_path"]) / ".tmp_keep_calibration_models"))
    runtime["path"].setdefault("training_states", str(Path(runtime["root_path"]) / ".tmp_keep_calibration_states"))
    runtime["path"].setdefault("log", str(Path(runtime["root_path"]) / ".tmp_keep_calibration_logs"))
    _remap_local_cp_path(runtime)
    return runtime


def _norms(gradient: torch.Tensor) -> torch.Tensor:
    """Return one L2 norm per batch sample from an output gradient."""

    return gradient.detach().float().flatten(1).norm(p=2, dim=1)


def _calibrate(
    model: Any,
    batch: dict[str, Any],
    device: torch.device,
    current_iter: int,
) -> dict[str, Any]:
    """Measure native+CP and Keep output gradients and candidate ratios."""

    # GT degradation is part of the formal trainer path; keep all generated
    # LQ/RGB metadata from feed_data and only remove no data fields.
    model.feed_data(batch)
    model.net_g.eval()
    output = model.net_g(model.lq)
    native = model.cri_y4uv_native(
        output,
        model.gt,
        valid_mask=model.valid_mask,
        text_mask=model.text_mask,
    )
    output444 = y4uv_to_yuv444(output)
    output_rgb = YUV2RGB(output444, format="RGB", is_train=True, uv_normalized=True)
    cp_weight = model._schedule(float(model.opt["train"].get("cp_weight", 0.0)), current_iter)
    if model.cri_cp is None or cp_weight <= 0.0:
        raise RuntimeError("calibration requires the configured native+CP branch")
    if bool(getattr(model.cri_cp, "expects_segment_metadata", False)):
        cp_raw = model.cri_cp(output_rgb, model.gt_rgb, model.segment_metadata)
    else:
        lines = model._build_cp_annotations(model.segment_metadata, output.shape[0])
        cp_raw = model.cri_cp(output_rgb, model.gt_rgb, lines)
    baseline = native + cp_weight * cp_raw
    baseline_gradient = torch.autograd.grad(baseline, output, retain_graph=True, allow_unused=False)[0]
    if model.cri_keep is None:
        raise RuntimeError("calibration requires the configured Keep branch")
    keep_raw, details = model.cri_keep(
        output_rgb,
        model.gt_rgb,
        model.lq_rgb,
        text_mask_rgb=model.text_mask_rgb,
        valid_mask_rgb=model.valid_mask_rgb,
        structure_available=getattr(model, "structure_available", []),
        return_details=True,
    )
    keep_gradient = torch.autograd.grad(keep_raw, output, retain_graph=False, allow_unused=False)[0]
    baseline_norm = _norms(baseline_gradient)
    keep_norm = _norms(keep_gradient)
    if not bool(torch.isfinite(baseline_norm).all() and torch.isfinite(keep_norm).all()):
        raise RuntimeError("non-finite output gradient norm")
    if not bool((baseline_norm > 0.0).all()):
        raise RuntimeError(f"native+CP output gradient has zero sample: {baseline_norm.tolist()}")
    if not bool((keep_norm > 0.0).any()):
        raise RuntimeError(
            "Keep selected no support on this degraded batch; choose a batch with visible LQ strokes"
        )
    ratios = keep_norm / baseline_norm.clamp_min(torch.finfo(baseline_norm.dtype).eps)
    candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0, 2.0, 4.0]
    candidate_ratios = {str(weight): _float((ratios * weight).median()) for weight in candidates}
    valid_candidates = [
        (abs(value - 0.175), weight, value)
        for weight, value in ((weight, candidate_ratios[str(weight)]) for weight in candidates)
        if 0.10 <= value <= 0.25
    ]
    if not valid_candidates:
        raise RuntimeError(f"no candidate reaches median ratio [0.10,0.25]: {candidate_ratios}")
    _, selected_weight, selected_ratio = min(valid_candidates)
    scalar_details = {
        name: _float(value)
        for name, value in details.items()
        if isinstance(value, torch.Tensor) and value.ndim == 0
    }
    return {
        "current_iter": int(current_iter),
        "output_shape": list(output.shape),
        "decoded_rgb_shape": list(output_rgb.shape),
        "native": _float(native),
        "cp_raw": _float(cp_raw),
        "cp_weight": float(cp_weight),
        "native_plus_cp": _float(baseline),
        "keep_raw": _float(keep_raw),
        "baseline_output_gradient_l2_per_sample": [_float(value) for value in baseline_norm],
        "keep_output_gradient_l2_per_sample": [_float(value) for value in keep_norm],
        "raw_keep_to_native_plus_cp_ratio_per_sample": [_float(value) for value in ratios],
        "median_baseline_output_gradient_l2": _float(baseline_norm.median()),
        "median_keep_output_gradient_l2": _float(keep_norm.median()),
        "median_raw_ratio": _float(ratios.median()),
        "candidate_median_ratios": candidate_ratios,
        "selected_keep_weight": float(selected_weight),
        "selected_median_ratio": float(selected_ratio),
        "keep_diagnostics": scalar_details,
    }


def main() -> int:
    """Run calibration and write a JSON report."""

    args = _parser().parse_args()
    option_path = Path(args.opt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    opt = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    if not isinstance(opt, dict):
        raise ValueError("training YAML must decode to a mapping")
    keep_opt = copy.deepcopy(opt.get("train", {}).get("keep_opt") or opt.get("train", {}).get("keep_loss_opt") or {})
    if keep_opt.pop("type", None) != "KeepLoss":
        raise ValueError("calibration config must select KeepLoss")
    seed = int(opt.get("manual_seed", 0))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    batch, dataset, selected_indices = _build_batch(opt, args.batch_size, args.max_samples, args.indices)
    runtime = _runtime_options(opt, device)
    model = build_model(runtime)
    report = _calibrate(model, batch, device, args.current_iter)
    payload = {
        "status": "pass",
        "script": "calibrate_keep_loss.py",
        "config": str(option_path),
        "experiment_name": opt.get("name"),
        "device": str(device),
        "seed": seed,
        "dataset": {
            "type": opt["datasets"]["train"].get("type"),
            "length": len(dataset),
            "sample_count": len(selected_indices),
            "selected_indices": selected_indices,
            "sources": opt["datasets"]["train"].get("sources", []),
        },
        "keep_opt": keep_opt,
        "calibration": report,
        "acceptance": {
            "metric": "per-sample L2 norm of d(loss)/d(six-channel network output), median ratio",
            "target_interval": [0.10, 0.25],
            "selected_weight_in_interval": 0.10 <= report["selected_median_ratio"] <= 0.25,
            "no_optimizer_step": True,
        },
    }
    path = output_dir / "keep_calibration.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(path), "selected_keep_weight": report["selected_keep_weight"], "selected_median_ratio": report["selected_median_ratio"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
