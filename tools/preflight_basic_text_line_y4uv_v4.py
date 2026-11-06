"""Preflight one V4 YAML against its real filtered dataset and network graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import realesrgan.archs  # noqa: F401 - registry side effect
import realesrgan.data  # noqa: F401 - registry side effect
import realesrgan.models  # noqa: F401 - registry side effect
from basicsr.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY


def run_preflight(
    config_path: Path,
    *,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    """@shanlan-comment Load real V4 data, stack one batch, and run finite model inference."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_opt = dict(config["datasets"]["train"])
    dataset_opt["phase"] = "train"
    dataset_type = str(dataset_opt["type"])
    dataset_cls = DATASET_REGISTRY.get(dataset_type)
    if dataset_cls is None:
        raise KeyError(f"dataset is not registered: {dataset_type}")
    dataset = dataset_cls(dataset_opt)
    count = min(int(batch_size), len(dataset))
    if count <= 0:
        raise RuntimeError("filtered dataset is empty")
    batch = dataset.collate_fn([dataset[index] for index in range(count)])

    network_opt = dict(config["network_g"])
    network_type = str(network_opt.pop("type"))
    network_cls = ARCH_REGISTRY.get(network_type)
    if network_cls is None:
        raise KeyError(f"network is not registered: {network_type}")
    model = network_cls(**network_opt).eval()
    runtime_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(runtime_device)
    input_tensor = batch["lq"].to(runtime_device)
    with torch.inference_mode():
        output = model(input_tensor)
    expected = (count, 6, 48, 240)
    if tuple(output.shape) != expected:
        raise RuntimeError(f"V4 output shape mismatch: {tuple(output.shape)} vs {expected}")
    if not torch.isfinite(output).all():
        raise RuntimeError("V4 output contains NaN or Inf")
    model_type = str(config.get("model_type", ""))
    if MODEL_REGISTRY.get(model_type) is None:
        raise KeyError(f"training model is not registered: {model_type}")
    orientations = [str(value) for value in batch.get("orientation", [])]
    return {
        "status": "passed",
        "config": str(config_path),
        "architecture": network_type,
        "training_model": model_type,
        "dataset": dataset_type,
        "device": str(runtime_device),
        "batch_size": count,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "finite": True,
        "haar_switch": bool(getattr(model, "haar_switch", False)),
        "stage_shapes": [list(shape) for shape in model.stage_shapes],
        "stage_volumes": list(model.stage_volumes),
        "dataset_length": len(dataset),
        "dataset_filter_stats": dataset.size_filter_stats,
        "cp_eligible_in_batch": int(sum(bool(value) for value in batch.get("cp_eligible", []))),
        "orientations_in_batch": orientations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """@shanlan-comment Run the V4 preflight and print a JSON evidence record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(json.dumps(run_preflight(args.opt, batch_size=args.batch_size, device=args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
