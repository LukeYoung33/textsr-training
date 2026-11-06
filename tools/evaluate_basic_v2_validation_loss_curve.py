"""Evaluate BASIC V2 reconstruction loss across saved checkpoints.

The training jobs intentionally deferred periodic validation, so this tool
reconstructs an auditable validation curve from a deterministic, source-balanced
subset of the configured validation dataset.  It evaluates the same box-aware
YUV L1 definition used as ``l_base`` during training.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basicsr.archs import build_network

import realesrgan.archs  # noqa: F401,E402
from realesrgan.archs.basic_text_line_r4_student_v2_arch import load_strict_checkpoint  # noqa: E402
from realesrgan.data.basic_text_line_remote58_dataset import BasicTextLineRemote58Dataset  # noqa: E402


CHECKPOINT_RE = re.compile(r"^net_g_(\d+)\.pth$")


def checkpoint_series(directory: Path, stride: int) -> list[tuple[int, Path]]:
    """Return the first, regular-stride, and final numeric checkpoints."""
    found: list[tuple[int, Path]] = []
    for path in directory.glob("net_g_*.pth"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    if not found:
        raise FileNotFoundError(f"no numeric net_g checkpoints in {directory}")
    selected = [item for item in found if item[0] % stride == 0]
    for endpoint in (found[0], found[-1]):
        if endpoint not in selected:
            selected.append(endpoint)
    return sorted(selected)


def balanced_indices(dataset: BasicTextLineRemote58Dataset, samples_per_source: int) -> list[int]:
    """Choose deterministic evenly-spaced samples from every configured source."""
    groups: dict[int, list[int]] = {}
    for index, entry in enumerate(dataset._entries):
        groups.setdefault(int(entry["source_index"]), []).append(index)
    selected: list[int] = []
    for source_index in sorted(groups):
        values = groups[source_index]
        count = len(values) if samples_per_source <= 0 else min(samples_per_source, len(values))
        offsets = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
        selected.extend(values[int(offset)] for offset in offsets)
    return sorted(set(selected))


def _base_loss_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    text_mask: torch.Tensor,
    loss_opt: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return numerator and denominator of BoxAwareMaskedL1LossV2."""
    valid = valid_mask.to(device=prediction.device, dtype=prediction.dtype).clamp_(0.0, 1.0)
    text = text_mask.to(device=prediction.device, dtype=prediction.dtype).clamp_(0.0, 1.0) * valid
    foreground = float(loss_opt.get("fg_weight", 1.0))
    background = float(loss_opt.get("bg_weight", 0.5))
    channel_values = loss_opt.get("channel_weight", [1.0, 0.25, 0.25])
    channel = prediction.new_tensor([float(value) for value in channel_values]).view(1, -1, 1, 1)
    region = valid * (text * foreground + (1.0 - text) * background)
    effective = region * channel
    numerator = ((prediction - target).abs() * effective).sum() * float(loss_opt.get("loss_weight", 1.0))
    return numerator, effective.sum()


def evaluate_checkpoint(
    network: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    loss_opt: dict[str, Any],
) -> tuple[float, int]:
    """Evaluate one checkpoint and return global masked L1 plus sample count."""
    numerator = 0.0
    denominator = 0.0
    samples = 0
    network.eval()
    with torch.inference_mode():
        for batch in loader:
            lq = batch["lq"].to(device=device, non_blocking=True)
            target = batch["gt"].to(device=device, non_blocking=True)
            prediction = network(lq)
            num, den = _base_loss_sums(
                prediction,
                target,
                batch["valid_mask"],
                batch["text_mask"],
                loss_opt,
            )
            numerator += float(num.detach().cpu())
            denominator += float(den.detach().cpu())
            samples += int(lq.shape[0])
    if denominator <= 0.0:
        raise RuntimeError("validation effective loss weight is zero")
    return numerator / denominator, samples


def main() -> int:
    """Evaluate selected checkpoints and write an append-safe CSV curve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoints-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-source", type=int, default=96)
    parser.add_argument("--checkpoint-stride", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.checkpoint_stride <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("stride/batch-size must be positive and num-workers non-negative")
    config_path = args.config.expanduser().resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    val_opt = dict(document["datasets"]["val"])
    val_opt["phase"] = "val"
    dataset = BasicTextLineRemote58Dataset(val_opt)
    indices = balanced_indices(dataset, args.samples_per_source)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=dataset.collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    network = build_network(dict(document["network_g"])).to(device)
    checkpoints = checkpoint_series(args.checkpoints_dir.expanduser().resolve(), args.checkpoint_stride)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[int, dict[str, str]] = {}
    if output.is_file():
        with output.open("r", newline="", encoding="utf-8") as handle:
            existing = {int(row["iteration"]): row for row in csv.DictReader(handle)}

    rows: list[dict[str, object]] = []
    for iteration, checkpoint in checkpoints:
        if iteration in existing:
            rows.append(existing[iteration])
            print(f"skip iteration={iteration} existing={existing[iteration]['val_l_base']}", flush=True)
            continue
        sha256 = load_strict_checkpoint(network, checkpoint, map_location="cpu")
        value, sample_count = evaluate_checkpoint(network, loader, device, dict(document["train"]["pixel_opt"]))
        row = {
            "iteration": iteration,
            "val_l_base": f"{value:.10g}",
            "samples": sample_count,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256,
            "resize_mode": str(val_opt.get("resize_mode", "")),
            "canvas_hw": "x".join(str(value) for value in val_opt.get("canvas_hw", [])),
        }
        rows.append(row)
        print(f"iteration={iteration} val_l_base={value:.8f} samples={sample_count}", flush=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda item: int(item["iteration"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
