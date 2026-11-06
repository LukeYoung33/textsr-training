"""Generate small RGB montages for the isolated BASIC V2 Y4UV V3 models.

The utility uses the exact resize-to-96x480 dataset and the six-channel
Y4UV network contract.  It never writes to an experiment directory or loads
an optimizer/training state; the supplied model checkpoint is read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realesrgan.archs  # noqa: F401  # register the Y4UV architecture
import realesrgan.data  # noqa: F401  # register the Y4UV dataset
from basicsr.archs import build_network
from basicsr.data import build_dataset
from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444, y4uv_mask_to_yuv444
from realesrgan.utils import YUV2RGB


def _sha256(path: Path) -> str:
    """Return a checkpoint digest for the visual artifact provenance report."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_network(config_path: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Build the configured Y4UV network and strictly load its parameter mapping."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    network = build_network(config["network_g"]).to(device).eval()
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict):
        state = state.get("params_ema", state.get("params", state.get("state_dict", state)))
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a parameter mapping: {checkpoint}")
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    network.load_state_dict(state, strict=True)
    return network


def _select_indices(length: int, count: int, explicit: str | None) -> list[int]:
    """Select reproducible samples spread over one manifest."""
    if length <= 0:
        raise ValueError("dataset is empty")
    if explicit:
        indices = [int(value.strip()) for value in explicit.split(",") if value.strip()]
        if not indices or any(index < 0 or index >= length for index in indices):
            raise ValueError(f"indices must be within [0,{length})")
        return list(dict.fromkeys(indices))
    count = max(1, min(int(count), length))
    return sorted(set(int(round(value)) for value in np.linspace(0, length - 1, count)))


def _rgb_from_y4uv(value: torch.Tensor) -> np.ndarray:
    """Decode one normalized Y4UV sample to an RGB uint8 HWC image."""
    batched = value if value.ndim == 4 else value.unsqueeze(0)
    if batched.shape[0] != 1 or batched.shape[1] != 6:
        raise ValueError(f"expected one [6,H/2,W/2] sample, got {tuple(value.shape)}")
    yuv444 = y4uv_to_yuv444(batched)
    rgb = YUV2RGB(yuv444, format="RGB", is_train=False, uv_normalized=True)[0]
    return (
        rgb.detach().cpu().permute(1, 2, 0).numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)


def _rgb_from_chw(value: torch.Tensor) -> np.ndarray:
    """Convert one normalized RGB CHW tensor to RGB uint8 HWC."""
    return (value.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    """Write an RGB image as a PNG without changing its geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {path}")


def _label(image: np.ndarray, title: str) -> np.ndarray:
    """Add a compact label for montage inspection."""
    result = np.ascontiguousarray(np.array(image, copy=True))
    width = min(result.shape[1] - 1, 185)
    cv2.rectangle(result, (0, 0), (width, 24), (0, 0, 0), -1)
    cv2.putText(result, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Report normalized YUV444 and RGB MAE/PSNR for one prediction."""
    pred444 = y4uv_to_yuv444(pred.unsqueeze(0))[0]
    gt444 = y4uv_to_yuv444(gt.unsqueeze(0))[0]
    yuv_error = pred444.float() - gt444.float()
    yuv_mse = float(torch.mean(yuv_error.square()).item())
    pred_rgb = YUV2RGB(pred444.unsqueeze(0), format="RGB", is_train=False, uv_normalized=True)[0]
    gt_rgb = YUV2RGB(gt444.unsqueeze(0), format="RGB", is_train=False, uv_normalized=True)[0]
    rgb_error = pred_rgb.float() - gt_rgb.float()
    rgb_mse = float(torch.mean(rgb_error.square()).item())
    return {
        "yuv444_mae": float(torch.mean(yuv_error.abs()).item()),
        "yuv444_psnr": 10.0 * math.log10(1.0 / max(yuv_mse, 1e-12)),
        "rgb_mae": float(torch.mean(rgb_error.abs()).item()),
        "rgb_psnr": 10.0 * math.log10(1.0 / max(rgb_mse, 1e-12)),
    }


def _dump_sample(root: Path, split: str, number: int, index: int, item: dict[str, Any], prediction: torch.Tensor) -> dict[str, Any]:
    """Write source/decoded/prediction/target/difference images for one sample."""
    sample_dir = root / split / f"sample_{number:03d}_idx_{index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    lq_source = _rgb_from_chw(item["lq_rgb"])
    gt_source = _rgb_from_chw(item["gt_rgb"])
    lq_decoded = _rgb_from_y4uv(item["lq"])
    gt_decoded = _rgb_from_y4uv(item["gt"])
    pred_rgb = _rgb_from_y4uv(prediction)
    diff = np.abs(pred_rgb.astype(np.float32) - gt_decoded.astype(np.float32))
    diff = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    images = {
        "lq_source_rgb": lq_source,
        "lq_decoded_rgb": lq_decoded,
        "gt_source_rgb": gt_source,
        "gt_decoded_rgb": gt_decoded,
        "pred_rgb": pred_rgb,
        "diff_x4_rgb": diff,
    }
    for name, image in images.items():
        _write_rgb(sample_dir / f"{name}.png", image)
    montage = np.concatenate(
        [
            _label(lq_source, "LQ source"),
            _label(lq_decoded, "LQ decoded"),
            _label(pred_rgb, "PRED"),
            _label(gt_decoded, "GT"),
            _label(diff, "ABS DIFF x4"),
        ],
        axis=1,
    )
    _write_rgb(sample_dir / "montage.png", montage)
    np.savez_compressed(
        sample_dir / "tensors.npz",
        lq_y4uv=item["lq"].detach().cpu().numpy(),
        pred_y4uv=prediction.detach().cpu().numpy(),
        gt_y4uv=item["gt"].detach().cpu().numpy(),
    )
    record = {
        "split": split,
        "index": index,
        "segment_id": str(item.get("segment_id", "")),
        "line_id": str(item.get("line_id", "")),
        "transcription": str(item.get("transcription", "")),
        "lq_path": str(item.get("lq_path", "")),
        "gt_path": str(item.get("gt_path", "")),
        "input_shape": list(item["lq"].shape),
        "output_shape": list(prediction.shape),
        "finite": bool(torch.isfinite(prediction).all().item()),
        "pred_range": [float(prediction.min().item()), float(prediction.max().item())],
        "metrics": _metrics(prediction, item["gt"]),
        "files": sorted(str(path.relative_to(root)) for path in sample_dir.iterdir()),
    }
    # ASCII escaping keeps malformed legacy OCR strings from invalidating the report JSON.
    (sample_dir / "sample.json").write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run bounded inference for one config/checkpoint and one dataset split."""
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = build_dataset(config["datasets"][args.split])
    indices = _select_indices(len(dataset), args.count, args.indices)
    device = torch.device(args.device)
    network = _load_network(args.config, args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for number, index in enumerate(indices):
            item = dataset[index]
            batch = dataset.collate_fn([item]) if hasattr(dataset, "collate_fn") else {"lq": item["lq"].unsqueeze(0)}
            prediction = network(batch["lq"].to(device))[0].detach().cpu()
            if tuple(prediction.shape) != (6, 48, 240):
                raise RuntimeError(f"unexpected Y4UV output shape: {tuple(prediction.shape)}")
            records.append(_dump_sample(args.output_dir, args.split, number, index, item, prediction))
    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "split": args.split,
        "device": str(device),
        "dataset_length": len(dataset),
        "indices": indices,
        "records": records,
    }
    (args.output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    """Parse the bounded visualization contract and run it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--indices", default=None, help="comma-separated dataset indices; overrides --count")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({key: summary[key] for key in ("checkpoint", "split", "device", "dataset_length", "indices")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
