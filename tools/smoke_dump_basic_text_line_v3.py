"""Run a bounded BASIC V3 text-line smoke and dump inspectable YUV samples.

The dump is intentionally independent from the long-running formal job. It
uses the same segment dataset, R4 Student architecture, packed NV21 input,
YUV444 target/output contract, and BoxAware baseline loss. It saves both the
pre-update and post-update predictions so that data, color conversion, masks,
and one short training path can be inspected together.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basicsr.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY
from realesrgan.losses.basic_text_line_p0_losses import BoxAwareMaskedL1LossV2


DEFAULT_MANIFEST = "/AI_DNS/guang.yang/TextSR/data/IQ_data_grouped_560/basic_v2_segment_manifest_20260815/val.jsonl"
DEFAULT_CHECKPOINT = "/home/guang.yang/RealCE/experiments/train_bucket_3/pretrain_realce_ECBSRUnet_student_YUV_r3/models/net_g_latest.pth"
DEFAULT_OUTPUT = "/home/guang.yang/workspace/Real-ESRGAN/results/basic_text_line_v3_remote58_dump_20260816"


def _parse_args() -> argparse.Namespace:
    """Parse the bounded dump contract and reproducibility controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def _load_student(checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Build the fixed R4 Student and strictly load the frozen checkpoint."""
    network = ARCH_REGISTRY.get("BasicTextLineR4StudentV2")(
        canvas_hw=[96, 320],
        in_channel=12,
        out_channel=12,
        layer_num=3,
        expand_channel=[0.5, 1.0, 2.0, 4.0],
        depth_factor=1,
        scale_factor=2,
        with_idt=False,
        act_type="prelu",
        scale=2,
        with_proj=True,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("params_ema", state.get("params", state.get("state_dict", state)))
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {checkpoint}")
    network.load_state_dict(state, strict=True)
    return network.to(device)


def _select_indices(length: int, count: int) -> List[int]:
    """Select deterministic samples spread across the validation manifest."""
    if length <= 0:
        raise ValueError("manifest has no accepted records")
    count = max(1, min(int(count), length))
    return sorted(set(int(round(v)) for v in np.linspace(0, length - 1, count)))


def _u8(value: np.ndarray) -> np.ndarray:
    """Convert a normalized image array into uint8 without changing geometry."""
    return np.clip(np.rint(np.asarray(value, np.float32) * 255.0), 0, 255).astype(np.uint8)


def _nv21_to_rgb(packed: torch.Tensor) -> np.ndarray:
    """Decode one canonical packed NV21 tensor to an RGB image for inspection."""
    value = packed.detach().cpu().numpy()
    if value.ndim == 3:
        value = value[0]
    return cv2.cvtColor(_u8(value), cv2.COLOR_YUV2RGB_NV21)


def _yuv444_to_rgb(yuv: torch.Tensor | np.ndarray) -> np.ndarray:
    """Decode one planar normalized YUV444 CHW tensor to RGB."""
    value = yuv.detach().cpu().numpy() if isinstance(yuv, torch.Tensor) else np.asarray(yuv)
    if value.ndim == 4:
        value = value[0]
    value = np.transpose(value, (1, 2, 0))
    return cv2.cvtColor(_u8(value), cv2.COLOR_YUV2RGB)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    """Write RGB data as a standard BGR PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {path}")


def _write_gray(path: Path, value: np.ndarray) -> None:
    """Write a normalized mask as an 8-bit grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), _u8(value)):
        raise IOError(f"failed to write {path}")


def _label(image: np.ndarray, title: str) -> np.ndarray:
    """Add a small ASCII label to one RGB image for montage inspection."""
    result = np.array(image, copy=True)
    cv2.rectangle(result, (0, 0), (min(result.shape[1] - 1, 180), 24), (0, 0, 0), -1)
    cv2.putText(result, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """Compute valid-region YUV metrics for one dumped prediction."""
    mask = np.asarray(valid, np.float32)[..., None]
    error = np.asarray(pred, np.float32) - np.asarray(target, np.float32)
    denom = max(float(mask.sum()) * 3.0, 1.0)
    l1 = float((np.abs(error) * mask).sum() / denom)
    mse = float(((error * error) * mask).sum() / denom)
    return {"valid_l1_yuv444": l1, "valid_mse_yuv444": mse, "valid_psnr_yuv444": 10.0 * math.log10(1.0 / max(mse, 1e-12))}


def _dump_stage(
    root: Path,
    stage: str,
    sample_number: int,
    item: Dict[str, Any],
    prediction: torch.Tensor,
) -> Dict[str, Any]:
    """Write visual images and exact normalized tensors for one sample/stage."""
    prefix = f"sample_{sample_number:03d}_{stage}"
    sample_dir = root / f"sample_{sample_number:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    lq_rgb = _nv21_to_rgb(item["lq"])
    gt_rgb = _yuv444_to_rgb(item["gt"])
    pred_rgb = _yuv444_to_rgb(prediction)
    gt_yuv = item["gt"].detach().cpu().numpy()
    pred_yuv = prediction.detach().cpu().numpy()
    valid = item["valid_mask"].detach().cpu().numpy()[0]
    text = item["text_mask"].detach().cpu().numpy()[0]
    diff_rgb = np.clip(np.abs(pred_rgb.astype(np.float32) - gt_rgb.astype(np.float32)) * 4.0, 0, 255).astype(np.uint8)
    _write_rgb(sample_dir / f"{prefix}_lq_rgb.png", lq_rgb)
    _write_rgb(sample_dir / f"{prefix}_gt_rgb.png", gt_rgb)
    _write_rgb(sample_dir / f"{prefix}_pred_rgb.png", pred_rgb)
    _write_rgb(sample_dir / f"{prefix}_diff_x4_rgb.png", diff_rgb)
    _write_gray(sample_dir / f"{prefix}_valid_mask.png", valid)
    _write_gray(sample_dir / f"{prefix}_text_mask.png", text)
    panels = [_label(lq_rgb, "LQ NV21"), _label(gt_rgb, "GT YUV444"), _label(pred_rgb, "PRED YUV444"), _label(diff_rgb, "ABS DIFF x4")]
    montage = np.concatenate(panels, axis=1)
    _write_rgb(sample_dir / f"{prefix}_montage.png", montage)
    np.savez_compressed(
        sample_dir / f"{prefix}_tensors.npz",
        lq_nv21=item["lq"].detach().cpu().numpy(),
        gt_yuv444=gt_yuv,
        pred_yuv444=pred_yuv,
        valid_mask=valid,
        text_mask=text,
    )
    metrics = _metrics(pred_yuv.transpose(1, 2, 0), gt_yuv.transpose(1, 2, 0), valid)
    left, top, right, bottom = (int(value) for value in item["crop_xyxy_even"])
    return {
        "stage": stage,
        "sample_number": sample_number,
        "segment_id": item["segment_id"],
        "line_id": item["line_id"],
        "orientation": item["orientation"],
        "valid_hw": [bottom - top, right - left],
        "crop_xyxy_even": list(item["crop_xyxy_even"]),
        "cp_eligible": bool(item["cp_eligible"]),
        "transcription": item["transcription"],
        "files": sorted(str(path.relative_to(root)) for path in sample_dir.glob(f"{prefix}_*")),
        **metrics,
    }


def main() -> int:
    """Run the deterministic inference/training dump and write its summary."""
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    dataset = DATASET_REGISTRY.get("BasicTextLineSegmentDatasetV3")(
        {
            "manifest": args.manifest,
            "canvas_hw": [96, 320],
            "min_short_side_ratio": 0.5,
            "min_long_side_ratio": 0.5,
        }
    )
    indices = _select_indices(len(dataset), args.num_samples)
    items = [dataset[index] for index in indices]
    network = _load_student(args.checkpoint, device)
    optimizer = torch.optim.Adam((parameter for parameter in network.parameters() if parameter.requires_grad), lr=1e-4)
    loss_fn = BoxAwareMaskedL1LossV2(loss_weight=1.0, fg_weight=1.0, bg_weight=0.5, channel_weight=[1.0, 0.25, 0.25]).to(device)
    records: List[Dict[str, Any]] = []

    network.eval()
    with torch.inference_mode():
        for sample_number, item in enumerate(items):
            batch = dataset.collate_fn([item])
            prediction = network(batch["lq"].to(device))[0].detach().cpu()
            records.append(_dump_stage(root, "before", sample_number, item, prediction))

    train_losses: List[float] = []
    network.train()
    for step in range(max(0, int(args.train_steps))):
        item = items[step % len(items)]
        batch = dataset.collate_fn([item])
        lq = batch["lq"].to(device)
        gt = batch["gt"].to(device)
        valid = batch["valid_mask"].to(device)
        text = batch["text_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = network(lq)
        loss = loss_fn(prediction, gt, valid_mask=valid, text_mask=text)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite smoke loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    network.eval()
    with torch.inference_mode():
        for sample_number, item in enumerate(items):
            batch = dataset.collate_fn([item])
            prediction = network(batch["lq"].to(device))[0].detach().cpu()
            records.append(_dump_stage(root, "after", sample_number, item, prediction))

    summary = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "output": str(root),
        "device": str(device),
        "dataset_length": len(dataset),
        "indices": indices,
        "num_samples": len(items),
        "train_steps": int(args.train_steps),
        "train_losses": train_losses,
        "model_parameters": sum(parameter.numel() for parameter in network.parameters()),
        "input_shape": [1, 1, 144, 320],
        "output_shape": [1, 3, 96, 320],
        "records": records,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("output", "indices", "train_steps", "train_losses", "model_parameters", "input_shape", "output_shape")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
