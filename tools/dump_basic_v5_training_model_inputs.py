"""Materialize exact V5 dataset outputs at the model-input boundary for audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.data.basic_text_line_y4uv_v4_cached_dataset import (
    BasicTextLineSegmentDatasetY4UVV4CachedNativeCrop96x480,
    _native_to_rgb,
)
from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import BasicTextLineR4StudentY4UVV4


def _to_bgr(tensor: torch.Tensor) -> np.ndarray:
    rgb = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _read_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _panel(images: list[tuple[str, np.ndarray]], target: Path) -> None:
    width = max(image.shape[1] for _, image in images)
    height = max(image.shape[0] for _, image in images)
    tiles = []
    for title, image in images:
        canvas = np.full((height + 30, width, 3), 245, dtype=np.uint8)
        canvas[:image.shape[0], :image.shape[1]] = image
        cv2.putText(canvas, title, (6, height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    cv2.imwrite(str(target), np.concatenate((np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)), axis=0))


def _stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().numpy()
    return {"shape": list(value.shape), "min": [float(x) for x in value.reshape(value.shape[0], -1).min(axis=1)], "max": [float(x) for x in value.reshape(value.shape[0], -1).max(axis=1)], "mean": [float(x) for x in value.reshape(value.shape[0], -1).mean(axis=1)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contains", action="append", required=True, help="substring matched against source_line_id")
    parser.add_argument("--augmentation-seed", type=int, default=20260901)
    parser.add_argument("--checkpoint", type=Path, help="optional checkpoint for direct, pre-compositor prediction")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_opt = dict(config["datasets"]["train"])
    train_opt["augmentation_seed"] = int(args.augmentation_seed)
    dataset = BasicTextLineSegmentDatasetY4UVV4CachedNativeCrop96x480(train_opt)
    model = None
    if args.checkpoint:
        model = BasicTextLineR4StudentY4UVV4(canvas_hw=(96, 480), haar_switch=True).eval()
        state = torch.load(args.checkpoint, map_location="cpu")
        params = state.get("params_ema") or state.get("params") or state
        model.load_state_dict(params, strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for needle in args.contains:
        index = next((i for i, item in enumerate(dataset.records) if needle in item["source_line_id"]), None)
        if index is None:
            raise ValueError(f"no training record matches {needle!r}")
        sample = dataset[index]
        name = f"{index:05d}_{sample['segment_id'].replace(':', '_')}"
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        raw_lq, raw_gt = _read_bgr(sample["lq_path"]), _read_bgr(sample["gt_path"])
        model_lq, model_gt = _to_bgr(sample["lq_rgb"]), _to_bgr(sample["gt_rgb"])
        cv2.imwrite(str(folder / "01_cached_native_lq.png"), raw_lq)
        cv2.imwrite(str(folder / "02_cached_native_gt.png"), raw_gt)
        cv2.imwrite(str(folder / "03_model_input_lq_rgb.png"), model_lq)
        cv2.imwrite(str(folder / "04_model_target_gt_rgb.png"), model_gt)
        np.save(folder / "model_input_lq_y4uv.npy", sample["lq"].numpy())
        np.save(folder / "model_target_gt_y4uv.npy", sample["gt"].numpy())
        if model is not None:
            with torch.inference_mode():
                prediction = model(sample["lq"].unsqueeze(0))[0].cpu()
            prediction_rgb = _to_bgr(_native_to_rgb(prediction.numpy()))
            cv2.imwrite(str(folder / "05_direct_model_prediction_rgb.png"), prediction_rgb)
            np.save(folder / "direct_model_prediction_y4uv.npy", prediction.numpy())
            model_io_target = np.concatenate((model_lq, prediction_rgb, model_gt), axis=1)
            cv2.putText(model_io_target, "input LQ", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(model_io_target, "direct model output", (490, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(model_io_target, "training GT", (970, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(folder / "06_direct_model_io_target.png"), model_io_target)
        _panel([
            ("cached native LQ", raw_lq), ("cached native GT", raw_gt),
            ("model input LQ: decoded from exact Y4UV", model_lq),
            ("model target GT: decoded from exact Y4UV", model_gt),
        ], folder / "panel.png")
        metadata = {key: value for key, value in sample.items() if key not in {"lq", "gt", "lq_rgb", "gt_rgb", "valid_mask", "text_mask"}}
        metadata.update({
            "augmentation_seed": int(args.augmentation_seed),
            "model_input_lq_y4uv": _stats(sample["lq"]),
            "model_target_gt_y4uv": _stats(sample["gt"]),
            "direct_checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "audit_contract": "model_input/target RGB are regenerated from the exact six-channel tensors supplied by dataset.__getitem__",
        })
        (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"match": needle, "index": index, "folder": str(folder), "augmentation": sample.get("augmentation_metadata", None)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
