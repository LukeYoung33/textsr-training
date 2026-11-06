"""Smoke-dump a six-channel BASIC V2 Y4UV tensor and an RGB preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
from realesrgan.utils import YUV2RGB


def _load_tensor(path: Path) -> torch.Tensor:
    """@luna-comment Load a CHW/BCHW Y4UV tensor from a tensor or common checkpoint mapping."""
    try:
        value: Any = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(str(path), map_location="cpu")
    if isinstance(value, dict):
        for key in ("y4uv", "output", "pred", "tensor"):
            if isinstance(value.get(key), torch.Tensor):
                value = value[key]
                break
    if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4) or value.shape[-3] != 6:
        raise ValueError("input must be a CHW/BCHW six-channel Y4UV tensor")
    return value.float()


def y4uv_to_rgb_preview(y4uv: torch.Tensor) -> np.ndarray:
    """@luna-comment Decode one Y4UV sample to an RGB uint8 HWC preview."""
    value = y4uv if y4uv.ndim == 4 else y4uv.unsqueeze(0)
    if value.shape[0] != 1:
        raise ValueError("preview helper accepts exactly one sample")
    yuv444 = y4uv_to_yuv444(value)
    rgb = YUV2RGB(yuv444, format="RGB", is_train=False, uv_normalized=True)[0]
    return (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)


def dump_y4uv(tensor_path: Path, output_dir: Path) -> dict[str, Any]:
    """@luna-comment Save the native tensor, decoded RGB preview, and shape report."""
    tensor = _load_tensor(tensor_path)
    sample = tensor[0] if tensor.ndim == 4 else tensor
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_output = output_dir / "y4uv.pt"
    preview_output = output_dir / "preview_rgb.png"
    torch.save(sample, tensor_output)
    preview = y4uv_to_rgb_preview(sample)
    cv2.imwrite(str(preview_output), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    report = {
        "source": str(tensor_path),
        "tensor": str(tensor_output),
        "preview": str(preview_output),
        "input_contract": ["B", 6, int(tensor.shape[-2]), int(tensor.shape[-1])] if tensor.ndim == 4 else [6, int(tensor.shape[-2]), int(tensor.shape[-1])],
        "decoded_preview_contract": [int(preview.shape[0]), int(preview.shape[1]), 3],
    }
    (output_dir / "dump_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """@luna-comment Parse smoke-dump arguments and print the report."""
    parser = argparse.ArgumentParser(description="Dump BASIC V2 Y4UV V3 tensor")
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dump_y4uv(args.tensor, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
