"""Write color-correct, per-sample RGB validation comparisons for BASIC V2.

The training dataset stores LQ as packed NV21 and GT/model predictions as
normalized YUV444 (with U/V centered at 0.5).  BasicSR's generic validation
writer treats those tensors as RGB, which is not a valid visualization.  This
tool explicitly decodes both representations before writing PNG artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

import realesrgan.data  # noqa: F401  Register the project dataset.
from realesrgan.data.basic_text_line_remote58_dataset import BasicTextLineRemote58Dataset
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.utils import YUV2RGB
from tools.basic_v2_eval_network import build_eval_network, resolve_canvas_hw


def _parse_args() -> argparse.Namespace:
    """@guang-yang-comment Parse the bounded checkpoint-to-RGB-pair dump contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Training YAML containing datasets.val and network_g.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means every validation sample.")
    return parser.parse_args()


def _safe_component(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value or fallback


def _u8(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    """@guang-yang-comment Persist one RGB image using OpenCV's BGR file interface."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(_u8(rgb), cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write RGB image: {path}")


def _write_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), _u8(value)):
        raise IOError(f"failed to write grayscale image: {path}")


def _nv21_to_rgb(packed: torch.Tensor) -> np.ndarray:
    """@guang-yang-comment Decode a normalized packed NV21 tensor into RGB."""
    value = packed.detach().cpu().numpy()
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3:
        value = value[0]
    return cv2.cvtColor(_u8(value), cv2.COLOR_YUV2RGB_NV21).astype(np.float32) / 255.0


def _yuv444_to_rgb(yuv: torch.Tensor) -> np.ndarray:
    """@guang-yang-comment Decode BASIC normalized YUV444 with its explicit U/V offset."""
    batch = yuv if yuv.ndim == 4 else yuv.unsqueeze(0)
    rgb = YUV2RGB(batch.detach(), format="RGB", is_train=False, uv_normalized=True)[0]
    return rgb.detach().cpu().permute(1, 2, 0).numpy()


def _label(image: np.ndarray, title: str) -> np.ndarray:
    result = np.ascontiguousarray(_u8(image))
    cv2.rectangle(result, (0, 0), (min(result.shape[1] - 1, 180), 23), (0, 0, 0), -1)
    cv2.putText(result, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return result.astype(np.float32) / 255.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_dataset(config_path: Path) -> BasicTextLineRemote58Dataset:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    val_opt = dict(config["datasets"]["val"])
    if val_opt.get("type") != "BasicTextLineRemote58Dataset":
        raise ValueError(f"expected BasicTextLineRemote58Dataset validation data, got {val_opt.get('type')!r}")
    return BasicTextLineRemote58Dataset(val_opt)


def dump_pairs(args: argparse.Namespace) -> dict[str, Any]:
    """@guang-yang-comment Infer every validation item and preserve RGB input/GT/output directories."""
    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = _load_dataset(config_path)
    canvas_hw = resolve_canvas_hw(config_path)
    if tuple(dataset.canvas_hw) != canvas_hw:
        raise ValueError(f"dataset canvas {dataset.canvas_hw} does not match network canvas {canvas_hw}")
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        network_type = str(yaml.safe_load(handle)["network_g"]["type"])
    uses_y4uv = "Y4UV" in network_type
    sample_count = len(dataset) if args.max_samples <= 0 else min(int(args.max_samples), len(dataset))
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    output_root.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for index in range(sample_count):
            item = dataset[index]
            input_rgb = _nv21_to_rgb(item["lq"])
            gt_rgb = _yuv444_to_rgb(item["gt"])
            if uses_y4uv:
                lq_y4uv = rgb_to_y4uv(torch.from_numpy(np.ascontiguousarray(input_rgb)).permute(2, 0, 1).unsqueeze(0)).to(device)
                output_rgb = _yuv444_to_rgb(y4uv_to_yuv444(network(lq_y4uv)))
            else:
                output_rgb = _yuv444_to_rgb(network(item["lq"].unsqueeze(0).to(device)))
            diff_rgb = np.clip(np.abs(output_rgb - gt_rgb) * 4.0, 0.0, 1.0)
            source_name = _safe_component(str(item["source_name"]), "source")
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            stem = _safe_component(Path(str(item["line_id"])).stem, f"sample_{index:06d}")
            sample_dir = output_root / source_name / f"{index:06d}_{stem}"
            _write_rgb(sample_dir / "input.png", input_rgb)
            _write_rgb(sample_dir / "gt.png", gt_rgb)
            _write_rgb(sample_dir / "output.png", output_rgb)
            _write_rgb(sample_dir / "diff_x4.png", diff_rgb)
            _write_gray(sample_dir / "valid_mask.png", item["valid_mask"][0].numpy())
            _write_gray(sample_dir / "text_mask.png", item["text_mask"][0].numpy())
            montage = np.concatenate([
                _label(input_rgb, "INPUT (NV21 -> RGB)"),
                _label(gt_rgb, "GT (YUV444 -> RGB)"),
                _label(output_rgb, "OUTPUT (YUV444 -> RGB)"),
                _label(diff_rgb, "ABS DIFF x4"),
            ], axis=1)
            _write_rgb(sample_dir / "comparison.png", montage)
            metadata = {
                "index": index,
                "source_name": str(item["source_name"]),
                "line_id": str(item["line_id"]),
                "lq_path": str(item["lq_path"]),
                "gt_path": str(item["gt_path"]),
                "canvas_hw": list(canvas_hw),
                "valid_hw": [int(value) for value in item["valid_hw"]],
                "pad_ltrb": [int(value) for value in item["pad_ltrb"]],
                "resize_mode": str(item["resize_mode"]),
                "color_contract": {
                    "input": "packed NV21 decoded by OpenCV COLOR_YUV2RGB_NV21",
                    "gt": "normalized YUV444 decoded by realesrgan.utils.YUV2RGB with uv_normalized=true",
                    "output": "normalized YUV444 decoded by realesrgan.utils.YUV2RGB with uv_normalized=true",
                },
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append({"index": index, "directory": str(sample_dir.relative_to(output_root)), **metadata})
            if (index + 1) % 100 == 0 or index + 1 == sample_count:
                print(f"[validation-rgb-pairs] {index + 1}/{sample_count}", flush=True)

    summary = {
        "status": "COMPLETE",
        "sample_count": sample_count,
        "source_counts": source_counts,
        "canvas_hw": list(canvas_hw),
        "config": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "color_contract": "input NV21 and GT/output normalized YUV444 explicitly decoded to RGB before PNG write",
        "network_input_contract": "RGB converted to packed Y4UV before inference" if uses_y4uv else "dataset packed tensor passed directly to network",
        "records": records,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = dump_pairs(_parse_args())
    print(json.dumps({key: result[key] for key in ("status", "sample_count", "source_counts", "canvas_hw")}, ensure_ascii=False))
