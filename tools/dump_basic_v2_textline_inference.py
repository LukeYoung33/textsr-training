"""Run one BASIC V2 checkpoint on a directory of standalone text-line images.

This is a no-reference evaluation route: each source line is directly resized
to the 96x480 BASIC canvas, converted to packed NV21, inferred as normalized
YUV444, explicitly converted back to RGB, and retained as a per-image pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.utils import YUV2RGB
from tools.basic_v2_eval_network import build_eval_network, resolve_canvas_hw
from tools.infer_basic_text_line_full_image import rgb_to_nv21


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, action="append", type=Path,
                        help="Standalone TextLines directory; repeat for multiple sets.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-mode", choices=("stretch", "aspect_fit", "height96_tile_pad"), required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all images.")
    return parser.parse_args()


def _u8(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    """@guang-yang-comment Write a normalized RGB image through OpenCV's BGR interface."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(_u8(rgb), cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {path}")


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _resize_rgb(image: np.ndarray, canvas_hw: tuple[int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    target_h, target_w = canvas_hw
    interpolation = cv2.INTER_AREA if target_h <= height and target_w <= width else cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation).astype(np.float32, copy=False)


def _prepare_input(image: np.ndarray, canvas_hw: tuple[int, int], resize_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """@guang-yang-comment Create either full stretch or aspect-fit edge-padded BASIC input."""
    source_h, source_w = image.shape[:2]
    canvas_h, canvas_w = canvas_hw
    if resize_mode == "stretch":
        return _resize_rgb(image, canvas_hw), {
            "resize_mode": "stretch",
            "content_xyxy": [0, 0, canvas_w, canvas_h],
            "pad_ltrb": [0, 0, 0, 0],
            "scale_yx": [canvas_h / float(source_h), canvas_w / float(source_w)],
        }
    scale = min(canvas_h / float(source_h), canvas_w / float(source_w))
    resized_h = max(2, min(canvas_h, int(round(source_h * scale))))
    resized_w = max(2, min(canvas_w, int(round(source_w * scale))))
    if resized_h % 2:
        resized_h = min(canvas_h, resized_h + 1)
    if resized_w % 2:
        resized_w = min(canvas_w, resized_w + 1)
    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)
    top = (canvas_h - resized_h) // 2
    bottom = canvas_h - resized_h - top
    left = (canvas_w - resized_w) // 2
    right = canvas_w - resized_w - left
    # Edge replication makes the NV21 canvas continuous without injecting a
    # synthetic black region into this comparative no-reference test.
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return canvas, {
        "resize_mode": "aspect_fit",
        "content_xyxy": [left, top, left + resized_w, top + resized_h],
        "pad_ltrb": [left, top, right, bottom],
        "pad_mode": "edge_replicate",
        "scale_yx": [resized_h / float(source_h), resized_w / float(source_w)],
    }


def _yuv444_to_rgb(yuv: torch.Tensor) -> np.ndarray:
    rgb = YUV2RGB(yuv.detach(), format="RGB", is_train=False, uv_normalized=True)[0]
    return rgb.detach().cpu().permute(1, 2, 0).numpy()


def _height96_tile_pad(
    source: np.ndarray,
    network: torch.nn.Module,
    device: torch.device,
    canvas_hw: tuple[int, int],
    sample_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """@guang-yang-comment Preserve aspect at height 96, tile long lines, and edge-pad short tiles."""
    canvas_h, canvas_w = canvas_hw
    source_h, source_w = source.shape[:2]
    scaled_w = max(2, int(round(source_w * canvas_h / float(source_h))))
    resized = cv2.resize(
        source, (scaled_w, canvas_h),
        interpolation=cv2.INTER_AREA if canvas_h <= source_h else cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)
    if scaled_w <= canvas_w:
        starts = [0]
    else:
        stride = canvas_w - 48
        starts = list(range(0, scaled_w - canvas_w + 1, stride))
        if starts[-1] != scaled_w - canvas_w:
            starts.append(scaled_w - canvas_w)
    prediction_sum = np.zeros((3, canvas_h, scaled_w), dtype=np.float32)
    prediction_weight = np.zeros((scaled_w,), dtype=np.float32)
    tiles: list[dict[str, Any]] = []
    for tile_index, start in enumerate(starts):
        end = min(start + canvas_w, scaled_w)
        valid_w = end - start
        tile = resized[:, start:end]
        if valid_w < canvas_w:
            tile = cv2.copyMakeBorder(tile, 0, 0, 0, canvas_w - valid_w, cv2.BORDER_REPLICATE)
        packed = torch.from_numpy(rgb_to_nv21(tile)).unsqueeze(0).to(device)
        prediction = network(packed)[0].detach().cpu().numpy()
        prediction_sum[:, :, start:end] += prediction[:, :, :valid_w]
        prediction_weight[start:end] += 1.0
        tile_dir = sample_dir / "tiles" / f"tile_{tile_index:03d}"
        _write_rgb(tile_dir / "input.png", tile)
        _write_rgb(tile_dir / "output.png", _yuv444_to_rgb(torch.from_numpy(prediction).unsqueeze(0)))
        tiles.append({
            "tile_index": tile_index,
            "x_range": [start, end],
            "valid_width": valid_w,
            "pad_right": canvas_w - valid_w,
        })
    merged_yuv = prediction_sum / np.maximum(prediction_weight[None, None, :], 1.0)
    merged_rgb = _yuv444_to_rgb(torch.from_numpy(merged_yuv).unsqueeze(0))
    return resized, merged_rgb, {
        "resize_mode": "height96_tile_pad",
        "scaled_hw": [canvas_h, scaled_w],
        "scale_yx": [canvas_h / float(source_h), scaled_w / float(source_w)],
        "tile_canvas_hw": [canvas_h, canvas_w],
        "tile_overlap": 48 if len(starts) > 1 else 0,
        "tile_pad_mode": "edge_replicate",
        "tiles": tiles,
    }


def _label(image: np.ndarray, title: str) -> np.ndarray:
    result = np.ascontiguousarray(_u8(image))
    cv2.rectangle(result, (0, 0), (min(result.shape[1] - 1, 220), 23), (0, 0, 0), -1)
    cv2.putText(result, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return result.astype(np.float32) / 255.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_textlines(args: argparse.Namespace) -> dict[str, Any]:
    """@guang-yang-comment Run direct-resize no-reference inference and retain every visual pair."""
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    input_dirs = [path.resolve() for path in args.input_dir]
    output_root = args.output_root.resolve()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    canvas_hw = resolve_canvas_hw(config)
    grouped_paths = [
        (input_dir, sorted(path for path in input_dir.iterdir()
                           if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}))
        for input_dir in input_dirs
    ]
    if any(not paths for _, paths in grouped_paths):
        missing = [str(input_dir) for input_dir, paths in grouped_paths if not paths]
        raise ValueError(f"no images in: {missing}")
    paths = [(input_dir, path) for input_dir, entries in grouped_paths for path in entries]
    if args.max_samples > 0:
        paths = paths[:min(len(paths), int(args.max_samples))]
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, (input_dir, path) in enumerate(paths):
            source = _read_rgb(path)
            sample_dir = output_root / input_dir.name / f"{index:06d}_{path.stem}"
            if args.resize_mode == "height96_tile_pad":
                input_rgb, output_rgb, geometry = _height96_tile_pad(source, network, device, canvas_hw, sample_dir)
            else:
                input_rgb, geometry = _prepare_input(source, canvas_hw, args.resize_mode)
                packed = torch.from_numpy(rgb_to_nv21(input_rgb)).unsqueeze(0).to(device)
                output_rgb = _yuv444_to_rgb(network(packed))
            _write_rgb(sample_dir / "input.png", input_rgb)
            _write_rgb(sample_dir / "output.png", output_rgb)
            _write_rgb(sample_dir / "comparison.png", np.concatenate([
                _label(input_rgb, "INPUT (direct resize RGB)"),
                _label(output_rgb, "OUTPUT (YUV444 -> RGB)"),
            ], axis=1))
            metadata = {
                "index": index,
                "source_path": str(path),
                "source_set": input_dir.name,
                "source_hw": [int(source.shape[0]), int(source.shape[1])],
                "canvas_hw": list(canvas_hw),
                **geometry,
                "color_contract": {
                    "input": "RGB prepared by the recorded resize_mode then encoded as packed NV21 for inference",
                    "output": "normalized YUV444 decoded by realesrgan.utils.YUV2RGB with uv_normalized=true",
                },
                "reference_metrics": "unavailable: supplied TextLines directory has no GT/reference images",
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append({"directory": sample_dir.name, **metadata})
            if (index + 1) % 100 == 0 or index + 1 == len(paths):
                print(f"[textline-inference] {index + 1}/{len(paths)}", flush=True)
    summary = {
        "status": "COMPLETE",
        "sample_count": len(paths),
        "input_dirs": [str(path) for path in input_dirs],
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "canvas_hw": list(canvas_hw),
        "resize_mode": args.resize_mode,
        "reference_metrics": "unavailable: supplied TextLines directory has no GT/reference images",
        "records": records,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = dump_textlines(_parse_args())
    print(json.dumps({key: summary[key] for key in ("status", "sample_count", "canvas_hw")}, ensure_ascii=False))
