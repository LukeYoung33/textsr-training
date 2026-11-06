"""Native 96x480 OCR-box tiles with P17-style overlap and boundary fusion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.data.basic_text_line_full_frame import build_native_tiled_full_frame_segments, pad_even_rgb, parse_ocr_json
from tools import infer_basic_text_line_full_image as src
from tools import run_cached_text_line_bimodal_scurve as base
from tools import run_cached_text_line_verified_normal_scurve_v10 as v10


def _tile_weight(height: int, width: int) -> np.ndarray:
    """@luna-comment Produce P17-style nonzero two-dimensional overlap weights for native tiles.

    Input: Native tile height and width. Output: A positive 2-D cosine-window weight plane.
    Key Steps: Form separable sine windows and floor them at 0.1 before multiplication.
    Key Params: Native tile geometry is at most 96x480; the 0.1 floor preserves coverage at image edges.
    State/Dependencies: Pure NumPy computation used only for normalized overlap accumulation.
    """
    y = np.maximum(np.sin(np.linspace(0.0, np.pi, max(2, height), dtype=np.float32)), 0.1)[:height]
    x = np.maximum(np.sin(np.linspace(0.0, np.pi, max(2, width), dtype=np.float32)), 0.1)[:width]
    return np.outer(y, x).astype(np.float32)


def _box_boundary_weight(tile_xyxy: tuple[int, int, int, int], box_xyxy: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    """@luna-comment Taper a processed OCR box to its untouched surrounding source frame.

    Input: Tile/frame bounds, original OCR AABB, and native tile shape. Output: Inner 32-pixel cosine alpha.
    Key Steps: Measure distance to the OCR-box perimeter in frame coordinates and map it to a raised cosine.
    Key Params: The ramp is 32 source pixels, strictly inside the OCR box, so no model data is invented outside it.
    State/Dependencies: Read-only coordinate math; P17 tile results remain unchanged before final compositing.
    """
    left, top, _, _ = tile_xyxy
    box_left, box_top, box_right, box_bottom = box_xyxy
    height, width = shape
    yy, xx = np.mgrid[top:top + height, left:left + width]
    distance = np.minimum.reduce([xx - box_left, box_right - 1 - xx, yy - box_top, box_bottom - 1 - yy]).astype(np.float32)
    t = np.clip(distance / 32.0, 0.0, 1.0)
    return (.5 - .5 * np.cos(np.pi * t)).astype(np.float32)


def run(input_dir: Path, ocr_dir: Path, network_config: Path, checkpoint: Path, output_dir: Path, expected_images: int, device: str) -> None:
    """@luna-comment Run native OCR-box tiles without box resizing or persistent cache.

    Input: Image/OCR paths, model config/checkpoint, target directory, expected count, and CUDA device.
    Output: One immediately composed PNG and route record per source frame.
    Key Steps: Keep OCR boxes in source coordinates, split them into overlapping native 96x480 tiles,
      run model and V10 per tile, normalize tile/box overlaps, then feather each box into original frame.
    Key Params: No OCR recovery; native tiles use 64x24 overlap and a 32-pixel box-edge source blend.
    State/Dependencies: Model is loaded once; no raw probe, no geometry recovery, and no frames cache is written.
    """
    images = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(images) != expected_images:
        raise ValueError("unexpected image count")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch_device = torch.device(device)
    network, checkpoint_digest = src._load_network(checkpoint, torch_device, network_config)
    config = src.load_lq_preprocess_config(network_config)
    runner = src.NativeY4UVTileRunner(network, torch_device, config)
    canvas = src.resolve_canvas_hw(network_config)
    summary = {"mode": "native_ocrbox_tiles_v10", "geometry_mode": "direct_ocr_box_no_recovery", "resize_mode": "native_96x480_tiles_no_resize", "raw_probe_calls": 0, "images": []}
    for image in images:
        frame, original_hw = pad_even_rgb(src._read_rgb(image))
        frame_yuv = src.rgb_to_yuv444(frame)
        lines = parse_ocr_json(json.loads((ocr_dir / f"{image.stem}.json").read_text(encoding="utf-8")), frame.shape[:2])
        accumulation = np.zeros_like(frame_yuv)
        weights = np.zeros(frame_yuv.shape[1:], dtype=np.float32)
        routes = []
        for line in lines:
            segments, reason = build_native_tiled_full_frame_segments(line.polygon_xy, frame.shape[:2], canvas, overlap_x=64, overlap_y=24)
            if not segments:
                routes.append({"line_index": line.line_index, "status": "identity_skip", "identity_skip_reason": reason or "no_native_segments"})
                continue
            box_left = max(0, int(np.floor(line.polygon_xy[:, 0].min())))
            box_top = max(0, int(np.floor(line.polygon_xy[:, 1].min())))
            box_right = min(frame.shape[1], int(np.ceil(line.polygon_xy[:, 0].max())))
            box_bottom = min(frame.shape[0], int(np.ceil(line.polygon_xy[:, 1].max())))
            for segment in segments:
                left, top, right, bottom = segment.crop_xyxy
                height, width = segment.valid_hw
                native = np.clip(frame[top:bottom, left:right], 0.0, 1.0)
                canvas_rgb = src._pad_canvas(native, canvas, edge_replicate=True)
                prediction = np.clip(runner.run_rgb(canvas_rgb), 0.0, 1.0)[:, :height, :width]
                valid = np.ones((height, width), dtype=bool)
                reconstructed, _, report = v10.fill_v10(prediction, frame_yuv[:, top:bottom, left:right], valid)
                alpha = _tile_weight(height, width) * _box_boundary_weight((left, top, right, bottom), (box_left, box_top, box_right, box_bottom), (height, width))
                accumulation[:, top:bottom, left:right] += reconstructed * alpha[None]
                weights[top:bottom, left:right] += alpha
            routes.append({"line_index": line.line_index, "status": "processed", "geometry_mode": "direct_ocr_box_native_tiles", "box_xyxy": [box_left, box_top, box_right, box_bottom], "segments": len(segments), "tile_size": [96, 480], "resize_applied": False, "tile_overlap": [24, 64], "box_boundary_fusion": "inner_cosine_32px"})
        result = frame_yuv.copy()
        covered = weights > 1e-6
        result[:, covered] = accumulation[:, covered] / weights[covered][None]
        base._rgb_png(output_dir / f"{image.stem}.png", result[:, :original_hw[0], :original_hw[1]])
        (output_dir / f"{image.stem}.routes.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["images"].append({"image_id": image.stem, "lines": len(routes), "fused_pixels": int(covered.sum()), "overlap_pixels": int((weights > 1.0).sum())})
    summary.update({"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_digest, "network_config": str(network_config), "runner_calls": runner.calls})
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--network-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    run(**vars(parser.parse_args()))
