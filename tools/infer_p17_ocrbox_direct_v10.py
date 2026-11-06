"""Direct OCR-box P17 inference with immediate V10 fusion paste-back."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.data.basic_text_line_full_frame import attach_line_index, build_p17_height96_tiled_full_frame_segments, pad_even_rgb, parse_ocr_json
from tools import build_p17_maskguided_nodeblur_linecache_v10 as tiles
from tools import infer_basic_text_line_full_image as src
from tools import run_cached_text_line_bimodal_scurve as base
from tools import run_cached_text_line_verified_normal_scurve_v10 as v10
from tools.run_cached_text_line_verified_normal_scurve_v10_ocrbox_fusion import MODE, _blend_weight


def run(input_dir: Path, ocr_dir: Path, network_config: Path, checkpoint: Path, output_dir: Path, expected_images: int, device: str) -> None:
    """@luna-comment Run direct OCR-box inference without persisting intermediate line cache.

    Input: Image/OCR directories, matching model configuration/checkpoint, target directory, and CUDA device.
    Output: One final fused PNG and routes JSON per image, plus a run receipt.
    Key Steps: Keep each OCR box prediction in memory, apply unchanged V10 luminance reconstruction,
      accumulate overlap-safe cosine support weights, then immediately emit the composed full image.
    Key Params: OCR boxes are never recovered; P17 uses 96x480 bicubic tiles and fusion uses a 16-pixel ramp.
    State/Dependencies: Loads the network once; intentionally writes no frames cache and makes zero raw-mask probes.
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
    summary = {"mode": MODE, "geometry_mode": "direct_ocr_box_no_recovery", "raw_probe_calls": 0, "images": []}
    for image in images:
        frame, original_hw = pad_even_rgb(src._read_rgb(image))
        lines = parse_ocr_json(json.loads((ocr_dir / f"{image.stem}.json").read_text(encoding="utf-8")), frame.shape[:2])
        frame_yuv = src.rgb_to_yuv444(frame)
        accumulation = np.zeros_like(frame_yuv)
        weights = np.zeros(frame_yuv.shape[1:], dtype=np.float32)
        routes = []
        for line in lines:
            segments = attach_line_index(build_p17_height96_tiled_full_frame_segments(line.polygon_xy, frame.shape[:2], canvas, overlap=64, vertical_context=.12, horizontal_context=.03)[0], line.line_index)
            if not segments:
                continue
            prediction, valid, crop = tiles._cache_line(frame, line, segments, runner, canvas)
            left, top, right, bottom = crop
            if prediction.shape[1] >= 32767 or prediction.shape[2] >= 32767:
                reconstructed = prediction
                report = {"status": "identity_skip", "identity_skip_reason": "v10_remap_source_limit", "valid_pixels": int(valid.sum()), "accepted_profiles": 0}
            else:
                try:
                    reconstructed, _, report = v10.fill_v10(prediction, frame_yuv[:, top:bottom, left:right], valid)
                except cv2.error as error:
                    reconstructed = prediction
                    report = {"status": "identity_skip", "identity_skip_reason": "v10_remap_limit", "opencv_error": str(error).splitlines()[0], "valid_pixels": int(valid.sum()), "accepted_profiles": 0}
            alpha = _blend_weight(valid)
            accumulation[:, top:bottom, left:right] += reconstructed * alpha[None]
            weights[top:bottom, left:right] += alpha
            report.update({"line_index": line.line_index, "geometry_mode": "direct_ocr_box", "crop_xyxy": list(crop), "segments": len(segments), "paste_mode": "support_edge_cosine_fusion", "paste_alpha_pixels": int((alpha > 0).sum())})
            routes.append(report)
        result = frame_yuv.copy()
        covered = weights > 1e-6
        result[:, covered] = accumulation[:, covered] / weights[covered][None]
        base._rgb_png(output_dir / f"{image.stem}.png", result[:,:original_hw[0],:original_hw[1]])
        (output_dir / f"{image.stem}.routes.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["images"].append({"image_id": image.stem, "lines": len(routes), "overlap_pixels": int((weights > 1.0).sum())})
    summary.update({"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_digest, "network_config": str(network_config), "runner_calls": runner.calls})
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_receipt.json").write_text(json.dumps({"mode": MODE, "geometry_mode": "direct_ocr_box_no_recovery", "raw_probe_calls": 0}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--network-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    run(**vars(arguments))
