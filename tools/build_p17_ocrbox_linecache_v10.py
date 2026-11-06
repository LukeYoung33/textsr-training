"""Build a direct-OCR-box P17 line cache without geometry recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.data.basic_text_line_full_frame import attach_line_index, build_p17_height96_tiled_full_frame_segments, pad_even_rgb, parse_ocr_json
from tools import build_basic_text_line_inference_cache as old
from tools import build_p17_maskguided_nodeblur_linecache_v10 as source
from tools import infer_basic_text_line_full_image as src


def _sha(path: Path) -> str:
    """@luna-comment Return a content digest for cache provenance records.

    Input: Existing file path. Output: SHA-256 hex digest. Key Steps: Stream bytes into hashlib.
    Key Params: SHA-256. State/Dependencies: Read-only filesystem operation.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """@luna-comment Build direct OCR-box 96x480 predictions with no raw-mask box recovery.

    Input: Images, PaddleOCR v6 tiny JSON, matching model configuration/checkpoint, and output path.
    Output: Per-image and per-OCR-box model cache plus direct-geometry provenance receipt.
    Key Steps: Parse OCR polygons unchanged, tile each box at P17 height 96, run the model, and persist
      prediction/valid-support arrays. Key Params: Bicubic cache path and fixed P17 tile contract are inherited.
    State/Dependencies: Loads the model once; intentionally performs zero P17 raw probes and zero geometry recovery.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--network-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    images = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(images) != args.expected_images:
        raise ValueError("unexpected image count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    network, checkpoint_digest = src._load_network(args.checkpoint, device, args.network_config)
    config = src.load_lq_preprocess_config(args.network_config)
    runner = src.NativeY4UVTileRunner(network, device, config)
    canvas = src.resolve_canvas_hw(args.network_config)
    frames: list[dict] = []
    total_segments = 0
    for image in images:
        ocr_path = args.ocr_dir / f"{image.stem}.json"
        original = src._read_rgb(image)
        frame, original_hw = pad_even_rgb(original)
        lines = parse_ocr_json(json.loads(ocr_path.read_text(encoding="utf-8")), frame.shape[:2])
        directory = args.output_dir / "frames" / image.stem
        directory.mkdir(parents=True)
        src._write_rgb(directory / "frame_input.png", frame)
        np.save(directory / "frame_yuv444.npy", src.rgb_to_yuv444(frame))
        (directory / "geometry_recovery.json").write_text(json.dumps({"mode": "ocr_box_direct", "raw_probe_calls": 0, "lines": [{"line_index": line.line_index, "status": "direct_ocr_box"} for line in lines]}, ensure_ascii=False, indent=2), encoding="utf-8")
        records: list[dict] = []
        for line in lines:
            segments = attach_line_index(build_p17_height96_tiled_full_frame_segments(line.polygon_xy, frame.shape[:2], canvas, overlap=64, vertical_context=.12, horizontal_context=.03)[0], line.line_index)
            if not segments:
                continue
            prediction, valid, crop = source._cache_line(frame, line, segments, runner, canvas)
            line_dir = directory / "lines" / f"{line.line_index:04d}"
            line_dir.mkdir(parents=True)
            np.save(line_dir / "prediction_yuv444.npy", prediction)
            np.save(line_dir / "valid_support.npy", valid)
            geometry = {"line_index": line.line_index, "crop_xyxy": list(crop), "line_to_frame_offset": [crop[0], crop[1]], "detection_score": float(line.score), "text": line.text, "geometry_recovery": {"status": "direct_ocr_box"}}
            route = {"resize_mode": "p17_height96_tile_pad", "input_interpolation": "bicubic", "restore_interpolation": "bicubic", "postprocess": "none", "segments": [{"crop_xyxy": list(segment.crop_xyxy), "tile_index": segment.tile_index, "tile_count": segment.tile_count} for segment in segments]}
            (line_dir / "geometry.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")
            (line_dir / "route.json").write_text(json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8")
            artifacts = {name: (old.array_record(line_dir / name) if name.endswith(".npy") else old.json_record(line_dir / name)) for name in ("prediction_yuv444.npy", "valid_support.npy", "geometry.json", "route.json")}
            (line_dir / "artifact.sha256.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
            records.append({"line_index": line.line_index, "directory": str(line_dir.relative_to(args.output_dir)).replace("\\", "/")})
            total_segments += len(segments)
        metadata = {"image_id": image.stem, "directory": str(directory.relative_to(args.output_dir)).replace("\\", "/"), "input_sha256": _sha(image), "ocr_sha256": _sha(ocr_path), "original_hw": list(original_hw), "lines": records}
        (directory / "frame.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        frames.append(metadata)
    manifest = {"cache_id": "p17_ocrbox_direct_linecache_v10_20260924", "schema_version": 1, "detector": "PaddleOCR v6 tiny", "geometry_mode": "direct_ocr_box_no_recovery", "raw_probe_calls": 0, "checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha(args.checkpoint), "network_config": str(args.network_config), "network_config_sha256": _sha(args.network_config), "resize_mode": "p17_height96_tile_pad", "input_interpolation": "bicubic", "restore_interpolation": "bicubic", "postprocess": "none", "p17_wiener_bypass_enabled": bool(config.get("p17_wiener_bypass_enabled")), "frames": frames}
    (args.output_dir / "cache_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "cache_manifest.sha256").write_text(hashlib.sha256(old.canonical(manifest)).hexdigest(), encoding="utf-8")
    (args.output_dir / "cache_build_receipt.json").write_text(json.dumps({"status": "complete", "raw_probe_calls": 0, "geometry_mode": "direct_ocr_box_no_recovery", "segments": total_segments, "runner_calls": runner.calls, "checkpoint_sha256": checkpoint_digest}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
