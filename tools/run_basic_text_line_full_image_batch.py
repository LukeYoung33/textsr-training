"""Batch wrapper for OCR-driven BASIC full-image inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

try:
    from tools.infer_basic_text_line_full_image import NativeY4UVTileRunner, _load_network, infer_full_image, load_lq_preprocess_config
    from tools.basic_v2_eval_network import load_network_config
except ImportError:  # direct ``python tools/run_...py`` execution
    from infer_basic_text_line_full_image import NativeY4UVTileRunner, _load_network, infer_full_image, load_lq_preprocess_config
    from basic_v2_eval_network import load_network_config


def _image_paths(input_dir: Path) -> list[Path]:
    paths = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(input_dir.glob(suffix))
    return sorted(set(paths))


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """@guang-yang-comment Run every input image with one shared strict-loaded network.

    Input: argparse namespace containing input/OCR/output roots, checkpoint, and
    optional network config. Output: per-image rows plus a final run_summary.json.
    Key Steps: resolve the selected registry graph, process all images, preserve
    debug artifacts, and enforce optional frozen image count. Key Params:
    ``expected_images`` rejects a mismatched frozen testset. State/Dependencies:
    full-image inference helper and filesystem output root.
    """

    inputs = _image_paths(args.input_dir)
    if not inputs:
        raise FileNotFoundError(f"no input images found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    network_config = getattr(args, "network_config", None)
    if network_config is None:
        network, checkpoint_sha256 = _load_network(args.checkpoint, device)
    else:
        network, checkpoint_sha256 = _load_network(args.checkpoint, device, network_config)
    network_type = str(load_network_config(network_config).get("type", ""))
    lq_preprocess_config = load_lq_preprocess_config(network_config)
    tile_runner = (
        NativeY4UVTileRunner(network, device, lq_preprocess_config)
        if "Y4UV" in network_type.upper() else None
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, input_path in enumerate(inputs, start=1):
        ocr_path = args.ocr_dir / f"{input_path.stem}.json"
        output_path = args.output_dir / f"{input_path.stem}_basic_v2.png"
        if not ocr_path.is_file():
            rows.append({"input": str(input_path), "status": "failed", "reason": "OCR_JSON_MISSING"})
            continue
        if args.skip_existing and output_path.is_file() and output_path.with_suffix(output_path.suffix + ".json").is_file():
            rows.append({"input": str(input_path), "ocr_json": str(ocr_path), "output": str(output_path), "status": "skipped"})
            continue
        debug_dir = args.debug_root / input_path.stem if args.debug_root else None
        try:
            summary = infer_full_image(
                input_path,
                ocr_path,
                args.checkpoint,
                output_path,
                device_name=args.device,
                context_ratio=args.context_ratio,
                overlap=args.overlap,
                blend_radius=args.blend_radius,
                min_score=args.min_score,
                resize_bypass=args.resize_bypass,
                resize_min_ratio=args.resize_min_ratio,
                resize_max_ratio=args.resize_max_ratio,
                resize_target_short_ratio=args.resize_target_short_ratio,
                debug_dir=debug_dir,
                network=network,
                checkpoint_sha256=checkpoint_sha256,
                network_config=network_config,
                resize_mode=getattr(args, "resize_mode", "native"),
                ocr_format=getattr(args, "ocr_format", "generic"),
                p17_vertical_context=getattr(args, "p17_vertical_context", 0.12),
                p17_horizontal_context=getattr(args, "p17_horizontal_context", 0.03),
                mask_guided_ocr_recovery=getattr(args, "mask_guided_ocr_recovery", False),
                mask_guided_geometry_stats=getattr(args, "mask_guided_geometry_stats", None),
                native_tile_height_threshold=getattr(args, "native_tile_height_threshold", 128),
                large_tile_canvas_hw=(getattr(args, "large_tile_canvas", 256),) * 2,
                large_tile_overlap=getattr(args, "large_tile_overlap", 96),
                generic_ocr_nms_iou=getattr(args, "generic_ocr_nms_iou", 0.45),
                sharp_text_bypass_laplacian=getattr(args, "sharp_text_bypass_laplacian", 0.0),
                sharp_text_bypass_max_height=getattr(args, "sharp_text_bypass_max_height", 128),
                tile_runner=tile_runner,
                restore_interpolation=getattr(args, "restore_interpolation", "linear"),
                input_interpolation=getattr(args, "input_interpolation", "default"),
                postprocess=getattr(args, "postprocess", "none"),
                postprocess_gamma=getattr(args, "postprocess_gamma", 1.3),
                postprocess_strength=getattr(args, "postprocess_strength", 0.5),
                postprocess_max_delta=getattr(args, "postprocess_max_delta", 8.0 / 255.0),
            )
            rows.append({
                "input": str(input_path),
                "ocr_json": str(ocr_path),
                "output": str(output_path),
                "status": "passed",
                "lines_total": summary["lines_total"],
                "lines_processed": summary["lines_processed"],
                "lines_bypassed": summary["lines_bypassed"],
                "segments_total": summary["segments_total"],
                "segments_native": summary["segments_native"],
                "segments_resized": summary["segments_resized"],
                "lines_resized": summary["lines_resized"],
                "processed_fraction": summary["processed_fraction"],
                "elapsed_sec": summary["elapsed_sec"],
            })
        except Exception as exc:  # keep the aggregate report useful for the remaining images
            rows.append({
                "input": str(input_path),
                "ocr_json": str(ocr_path),
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
        aggregate = {
            "status": "running",
            "device": str(device),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "tile_runner": "native_y4uv_adapter" if tile_runner is not None else "direct_network",
            "lq_preprocess": lq_preprocess_config,
            "input_dir": str(args.input_dir),
            "ocr_dir": str(args.ocr_dir),
            "output_dir": str(args.output_dir),
            "completed_images": index,
            "total_images": len(inputs),
            "rows": rows,
        }
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"index": index, "total": len(inputs), **rows[-1]}, ensure_ascii=False), flush=True)

    passed = sum(row.get("status") == "passed" for row in rows)
    failed = sum(row.get("status") == "failed" for row in rows)
    skipped = sum(row.get("status") == "skipped" for row in rows)
    aggregate = {
        "status": "passed" if failed == 0 else "completed_with_failures",
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "tile_runner": "native_y4uv_adapter" if tile_runner is not None else "direct_network",
        "lq_preprocess": lq_preprocess_config,
        "input_dir": str(args.input_dir),
        "ocr_dir": str(args.ocr_dir),
        "output_dir": str(args.output_dir),
        "total_images": len(inputs),
        "passed_images": passed,
        "failed_images": failed,
        "skipped_images": skipped,
        "elapsed_sec": float(time.perf_counter() - started),
        "rows": rows,
        "expected_images": getattr(args, "expected_images", None),
    }
    expected_images = getattr(args, "expected_images", None)
    if expected_images is not None and len(inputs) != expected_images:
        aggregate["status"] = "completed_with_failures"
        aggregate["failure_reason"] = f"expected {expected_images} images, found {len(inputs)}"
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch BASIC full-image OCR pipeline")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--network-config", type=Path, default=None,
                        help="Optional training YAML; network_g is built via the registry")
    parser.add_argument("--resize-mode", choices=("native", "native_tile", "native_tile_height192x960", "stretch", "height96_tile_pad", "p17_height96_tile_pad"), default="native",
                        help="native preserves crop geometry; stretch resizes every model tile to the full canvas")
    parser.add_argument("--restore-interpolation", choices=("linear", "bicubic"), default="linear",
                        help="Interpolation used only when a resized model prediction is restored to source geometry")
    parser.add_argument("--input-interpolation", choices=("default", "bicubic"), default="default")
    parser.add_argument("--postprocess", choices=("none", "y_scurve", "otsu_foreground", "source_edge", "subpixel_transition", "p11_edge_narrow_v2", "p11_edge_narrow_v3", "p11_edge_narrow_v3_1", "p11_valid_support_polarity_guard_edge_narrow_v4_2", "p11_valid_maskraw_compare_v5", "p11_valid_maskdirectional_compare_v5"), default="none")
    parser.add_argument("--postprocess-gamma", type=float, default=1.3)
    parser.add_argument("--postprocess-strength", type=float, default=.5)
    parser.add_argument("--postprocess-max-delta", type=float, default=8.0 / 255.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-ratio", type=float, default=0.15)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--blend-radius", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--ocr-format", choices=("generic", "p17_annotations"), default="generic",
                        help="Use manifest OCR annotations directly; never run an OCR detector.")
    parser.add_argument("--p17-vertical-context", type=float, default=0.12)
    parser.add_argument("--p17-horizontal-context", type=float, default=0.03)
    parser.add_argument("--mask-guided-ocr-recovery", action="store_true",
                        help="Recover only horizontal OCR y-bounds from the P17 raw mask before P17 segment construction.")
    parser.add_argument("--mask-guided-geometry-stats", type=Path, default=None,
                        help="Versioned geometry quantiles JSON generated from the P17 training manifest.")
    parser.add_argument("--native-tile-height-threshold", type=int, default=128,
                        help="Resize OCR regions up to this height; preserve scale and 2-D tile taller regions.")
    parser.add_argument("--large-tile-canvas", type=int, default=256,
                        help="Square V5 canvas for taller OCR regions; must be a multiple of eight.")
    parser.add_argument("--large-tile-overlap", type=int, default=96,
                        help="Two-dimensional overlap for the large native tile route.")
    parser.add_argument("--generic-ocr-nms-iou", type=float, default=0.45,
                        help="Suppress duplicate generic OCR envelopes at this AABB IoU; non-positive disables it.")
    parser.add_argument("--sharp-text-bypass-laplacian", type=float, default=0.0,
                        help="Bypass small generic OCR lines above this local Laplacian variance; zero disables it.")
    parser.add_argument("--sharp-text-bypass-max-height", type=int, default=128,
                        help="Maximum OCR-envelope height eligible for the sharp-text bypass.")
    parser.add_argument("--resize-bypass", action="store_true",
                        help="Uniformly resize eligible native short-axis bypasses before model inference")
    parser.add_argument("--resize-min-ratio", type=float, default=0.5)
    parser.add_argument("--resize-max-ratio", type=float, default=1.5)
    parser.add_argument("--resize-target-short-ratio", type=float, default=0.75)
    parser.add_argument("--debug-root", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--expected-images", type=int, default=None,
                        help="Fail the batch if the input count differs from this frozen count")
    args = parser.parse_args(argv)
    summary = run_batch(args)
    print(json.dumps({key: summary[key] for key in (
        "status", "total_images", "passed_images", "failed_images", "skipped_images", "elapsed_sec",
    )}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
