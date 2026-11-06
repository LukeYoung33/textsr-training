"""Dump all accepted HX OCR text-line crops for one training checkpoint.

This is deliberately line-level only: it never runs the fixed-canvas network
on a full HX image and never writes a full-image restoration result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import registry modules before constructing the custom architecture/dataset.
import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
from realesrgan.data.basic_text_line_segment_dataset_v3 import BasicTextLineSegmentDatasetV3
from tools.basic_v2_eval_network import build_eval_network, resolve_canvas_hw


CANVAS_HW = (96, 320)


def _u8(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    """Write either normalized float RGB or already-quantized uint8 RGB once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(rgb)
    rgb_u8 = value if value.dtype == np.uint8 else _u8(value)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"failed to write RGB image: {path}")


def _write_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), _u8(value))
    if not ok:
        raise IOError(f"failed to write mask: {path}")


def _yuv444_to_rgb(yuv_chw: np.ndarray) -> np.ndarray:
    """Decode normalized planar YUV444 and return normalized RGB float data.

    ``_write_rgb`` consumes RGB arrays in ``[0, 1]`` and performs the single
    float-to-uint8 conversion.  Returning OpenCV's uint8 result here would
    make the writer multiply it by 255 a second time, saturating predictions
    to white.  Keep this helper's output contract aligned with ``lq_rgb`` and
    ``gt_rgb`` returned by ``BasicTextLineSegmentDatasetV3``.
    """
    yuv = np.transpose(np.asarray(yuv_chw, dtype=np.float32), (1, 2, 0))
    rgb_u8 = cv2.cvtColor(_u8(yuv), cv2.COLOR_YUV2RGB)
    return rgb_u8.astype(np.float32) / 255.0


def _label(image: np.ndarray, title: str) -> np.ndarray:
    result = np.array(image, copy=True)
    cv2.rectangle(result, (0, 0), (min(result.shape[1] - 1, 180), 23), (0, 0, 0), -1)
    cv2.putText(result, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    denom = float(np.asarray(mask, dtype=np.float32).sum())
    return float((np.asarray(value, dtype=np.float32) * mask).sum() / max(denom, 1.0))


def _sobel_magnitude(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def _metrics(pred_yuv: np.ndarray, gt_yuv: np.ndarray, valid: np.ndarray, text: np.ndarray) -> dict[str, float]:
    valid = np.asarray(valid, dtype=np.float32)
    text = valid * np.asarray(text, dtype=np.float32)
    error = np.abs(np.asarray(pred_yuv, dtype=np.float32) - np.asarray(gt_yuv, dtype=np.float32))
    text_yuv = error * np.asarray(text)[None]
    valid_yuv = error * np.asarray(valid)[None]
    pred_y = np.asarray(pred_yuv[0], dtype=np.float32)
    gt_y = np.asarray(gt_yuv[0], dtype=np.float32)
    text_grad = np.abs(_sobel_magnitude(pred_y) - _sobel_magnitude(gt_y))
    text_count = max(float(text.sum()), 1.0)
    valid_count = max(float(valid.sum()), 1.0)
    text_weight = np.asarray([1.0, 0.25, 0.25], dtype=np.float32)[:, None, None]
    weighted_text = (error * text[None] * text_weight).sum() / max(float((text[None] * text_weight).sum()), 1.0)
    valid_mse_y = float((((pred_y - gt_y) ** 2) * valid).sum() / valid_count)
    return {
        "text_y_mae": float((error[0] * text).sum() / text_count),
        "text_yuv_weighted_mae": float(weighted_text),
        "valid_y_mae": float((error[0] * valid).sum() / valid_count),
        "text_y_gradient_mae": float((text_grad * text).sum() / text_count),
        "valid_psnr_y": float(10.0 * math.log10(1.0 / max(valid_mse_y, 1e-12))),
    }


def _safe_component(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value or fallback


def _source_group(path: Path) -> str:
    """Return the dataset/source directory that owns an LQ image path."""
    # The manifests use .../<source_group>/lq/<stem>.png.  Including the
    # source group prevents validation manifests that contain several
    # same-named images from different degradation sets from overwriting each
    # other in the visualization tree.
    try:
        return _safe_component(path.parent.parent.name, "source")
    except IndexError:
        return "source"


def _make_network(
    device: torch.device,
    checkpoint: Path,
    network_config: Path | None = None,
) -> tuple[torch.nn.Module, str]:
    """@guang-yang-comment Build configured registry network and strictly load its checkpoint.

    Input: target device, checkpoint, and optional training YAML. Output: eval-mode
    network plus checkpoint digest. Key Steps: resolve ``network_g`` and enforce
    strict state compatibility. Key Params: absent config retains baseline graph.
    State/Dependencies: shared registry builder and torch checkpoint format.
    """

    return build_eval_network(checkpoint, device, network_config)


def dump_checkpoint(
    manifest: Path,
    manifest_summary: Path | None,
    checkpoint: Path,
    output_root: Path,
    iteration: int,
    device_name: str,
    network_config: Path | None = None,
    expected_segments: int | None = None,
    resize_mode: str | None = None,
) -> dict[str, Any]:
    """@guang-yang-comment Dump every accepted HX segment with full visual/metric artifacts.

    Input: frozen segment manifest, checkpoint, output root, iteration, device,
    and optional network config/count contract. Output: per-segment input/output/
    GT/masks/montages, metrics, and aggregate summary. Key Steps: build the
    selected registry network, process all records, and fail after writing if a
    frozen count mismatches. Key Params: expected segment count defaults unset.
    State/Dependencies: local manifest, model checkpoint, OpenCV, and torch.
    """
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    canvas_hw = resolve_canvas_hw(network_config)
    resize_mode = str(resize_mode or ("stretch" if canvas_hw != CANVAS_HW else "aspect_fit")).strip().lower()
    if resize_mode not in ("aspect_fit", "stretch"):
        raise ValueError("resize_mode must be 'aspect_fit' or 'stretch'")
    dataset = BasicTextLineSegmentDatasetV3({
        "manifest": str(manifest),
        "canvas_hw": list(canvas_hw),
        "resize_mode": resize_mode,
        "min_short_side_ratio": 0.5,
        "min_long_side_ratio": 0.5,
    })
    network, checkpoint_sha256 = _make_network(device, checkpoint, network_config)
    iteration_root = output_root / f"iter_{int(iteration):09d}"
    iteration_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    image_records: dict[str, list[dict[str, Any]]] = {}
    source_groups_by_stem: dict[str, set[str]] = {}
    for manifest_record in dataset.records:
        manifest_path = Path(str(manifest_record["lq"]))
        base_stem = _safe_component(manifest_path.stem, "unknown_image")
        source_groups_by_stem.setdefault(base_stem, set()).add(_source_group(manifest_path))
    seen_line_keys: dict[tuple[str, int, int], int] = {}
    manifest_summary_data: dict[str, Any] = {}
    if manifest_summary and manifest_summary.is_file():
        manifest_summary_data = json.loads(manifest_summary.read_text(encoding="utf-8"))
        frame_manifest_value = manifest_summary_data.get("frame_manifest")
        frame_manifest = Path(frame_manifest_value) if frame_manifest_value else None
        if frame_manifest and frame_manifest.is_file():
            for line in frame_manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                frame = json.loads(line)
                image_stem = _safe_component(str(frame.get("stem", "")), "unknown_image")
                image_records.setdefault(image_stem, [])
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            prediction = network(item["lq"].unsqueeze(0).to(device))[0].detach().cpu().numpy()
            prediction = np.clip(prediction, 0.0, 1.0)
            lq_rgb = item["lq_rgb"].numpy().transpose(1, 2, 0)
            gt_rgb = item["gt_rgb"].numpy().transpose(1, 2, 0)
            gt_yuv = item["gt"].numpy()
            valid = item["valid_mask"][0].numpy()
            text = item["text_mask"][0].numpy()
            metric = _metrics(prediction, gt_yuv, valid, text)

            lq_path = Path(str(item["lq_path"]))
            base_stem = _safe_component(lq_path.stem, f"image_{index:04d}")
            source_group = _source_group(lq_path)
            if len(source_groups_by_stem.get(base_stem, {source_group})) > 1:
                stem = _safe_component(f"{source_group}__{base_stem}", base_stem)
            else:
                stem = base_stem
            segment_id = str(item.get("segment_id", ""))
            line_match = re.search(r"::line(\d+)", segment_id)
            line_index = int(line_match.group(1)) if line_match else index
            tile_index = int(item.get("tile_index", 0))
            image_dir = iteration_root / stem
            line_key = (stem, line_index, tile_index)
            duplicate_index = seen_line_keys.get(line_key, 0)
            seen_line_keys[line_key] = duplicate_index + 1
            suffix = f"_record_{index:06d}" if duplicate_index else ""
            line_dir = image_dir / f"line_{line_index:04d}_tile_{tile_index:02d}{suffix}"
            line_dir.mkdir(parents=True, exist_ok=True)
            pred_rgb = _yuv444_to_rgb(prediction)
            _write_rgb(line_dir / "input.png", lq_rgb)
            _write_rgb(line_dir / "output.png", pred_rgb)
            _write_rgb(line_dir / "gt.png", gt_rgb)
            _write_gray(line_dir / "valid_mask.png", valid)
            _write_gray(line_dir / "text_mask.png", text)
            diff = np.clip(np.abs(pred_rgb - gt_rgb) * 4.0, 0.0, 1.0)
            montage = np.concatenate([
                _label(lq_rgb, "INPUT"),
                _label(gt_rgb, "GT"),
                _label(pred_rgb, "OUTPUT"),
                _label(diff, "ABS DIFF x4"),
            ], axis=1)
            _write_rgb(line_dir / "montage.png", montage)

            record = {
                "index": index,
                "manifest_index": index,
                "stem": stem,
                "line_index": line_index,
                "tile_index": tile_index,
                "tile_count": int(item.get("tile_count", 1)),
                "orientation": str(item.get("orientation", "")),
                "source_line_id": str(item.get("line_id", "")),
                "segment_id": segment_id,
                "crop_xyxy_even": [int(v) for v in item["crop_xyxy_even"]],
                "source_valid_hw": [
                    int(item["crop_xyxy_even"][3] - item["crop_xyxy_even"][1]),
                    int(item["crop_xyxy_even"][2] - item["crop_xyxy_even"][0]),
                ],
                "model_valid_hw": list(canvas_hw) if dataset.resize_mode == "stretch" else [
                    int(item["crop_xyxy_even"][3] - item["crop_xyxy_even"][1]),
                    int(item["crop_xyxy_even"][2] - item["crop_xyxy_even"][0]),
                ],
                "valid_hw": list(canvas_hw) if dataset.resize_mode == "stretch" else [
                    int(item["crop_xyxy_even"][3] - item["crop_xyxy_even"][1]),
                    int(item["crop_xyxy_even"][2] - item["crop_xyxy_even"][0]),
                ],
                "resize_mode": dataset.resize_mode,
                "resize_scale": [float(value) for value in item.get("resize_scale", [1.0, 1.0])],
                "cp_eligible": bool(item.get("cp_eligible", False)),
                "transcription": str(item.get("transcription", "")),
                "files": sorted(str(path.relative_to(iteration_root)) for path in line_dir.iterdir()),
                **metric,
            }
            (line_dir / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(record)
            image_records.setdefault(stem, []).append(record)

    for stem, image_items in image_records.items():
        image_dir = iteration_root / stem
        image_dir.mkdir(parents=True, exist_ok=True)
        image_summary = {
            "stem": stem,
            "segments": len(image_items),
            "records": image_items,
        }
        (image_dir / "image_summary.json").write_text(
            json.dumps(image_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary: dict[str, Any] = {
        "iteration": int(iteration),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest": str(manifest),
        "manifest_summary": str(manifest_summary) if manifest_summary else None,
        "device": str(device),
        "canvas_hw": list(canvas_hw),
        "resize_mode": dataset.resize_mode,
        "input_shape": [1, 1, canvas_hw[0] + canvas_hw[0] // 2, canvas_hw[1]],
        "output_shape": [1, 3, canvas_hw[0], canvas_hw[1]],
        "accepted_segments": len(records),
        "source_images": len(image_records),
        "source_lines": len({record["source_line_id"] for record in records}),
        "records": records,
        "size_filter_stats": dataset.size_filter_stats,
        "filtered_records": dataset.filtered_records,
        "manifest_summary_data": manifest_summary_data,
        "expected_segments": expected_segments,
    }
    if records:
        for metric_name in ("text_y_mae", "text_yuv_weighted_mae", "valid_y_mae", "text_y_gradient_mae", "valid_psnr_y"):
            summary[f"mean_{metric_name}"] = float(np.mean([record[metric_name] for record in records]))
    (iteration_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if expected_segments is not None and len(records) != int(expected_segments):
        raise ValueError(f"expected {expected_segments} accepted segments, got {len(records)}")
    print(json.dumps({key: summary[key] for key in (
        "iteration", "accepted_segments", "source_images", "source_lines",
        "mean_text_y_mae", "mean_text_y_gradient_mae",
    ) if key in summary}, ensure_ascii=False, indent=2))
    return summary


def main(argv=None) -> int:
    """@guang-yang-comment Parse the HX dump CLI and return its process status.

    Input: manifest/checkpoint/output arguments, optional network config, and
    expected count. Output: artifacts from ``dump_checkpoint`` and exit code.
    Key Steps: parse options and invoke one dump. Key Params: network config is
    optional for baseline compatibility. State/Dependencies: local filesystem.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--network-config", type=Path, default=None,
                        help="Optional training YAML; network_g is built via the registry")
    parser.add_argument("--expected-segments", type=int, default=None,
                        help="Fail after writing artifacts if accepted segment count differs")
    parser.add_argument("--resize-mode", choices=("aspect_fit", "stretch"), default=None,
                        help="Resize mode; 96x480 contracts use stretch")
    args = parser.parse_args(argv)
    dump_checkpoint(
        args.manifest,
        args.manifest_summary,
        args.checkpoint,
        args.output_root,
        args.iteration,
        args.device,
        args.network_config,
        args.expected_segments,
        args.resize_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
