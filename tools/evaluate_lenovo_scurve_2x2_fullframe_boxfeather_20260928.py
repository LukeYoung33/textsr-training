"""Restore materialized Lenovo validation lines and paste them into source validation frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.basic_v2_eval_network import build_eval_network
from realesrgan.data.basic_text_line_full_frame import parse_p17_ocr_annotations
from tools.infer_basic_text_line_full_image import (
    NativeY4UVTileRunner, _read_rgb, _write_rgb, infer_full_image,
    load_lq_preprocess_config, rgb_to_yuv444, yuv444_to_rgb,
)


SOURCE_ROOTS = {
    "lenovo260926": (
        Path("/AI_DNS/guang.yang/TextSR/data/Lenovo_data_260926/training_20260927_validcrop_v4"),
        Path("/AI_DNS/guang.yang/TextSR/data/Lenovo_data_260926/training_20260927_batch3_validcrop_v4_brightfix"),
    ),
    "lenovo20260920": (Path("/home/guang.yang/data/lenovo_textsr_20260920_final_validcrop_ocrfix_v4_formula_groups"),),
}
MANIFEST_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_mild_scurve_2x2_20260927/input_manifests")
MATERIALIZED_260926_ROOT = Path("/home/guang.yang/data/lenovo_260926_validcrop_v4_all3_mild_scurve_v1_20260927")
MATERIALIZED_20260920_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_mild_scurve_2x2_20260927/ocrfix_v4_val_lines_mild_scurve_20260928")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _asset(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_gray(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise IOError(path)


def _box_feather(shape: tuple[int, int], crop_xyxy: tuple[int, int, int, int],
                 polygon: np.ndarray, valid: np.ndarray, expansion_px: float = 4.0) -> tuple[np.ndarray, list[float]]:
    """Keep the expanded OCR AABB opaque and fade it to the materialized crop edges."""
    full_h, full_w = shape
    x0, y0, x1, y1 = crop_xyxy
    points = np.asarray(polygon, dtype=np.float32)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    left = max(0.0, float(minimum[0]) - min(expansion_px, float(minimum[0]) * .5))
    top = max(0.0, float(minimum[1]) - min(expansion_px, float(minimum[1]) * .5))
    right = min(float(full_w), float(maximum[0]) + min(expansion_px, (full_w - float(maximum[0])) * .5))
    bottom = min(float(full_h), float(maximum[1]) + min(expansion_px, (full_h - float(maximum[1])) * .5))
    xx = np.arange(x0, x1, dtype=np.float32) + .5
    yy = np.arange(y0, y1, dtype=np.float32) + .5
    left_weight = np.clip(xx / left, 0.0, 1.0) if left > 0 else np.ones_like(xx)
    right_weight = np.clip((full_w - xx) / (full_w - right), 0.0, 1.0) if right < full_w else np.ones_like(xx)
    top_weight = np.clip(yy / top, 0.0, 1.0) if top > 0 else np.ones_like(yy)
    bottom_weight = np.clip((full_h - yy) / (full_h - bottom), 0.0, 1.0) if bottom < full_h else np.ones_like(yy)
    alpha = np.minimum(left_weight, right_weight)[None, :] * np.minimum(top_weight, bottom_weight)[:, None]
    support = np.asarray(valid, dtype=bool)
    if not support.all():
        distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
        alpha *= np.clip(distance / 4.0, 0.0, 1.0)
    return alpha.astype(np.float32), [left, top, right, bottom]


def _full_box_feather(valid: np.ndarray, radius: float = 8.0) -> np.ndarray:
    """Feather only the outside of the complete model-predicted crop box."""
    support = np.asarray(valid, dtype=np.uint8)
    if support.ndim != 2:
        raise ValueError("full-box support must be a 2-D mask")
    padded = np.pad(support, 1, mode="constant")
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    return np.clip(distance / radius, 0.0, 1.0).astype(np.float32)


def _comparison(lq: np.ndarray, used: np.ndarray, output: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """@luna-comment
    Input: Four aligned RGB float images at the same full-frame shape.
    Output: One width-concatenated BGR uint8 visual comparison.
    Key Steps: Scale each panel equally, concatenate, and convert color order.
    Key Params: A maximum 850-pixel panel side bounds comparison image size.
    State/Dependencies: Uses OpenCV; no persistent state or source mutation.
    """
    scale = min(1.0, 850.0 / lq.shape[0], 850.0 / lq.shape[1])
    size = (max(1, round(lq.shape[1] * scale)), max(1, round(lq.shape[0] * scale)))
    panels = [cv2.resize((np.clip(image, 0, 1) * 255).astype(np.uint8), size,
                         interpolation=cv2.INTER_AREA) for image in (lq, used, output, gt)]
    return cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR)


def _evaluate_parent(parent: dict[str, Any], lines: list[dict[str, Any]], source_root: Path,
                     output_root: Path, checkpoint: Path, config: Path, checkpoint_hash: str,
                     network: torch.nn.Module, runner: NativeY4UVTileRunner, device: torch.device,
                     dataset_name: str, mode: str, paste_strategy: str = "ocr_box") -> dict[str, Any]:
    """@luna-comment
    Input: One parent sample, its materialized OCR lines, runtime assets, and raw/scurve mode.
    Output: A per-parent report and saved original/used/model/GT comparison artifacts.
    Key Steps: Read aligned assets, infer each line, paste the line input and prediction with shared alpha, and write results.
    Key Params: The line envelope and inference blend radius define the parent update region.
    State/Dependencies: Reads manifest assets, calls the frozen inference runner, and writes under output_root.
    """
    parent_id = str(parent["sample_id"])
    if paste_strategy not in ("ocr_box", "full_box"):
        raise ValueError(f"unsupported paste strategy: {paste_strategy}")
    sample_root = output_root / dataset_name / source_root.name / parent_id
    files = parent["files"]
    lq_path, gt_path = _asset(source_root, files["lq"]), _asset(source_root, files["gt"])
    frame_rgb = _read_rgb(lq_path)
    gt_rgb = _read_rgb(gt_path)
    if frame_rgb.shape != gt_rgb.shape:
        raise ValueError(f"parent LQ/GT shape mismatch: {parent_id}")
    frame_yuv = rgb_to_yuv444(frame_rgb)
    output_sum = np.zeros_like(frame_yuv)
    used_sum = np.zeros_like(frame_yuv)
    weight_sum = np.zeros(frame_rgb.shape[:2], np.float32)
    line_reports = []
    for row in lines:
        line_id = str(row["sample_id"])
        line_path = Path(row["files"]["lq"])
        ocr_path = Path(row["files"]["ocr"])
        if ("LQ_scurve" in str(line_path)) != (mode == "scurve"):
            raise ValueError(f"input mode mismatch: {line_path}")
        envelope = tuple(map(int, row["line_envelope_xyxy_in_parent"]))
        x0, y0, x1, y1 = envelope
        line_rgb = _read_rgb(line_path)
        if line_rgb.shape[:2] != (y1 - y0, x1 - x0):
            raise ValueError(f"line envelope size mismatch: {line_id}")
        if not (0 <= x0 < x1 <= frame_rgb.shape[1] and 0 <= y0 < y1 <= frame_rgb.shape[0]):
            raise ValueError(f"line envelope outside parent: {line_id}")
        raw_line = _read_rgb(Path(row["files"]["lq_raw"]))
        source_line = frame_rgb[y0:y1, x0:x1]
        if np.max(np.abs(raw_line - source_line)) > 0:
            raise ValueError(f"materialized raw line does not match parent pixels: {line_id}")
        calls = 0
        updated_pixels = 0
        line_output_yuv = rgb_to_yuv444(line_rgb)
        protected_box: list[float] | None = None

        def paste(crop_xyxy: tuple[int, int, int, int], prediction: np.ndarray,
                  valid: np.ndarray, line: Any) -> None:
            """Accumulate one line using an opaque expanded box and outer-crop feather."""
            nonlocal calls, updated_pixels, protected_box
            lx0, ly0, lx1, ly1 = crop_xyxy
            # The line inference pads odd dimensions to even; never map padding into the parent.
            lx1, ly1 = min(lx1, line_rgb.shape[1]), min(ly1, line_rgb.shape[0])
            h, w = ly1 - ly0, lx1 - lx0
            if h <= 0 or w <= 0:
                raise ValueError(f"empty prediction crop: {line_id}")
            parent_crop = (x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1)
            prediction = prediction[:, :h, :w]
            if paste_strategy == "full_box":
                alpha = _full_box_feather(valid[:h, :w])
                protected_box = [float(lx0), float(ly0), float(lx1), float(ly1)]
            else:
                alpha, protected_box = _box_feather(line_rgb.shape[:2], (lx0, ly0, lx1, ly1),
                                                     line.polygon_xy, valid[:h, :w])
            px0, py0, px1, py1 = parent_crop
            output_sum[:, py0:py1, px0:px1] += prediction * alpha[None]
            input_patch = rgb_to_yuv444(line_rgb[ly0:ly1, lx0:lx1])
            used_sum[:, py0:py1, px0:px1] += input_patch * alpha[None]
            weight_sum[py0:py1, px0:px1] += alpha
            line_output_yuv[:, ly0:ly1, lx0:lx1] = prediction * alpha[None] + input_patch * (1.0 - alpha[None])
            calls += 1
            updated_pixels += int(np.count_nonzero(alpha))

        line_output = sample_root / "line_outputs" / f"{line_id}.png"
        trace = infer_full_image(
            line_path, ocr_path, checkpoint, line_output, device_name=str(device),
            overlap=64, blend_radius=8, network=network, checkpoint_sha256=checkpoint_hash,
            network_config=config, tile_runner=runner, resize_mode="p17_height96_tile_pad",
            ocr_format="p17_annotations", p17_vertical_context=.12, p17_horizontal_context=.03,
            line_prediction_support_callback=paste if paste_strategy == "ocr_box" else None,
            line_prediction_full_box_callback=paste if paste_strategy == "full_box" else None,
        )
        if calls != 1 or trace["lines_processed"] != 1:
            raise RuntimeError(f"expected one processed OCR line for {line_id}, got callbacks={calls}, trace={trace['lines_processed']}")
        _write_rgb(line_output, yuv444_to_rgb(line_output_yuv))
        line_reports.append({"line_id": line_id, "lq": str(line_path), "envelope_xyxy": envelope,
                             "model_calls": trace["segments_total"], "updated_pixels": updated_pixels,
                             "protected_box_xyxy_in_line": protected_box,
                             "line_output": str(line_output)})
    processed = weight_sum > 0
    opacity = np.clip(weight_sum, 0.0, 1.0)
    denominator = np.maximum(weight_sum, 1e-6)
    output_yuv = frame_yuv * (1.0 - opacity[None]) + (output_sum / denominator[None]) * opacity[None]
    used_yuv = frame_yuv * (1.0 - opacity[None]) + (used_sum / denominator[None]) * opacity[None]
    output_rgb = yuv444_to_rgb(output_yuv)
    used_rgb = yuv444_to_rgb(used_yuv)
    output_rgb[~processed] = frame_rgb[~processed]
    used_rgb[~processed] = frame_rgb[~processed]
    sample_root.mkdir(parents=True, exist_ok=True)
    _write_rgb(sample_root / "output.png", output_rgb)
    _write_rgb(sample_root / "lq_original.png", frame_rgb)
    _write_rgb(sample_root / "lq_used.png", used_rgb)
    _write_rgb(sample_root / "gt.png", gt_rgb)
    _write_gray(sample_root / "changed_pixels.png", (processed.astype(np.uint8) * 255))
    comparison_path = sample_root / "comparison_lq_used_output_gt.jpg"
    if not cv2.imwrite(str(comparison_path), _comparison(frame_rgb, used_rgb, output_rgb, gt_rgb),
                       [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(comparison_path)
    report = {"sample_id": parent_id, "dataset": dataset_name, "status": "passed",
              "parent_lq": str(lq_path), "parent_gt": str(gt_path), "mode": mode,
              "paste_strategy": paste_strategy,
              "lines_selected": len(lines), "lines_processed": len(line_reports),
              "processed_pixels": int(processed.sum()), "model_calls": sum(item["model_calls"] for item in line_reports),
              "output": str(sample_root / "output.png"), "comparison": str(comparison_path), "lines": line_reports}
    (sample_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scurve", "raw"), required=True)
    parser.add_argument("--height", type=int, choices=(64, 96), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-parents-per-dataset", type=int)
    parser.add_argument("--min-ocr-height-260926", type=float,
                        help="Use every materialized 20260926 validation line with OCR height strictly above this value")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    network, checkpoint_hash = build_eval_network(args.checkpoint, device, args.config)
    preprocess = load_lq_preprocess_config(args.config)
    if preprocess["enabled"]:
        raise ValueError("this evaluation requires materialized input and disabled online preprocessing")
    runner = NativeY4UVTileRunner(network, device, preprocess)
    args.output_root.mkdir(parents=True)
    all_reports = []
    selected_lines_by_dataset = {}
    excluded_invalid_ocr: dict[str, list[str]] = {}
    for dataset_name, source_roots in SOURCE_ROOTS.items():
        selected_manifest = MANIFEST_ROOT / f"{dataset_name}_val_{args.mode}.jsonl"
        grouped: dict[tuple[Path, str], list[dict[str, Any]]] = defaultdict(list)
        if dataset_name == "lenovo20260920":
            selected_rows = []
            invalid_ids = []
            for row in _rows(MATERIALIZED_20260920_ROOT / "manifests/val.jsonl"):
                x0, y0, x1, y1 = row["line_envelope_xyxy_in_parent"]
                line_hw = (y1 - y0, x1 - x0)
                document = json.loads(Path(row["files"]["ocr"]).read_text(encoding="utf-8"))
                if len(parse_p17_ocr_annotations(document, line_hw)) != 1:
                    invalid_ids.append(row["sample_id"])
                    continue
                selected_rows.append(row)
            excluded_invalid_ocr[dataset_name] = invalid_ids
            if args.mode == "raw":
                for row in selected_rows:
                    row["files"]["lq"] = row["files"]["lq_raw"]
        elif dataset_name == "lenovo260926" and args.min_ocr_height_260926 is not None:
            selected_rows = []
            for source_row in _rows(MATERIALIZED_260926_ROOT / "manifests/val.jsonl"):
                bbox = source_row["ocr_crop_xyxy"]
                if float(bbox[3]) - float(bbox[1]) <= args.min_ocr_height_260926:
                    continue
                row = dict(source_row)
                row["files"] = dict(row["files"])
                if args.mode == "raw":
                    row["files"]["lq"] = row["files"]["lq_raw"]
                selected_rows.append(row)
        else:
            selected_rows = _rows(selected_manifest)
        selected_lines_by_dataset[dataset_name] = len(selected_rows)
        for row in selected_rows:
            source = Path(row["source_release_root"]) if row.get("source_release_root") else source_roots[0]
            if source not in source_roots:
                raise ValueError(f"unexpected parent source: {source}")
            grouped[(source, str(row["parent_sample_id"]))].append(row)
        parent_rows = [(source_root, parent)
                       for source_root in source_roots
                       for parent in _rows(source_root / "manifests/val.jsonl")]
        known = {(root, str(parent["sample_id"])) for root, parent in parent_rows}
        if set(grouped) - known:
            raise ValueError(f"selected lines have no parent images: {sorted(set(grouped) - known)[:5]}")
        if args.max_parents_per_dataset is not None:
            parent_rows = [item for item in parent_rows if (item[0], str(item[1]["sample_id"])) in grouped][:args.max_parents_per_dataset]
        for index, (source_root, parent) in enumerate(parent_rows, 1):
            parent_id = str(parent["sample_id"])
            try:
                report = _evaluate_parent(parent, grouped.get((source_root, parent_id), []), source_root,
                                          args.output_root, args.checkpoint, args.config,
                                          checkpoint_hash, network, runner, device,
                                          dataset_name, args.mode)
            except Exception as exc:
                report = {"sample_id": parent_id, "dataset": dataset_name,
                          "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            all_reports.append(report)
            print(json.dumps({"dataset": dataset_name, "index": index, "total": len(parent_rows),
                              "sample_id": parent_id, "status": report["status"],
                              "lines": report.get("lines_processed"), "error": report.get("error")},
                             ensure_ascii=False), flush=True)
            (args.output_root / "progress.json").write_text(
                json.dumps({"arms": f"{args.mode}_h{args.height}", "completed": len(all_reports),
                            "passed": sum(row["status"] == "passed" for row in all_reports),
                            "failed": sum(row["status"] == "failed" for row in all_reports)}, indent=2), encoding="utf-8")
    summary = {"arm": f"{args.mode}_h{args.height}", "checkpoint": str(args.checkpoint),
               "checkpoint_sha256": checkpoint_hash, "config": str(args.config),
               "config_sha256": _sha256(args.config), "input_mode": args.mode,
               "input_contract": "materialized LQ_scurve" if args.mode == "scurve" else "materialized LQ_raw",
               "parent_image_contract": "source validcrop val/LQ, materialized line envelope mapped to parent coordinates",
               "min_ocr_height_260926_exclusive": args.min_ocr_height_260926,
               "selected_lines_by_dataset": selected_lines_by_dataset,
               "excluded_invalid_ocr": excluded_invalid_ocr,
               "pasteback": "expanded OCR AABB protected by 4px when context permits; separable feather to line envelope; overlapping line predictions weighted together",
               "parents": len(all_reports), "passed": sum(row["status"] == "passed" for row in all_reports),
               "failed": sum(row["status"] == "failed" for row in all_reports),
               "lines_processed": sum(row.get("lines_processed", 0) for row in all_reports),
               "model_calls": runner.calls, "rows": all_reports}
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
