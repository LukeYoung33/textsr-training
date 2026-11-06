"""Render three OCR routing variants for one 50K finetune checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.basic_v2_eval_network import build_eval_network
from tools.infer_basic_text_line_full_image import NativeY4UVTileRunner, infer_full_image, load_lq_preprocess_config


DATA_ROOT = Path("/home/guang.yang/data/lenovo_mojito_textsr_20260928_whiteboard_v4")
MANIFEST = DATA_ROOT / "manifests/train.jsonl"
ROUTES = ("normal", "large_tile", "overlap_tile")


def records() -> list[dict]:
    rows = []
    for line in MANIFEST.open(encoding="utf-8"):
        row = json.loads(line)
        files = row["files"]
        rows.append({"sample_id": row["sample_id"],
                     **{key: DATA_ROOT / files[key] for key in ("lq", "gt", "ocr")}})
    if len(rows) != 33 or len({row["sample_id"] for row in rows}) != 33:
        raise ValueError("expected exactly 33 unique source images")
    return rows


def _box(item: dict) -> tuple[float, float, float, float] | None:
    raw = item.get("polygon") or item.get("bbox_xyxy")
    if raw is None:
        return None
    array = np.asarray(raw, np.float32)
    if array.shape == (4,):
        x0, y0, x1, y1 = map(float, array)
    elif array.ndim == 2 and array.shape[1] == 2:
        x0, y0 = map(float, array.min(axis=0))
        x1, y1 = map(float, array.max(axis=0))
    else:
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _large_overlap(first: tuple[float, ...], second: tuple[float, ...], threshold: float) -> float:
    x = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    y = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    if not x or not y:
        return 0.0
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (first, second)]
    ratio = x * y / max(1.0, min(areas))
    return ratio if ratio >= threshold else 0.0


def merge_overlapping_ocr(data: dict, threshold: float = 0.30) -> tuple[dict, dict, set[int]]:
    """Merge connected large-overlap OCR rectangles; keep all other lines."""
    items = data.get("lines")
    if not isinstance(items, list):
        raise ValueError("source OCR must contain a lines list")
    boxes = [_box(item) if isinstance(item, dict) and not item.get("ignore") and item.get("visible", True) else None
             for item in items]
    parent = list(range(len(items)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs = []
    for i, first in enumerate(boxes):
        if first is None:
            continue
        for j in range(i + 1, len(boxes)):
            if boxes[j] is None:
                continue
            ratio = _large_overlap(first, boxes[j], threshold)
            if ratio:
                parent[root(j)] = root(i)
                pairs.append({"first": i, "second": j, "intersection_over_smaller": round(ratio, 4)})
    components: dict[int, list[int]] = {}
    for index, box in enumerate(boxes):
        if box is not None:
            components.setdefault(root(index), []).append(index)
    groups = [sorted(members) for members in components.values() if len(members) >= 2]
    groups.sort(key=lambda members: members[0])
    group_by_member = {member: group for group in groups for member in group}
    output_lines = []
    group_records = []
    forced_indices: set[int] = set()
    for index, item in enumerate(items):
        group = group_by_member.get(index)
        if group is None:
            output_lines.append(item)
            continue
        if index != group[0]:
            continue
        left = min(boxes[member][0] for member in group)
        top = min(boxes[member][1] for member in group)
        right = max(boxes[member][2] for member in group)
        bottom = max(boxes[member][3] for member in group)
        polygon = [[left, top], [right, top], [right, bottom], [left, bottom]]
        output_index = len(output_lines)
        forced_indices.add(output_index)
        output_lines.append({"polygon": polygon, "bbox_xyxy": [left, top, right, bottom],
                             "rec_score": max(float(items[member].get("rec_score", 1.0)) for member in group),
                             "text": " ".join(str(items[member].get("text", "")) for member in group),
                             "merged_from_indices": group})
        group_records.append({"output_line_index": output_index, "source_line_indices": group,
                              "union_xyxy": [left, top, right, bottom], "union_height": bottom - top})
    merged = {"image": data.get("image"), "lines": output_lines}
    report = {"overlap_threshold": threshold, "original_lines": len(items),
              "result_lines": len(output_lines), "overlap_pairs": pairs, "groups": group_records,
              "forced_native_tile_indices": sorted(forced_indices)}
    return merged, report, forced_indices


def _preview(lq_path: Path, output_path: Path, gt_path: Path, destination: Path) -> None:
    panels = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in (lq_path, output_path, gt_path)]
    if any(panel is None for panel in panels) or len({panel.shape for panel in panels}) != 1:
        raise ValueError("LQ/output/GT decode or shape mismatch")
    target_w = min(1100, panels[0].shape[1])
    scale = target_w / panels[0].shape[1]
    target_h = max(1, int(round(panels[0].shape[0] * scale)))
    resized = [cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA) for panel in panels]
    strip = np.full((34, 3 * target_w, 3), 245, np.uint8)
    for index, label in enumerate(("LQ", "OUTPUT", "GT")):
        cv2.putText(strip, label, (index * target_w + 14, 25), cv2.FONT_HERSHEY_SIMPLEX, .75, (30, 30, 30), 2)
    preview = np.vstack((strip, np.hstack(resized)))
    cv2.imwrite(str(destination), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--height", type=int, choices=(64, 96), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--sample-prefix")
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=ROUTES)
    parser.add_argument("--tile-debug", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    selected = records()
    if args.sample_id:
        selected = [row for row in selected if row["sample_id"] == args.sample_id]
        if len(selected) != 1:
            raise ValueError("sample ID not present exactly once")
    elif args.sample_prefix:
        selected = [row for row in selected if row["sample_id"].startswith(args.sample_prefix)]
        if not selected:
            raise ValueError("sample prefix not present")
    elif args.max_images:
        selected = selected[:args.max_images]
    device = torch.device(args.device)
    network, checkpoint_hash = build_eval_network(args.checkpoint, device, args.config)
    runner = NativeY4UVTileRunner(network, device, load_lq_preprocess_config(args.config))
    tile_canvas_hw = (args.height, 480)
    tile_overlap_x = 64
    tile_overlap_y = 24 if args.height == 96 else 16
    rows = []
    for image_index, row in enumerate(selected, 1):
        source_ocr = json.loads(row["ocr"].read_text(encoding="utf-8"))
        merged_ocr, merge_report, forced_indices = merge_overlapping_ocr(source_ocr)
        for route in args.routes:
            image_root = args.output_root / route / row["sample_id"]
            image_root.mkdir(parents=True, exist_ok=True)
            output = image_root / "output.png"
            result_path = image_root / "result.json"
            debug_dir = image_root / "tile_debug" if args.tile_debug else None
            if output.is_file() and result_path.is_file() and (debug_dir is None or debug_dir.is_dir()):
                rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            ocr_path = row["ocr"]
            if route == "overlap_tile":
                ocr_path = image_root / "ocr_merged.json"
                ocr_path.write_text(json.dumps(merged_ocr, ensure_ascii=False, indent=2), encoding="utf-8")
                (image_root / "merge_report.json").write_text(json.dumps(merge_report, ensure_ascii=False, indent=2), encoding="utf-8")
            trace = infer_full_image(
                row["lq"], ocr_path, args.checkpoint, output, device_name=args.device,
                network=network, checkpoint_sha256=checkpoint_hash, network_config=args.config,
                tile_runner=runner, ocr_format="p17_annotations",
                resize_mode="p17_height96_tile_pad" if route == "normal" else "native_tile",
                native_tile_height_threshold=128, large_tile_canvas_hw=tile_canvas_hw,
                large_tile_overlap=tile_overlap_x,
                large_tile_overlap_x=tile_overlap_x, large_tile_overlap_y=tile_overlap_y,
                force_native_tile_line_indices=forced_indices if route == "overlap_tile" else None,
                overlap=64, blend_radius=8, p17_vertical_context=.12, p17_horizontal_context=.03,
                debug_dir=debug_dir,
            )
            rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
            if rendered is None:
                raise ValueError(f"failed to decode output: {output}")
            full_jpeg = image_root / "output_full.jpg"
            if not cv2.imwrite(str(full_jpeg), rendered, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise ValueError(f"failed to save JPEG: {full_jpeg}")
            _preview(row["lq"], output, row["gt"], image_root / "preview.jpg")
            record = {"sample_id": row["sample_id"], "route": route, "status": "passed",
                      "output": str(output), "output_full_jpg": str(full_jpeg),
                      "preview": str(image_root / "preview.jpg"),
                      "lines_processed": trace["lines_processed"], "segments_total": trace["segments_total"],
                      "native_segments": trace["segments_native"], "resized_segments": trace["segments_resized"],
                      "groups": len(merge_report["groups"]) if route == "overlap_tile" else 0,
                      "tile_canvas_hw": list(tile_canvas_hw),
                      "tile_overlap_xy": [tile_overlap_x, tile_overlap_y],
                      "tile_debug_dir": str(debug_dir) if debug_dir else None,
                      "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash}
            result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(record)
            print(json.dumps({"arm": args.arm, "height": args.height, "image": image_index,
                              "total_images": len(selected), **record}, ensure_ascii=False), flush=True)
    expected = len(selected) * len(args.routes)
    summary = {"arm": args.arm, "height": args.height, "expected": expected, "completed": len(rows),
               "status": "complete" if len(rows) == expected else "incomplete", "rows": rows}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
