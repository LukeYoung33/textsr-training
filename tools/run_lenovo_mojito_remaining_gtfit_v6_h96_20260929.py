"""Run the remaining Mojito/PPT images on gtfit_v6 h96, then apply V5 postprocess."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.postprocess_lenovo_borderrail_normal_v5_20260929 import PASTE_MODE, apply_postprocess


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/guang.yang/realSR/bin/python")
SOURCE = Path("/AI_DNS/guang.yang/TextSR/results/lenovo_mojito_20260929_v5v6_test_ocr_scurve/preprocessed_ocrbox_tiled")
OLD_MANIFEST = Path("/home/guang.yang/data/lenovo_mojito_textsr_20260928_whiteboard_v4/manifests/train.jsonl")
OUTPUT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_mojito_remaining14_gtfit_v6_h96_20260929_v3")
CHECKPOINT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_8arm_mojito_finetune_20260928/lenovo_v5_r1_bglf_uv_gtfit_v6_borderrail_h96_mojito_ft100k_20260928/models/net_g_50000.pth")
CONFIG = REPO / "options/train_bucket_3/lenovo_borderrail_8arm_20260928/finetune/train_gtfit_v6_borderrail_h96_ft100k_20260928.yml"
CAMERA_STEM = re.compile(r"IMG_\d{8}_\d{9}")
POSTPROCESS_MODE = PASTE_MODE + "_local_residual_sigma4_pt26_conservative_polarity_v2"
# These four shots show the same slide. Keep the title, 40pt, 32pt, 26pt,
# and the later 28pt comparison line; leave all smaller slide text unchanged.
PPT_KEEP_OCR_INDEX = {
    "IMG_20260610_032725206": {0, 1, 2, 3, 13},
    "IMG_20260610_032933806": {0, 1, 2, 3, 13},
    "IMG_20260610_055939992": {0, 1, 2, 3, 14},
    "IMG_20260610_055946880": {0, 1, 2, 3, 12, 13},
}


def select_images(smoke: bool = False) -> tuple[list[dict], dict]:
    records = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))["rows"]
    old = [json.loads(line) for line in OLD_MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    old_names = {CAMERA_STEM.search(row["source_lq"]).group(0) for row in old}
    ppt_names = sorted(row["image"] for row in records if row["group"] == "ppt")
    if len(records) != 48 or len(ppt_names) != 24 or len(old_names) != 22:
        raise ValueError("source or exclusion population changed")
    allowed_ppt = set(ppt_names[:12])
    selected = [row for row in records if Path(row["image"]).stem not in old_names and
                (row["group"] != "ppt" or row["image"] in allowed_ppt)]
    if len(selected) != 14 or sum(row["group"] == "ppt" for row in selected) != 4:
        raise ValueError("expected 10 blackwhite and 4 PPT new images")
    excluded = [{"group": row["group"], "image": row["image"],
                 "reason": "previous_batch" if Path(row["image"]).stem in old_names else "later_12_blue_ppt"}
                for row in records if row not in selected]
    if smoke:
        selected = [next(row for row in selected if row["group"] == group)
                    for group in ("blackwhite", "ppt")]
    return selected, {"selected": [{"group": row["group"], "image": row["image"]} for row in selected],
                      "excluded": excluded, "source": str(SOURCE), "checkpoint": str(CHECKPOINT),
                      "config": str(CONFIG), "postprocess_version": POSTPROCESS_MODE,
                      "ppt_minimum_font_pt": 26, "ppt_later_28pt_kept": True}


def _preview(source: np.ndarray, model: np.ndarray, final: np.ndarray, destination: Path) -> None:
    width = min(800, source.shape[1])
    height = max(1, round(source.shape[0] * width / source.shape[1]))
    panels = [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
              for image in (source, model, final)]
    title = np.full((34, width * 3, 3), 245, np.uint8)
    for index, label in enumerate(("V6 BORDER-RAIL INPUT", "MODEL", "V5 POST")):
        cv2.putText(title, label, (index * width + 10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .66, (20, 20, 20), 2)
    if not cv2.imwrite(str(destination), np.vstack((title, np.hstack(panels))),
                       [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise ValueError(f"cannot save preview: {destination}")


def worker_main(worker: int, workers: int, smoke: bool, output_root: Path) -> int:
    import torch
    from tools.basic_v2_eval_network import build_eval_network
    from tools.infer_basic_text_line_full_image import NativeY4UVTileRunner, infer_full_image, load_lq_preprocess_config

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    rows, _ = select_images(smoke)
    assigned = [row for index, row in enumerate(rows) if index % workers == worker]
    if not assigned:
        return 0
    device = torch.device("cuda:0")
    network, checkpoint_hash = build_eval_network(CHECKPOINT, device, CONFIG)
    runner = NativeY4UVTileRunner(network, device, load_lq_preprocess_config(CONFIG))
    for ordinal, row in enumerate(assigned, 1):
        group, image_name = row["group"], row["image"]
        stem = Path(image_name).stem
        directory = output_root / "images" / group / stem
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        if result_path.is_file() and (directory / "output.png").is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("postprocess_version") == POSTPROCESS_MODE and existing.get("checkpoint_sha256") == checkpoint_hash:
                print(json.dumps({"worker": worker, "image": image_name, "status": "existing"}), flush=True)
                continue
        source_path = SOURCE / "fullframe" / "v6_borderrail" / group / f"{stem}.png"
        metadata_path = SOURCE / "metadata" / group / f"{stem}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ocr_path = Path(metadata["ocr"])
        annotations = json.loads(ocr_path.read_text(encoding="utf-8"))
        source_annotations = annotations["ocr"]["annotations"]
        selected_indices = None
        if group == "ppt":
            selected_indices = PPT_KEEP_OCR_INDEX[stem]
            if not selected_indices.issubset({int(line["ocr_index"]) for line in source_annotations}):
                raise ValueError(f"PPT OCR indices changed: {stem}")
            annotations["ocr"]["annotations"] = [line for line in source_annotations
                                                    if int(line["ocr_index"]) in selected_indices]
            annotations["ocr"]["annotation_count"] = len(annotations["ocr"]["annotations"])
            ocr_path = directory / "selected_ocr.json"
            ocr_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None or source.shape[:2] != (metadata["height"], metadata["width"]):
            raise ValueError(f"input decode/size mismatch: {source_path}")
        trace = infer_full_image(
            source_path, ocr_path, CHECKPOINT, directory / "model_output.png",
            device_name="cuda:0", network=network, checkpoint_sha256=checkpoint_hash,
            network_config=CONFIG, tile_runner=runner, ocr_format="p17_annotations",
            resize_mode="p17_height96_tile_pad", native_tile_height_threshold=128,
            large_tile_canvas_hw=(96, 480), large_tile_overlap=64,
            large_tile_overlap_x=64, large_tile_overlap_y=24,
            overlap=64, blend_radius=8, p17_vertical_context=.12, p17_horizontal_context=.03,
        )
        model = cv2.imread(str(directory / "model_output.png"), cv2.IMREAD_COLOR)
        if model is None or model.shape != source.shape:
            raise ValueError(f"model output decode/size mismatch: {image_name}")
        boxes = [{"bbox_xyxy": region["ocr_box"]} for region in metadata["regions"]
                 if selected_indices is None or int(region["ocr_index"]) in selected_indices]
        final, reports, frame_weight, frame_report = apply_postprocess(
            model, boxes, background_sigma=4.0, ambiguity_mode="conservative_edge")
        processed = sum(report["raw_changed_pixels"] > 0 and report["blended_pixels"] > 0 for report in reports)
        if processed != len(boxes):
            raise ValueError(f"postprocess did not cover every OCR box: {image_name} {processed}/{len(boxes)}")
        output_path = directory / "output.png"
        if not cv2.imwrite(str(output_path), final):
            raise ValueError(f"cannot save final image: {output_path}")
        cv2.imwrite(str(directory / "board_frame_protection.png"),
                    np.rint((1.0 - frame_weight) * 255).astype(np.uint8))
        _preview(source, model, final, directory / "preview.jpg")
        record = {"group": group, "image": image_name, "input": str(source_path),
                  "metadata": str(metadata_path), "ocr": str(ocr_path), "checkpoint": str(CHECKPOINT),
                  "checkpoint_sha256": checkpoint_hash, "postprocess_version": POSTPROCESS_MODE,
                  "selected_ocr_indices": sorted(selected_indices) if selected_indices is not None else None,
                  "skipped_small_ocr_lines": len(source_annotations) - len(selected_indices) if selected_indices is not None else 0,
                  "ocr_boxes": len(boxes), "processed_boxes": processed,
                  "model_lines_processed": trace.get("lines_processed"),
                  "model_segments_total": trace.get("segments_total"), "board_frame": frame_report,
                  "line_reports": reports, "output": str(output_path)}
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"worker": worker, "index": ordinal, "total": len(assigned),
                          "image": image_name, "ocr_boxes": len(boxes), "status": "complete"}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in 1..4")
    for path in (SOURCE / "summary.json", OLD_MANIFEST, CHECKPOINT, CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows, selection = select_images(args.smoke)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.worker is not None:
        return worker_main(args.worker, args.workers, args.smoke, args.output_root)
    processes = []
    for worker in range(min(args.workers, len(rows))):
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(worker), PYTHONPATH=str(REPO), OMP_NUM_THREADS="2")
        log_path = args.output_root / f"worker_{worker}.log"
        command = [str(PYTHON), "-u", str(Path(__file__)), "--workers", str(args.workers),
                   "--worker", str(worker), "--output-root", str(args.output_root)]
        if args.smoke:
            command.append("--smoke")
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((worker, process, handle, log_path))
    failed = []
    for worker, process, handle, log_path in processes:
        code = process.wait()
        handle.close()
        print(json.dumps({"worker": worker, "exit_code": code, "log": str(log_path)}), flush=True)
        if code:
            failed.append(worker)
    completed = list((args.output_root / "images").glob("*/*/result.json"))
    expected = len(rows)
    summary = {"expected": expected, "completed": len(completed), "failed_workers": failed,
               "rows": [json.loads(path.read_text(encoding="utf-8")) for path in completed]}
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not failed and len(completed) == expected:
        flat = args.output_root / "outputs_by_source_name"
        flat.mkdir(exist_ok=True)
        for record in summary["rows"]:
            source = Path(record["output"])
            shutil.copy2(source, flat / Path(record["image"]).with_suffix(".png"))
    print(json.dumps({"expected": expected, "completed": len(completed), "failed_workers": failed}), flush=True)
    return 0 if not failed and len(completed) == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
