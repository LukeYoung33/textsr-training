"""Render matched validation parent frames for the eight borderrail pretrains."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.basic_v2_eval_network import build_eval_network
from realesrgan.data.basic_text_line_full_frame import parse_p17_ocr_annotations
from tools.evaluate_lenovo_scurve_2x2_fullframe_boxfeather_20260928 import _evaluate_parent
from tools.infer_basic_text_line_full_image import NativeY4UVTileRunner, load_lq_preprocess_config


DATASETS = {
    "lenovo260926": (
        Path("/AI_DNS/guang.yang/TextSR/data/Lenovo_data_260926/training_20260927_batch3_validcrop_v4_brightfix_v5_v6_ocr_matrix_20260928"),
        (Path("/AI_DNS/guang.yang/TextSR/data/Lenovo_data_260926/training_20260927_validcrop_v4"),
         Path("/AI_DNS/guang.yang/TextSR/data/Lenovo_data_260926/training_20260927_batch3_validcrop_v4_brightfix")),
    ),
    "lenovo20260920": (
        Path("/AI_DNS/guang.yang/TextSR/data/Lenovo_data_260920/training_20260920_ocrfix_v4_formula_groups_v5_v6_ocr_matrix_20260928"),
        (Path("/home/guang.yang/data/lenovo_textsr_20260920_final_validcrop_ocrfix_v4_formula_groups"),),
    ),
}
ARMS = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
EXPERIMENTS = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_8arm_20260928")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evenly_spaced(items: list, limit: int) -> list:
    if limit == 0 or len(items) <= limit:
        return items
    if limit == 1:
        return items[:1]
    return [items[round(index * (len(items) - 1) / (limit - 1))] for index in range(limit)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--height", type=int, choices=(64, 96), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parents-per-dataset", type=int, default=0,
                        help="zero evaluates every validation parent image")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path,
                        help="generator checkpoint; defaults to the original 250K experiment path")
    parser.add_argument("--config", type=Path,
                        help="training YAML providing network_g and lq_preprocess")
    parser.add_argument("--dataset-0926-root", type=Path, default=DATASETS["lenovo260926"][0])
    parser.add_argument("--dataset-0920-root", type=Path, default=DATASETS["lenovo20260920"][0])
    parser.add_argument("--source-0926-root", type=Path, default=DATASETS["lenovo260926"][1][0])
    parser.add_argument("--source-0926-alt-root", type=Path, default=DATASETS["lenovo260926"][1][1])
    parser.add_argument("--source-0920-root", type=Path, default=DATASETS["lenovo20260920"][1][0])
    args = parser.parse_args()
    if args.parents_per_dataset < 0:
        raise ValueError("parents-per-dataset must be nonnegative")
    name = f"lenovo_v5_r1_bglf_uv_{args.arm}_h{args.height}_250k_20260928"
    checkpoint = args.checkpoint or EXPERIMENTS / name / "models/net_g_250000.pth"
    config = args.config or ROOT / "options/train_bucket_3/lenovo_borderrail_8arm_20260928" / f"train_{args.arm}_h{args.height}_250k_20260928.yml"
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(f"checkpoint or config missing: {checkpoint}, {config}")
    device = torch.device(args.device)
    network, checkpoint_hash = build_eval_network(checkpoint, device, config)
    preprocess = load_lq_preprocess_config(config)
    if preprocess["enabled"]:
        raise ValueError("on-the-fly preprocessing must be disabled")
    runner = NativeY4UVTileRunner(network, device, preprocess)
    all_reports = []
    selection = {}
    excluded_invalid_ocr = {}
    datasets = {
        "lenovo260926": (args.dataset_0926_root, (args.source_0926_root, args.source_0926_alt_root)),
        "lenovo20260920": (args.dataset_0920_root, (args.source_0920_root,)),
    }
    for dataset, (data_root, source_roots) in datasets.items():
        manifest = data_root / args.arm / "manifests/val.jsonl"
        grouped = defaultdict(list)
        excluded = []
        for row in rows(manifest):
            line = dict(row)
            line["files"] = {key: str((manifest.parent.parent / value).resolve())
                             for key, value in row["files"].items()}
            parent_id = str(row["parent_sample_id"])
            grouped.setdefault(parent_id, [])
            width, height = map(int, row["cropped_size_wh"])
            ocr_document = json.loads(Path(line["files"]["ocr"]).read_text(encoding="utf-8"))
            if len(parse_p17_ocr_annotations(ocr_document, (height, width))) != 1:
                excluded.append(str(row["sample_id"]))
                continue
            grouped[parent_id].append(line)
        excluded_invalid_ocr[dataset] = excluded
        reference_ids = sorted(grouped)
        chosen = evenly_spaced(reference_ids, args.parents_per_dataset)
        if any(parent_id not in grouped for parent_id in chosen):
            raise ValueError(f"selected parent absent from {args.arm} validation: {dataset}")
        selection[dataset] = chosen
        parent_lookup = {}
        for source_root in source_roots:
            for parent in rows(source_root / "manifests/val.jsonl"):
                parent_lookup[str(parent["sample_id"])] = source_root, parent
        missing = sorted(set(chosen) - set(parent_lookup))
        if missing:
            raise ValueError(f"parent source missing for {dataset}: {missing[:5]}")
        for index, parent_id in enumerate(chosen, 1):
            if parent_id not in parent_lookup:
                raise ValueError(f"parent source missing: {dataset}/{parent_id}")
            source_root, parent = parent_lookup[parent_id]
            target = args.output_root / dataset / source_root.name / parent_id
            report_path = target / "summary.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = _evaluate_parent(parent, grouped[parent_id], source_root,
                                          args.output_root, checkpoint, config, checkpoint_hash,
                                          network, runner, device, dataset, "raw", "full_box")
            all_reports.append(report)
            print(json.dumps({"arm": args.arm, "height": args.height, "dataset": dataset,
                              "index": index, "total": len(chosen), "parent": parent_id,
                              "lines": report["lines_processed"], "output": report["output"]}), flush=True)
    summary = {"arm": args.arm, "height": args.height, "checkpoint": str(checkpoint),
               "checkpoint_sha256": checkpoint_hash, "config": str(config),
               "paste_strategy": "full_box_8px_outer_feather",
               "data_roots": {key: str(value[0]) for key, value in datasets.items()},
               "source_roots": {key: [str(root) for root in value[1]] for key, value in datasets.items()},
               "excluded_invalid_ocr": excluded_invalid_ocr,
               "selection": selection, "expected": sum(map(len, selection.values())),
               "completed": len(all_reports), "model_calls": runner.calls,
               "rows": all_reports}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
