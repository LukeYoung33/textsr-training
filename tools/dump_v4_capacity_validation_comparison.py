"""Write RGB-clamped real-paired validation montages for four V4 capacities.

Each output row is ``LQ | GT | V4-base | V4-10x | V4-20x | V4-50x``.  The
dataset's packed-NV21 LQ and normalized-YUV444 GT/predictions are decoded with
their respective colour contracts before writing, so no generic-RGB writer is
involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444

from dump_basic_v2_validation_rgb_pairs import (
    _label,
    _load_dataset,
    _nv21_to_rgb,
    _safe_component,
    _write_rgb,
    _yuv444_to_rgb,
)
from basic_v2_eval_network import build_eval_network, resolve_canvas_hw


def _parse_model(value: str) -> tuple[str, Path, Path]:
    try:
        name, config, checkpoint = value.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("model must be NAME=CONFIG=CHECKPOINT") from exc
    return name, Path(config), Path(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0, help="0 writes all paired validation samples")
    parser.add_argument("--layout", choices=("horizontal", "vertical"), default="horizontal")
    parser.add_argument("--write-individual", action="store_true", help="also save LQ, GT, and each model output separately")
    args = parser.parse_args()
    if len(args.model) < 2:
        raise ValueError("at least two models are required for a comparison")

    device = torch.device(args.device)
    first_name, first_config, _ = args.model[0]
    dataset = _load_dataset(first_config)
    canvas_hw = resolve_canvas_hw(first_config)
    if tuple(dataset.canvas_hw) != canvas_hw:
        raise ValueError(f"dataset canvas {dataset.canvas_hw} != model canvas {canvas_hw}")
    models: list[tuple[str, torch.nn.Module, str]] = []
    for name, config, checkpoint in args.model:
        if resolve_canvas_hw(config) != canvas_hw:
            raise ValueError(f"{name} has a different canvas")
        network, digest = build_eval_network(checkpoint, device, config)
        models.append((name, network, digest))

    count = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index in range(count):
            item = dataset[index]
            lq_rgb = _nv21_to_rgb(item["lq"])
            lq = rgb_to_y4uv(torch.from_numpy(np.ascontiguousarray(lq_rgb)).permute(2, 0, 1).unsqueeze(0)).to(device)
            gt_rgb = _yuv444_to_rgb(item["gt"])
            panels = [_label(lq_rgb, "LQ (NV21 -> RGB)"), _label(gt_rgb, "GT")]
            output_rgbs: dict[str, np.ndarray] = {}
            for name, network, _ in models:
                output_rgb = _yuv444_to_rgb(y4uv_to_yuv444(network(lq)))
                output_rgbs[name] = output_rgb
                panels.append(_label(output_rgb, name))
            image = np.concatenate(panels, axis=1 if args.layout == "horizontal" else 0)
            stem = _safe_component(Path(str(item["line_id"])).stem, f"sample_{index:06d}")
            filename = f"{index:06d}_{stem}.png"
            _write_rgb(args.output_root / "images" / filename, image)
            if args.write_individual:
                _write_rgb(args.output_root / "lq" / filename, lq_rgb)
                _write_rgb(args.output_root / "gt" / filename, gt_rgb)
                for name, output_rgb in output_rgbs.items():
                    _write_rgb(args.output_root / "outputs" / _safe_component(name, "model") / filename, output_rgb)
            records.append({"index": index, "file": f"images/{filename}", "line_id": str(item["line_id"]),
                            "source_name": str(item["source_name"]), "lq_path": str(item["lq_path"]),
                            "gt_path": str(item["gt_path"])})
            if (index + 1) % 50 == 0 or index + 1 == count:
                print(f"[v4-capacity-val] {index + 1}/{count}", flush=True)
    report = {"status": "COMPLETE", "sample_count": count, "canvas_hw": list(canvas_hw),
              "models": [{"name": name, "config": str(config), "checkpoint": str(checkpoint), "sha256": digest}
                         for (name, config, checkpoint), (_, _, digest) in zip(args.model, models)],
              "layout": args.layout, "write_individual": args.write_individual,
              "color_contract": "LQ NV21 and GT/output YUV444 are decoded to RGB and clamped before PNG write",
              "records": records}
    (args.output_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "sample_count", "canvas_hw")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
