"""Evaluate the latest HX CP checkpoints with per-sample diagnostics.

The report compares input RGB, the raw latest checkpoint output decoded with
the training YUV2RGB path, and GT using the current CRNN tile CP loss.  It
also reads the matching visualization dump and records dump-vs-raw values.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realesrgan.archs.basic_text_line_r4_student_v2_arch import (
    BasicTextLineR4StudentV2,
    load_strict_checkpoint,
)
from realesrgan.data.basic_text_line_segment_dataset_v3 import (
    BasicTextLineSegmentDatasetV3,
)
from realesrgan.losses.crnn_content_perceptual_loss import (
    FrozenCRNNFeatureExtractor,
    _DEFAULT_LAYER_WEIGHTS,
)
from realesrgan.utils import YUV2RGB


CANVAS_HW = (96, 320)
BATCH_SIZE = 8


def iteration_of(path: Path) -> int:
    match = re.search(r"(?:net_g_|iter_)(\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(root: Path) -> Path:
    candidates = [path for path in (root / "models").glob("net_g_*.pth") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint under {root / 'models'}")
    return max(candidates, key=iteration_of)


def latest_dump(root: Path) -> Path:
    candidates = [
        path for path in (root / "visualizations").glob("iter_*")
        if (path / "summary.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no visualization dump under {root / 'visualizations'}")
    return max(candidates, key=iteration_of)


def make_network(device: torch.device) -> torch.nn.Module:
    return BasicTextLineR4StudentV2(
        canvas_hw=CANVAS_HW,
        in_channel=12,
        out_channel=12,
        layer_num=3,
        expand_channel=[0.5, 1.0, 2.0, 4.0],
        depth_factor=1,
        scale_factor=2,
        with_idt=False,
        act_type="prelu",
        scale=2,
        with_proj=True,
    ).to(device).eval()


def valid_hw(item: dict[str, Any]) -> list[int]:
    crop = item["crop_xyxy_even"]
    return [int(crop[3] - crop[1]), int(crop[2] - crop[0])]


def resize_valid(image: torch.Tensor, hw: list[int]) -> tuple[torch.Tensor, int]:
    height = min(int(hw[0]), int(image.shape[-2]))
    width = min(int(hw[1]), int(image.shape[-1]))
    crnn_width = max(16, min(1024, int(round(width * 32.0 / height))))
    crop = image[:, :height, :width].unsqueeze(0)
    resized = F.interpolate(
        crop,
        size=(32, crnn_width),
        mode="bicubic",
        align_corners=True,
    )
    return resized.squeeze(0), crnn_width


def load_png(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def feature_loss(
    extractor: FrozenCRNNFeatureExtractor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    value = torch.zeros(prediction.shape[0], device=prediction.device)
    prediction_features = extractor(prediction)
    with torch.no_grad():
        target_features = extractor(target)
    for weight, prediction_feature, target_feature in zip(
        _DEFAULT_LAYER_WEIGHTS,
        prediction_features,
        target_features,
    ):
        value = value + float(weight) * (
            prediction_feature - target_feature
        ).square().flatten(1).mean(dim=1)
    return value.detach().cpu()


def sobel_magnitude(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32)
    grad_x = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def masked_mae(value: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    denominator = max(float(mask.sum()), 1.0)
    return float((np.abs(value - target) * mask).sum() / denominator)


def classify(row: dict[str, Any]) -> str:
    cp_change = float(row["relative_cp_improvement"])
    y_change = float(row["output_text_y_mae"] - row["input_text_y_mae"])
    cp_state = "improved" if cp_change > 0.05 else "regressed" if cp_change < -0.05 else "flat"
    y_state = "improved" if y_change < -0.005 else "regressed" if y_change > 0.005 else "flat"
    return f"{cp_state}_cp__{y_state}_y"


def dump_cp_values(
    dump_root: Path,
    extractor: FrozenCRNNFeatureExtractor,
    segment_id: str,
    device: torch.device,
) -> dict[str, float]:
    summary = json.loads((dump_root / "summary.json").read_text(encoding="utf-8"))
    record = next(
        (
            value for value in summary.get("records", [])
            if str(value.get("segment_id")) == segment_id
        ),
        None,
    )
    if record is None:
        return {}
    files = {
        Path(str(value)).name: dump_root / str(value)
        for value in record.get("files", [])
    }
    if not all(
        files.get(name) is not None and files[name].is_file()
        for name in ("input.png", "output.png", "gt.png")
    ):
        return {}
    hw = [int(record["valid_hw"][0]), int(record["valid_hw"][1])]
    input_crop, input_width = resize_valid(load_png(files["input.png"]), hw)
    output_crop, output_width = resize_valid(load_png(files["output.png"]), hw)
    target_crop, target_width = resize_valid(load_png(files["gt.png"]), hw)
    if len({input_width, output_width, target_width}) != 1:
        return {}
    with torch.inference_mode():
        input_value = feature_loss(
            extractor,
            input_crop.unsqueeze(0).to(device),
            target_crop.unsqueeze(0).to(device),
        )[0]
        output_value = feature_loss(
            extractor,
            output_crop.unsqueeze(0).to(device),
            target_crop.unsqueeze(0).to(device),
        )[0]
    return {
        "dump_input_cp": float(input_value),
        "dump_output_cp": float(output_value),
        "dump_output_minus_input_cp": float(output_value - input_value),
        "dump_output_overshoot_ratio": float(
            ((output_crop < 0.0) | (output_crop > 1.0)).float().mean()
        ),
    }


def run_experiment(
    name: str,
    root: Path,
    base_items: list[dict[str, Any]],
    extractor: FrozenCRNNFeatureExtractor,
    device: torch.device,
    dump_root: Path | None = None,
    report_tag: str = "latest",
    manifest_path: Path | None = None,
    eligible_total: int | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    checkpoint = latest_checkpoint(root)
    checkpoint_iter = iteration_of(checkpoint)
    dump_iter = iteration_of(dump_root) if dump_root is not None else None
    print(
        json.dumps(
            {
                "experiment": name,
                "checkpoint": str(checkpoint),
                "checkpoint_iter": checkpoint_iter,
                "dump": str(dump_root) if dump_root is not None else None,
                "dump_iter": dump_iter,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    network = make_network(device)
    load_strict_checkpoint(network, checkpoint, map_location="cpu")

    prediction_yuv: dict[str, torch.Tensor] = {}
    prediction_rgb: dict[str, torch.Tensor] = {}
    for start in range(0, len(base_items), BATCH_SIZE):
        chunk = base_items[start : start + BATCH_SIZE]
        lq = torch.stack([item["lq"] for item in chunk]).to(device)
        with torch.inference_mode():
            output_yuv = network(lq)
            output_rgb = YUV2RGB(
                output_yuv,
                format="RGB",
                is_train=True,
                uv_normalized=True,
            )
        for index, item in enumerate(chunk):
            key = item["key"]
            prediction_yuv[key] = output_yuv[index].detach().cpu()
            prediction_rgb[key] = output_rgb[index].detach().cpu()

    by_width: dict[int, list[dict[str, Any]]] = {}
    for item in base_items:
        by_width.setdefault(item["crnn_width"], []).append(item)
    cp_values: dict[str, tuple[float, float]] = {}
    for _, bucket in sorted(by_width.items()):
        for start in range(0, len(bucket), BATCH_SIZE):
            chunk = bucket[start : start + BATCH_SIZE]
            input_batch = torch.stack([item["input_crnn"] for item in chunk]).to(device)
            target_batch = torch.stack([item["target_crnn"] for item in chunk]).to(device)
            output_batch = torch.stack(
                [resize_valid(prediction_rgb[item["key"]], item["valid_hw"])[0] for item in chunk]
            ).to(device)
            with torch.inference_mode():
                input_loss = feature_loss(extractor, input_batch, target_batch)
                output_loss = feature_loss(extractor, output_batch, target_batch)
            for index, item in enumerate(chunk):
                cp_values[item["key"]] = (
                    float(input_loss[index]),
                    float(output_loss[index]),
                )

    rows: list[dict[str, Any]] = []
    for item in base_items:
        key = item["key"]
        input_cp, output_cp = cp_values[key]
        output_yuv = prediction_yuv[key].numpy()
        target_yuv = item["target_yuv"]
        mask = item["text_mask"] * item["valid_mask"]
        input_text_y_mae = masked_mae(item["input_y"], target_yuv[0], mask)
        output_text_y_mae = masked_mae(output_yuv[0], target_yuv[0], mask)
        input_gradient_mae = masked_mae(
            sobel_magnitude(item["input_y"]),
            sobel_magnitude(target_yuv[0]),
            mask,
        )
        output_gradient_mae = masked_mae(
            sobel_magnitude(output_yuv[0]),
            sobel_magnitude(target_yuv[0]),
            mask,
        )
        resized_output, _ = resize_valid(prediction_rgb[key], item["valid_hw"])
        row: dict[str, Any] = {
            "segment_id": key,
            "transcription": item["transcription"],
            "orientation": item["orientation"],
            "tile_index": item["tile_index"],
            "tile_count": item["tile_count"],
            "valid_hw": item["valid_hw"],
            "crnn_width": item["crnn_width"],
            "text_area": float(mask.sum()),
            "input_cp": input_cp,
            "latest_output_cp": output_cp,
            "output_minus_input_cp": output_cp - input_cp,
            "relative_cp_improvement": (input_cp - output_cp) / max(abs(input_cp), 1e-12),
            "input_text_y_mae": input_text_y_mae,
            "output_text_y_mae": output_text_y_mae,
            "output_minus_input_text_y_mae": output_text_y_mae - input_text_y_mae,
            "input_text_y_gradient_mae": input_gradient_mae,
            "output_text_y_gradient_mae": output_gradient_mae,
            "output_minus_input_text_y_gradient_mae": output_gradient_mae - input_gradient_mae,
            "raw_output_overshoot_ratio": float(
                ((resized_output < 0.0) | (resized_output > 1.0)).float().mean()
            ),
        }
        if dump_root is not None:
            row.update(dump_cp_values(dump_root, extractor, key, device))
        row["category"] = classify(row)
        rows.append(row)

    rows.sort(key=lambda value: value["relative_cp_improvement"])
    categories: dict[str, int] = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    aggregate = {
        "mean_input_cp": float(np.mean([row["input_cp"] for row in rows])),
        "mean_latest_output_cp": float(np.mean([row["latest_output_cp"] for row in rows])),
        "mean_relative_cp_improvement": float(
            np.mean([row["relative_cp_improvement"] for row in rows])
        ),
        "cp_improved_count": sum(row["latest_output_cp"] < row["input_cp"] for row in rows),
        "cp_regressed_count": sum(row["latest_output_cp"] > row["input_cp"] for row in rows),
        "mean_input_text_y_mae": float(np.mean([row["input_text_y_mae"] for row in rows])),
        "mean_latest_output_text_y_mae": float(
            np.mean([row["output_text_y_mae"] for row in rows])
        ),
        "y_improved_count": sum(row["output_minus_input_text_y_mae"] < 0 for row in rows),
        "y_regressed_count": sum(row["output_minus_input_text_y_mae"] > 0 for row in rows),
        "categories": categories,
    }
    report = {
        "experiment": name,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iter,
        "dump_root": str(dump_root) if dump_root is not None else None,
        "dump_iteration": dump_iter,
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "eligible_total": eligible_total,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "device": str(device),
        "eligible_count": len(rows),
        "aggregate": aggregate,
        "records": rows,
    }
    json_path = root / f"cp_{report_tag}_per_sample_analysis_20260818_iter{checkpoint_iter}.json"
    csv_path = root / f"cp_{report_tag}_per_sample_analysis_20260818_iter{checkpoint_iter}.csv"
    markdown_path = root / f"cp_{report_tag}_per_sample_analysis_20260818_iter{checkpoint_iter}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [key for key in rows[0] if key != "valid_hw"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "x".join(map(str, value)) if key == "valid_hw" else value
                    for key, value in row.items()
                    if key in fieldnames
                }
            )

    lines = [
        f"# CP latest per-sample analysis: {name}",
        "",
        f"checkpoint: {checkpoint} (iter {checkpoint_iter})",
        f"dump: {dump_root} (iter {dump_iter})" if dump_root is not None else "dump: not used",
        f"manifest: {manifest_path}" if manifest_path is not None else "",
        f"eligible total: {eligible_total}" if eligible_total is not None else "",
        f"sample size: {sample_size}" if sample_size is not None else "",
        f"sample seed: {sample_seed}" if sample_seed is not None else "",
        f"eligible samples: {len(rows)}",
        f"mean input CP: {aggregate['mean_input_cp']:.6f}",
        f"mean latest output CP: {aggregate['mean_latest_output_cp']:.6f}",
        f"mean relative CP improvement: {aggregate['mean_relative_cp_improvement']:.4%}",
        "",
        "## Categories",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(categories.items()))
    lines.extend(
        [
            "",
            "## Worst CP regressions",
            "",
            "| segment | text | input CP | output CP | relative | input Y MAE | output Y MAE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows[:10]:
        lines.append(
            f"| {row['segment_id']} | {row['transcription']} | "
            f"{row['input_cp']:.3f} | {row['latest_output_cp']:.3f} | "
            f"{row['relative_cp_improvement']:.1%} | "
            f"{row['input_text_y_mae']:.4f} | {row['output_text_y_mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best CP improvements",
            "",
            "| segment | text | input CP | output CP | relative | input Y MAE | output Y MAE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in list(reversed(rows[-10:])):
        lines.append(
            f"| {row['segment_id']} | {row['transcription']} | "
            f"{row['input_cp']:.3f} | {row['latest_output_cp']:.3f} | "
            f"{row['relative_cp_improvement']:.1%} | "
            f"{row['input_text_y_mae']:.4f} | {row['output_text_y_mae']:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment": name,
                "checkpoint_iter": checkpoint_iter,
                "dump_iter": dump_iter,
                "eligible": len(rows),
                "aggregate": aggregate,
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/"
            "basic_text_line_v3_hx250k_common_20260817/segments.jsonl"
        ),
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--experiment-root", type=Path, default=Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments"))
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--no-dump", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device if args.device != "cuda:2" or torch.cuda.is_available() else "cpu")
    roots = {
        "cp": args.experiment_root / "basic_text_line_v3_hx250k_cp_direct_20260817",
        "cp_ygradient": args.experiment_root / "basic_text_line_v3_hx250k_cp_ygradient_direct_20260817",
    }
    extractor = FrozenCRNNFeatureExtractor(
        str(REPO_ROOT / "weights/scene_base_CRNN.pth"),
        str(REPO_ROOT / "realesrgan/metrics/benchmark.txt"),
    ).to(device).eval()
    dataset = BasicTextLineSegmentDatasetV3(
        {
            "manifest": str(args.manifest),
            "canvas_hw": list(CANVAS_HW),
            "min_short_side_ratio": 0.5,
            "min_long_side_ratio": 0.5,
        }
    )
    eligible_indices = [
        index
        for index, record in enumerate(dataset.records)
        if (
            bool(record.get("cp_eligible", False))
            and str(record.get("orientation", "")).upper() in ("HORIZONTAL", "OBLIQUE")
        )
    ]
    eligible_total = len(eligible_indices)
    if args.sample_size < 0:
        raise ValueError("sample-size must be non-negative")
    if args.sample_size and args.sample_size < len(eligible_indices):
        rng = np.random.default_rng(args.seed)
        selected_positions = np.sort(
            rng.choice(len(eligible_indices), size=args.sample_size, replace=False)
        )
        selected_indices = [eligible_indices[int(position)] for position in selected_positions]
    else:
        selected_indices = eligible_indices

    base_items: list[dict[str, Any]] = []
    for index in selected_indices:
        record = dataset.records[index]
        if not (
            bool(record.get("cp_eligible", False))
            and str(record.get("orientation", "")).upper() in ("HORIZONTAL", "OBLIQUE")
        ):
            continue
        item = dataset[index]
        geometry = valid_hw(item)
        input_crnn, crnn_width = resize_valid(item["lq_rgb"], geometry)
        target_crnn, target_width = resize_valid(item["gt_rgb"], geometry)
        if crnn_width != target_width:
            continue
        rgb_weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)[:, None, None]
        base_items.append(
            {
                "key": str(item["segment_id"]),
                "transcription": str(item.get("transcription", "")),
                "orientation": str(item.get("orientation", "")),
                "tile_index": int(item.get("tile_index", 0)),
                "tile_count": int(item.get("tile_count", 1)),
                "valid_hw": geometry,
                "crnn_width": crnn_width,
                "lq": item["lq"],
                "input_crnn": input_crnn,
                "target_crnn": target_crnn,
                "target_yuv": item["gt"].numpy(),
                "input_y": (item["lq_rgb"].numpy() * rgb_weights).sum(axis=0),
                "text_mask": item["text_mask"].numpy()[0],
                "valid_mask": item["valid_mask"].numpy()[0],
            }
        )
    sample_size = len(base_items)
    print(
        f"DEVICE={device} ELIGIBLE_TOTAL={eligible_total} "
        f"SAMPLE_SIZE={sample_size} TOTAL={len(dataset)} SEED={args.seed}",
        flush=True,
    )
    for name, root in roots.items():
        dump_root = None if args.no_dump else latest_dump(root)
        run_experiment(
            name,
            root,
            base_items,
            extractor,
            device,
            dump_root=dump_root,
            report_tag=args.tag,
            manifest_path=args.manifest,
            eligible_total=eligible_total,
            sample_size=sample_size,
            sample_seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
