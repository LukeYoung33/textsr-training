"""Diagnose whether a training-only stroke probability head matches rendered SR ink.

``p`` is the 1x1-head sigmoid.  ``q(SR)`` is a continuous, GT-anchor-derived
ink probability from the final RGB output.  It is only a paired-validation
diagnostic; neither map is used by deployment inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realesrgan.data  # noqa: F401
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.losses.stroke_geometry_losses import _local_ink_probability
from realesrgan.utils import YUV2RGB
from tools.basic_v2_eval_network import build_eval_network
from tools.dump_basic_v2_validation_rgb_pairs import _load_dataset, _nv21_to_rgb, _safe_component, _yuv444_to_rgb


def _u8(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)


def _rgb(value: np.ndarray, title: str) -> np.ndarray:
    image = cv2.cvtColor(_u8(value), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 180), 22), (0, 0, 0), -1)
    cv2.putText(image, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _map(value: np.ndarray, title: str) -> np.ndarray:
    image = cv2.applyColorMap(_u8(value), cv2.COLORMAP_TURBO)
    cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 180), 22), (0, 0, 0), -1)
    cv2.putText(image, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _metric(p: np.ndarray, q: np.ndarray, text: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    mask = valid > 0.5
    pv, qv, target = p[mask], q[mask], text[mask] > 0.5
    corr = float(np.corrcoef(pv, qv)[0, 1]) if pv.size > 1 and pv.std() > 1e-7 and qv.std() > 1e-7 else float("nan")
    def f1(value: np.ndarray) -> float:
        pred = value >= 0.5
        inter = float(np.logical_and(pred, target).sum())
        return 2.0 * inter / max(float(pred.sum() + target.sum()), 1.0)
    return {"p_q_mae": float(np.abs(pv - qv).mean()), "p_q_corr": corr, "p_f1_gt": f1(pv), "q_f1_gt": f1(qv), "text_fraction": float(target.mean())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-count", type=int, default=12)
    parser.add_argument("--worst-count", type=int, default=12)
    args = parser.parse_args()
    option = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not bool(option["network_g"].get("stroke_prob_enabled", False)):
        raise ValueError("this diagnostic requires a checkpoint/config with stroke_prob_enabled=true")
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = _load_dataset(args.config)
    network, digest = build_eval_network(args.checkpoint, device, args.config)
    records: list[dict] = []
    renders: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            lq_rgb = _nv21_to_rgb(item["lq"])
            lq = rgb_to_y4uv(torch.from_numpy(np.ascontiguousarray(lq_rgb)).permute(2, 0, 1).unsqueeze(0)).to(device)
            output_y4uv, logits = network(lq, return_stroke_logits=True)
            output_rgb_t = YUV2RGB(y4uv_to_yuv444(output_y4uv), format="RGB", is_train=False, uv_normalized=True)
            gt_rgb_t = YUV2RGB(item["gt"].unsqueeze(0).to(device), format="RGB", is_train=False, uv_normalized=True)
            text_t = item["text_mask"].unsqueeze(0).to(device)
            valid_t = item["valid_mask"].unsqueeze(0).to(device)
            q_t, _, _, _, _ = _local_ink_probability(output_rgb_t, gt_rgb_t, text_t, valid_t, anchor_radius=3, min_contrast=0.01, temperature=0.05)
            p = logits.sigmoid()[0, 0].cpu().numpy()
            q = q_t[0, 0].cpu().numpy()
            text, valid = text_t[0, 0].cpu().numpy(), valid_t[0, 0].cpu().numpy()
            row = {"index": index, "line_id": str(item["line_id"]), "source_name": str(item["source_name"]), **_metric(p, q, text, valid)}
            records.append(row)
            renders[index] = (lq_rgb, _yuv444_to_rgb(item["gt"]), _yuv444_to_rgb(y4uv_to_yuv444(output_y4uv)), p, q)
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"[probability-consistency] {index + 1}/{len(dataset)}", flush=True)
    fixed = np.linspace(0, len(records) - 1, min(args.fixed_count, len(records)), dtype=int).tolist()
    worst = sorted(records, key=lambda row: row["p_q_mae"], reverse=True)[:args.worst_count]
    selected = [("fixed", index) for index in fixed] + [("worst_mismatch", int(row["index"])) for row in worst]
    panel_root = args.output_root / "panels"
    panel_root.mkdir(exist_ok=True)
    map_root = args.output_root / "maps"
    for directory in (map_root / "p", map_root / "q_sr", map_root / "abs_p_minus_q", map_root / "raw_npz"):
        directory.mkdir(parents=True, exist_ok=True)
    for rank, (kind, index) in enumerate(selected):
        lq, gt, output, p, q = renders[index]
        panel = np.concatenate((_rgb(lq, "LQ"), _rgb(gt, "GT"), _rgb(output, "SR"), _map(p, "p: head probability"), _map(q, "q(SR): rendered ink"), _map(np.abs(p-q), "|p-q|")), axis=1)
        stem = _safe_component(Path(records[index]["line_id"]).stem, f"sample_{index:04d}")
        filename = f"{kind}_{rank:02d}_{index:04d}_{stem}.png"
        cv2.imwrite(str(panel_root / filename), panel)
        # Standalone visual maps plus exact floating-point arrays for inspection.
        cv2.imwrite(str(map_root / "p" / filename), _map(p, "p: head probability"))
        cv2.imwrite(str(map_root / "q_sr" / filename), _map(q, "q(SR): rendered ink"))
        cv2.imwrite(str(map_root / "abs_p_minus_q" / filename), _map(np.abs(p - q), "|p-q|"))
        np.savez_compressed(map_root / "raw_npz" / filename.replace(".png", ".npz"), p=p, q_sr=q, abs_p_minus_q=np.abs(p - q))
    finite_corr = [row["p_q_corr"] for row in records if np.isfinite(row["p_q_corr"])]
    summary = {"status": "COMPLETE", "sample_count": len(records), "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
               "meaning": {"p": "training-only 1x1-head sigmoid", "q": "GT-anchor-derived continuous ink probability of final rendered SR", "warning": "q is a paired diagnostic, not a deployment mask"},
               "aggregate": {"p_q_mae_mean": float(np.mean([row["p_q_mae"] for row in records])), "p_q_corr_mean": float(np.mean(finite_corr)), "p_f1_gt_mean": float(np.mean([row["p_f1_gt"] for row in records])), "q_f1_gt_mean": float(np.mean([row["q_f1_gt"] for row in records]))},
               "selected": [{"kind": kind, **records[index]} for kind, index in selected]}
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps(summary["aggregate"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
