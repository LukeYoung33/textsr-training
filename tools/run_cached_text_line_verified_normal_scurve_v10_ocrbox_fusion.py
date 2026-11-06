"""Run V10 postprocessing with feathered fusion across overlapping OCR boxes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_cached_text_line_bimodal_scurve as base
from tools import run_cached_text_line_verified_normal_scurve_v10 as v10

MODE = "cached_maskraw_verified_normal_scurve_strong_v10_ocrbox_fusion"


def _blend_weight(valid: np.ndarray) -> np.ndarray:
    """@luna-comment Create an outward-only support-edge ramp for overlap blending.

    Input: Boolean valid support for one cached OCR box. Output: Float alpha in [0,1].
    Key Steps: Measure interior distance and raise a cosine ramp during the first 16 pixels.
    Key Params: The fusion ramp is 16 output pixels and never changes the box prediction itself.
    State/Dependencies: Read-only OpenCV distance transform; used only when multiple boxes overlap.
    """
    distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
    t = np.clip(distance / 16.0, 0.0, 1.0)
    return (.5 - .5 * np.cos(np.pi * t)).astype(np.float32) * valid.astype(np.float32)


def run(manifest_path: Path, cache_root: Path, output_root: Path) -> None:
    """@luna-comment Reconstruct direct OCR boxes and blend only their overlapping valid supports.

    Input: Direct-OCR cache and V10 experiment manifest. Output: Full images, routes, fusion weights, and V10 debug.
    Key Steps: Apply unchanged V10 line reconstruction, accumulate each valid box using a support-edge alpha,
      normalize overlaps, and leave all uncovered source-frame pixels exact. Key Params: 16-pixel cosine edge ramp.
    State/Dependencies: Cache-only operation; OCR and the restoration network are never invoked.
    """
    experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
    if experiment.get("postprocess", {}).get("mode") != MODE:
        raise ValueError("manifest mode mismatch")
    cache = base.verify_cache(cache_root)
    output_root.mkdir(parents=True, exist_ok=False)
    summary = {"experiment_id": experiment["experiment_id"], "mode": MODE, "images": []}
    for frame in cache["frames"]:
        frame_yuv = np.load(cache_root / frame["directory"] / "frame_yuv444.npy")
        accumulation = np.zeros_like(frame_yuv)
        weights = np.zeros(frame_yuv.shape[1:], dtype=np.float32)
        routes = []
        for line in sorted(frame["lines"], key=lambda item: item["line_index"]):
            line_dir = cache_root / line["directory"]
            prediction = np.load(line_dir / "prediction_yuv444.npy")
            valid = np.load(line_dir / "valid_support.npy").astype(bool)
            left, top, right, bottom = json.loads((line_dir / "geometry.json").read_text(encoding="utf-8"))["crop_xyxy"]
            reconstructed, evidence, report = v10.fill_v10(prediction, frame_yuv[:, top:bottom, left:right], valid)
            alpha = _blend_weight(valid)
            accumulation[:, top:bottom, left:right] += reconstructed * alpha[None]
            weights[top:bottom, left:right] += alpha
            debug_dir = output_root / "debug" / frame["image_id"] / f"line_{line['line_index']:04d}"
            v10._write_debug(debug_dir, prediction, frame_yuv[:, top:bottom, left:right], reconstructed, evidence, report)
            report.update({"line_index": line["line_index"], "paste_mode": "support_edge_cosine_fusion", "paste_alpha_pixels": int((alpha > 0).sum())})
            (debug_dir / "line_telemetry.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            routes.append(report)
        result = frame_yuv.copy()
        covered = weights > 1e-6
        result[:, covered] = accumulation[:, covered] / weights[covered][None]
        base._rgb_png(output_root / f"{frame['image_id']}.png", result)
        base._png(output_root / "debug" / frame["image_id"] / "paste_weight_sum.png", np.rint(np.clip(weights, 0, 1) * 255).astype(np.uint8))
        (output_root / f"{frame['image_id']}.routes.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["images"].append({"image_id": frame["image_id"], "lines": len(routes), "overlap_pixels": int((weights > 1.0).sum())})
    summary["manifest_sha256"] = base._sha256(manifest_path)
    (output_root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "run_receipt.json").write_text(json.dumps({"manifest_sha256": summary["manifest_sha256"], "mode": MODE}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.cache_root, args.output_dir)
