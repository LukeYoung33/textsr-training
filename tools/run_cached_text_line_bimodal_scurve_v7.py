"""Run V7: cached bimodal S-curve with histogram diagnostics only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_cached_text_line_bimodal_scurve as base

MODE = "cached_maskraw_bimodal_scurve_transition_v7"
_p10_segment_text = base._p10_segment_text


def fill_v7(prediction: np.ndarray, frame_crop: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """@luna-comment Build one V7 cached-line result without quality-metric fallback.

    Input: Normalized prediction/frame YUV crops [3,H,W] and boolean valid support.
    Output: V7 reconstruction, raw-mask evidence, and telemetry with a histogram label.
    Key Steps: Reproduce V6 hard input gates and S-curve; record pass/warning;
      always write the valid mapped result when those hard gates succeed.
    Key Params: V6's frozen 32-sample, 256-bin, 1.10 contrast, gamma 1.8, and 12/18-ring values.
    State/Dependencies: Cache arrays and frozen P10 parameters only; no OCR, model, or alpha compositor.
    """
    if prediction.shape != frame_crop.shape or prediction.ndim != 3 or valid.shape != prediction.shape[1:]:
        raise ValueError("V7 crop/support shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(frame_crop).all():
        raise ValueError("V7 non-finite cache array")
    evidence, telemetry = base._evidence(valid), {"status": "identity_skip", "valid_pixels": int(valid.sum())}
    raw, meta = _p10_segment_text(base._rgb_to_luma(base._to_rgb(prediction)), base._params(), valid_mask=valid)
    if raw is None or not meta.get("segmentation_accepted", False):
        telemetry["identity_skip_reason"] = "p10_reject_or_raw_mask_contract_failure"
        return frame_crop.copy(), evidence, telemetry
    raw = np.asarray(raw, bool)
    low, high, domain = (np.asarray(meta["_p10_low_class"], bool), np.asarray(meta["_p10_high_class"], bool), np.asarray(meta["_p10_valid"], bool))
    if not (np.array_equal(domain, valid) and np.array_equal(low | high, valid) and not (low & high).any() and not (raw & ~valid).any() and raw.any() and (~raw & valid).any()):
        raise ValueError("V7 raw-mask contract failed")
    polarity = str(meta.get("polarity", ""))
    if not (polarity.startswith("dark_text_on_light") or polarity.startswith("light_text_on_dark")):
        telemetry["identity_skip_reason"] = "unknown_polarity"
        return frame_crop.copy(), evidence, telemetry
    foreground, background = raw, (~raw & valid)
    fi, bi = base._interior_samples(foreground, valid), base._interior_samples(background, valid)
    if fi.sum() < 32:
        fi = foreground.copy()
    if bi.sum() < 32:
        bi = background.copy()
    if fi.sum() < 32 or bi.sum() < 32:
        telemetry["identity_skip_reason"] = "insufficient_class_samples"
        return frame_crop.copy(), evidence, telemetry
    y = prediction[0]
    f0, fhist = base._smoothed_peak(y[fi])
    b0, bhist = base._smoothed_peak(y[bi])
    if not np.isfinite([f0, b0]).all() or abs(f0 - b0) < .08:
        telemetry["identity_skip_reason"] = "peak_gap_below_0.08"
        return frame_crop.copy(), evidence, telemetry
    lower, upper = sorted((int(np.floor(min(f0, b0) * 255)), int(np.floor(max(f0, b0) * 255))))
    if upper - lower < 3:
        telemetry["identity_skip_reason"] = "invalid_peak_or_valley_statistics"
        return frame_crop.copy(), evidence, telemetry
    joint = fhist + bhist
    candidates = np.flatnonzero(joint[lower + 1:upper] == joint[lower + 1:upper].min()) + lower + 1
    valley = (int(candidates[np.argmin(np.abs(candidates - (lower + upper) * .5))]) + .5) / 256.0
    try:
        mapped_y, mapping = base._mapping(y, b0, f0, valley)
    except ValueError as error:
        telemetry["identity_skip_reason"] = str(error)
        return frame_crop.copy(), evidence, telemetry
    if not np.isfinite(mapped_y).all() or np.any(np.diff(base._mapping(np.linspace(0, 1, 1024, dtype=np.float32)[None, :], b0, f0, valley)[0][0]) < -1e-7):
        telemetry["identity_skip_reason"] = "mapping_nonfinite_or_nonmonotonic"
        return frame_crop.copy(), evidence, telemetry
    before_values, after_values = np.concatenate((y[fi], y[bi])), np.concatenate((mapped_y[fi], mapped_y[bi]))
    before_transition = base._transition_ratio(before_values, b0, f0)
    after_transition = base._transition_ratio(after_values, mapping["B1"], mapping["F1"])
    concentration_before = float(np.mean(np.abs(y[fi] - f0) <= 2.0 / 255.0) + np.mean(np.abs(y[bi] - b0) <= 2.0 / 255.0)) / 2.0
    concentration_after = float(np.mean(np.abs(mapped_y[fi] - mapping["F1"]) <= 2.0 / 255.0) + np.mean(np.abs(mapped_y[bi] - mapping["B1"]) <= 2.0 / 255.0)) / 2.0
    histogram_pass = concentration_after + 1e-8 >= concentration_before and (before_transition <= 1e-8 or after_transition <= before_transition * .8 + 1e-8)
    rings = base._rings(raw, valid)
    core = raw.copy()
    for ring in rings[:12]:
        core |= ring
    mapped_prediction = np.stack((mapped_y, prediction[1], prediction[2]))
    output = frame_crop.copy()
    output[:, core] = mapped_prediction[:, core]
    for index, ring in enumerate(rings[12:], 13):
        weight = ((index - 12) / 6.0) ** 2
        output[:, ring] = (1.0 - weight) * mapped_prediction[:, ring] + weight * frame_crop[:, ring]
        evidence["weight"][ring] = weight
    sign = mapping["polarity_sign"]
    evidence.update({"raw": raw, "foreground_samples": fi, "background_samples": bi, "core": core,
                     "transition_before": valid & (sign * (y - b0) / mapping["D0"] >= .20) & (sign * (y - b0) / mapping["D0"] <= .80),
                     "transition_after": valid & (sign * (mapped_y - mapping["B1"]) / mapping["D1"] >= .20) & (sign * (mapped_y - mapping["B1"]) / mapping["D1"] <= .80)})
    telemetry.update({"status": "processed", "histogram_status": "histogram_pass" if histogram_pass else "histogram_warning",
                      "polarity": polarity, "raw_mask_pixels": int(raw.sum()), "foreground_stat_pixels": int(fi.sum()), "background_stat_pixels": int(bi.sum()),
                      "core_support_pixels": int(core.sum()), "outer_feather_pixels": int((evidence["weight"] > 0).sum()),
                      "peak_concentration_before": concentration_before, "peak_concentration_after": concentration_after,
                      "transition_ratio_before": before_transition, "transition_ratio_after": after_transition,
                      "ring_pixels": {str(i): int(r.sum()) for i, r in enumerate(rings, 1)}, "ring_weights": {str(i): ((i - 12) / 6.0) ** 2 for i in range(13, 19)},
                      "foreground_feather_pixels": 0, "protection_feather_pixels": 0, **mapping})
    return output, evidence, telemetry


def run(manifest_path: Path, cache_root: Path, output_root: Path) -> None:
    """@luna-comment Execute V7 in cache-line order and preserve the full review contract.

    Input: Frozen V7 manifest, immutable line cache, and a fresh V7 output directory.
    Output: Full frame, per-line debug/telemetry, route JSON, and cache-only receipts.
    Key Steps: Verify cache; invoke fill_v7; apply ascending first-write ownership; emit V7 artifacts.
    Key Params: The manifest mode binds the run to V7; all numerical parameters reside in fill_v7.
    State/Dependencies: Immutable cache and filesystem output only; never invokes OCR or model inference.
    """
    experiment = json.loads(manifest_path.read_text())
    if experiment.get("authorized_postprocess") != MODE:
        raise ValueError("manifest mode mismatch")
    cache_manifest = base.verify_cache(cache_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"experiment_id": experiment["experiment_id"], "mode": MODE, "images": []}
    for frame in cache_manifest["frames"]:
        cached_frame = np.load(cache_root / frame["directory"] / "frame_yuv444.npy")
        result, taken, routes = cached_frame.copy(), np.zeros(cached_frame.shape[1:], bool), []
        for line in sorted(frame["lines"], key=lambda item: item["line_index"]):
            directory = cache_root / line["directory"]
            prediction, valid = np.load(directory / "prediction_yuv444.npy"), np.load(directory / "valid_support.npy").astype(bool)
            left, top, right, bottom = json.loads((directory / "geometry.json").read_text())["crop_xyxy"]
            reconstructed, evidence, telemetry = fill_v7(prediction, cached_frame[:, top:bottom, left:right], valid)
            prior, write = taken[top:bottom, left:right], valid & ~taken[top:bottom, left:right]
            result[:, top:bottom, left:right][:, write] = reconstructed[:, write]
            taken[top:bottom, left:right][write] = True
            debug = output_root / "debug" / frame["image_id"] / f"line_{line['line_index']:04d}"
            for name, key in (("valid_support.png", "valid"), ("mask_raw.png", "raw"), ("foreground_stat_samples.png", "foreground_samples"), ("background_stat_samples.png", "background_samples"), ("core_support.png", "core"), ("transition_band_before_after.png", "transition_after")):
                base._png(debug / name, np.where(evidence[key], 255, 0).astype(np.uint8))
            base._png(debug / "outer_feather_weights.png", np.rint(evidence["weight"] * 255).astype(np.uint8))
            base._rgb_png(debug / "prediction_before.png", prediction); base._rgb_png(debug / "cache_frame_crop.png", cached_frame[:, top:bottom, left:right]); base._rgb_png(debug / "reconstructed_line.png", reconstructed)
            base._png(debug / "bimodal_y_before_after.png", np.vstack((np.rint(prediction[0] * 255), np.rint(reconstructed[0] * 255))).astype(np.uint8))
            base._histogram_debug(debug / "histogram_before_after.png", prediction[0, valid], reconstructed[0, valid], telemetry)
            base._png(debug / "delta_y_signed.png", base._delta_image(reconstructed[0] - prediction[0]))
            telemetry.update({"line_index": line["line_index"], "overlap_conflict_pixels": int((valid & prior).sum()), "write_pixels": int(write.sum()), "mask_raw_bit_identical": True, "uv_preserved_in_core": bool(np.array_equal(reconstructed[1:, evidence["core"]], prediction[1:, evidence["core"]]))})
            (debug / "line_telemetry.json").write_text(json.dumps(telemetry, ensure_ascii=False, indent=2)); routes.append(telemetry)
        base._rgb_png(output_root / f"{frame['image_id']}.png", result)
        (output_root / f"{frame['image_id']}.routes.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2)); summary["images"].append({"image_id": frame["image_id"], "lines": len(routes)})
    summary["manifest_sha256"] = base._sha256(manifest_path)
    (output_root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_root / "run_receipt.json").write_text(json.dumps({"manifest_sha256": summary["manifest_sha256"], "mode": MODE}, indent=2))
    (output_root / "command.log").write_text("offline V7 cached postprocess; OCR/model/alpha compositor not executed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--cache-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); run(args.manifest, args.cache_root, args.output_dir)
