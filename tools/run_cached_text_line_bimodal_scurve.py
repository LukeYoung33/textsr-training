"""Run the frozen V6 cached, mask-raw bimodal luminance experiment offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.lenovo_stage_a_preprocess_screen import _p10_segment_text, _rgb_to_luma, candidate_parameters, load_config

MODE = "cached_maskraw_bimodal_scurve_transition_v6"
KERNEL_3X3 = np.ones((3, 3), np.uint8)


def _sha256(path: Path) -> str:
    """@luna-comment Return the SHA-256 of one immutable cache or receipt file.

    Input: Existing filesystem path.
    Output: Lowercase hexadecimal SHA-256 digest.
    Key Steps: Read bytes exactly once and hash them.
    Key Params: None.
    State/Dependencies: Filesystem only; input content is not changed.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_rgb(yuv: np.ndarray) -> np.ndarray:
    """@luna-comment Convert normalized cached YUV444 data to normalized RGB.

    Input: Finite [3,H,W] YUV data in the cache's [0,1] representation.
    Output: Float32 RGB [H,W,3] used only by frozen P10 segmentation.
    Key Steps: Quantize to cache display precision then use OpenCV YUV conversion.
    Key Params: 255 is the immutable cache visualization scale.
    State/Dependencies: OpenCV conversion; the source array remains read-only.
    """
    pixels = np.rint(yuv.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(pixels, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0


def _png(path: Path, image: np.ndarray) -> None:
    """@luna-comment Write one already-rendered uint8 debug image.

    Input: Destination path and uint8 OpenCV image.
    Output: PNG artifact, with parent directories created if required.
    Key Steps: Create the narrow debug directory and delegate image encoding to OpenCV.
    Key Params: None.
    State/Dependencies: Writes only the requested experiment artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"failed to write {path}")


def _rgb_png(path: Path, yuv: np.ndarray) -> None:
    """@luna-comment Render a normalized YUV crop as a BGR PNG for review.

    Input: Normalized [3,H,W] cached YUV crop.
    Output: BGR PNG at the requested path.
    Key Steps: Convert YUV to RGB, quantize, convert RGB to BGR, and write.
    Key Params: The cache visualization uses the fixed 8-bit scale.
    State/Dependencies: Uses _to_rgb and _png only.
    """
    rgb = _to_rgb(yuv)
    _png(path, cv2.cvtColor(np.rint(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def _params() -> dict[str, Any]:
    """@luna-comment Load the existing frozen P10-compatible segmentation parameters.

    Input: Repository-local frozen stage-A parameter JSON.
    Output: The named immutable P11 candidate consumed by _p10_segment_text.
    Key Steps: Load config and select the established lambda_0p03 record.
    Key Params: Candidate name is frozen to preserve mask_raw comparability.
    State/Dependencies: Reads options JSON only; no model or OCR is loaded.
    """
    config = load_config(ROOT / "options" / ("lenovo_stage_a_preprocess_screen_p11_kernel_" + "wiener_scurve.json"))
    return next(item for item in candidate_parameters(config, "P11") if item["name"] == "lambda_0p03")


def _rings(mask: np.ndarray, valid: np.ndarray, count: int = 18) -> list[np.ndarray]:
    """@luna-comment Build disjoint outward 3x3 dilation rings inside valid support.

    Input: Raw text mask, valid support, and frozen number of outward rings.
    Output: Disjoint boolean rings 1 through count, all restricted to valid support.
    Key Steps: Dilate the accumulated geometry once per ring and subtract prior support.
    Key Params: V6 uses 18 rings: 1--12 core and 13--18 outer feather.
    State/Dependencies: OpenCV morphology only; neither input geometry is mutated.
    """
    current, output = mask.copy(), []
    for _ in range(count):
        expanded = cv2.dilate(current.astype(np.uint8), KERNEL_3X3).astype(bool) & valid
        output.append(expanded & ~current)
        current = expanded
    return output


def _interior_samples(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """@luna-comment Select 3x3 same-class samples for robust peak estimation.

    Input: One raw P10 class mask and the matching valid support.
    Output: Boolean class-interior samples; callers may fall back to raw class when sparse.
    Key Steps: Erode the class and require all 3x3 neighbours to be valid.
    Key Params: The V6 statistics neighbourhood is fixed at 3x3.
    State/Dependencies: OpenCV morphology only; this output never changes mask geometry.
    """
    return (cv2.erode(mask.astype(np.uint8), KERNEL_3X3).astype(bool) & valid)


def _smoothed_peak(values: np.ndarray) -> tuple[float, np.ndarray]:
    """@luna-comment Estimate a deterministic 256-bin class peak from prediction luminance.

    Input: Finite normalized 1-D luminance samples for one raw-mask class.
    Output: Bin-centre peak and one-pass [1,2,1]/4 smoothed histogram.
    Key Steps: Build a fixed histogram, smooth once, then resolve ties near the class median.
    Key Params: 256 bins and the three-tap kernel are frozen by V6.
    State/Dependencies: Numpy only; values and mask geometry remain unchanged.
    """
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=256, range=(0.0, 1.0))
    smooth = np.convolve(counts.astype(np.float64), np.asarray([1.0, 2.0, 1.0]) / 4.0, mode="same")
    candidates = np.flatnonzero(smooth == smooth.max())
    median_bin = int(np.clip(np.floor(float(np.median(values)) * 255.0), 0, 255))
    index = int(candidates[np.argmin(np.abs(candidates - median_bin))])
    return (index + 0.5) / 256.0, smooth


def _mapping(y: np.ndarray, background: float, foreground: float, valley: float) -> tuple[np.ndarray, dict[str, float]]:
    """@luna-comment Apply V6's continuous monotonic two-peak S curve to prediction Y.

    Input: Normalized prediction luminance and V6 B0/F0/valley statistical anchors.
    Output: Mapped luminance plus frozen B1/F1/D0/D1/tau telemetry.
    Key Steps: Orient dark/light polarity, expand contrast around the original midpoint,
      then compress each valley-side interval with the fixed exponent 1.8 curve.
    Key Params: D1=clip(1.10*D0,.12,.55); gamma=1.8; no mask value enters the map.
    State/Dependencies: Pure numpy; it does not write mask, UV, frame, OCR, or model data.
    """
    sign = 1.0 if foreground > background else -1.0
    delta0 = abs(foreground - background)
    delta1 = float(np.clip(1.10 * delta0, 0.12, 0.55))
    midpoint = (foreground + background) * 0.5
    b1 = float(np.clip(midpoint - sign * delta1 * .5, 0.0, 1.0))
    f1 = float(np.clip(midpoint + sign * delta1 * .5, 0.0, 1.0))
    if abs(f1 - b1) < .10:
        raise ValueError("post_clamp_peak_gap_below_0.10")
    t = np.clip(sign * (y - background) / delta0, 0.0, 1.0)
    tau = float(np.clip(sign * (valley - background) / delta0, (2.0 / 255.0) / delta0, 1.0 - (2.0 / 255.0) / delta0))
    def h(value: np.ndarray) -> np.ndarray:
        power = np.power(np.clip(value, 0.0, 1.0), 1.8)
        inverse = np.power(1.0 - np.clip(value, 0.0, 1.0), 1.8)
        return power / np.maximum(power + inverse, 1.0e-12)
    left = .5 * h(t / tau)
    right = .5 + .5 * h((t - tau) / (1.0 - tau))
    u = np.where(t <= tau, left, right)
    return np.clip(b1 + sign * delta1 * u, 0.0, 1.0).astype(np.float32), {
        "B0": float(background), "T0": float(valley), "F0": float(foreground),
        "B1": b1, "F1": f1, "D0": delta0, "D1": delta1, "tau": tau, "polarity_sign": sign,
    }


def _transition_ratio(y: np.ndarray, background: float, foreground: float) -> float:
    """@luna-comment Measure the V6 normalized middle-band fraction for one sample set.

    Input: Finite normalized luminance samples and ordered-or-reversed endpoint anchors.
    Output: Fraction inside the polarity-normalized .20--.80 transition band.
    Key Steps: Normalize between anchors, then count samples in the fixed open band.
    Key Params: .20/.80 bounds are frozen experiment metrics, not mapping parameters.
    State/Dependencies: Numpy only; no samples are changed.
    """
    sign = 1.0 if foreground > background else -1.0
    normalized = sign * (y - background) / max(abs(foreground - background), 1e-12)
    return float(np.mean((normalized >= .20) & (normalized <= .80)))


def _evidence(valid: np.ndarray) -> dict[str, np.ndarray]:
    """@luna-comment Initialize V6 debug arrays for both processed and skipped lines.

    Input: Valid support shape.
    Output: Complete zero-valued debug evidence dictionary.
    Key Steps: Preallocate all required binary/support/weight images with stable keys.
    Key Params: V6 exposes raw mask, statistics samples, core support, and feather weights.
    State/Dependencies: No external state; callers replace entries after successful analysis.
    """
    empty = np.zeros_like(valid, dtype=bool)
    return {"valid": valid, "raw": empty, "foreground_samples": empty, "background_samples": empty,
            "core": empty, "weight": np.zeros_like(valid, np.float32), "transition_before": empty,
            "transition_after": empty}


def fill_v6(prediction: np.ndarray, frame_crop: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """@luna-comment Produce exactly one V6 cached-line reconstruction from raw P10 geometry.

    Input: Normalized prediction/frame YUV [3,H,W] and boolean valid support.
    Output: Cached-frame based reconstruction, required evidence, and line telemetry.
    Key Steps: Obtain bit-identical raw P10 geometry; estimate class peaks and valley;
      map prediction Y continuously; write core support; feather only rings 13--18 to frame.
    Key Params: 32 samples, .08 peak gap, 1.10 contrast, gamma 1.8, core=12, outer=13..18.
    State/Dependencies: Immutable cached arrays and frozen P10 parameters only; no OCR/model/alpha path.
    """
    if prediction.shape != frame_crop.shape or prediction.ndim != 3 or valid.shape != prediction.shape[1:]:
        raise ValueError("V6 crop/support shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(frame_crop).all():
        raise ValueError("V6 non-finite cache array")
    evidence = _evidence(valid)
    telemetry: dict[str, Any] = {"status": "identity_skip", "valid_pixels": int(valid.sum())}
    raw, meta = _p10_segment_text(_rgb_to_luma(_to_rgb(prediction)), _params(), valid_mask=valid)
    if raw is None or not meta.get("segmentation_accepted", False):
        telemetry["identity_skip_reason"] = "p10_rejected"
        return frame_crop.copy(), evidence, telemetry
    raw = np.asarray(raw, bool)
    low, high, domain = (np.asarray(meta["_p10_low_class"], bool), np.asarray(meta["_p10_high_class"], bool), np.asarray(meta["_p10_valid"], bool))
    if not (np.array_equal(domain, valid) and np.array_equal(low | high, valid) and not (low & high).any() and not (raw & ~valid).any() and raw.any() and (~raw & valid).any()):
        raise ValueError("V6 P10 raw-mask contract failed")
    polarity = str(meta.get("polarity", ""))
    if not (polarity.startswith("dark_text_on_light") or polarity.startswith("light_text_on_dark")):
        telemetry["identity_skip_reason"] = "unknown_polarity"
        return frame_crop.copy(), evidence, telemetry
    foreground, background = raw, (~raw & valid)
    fi, bi = _interior_samples(foreground, valid), _interior_samples(background, valid)
    if fi.sum() < 32:
        fi = foreground.copy()
    if bi.sum() < 32:
        bi = background.copy()
    if fi.sum() < 32 or bi.sum() < 32:
        telemetry["identity_skip_reason"] = "insufficient_class_samples"
        return frame_crop.copy(), evidence, telemetry
    y = prediction[0]
    f0, fhist = _smoothed_peak(y[fi])
    b0, bhist = _smoothed_peak(y[bi])
    if not np.isfinite([f0, b0]).all() or abs(f0 - b0) < .08:
        telemetry["identity_skip_reason"] = "peak_gap_below_minimum"
        return frame_crop.copy(), evidence, telemetry
    lower, upper = sorted((int(np.floor(min(f0, b0) * 255)), int(np.floor(max(f0, b0) * 255))))
    if upper - lower < 3:
        telemetry["identity_skip_reason"] = "invalid_peak_or_valley_statistics"
        return frame_crop.copy(), evidence, telemetry
    joint = fhist + bhist
    valley_window = joint[lower + 1:upper]
    valley_candidates = np.flatnonzero(valley_window == valley_window.min()) + lower + 1
    midpoint_bin = (lower + upper) * .5
    valley_index = int(valley_candidates[np.argmin(np.abs(valley_candidates - midpoint_bin))])
    valley = (valley_index + .5) / 256.0
    try:
        mapped_y, mapping = _mapping(y, b0, f0, valley)
    except ValueError as error:
        telemetry["identity_skip_reason"] = str(error)
        return frame_crop.copy(), evidence, telemetry
    before_values = np.concatenate((y[fi], y[bi]))
    after_values = np.concatenate((mapped_y[fi], mapped_y[bi]))
    before_transition = _transition_ratio(before_values, b0, f0)
    after_transition = _transition_ratio(after_values, mapping["B1"], mapping["F1"])
    concentration_before = float(np.mean(np.abs(y[fi] - f0) <= 2.0 / 255.0) + np.mean(np.abs(y[bi] - b0) <= 2.0 / 255.0)) / 2.0
    concentration_after = float(np.mean(np.abs(mapped_y[fi] - mapping["F1"]) <= 2.0 / 255.0) + np.mean(np.abs(mapped_y[bi] - mapping["B1"]) <= 2.0 / 255.0)) / 2.0
    transition_failed = before_transition > 1e-8 and after_transition > before_transition * .8 + 1e-8
    if concentration_after + 1e-8 < concentration_before or transition_failed:
        telemetry.update({"status": "histogram_reject", "identity_skip_reason": "histogram_metric_reject", **mapping,
                          "peak_concentration_before": concentration_before, "peak_concentration_after": concentration_after,
                          "transition_ratio_before": before_transition, "transition_ratio_after": after_transition})
        return frame_crop.copy(), evidence, telemetry
    rings = _rings(raw, valid)
    core = raw.copy()
    for ring in rings[:12]:
        core |= ring
    output = frame_crop.copy()
    mapped_prediction = np.stack((mapped_y, prediction[1], prediction[2]))
    output[:, core] = mapped_prediction[:, core]
    for index, ring in enumerate(rings[12:], 13):
        weight = ((index - 12) / 6.0) ** 2
        output[:, ring] = (1.0 - weight) * mapped_prediction[:, ring] + weight * frame_crop[:, ring]
        evidence["weight"][ring] = weight
    sign = mapping["polarity_sign"]
    normal_before = sign * (y - b0) / mapping["D0"]
    normal_after = sign * (mapped_y - mapping["B1"]) / mapping["D1"]
    evidence.update({"raw": raw, "foreground_samples": fi, "background_samples": bi, "core": core,
                     "transition_before": valid & (normal_before >= .20) & (normal_before <= .80),
                     "transition_after": valid & (normal_after >= .20) & (normal_after <= .80)})
    telemetry.update({"status": "processed", "polarity": polarity, "raw_mask_pixels": int(raw.sum()),
                      "foreground_stat_pixels": int(fi.sum()), "background_stat_pixels": int(bi.sum()),
                      "core_support_pixels": int(core.sum()), "outer_feather_pixels": int((evidence["weight"] > 0).sum()),
                      "peak_concentration_before": concentration_before, "peak_concentration_after": concentration_after,
                      "transition_ratio_before": before_transition, "transition_ratio_after": after_transition,
                      "ring_pixels": {str(i): int(r.sum()) for i, r in enumerate(rings, 1)},
                      "ring_weights": {str(i): ((i - 12) / 6.0) ** 2 for i in range(13, 19)},
                      "foreground_feather_pixels": 0, "protection_feather_pixels": 0, **mapping})
    return output, evidence, telemetry


def verify_cache(cache_root: Path) -> dict[str, Any]:
    """@luna-comment Verify the immutable V6 line cache before any reconstruction work.

    Input: Cache root containing manifest, receipts, frame arrays, and line artifact hashes.
    Output: Parsed cache manifest, or an explicit hard failure on integrity mismatch.
    Key Steps: Check canonical manifest SHA, receipt presence, each line artifact SHA, shape, and finiteness.
    Key Params: Required artifact names are defined by the frozen V6 manifest.
    State/Dependencies: Reads cache files only; it never builds, repairs, or changes cache content.
    """
    manifest = json.loads((cache_root / "cache_manifest.json").read_text())
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if hashlib.sha256(canonical).hexdigest() != (cache_root / "cache_manifest.sha256").read_text().strip() or not (cache_root / "cache_build_receipt.json").is_file():
        raise ValueError("cache manifest/receipt invalid")
    for frame in manifest["frames"]:
        cached_frame = np.load(cache_root / frame["directory"] / "frame_yuv444.npy")
        if cached_frame.ndim != 3 or cached_frame.shape[0] != 3 or not np.isfinite(cached_frame).all():
            raise ValueError("invalid frame cache array")
        for line in frame["lines"]:
            directory = cache_root / line["directory"]
            for name, receipt in json.loads((directory / "artifact.sha256.json").read_text()).items():
                if _sha256(directory / name) != receipt["sha256"]:
                    raise ValueError(f"cache hash mismatch: {name}")
            prediction, valid = np.load(directory / "prediction_yuv444.npy"), np.load(directory / "valid_support.npy")
            if prediction.ndim != 3 or prediction.shape[0] != 3 or prediction.shape[1:] != valid.shape or not np.isfinite(prediction).all():
                raise ValueError("invalid line cache array")
    return manifest


def _delta_image(delta: np.ndarray) -> np.ndarray:
    """@luna-comment Render a signed luminance delta into a neutral-centred debug PNG.

    Input: Finite scalar delta image.
    Output: uint8 image where 128 is zero and extrema use full display range.
    Key Steps: Normalize by the line's maximum absolute delta and centre at .5.
    Key Params: A small epsilon preserves the all-zero identity display.
    State/Dependencies: Numpy only; delta data is not modified.
    """
    scale = max(float(np.abs(delta).max()), 1e-6)
    return np.rint(np.clip(delta / scale * .5 + .5, 0, 1) * 255).astype(np.uint8)


def _histogram_debug(path: Path, before: np.ndarray, after: np.ndarray, telemetry: dict[str, Any]) -> None:
    """@luna-comment Write a compact before/after combined luminance histogram diagnostic.

    Input: Pre/post luminance vectors and line peak/valley telemetry.
    Output: 512x320 BGR PNG with both curves and B0/T0/F0/B1/F1 guide lines.
    Key Steps: Bin values, draw normalized curves, then overlay statistical anchor lines.
    Key Params: Fixed 256 bins and image extent make lines comparable across all cached rows.
    State/Dependencies: Debug-only OpenCV drawing; no reconstruction data changes.
    """
    canvas = np.full((320, 512, 3), 250, np.uint8)
    for values, color in ((before, (160, 160, 160)), (after, (40, 80, 220))):
        hist, _ = np.histogram(np.clip(values, 0, 1), bins=256, range=(0, 1))
        height = np.rint(hist / max(int(hist.max()), 1) * 250).astype(int)
        points = np.array([[i * 2, 300 - height[i]] for i in range(256)], np.int32)
        cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
    for key, color in (("B0", (0, 160, 0)), ("T0", (0, 0, 0)), ("F0", (0, 160, 0)), ("B1", (0, 0, 200)), ("F1", (0, 0, 200))):
        if key in telemetry:
            x = int(np.clip(float(telemetry[key]) * 511, 0, 511))
            cv2.line(canvas, (x, 0), (x, 300), color, 1)
    _png(path, canvas)


def run(manifest_path: Path, cache_root: Path, output_root: Path) -> None:
    """@luna-comment Execute V6 in ascending cached-line order and preserve complete review artifacts.

    Input: Frozen V6 manifest, immutable cache root, and a fresh output directory.
    Output: Full-frame PNG, per-line diagnostics, telemetry, routes, and execution receipts.
    Key Steps: Verify cache; reconstruct each line; enforce first-write hard ownership; emit required evidence.
    Key Params: Mode and all V6 mathematical parameters are validated from the supplied manifest.
    State/Dependencies: Cache-only execution; no OCR, model inference, historical pasteback, or alpha compositor.
    """
    experiment = json.loads(manifest_path.read_text())
    if experiment.get("authorized_postprocess") != MODE:
        raise ValueError("manifest mode mismatch")
    cache_manifest = verify_cache(cache_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"experiment_id": experiment["experiment_id"], "mode": MODE, "images": []}
    for frame in cache_manifest["frames"]:
        cached_frame = np.load(cache_root / frame["directory"] / "frame_yuv444.npy")
        result, taken, routes = cached_frame.copy(), np.zeros(cached_frame.shape[1:], bool), []
        for line in sorted(frame["lines"], key=lambda item: item["line_index"]):
            line_dir = cache_root / line["directory"]
            prediction = np.load(line_dir / "prediction_yuv444.npy")
            valid = np.load(line_dir / "valid_support.npy").astype(bool)
            left, top, right, bottom = json.loads((line_dir / "geometry.json").read_text())["crop_xyxy"]
            crop = cached_frame[:, top:bottom, left:right]
            reconstructed, evidence, telemetry = fill_v6(prediction, crop, valid)
            prior = taken[top:bottom, left:right]
            write = valid & ~prior
            result[:, top:bottom, left:right][:, write] = reconstructed[:, write]
            taken[top:bottom, left:right][write] = True
            debug = output_root / "debug" / frame["image_id"] / f"line_{line['line_index']:04d}"
            for name, key in (("valid_support.png", "valid"), ("mask_raw.png", "raw"), ("foreground_stat_samples.png", "foreground_samples"), ("background_stat_samples.png", "background_samples"), ("core_support.png", "core"), ("transition_band_before_after.png", "transition_after")):
                _png(debug / name, np.where(evidence[key], 255, 0).astype(np.uint8))
            _png(debug / "outer_feather_weights.png", np.rint(evidence["weight"] * 255).astype(np.uint8))
            _rgb_png(debug / "prediction_before.png", prediction)
            _rgb_png(debug / "cache_frame_crop.png", crop)
            _rgb_png(debug / "reconstructed_line.png", reconstructed)
            _png(debug / "bimodal_y_before_after.png", np.vstack((np.rint(prediction[0] * 255), np.rint(reconstructed[0] * 255))).astype(np.uint8))
            _histogram_debug(debug / "histogram_before_after.png", prediction[0, valid], reconstructed[0, valid], telemetry)
            _png(debug / "delta_y_signed.png", _delta_image(reconstructed[0] - prediction[0]))
            telemetry.update({"line_index": line["line_index"], "overlap_conflict_pixels": int((valid & prior).sum()), "write_pixels": int(write.sum()), "mask_raw_bit_identical": True, "uv_preserved_in_core": bool(np.array_equal(reconstructed[1:, evidence["core"]], prediction[1:, evidence["core"]]))})
            (debug / "line_telemetry.json").write_text(json.dumps(telemetry, ensure_ascii=False, indent=2))
            routes.append(telemetry)
        _rgb_png(output_root / f"{frame['image_id']}.png", result)
        (output_root / f"{frame['image_id']}.routes.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2))
        summary["images"].append({"image_id": frame["image_id"], "lines": len(routes)})
    summary["manifest_sha256"] = _sha256(manifest_path)
    (output_root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_root / "run_receipt.json").write_text(json.dumps({"manifest_sha256": summary["manifest_sha256"], "mode": MODE}, indent=2))
    (output_root / "command.log").write_text("offline V6 cached postprocess; OCR/model/alpha compositor not executed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.manifest, arguments.cache_root, arguments.output_dir)
